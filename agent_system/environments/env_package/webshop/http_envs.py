# Copyright 2026 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebShop HTTP 薄客户端 (GPU 训练机侧), 对接 server.py 的集中式环境服务.

与 Ray 版 WebshopMultiProcessEnv 完全同构的向量化 API (step/reset/close +
goal 划分语义), 但每个 env 槽位只是远端服务上的一个会话 (sid) —— 本机不加载
商品数据、不建 Lucene 索引、不起 Ray actor (规避容器 pids.max 限制)。

reward 重映射与 info 增强已在服务端完成 (server.py 是 WebshopWorker 的逐行
等价物), 客户端不做任何语义加工, 纯 HTTP 转发。并发用线程池 (I/O bound)。
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

REQUEST_TIMEOUT_S = 300
RETRIES = 3
RETRY_BACKOFF_S = 2.0


class _HTTPSession:
    """一个远端环境会话: 持有 sid + 线程本地 requests.Session."""

    def __init__(self, base_url: str, token: str, mode: str, seed: int):
        self.base_url = base_url.rstrip('/')
        self.headers = {'X-Token': token}
        self._local = threading.local()
        self.mode = mode
        self.sid = self._post('/session', dict(mode=mode, seed=seed))['sid']

    def _http(self):
        if not hasattr(self._local, 's'):
            self._local.s = requests.Session()
        return self._local.s

    def _post(self, path: str, payload: dict):
        last_err = None
        for attempt in range(RETRIES):
            try:
                r = self._http().post(self.base_url + path, json=payload,
                                      headers=self.headers, timeout=REQUEST_TIMEOUT_S)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001 — 网络抖动统一重试
                last_err = e
                import time
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
        raise RuntimeError(f'webshop server call {path} failed after {RETRIES} tries: {last_err}')

    def reset(self, idx: int):
        out = self._post('/reset', dict(sid=self.sid, idx=int(idx)))
        return out['obs'], out['info']

    def step(self, action: str):
        out = self._post('/step', dict(sid=self.sid, action=action))
        return out['obs'], out['reward'], out['done'], out['info']

    def close(self):
        try:
            self._http().delete(f'{self.base_url}/session/{self.sid}',
                                headers=self.headers, timeout=30)
        except Exception:  # noqa: BLE001 — 服务端 TTL reaper 会兜底
            pass


class WebshopHTTPEnv:
    """向量化 HTTP 客户端, API 与 WebshopMultiProcessEnv 一致.

    刻意不继承 gym.Env: gym 只装在 WebShop 专用环境里, GPU 训练机走 HTTP
    路径时不应引入该依赖 (manager 只按鸭子类型调 step/reset/close)。
    """

    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        server_url: str,
        token: str = 'psgrpo',
        mode: str = 'small',
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1

        self._rng = np.random.RandomState(seed)
        self._pool = ThreadPoolExecutor(max_workers=min(self.num_processes, 64))

        # 与 Ray 版一致: 同组共享 seed (seed + i // group_n)
        def _mk(i):
            return _HTTPSession(server_url, token, mode, seed + (i // group_n))
        self._workers = list(self._pool.map(_mk, range(self.num_processes)))

        r = requests.get(f'{server_url.rstrip("/")}/goals', params=dict(mode=mode),
                         headers={'X-Token': token}, timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        n_goals = len(r.json()['goals'])

        # 目标划分语义与 Ray 版逐行一致
        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, n_goals)
        print(f'[webshop-http] mode={mode} goals={n_goals} idxs={self.goal_idxs}')

    def step(self, actions: list):
        if len(actions) != self.num_processes:
            raise ValueError(f'Expected {self.num_processes} actions, got {len(actions)}')
        results = list(self._pool.map(lambda wa: wa[0].step(wa[1]),
                                      zip(self._workers, actions)))
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()
        results = list(self._pool.map(lambda wi: wi[0].reset(wi[1]),
                                      zip(self._workers, idx)))
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        # getattr 兜底: __init__ 中途抛异常时 __del__ 仍会调用 close
        if getattr(self, '_closed', False) or not hasattr(self, '_workers'):
            return
        list(self._pool.map(lambda w: w.close(), self._workers))
        self._pool.shutdown(wait=False)
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


def build_webshop_http_envs(
    seed: int,
    env_num: int,
    group_n: int,
    server_url: str,
    token: str = 'psgrpo',
    mode: str = 'small',
    is_train: bool = True,
):
    return WebshopHTTPEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        server_url=server_url,
        token=token,
        mode=mode,
        is_train=is_train,
    )
