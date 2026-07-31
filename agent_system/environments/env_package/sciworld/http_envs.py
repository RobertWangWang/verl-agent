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

"""ScienceWorld HTTP 薄客户端 (GPU 训练机侧), 对接 sciworld/server.py.

与 webshop/http_envs.py 完全同构: 每个 env 槽位 = 远端一个会话 (独占 JVM),
本机不装 Java、不起 Ray actor。纯 HTTP 转发, 语义都在服务端。

split 语义 (prereg 决策): 用 ScienceWorld 原生 variation 划分 ——
train 从 get_variations_train 采样, eval 用 get_variations_test (dev 可选)。
组语义: 同组共享 variation idx (np.repeat), 与 WebShop/ALFWorld 同构。
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

REQUEST_TIMEOUT_S = 300
RETRIES = 12
RETRY_BACKOFF_S = 2.0
RETRY_BACKOFF_CAP_S = 15.0


class _HTTPSession:
    """一个远端环境会话: 持有 sid + 线程本地 requests.Session."""

    def __init__(self, base_url: str, token: str, task: str, simplification: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {'X-Token': token}
        self._local = threading.local()
        self.sid = self._post('/session', dict(task=task, simplification=simplification))['sid']

    def _http(self):
        if not hasattr(self._local, 's'):
            self._local.s = requests.Session()
        return self._local.s

    def _post(self, path: str, payload: dict):
        last_err, last_body = None, ''
        for attempt in range(RETRIES):
            try:
                r = self._http().post(self.base_url + path, json=payload,
                                      headers=self.headers, timeout=REQUEST_TIMEOUT_S)
                if r.status_code >= 400:
                    last_body = r.text[:300]
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001 — 网络抖动统一重试
                last_err = e
                time.sleep(min(RETRY_BACKOFF_S * (attempt + 1), RETRY_BACKOFF_CAP_S))
        raise RuntimeError(f'sciworld server call {path} failed after {RETRIES} tries: '
                           f'{last_err}; body={last_body}; payload={str(payload)[:300]}')

    def reset(self, idx: int):
        out = self._post('/reset', dict(sid=self.sid, idx=int(idx)))
        self._last_obs, self._last_info = out['obs'], out['info']
        return out['obs'], out['info']

    def step(self, action: str):
        try:
            out = self._post('/step', dict(sid=self.sid, action=action))
        except RuntimeError as e:
            if getattr(self, '_last_obs', None) is None:
                raise
            print(f'[sciworld-http] WARN step fallback (no-op): {e}')
            return self._last_obs, 0, False, dict(self._last_info or {}, step_fallback=True)
        self._last_obs, self._last_info = out['obs'], out['info']
        return out['obs'], out['reward'], out['done'], out['info']

    def close(self):
        try:
            self._http().delete(f'{self.base_url}/session/{self.sid}',
                                headers=self.headers, timeout=30)
        except Exception:  # noqa: BLE001 — 服务端 TTL reaper 兜底
            pass


class SciWorldHTTPEnv:
    """向量化 HTTP 客户端, API 与 WebshopHTTPEnv 一致 (manager 鸭子类型)."""

    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        server_url: str,
        token: str = 'psgrpo',
        task: str = 'boil',
        simplification: str = '',
        split: str = 'train',
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

        def _mk(_i):
            return _HTTPSession(server_url, token, task, simplification)
        self._workers = list(self._pool.map(_mk, range(self.num_processes)))

        r = requests.get(f'{server_url.rstrip("/")}/variations', params=dict(task=task),
                         headers={'X-Token': token}, timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        splits = r.json()
        key = split if split in splits else ('train' if is_train else 'test')
        self.variation_idxs = list(splits[key])
        if not self.variation_idxs:
            raise ValueError(f'sciworld task={task} split={key} has no variations')
        self._sample_replace = len(self.variation_idxs) < self.env_num
        if self._sample_replace:
            print(f'[sciworld-http] WARN task={task} split={key} has only '
                  f'{len(self.variation_idxs)} variations < env_num={self.env_num}; '
                  f'sampling WITH replacement')
        print(f'[sciworld-http] task={task} split={key} '
              f'variations={len(self.variation_idxs)}')

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
        idx = self._rng.choice(self.variation_idxs, size=self.env_num,
                               replace=self._sample_replace)
        idx = np.repeat(idx, self.group_n).tolist()
        results = list(self._pool.map(lambda wi: wi[0].reset(wi[1]),
                                      zip(self._workers, idx)))
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def close(self):
        if getattr(self, '_closed', False) or not hasattr(self, '_workers'):
            return
        list(self._pool.map(lambda w: w.close(), self._workers))
        self._pool.shutdown(wait=False)
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


def build_sciworld_http_envs(
    seed: int,
    env_num: int,
    group_n: int,
    server_url: str,
    token: str = 'psgrpo',
    task: str = 'boil',
    simplification: str = '',
    split: str = 'train',
    is_train: bool = True,
):
    return SciWorldHTTPEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        server_url=server_url,
        token=token,
        task=task,
        simplification=simplification,
        split=split,
        is_train=is_train,
    )
