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

"""WebShop HTTP 环境服务 (集中式 CPU 宿主, 供多台 GPU 训练机远程访问).

架构: 每个模式 (small=1000 商品 / all=118 万商品+human goals) 构建一个全局共享的
SimServer (商品数据 + Lucene 检索只加载一次); 每个训练 env 对应一个会话
(WebAgentTextEnv 传入共享 server, 会话本身零数据拷贝)。

组语义: WebAgentTextEnv.reset(session=idx) 由 idx 确定性选目标 —— 与 Ray 版
WebshopWorker.reset(idx) 完全一致, GRPO 组内同 idx 即同初始状态, 语义原样保留。

step 的 reward 重映射 (done & reward==1.0 -> 10.0, 否则 0) 与 info 增强
(available_actions / task_score / won) 是 WebshopWorker.step 的逐行等价物,
使 GPU 侧客户端可以退化为纯 HTTP 转发。

用法:
    python server.py --port 6006 [--preload small,all] [--token psgrpo]

模式 all 的检索路由到 indexes_full (见 _patched_search_engine; env 包默认
num_products=None 时硬编码 'indexes' 目录, 该目录与 small 数据配套, 不可覆盖)。
"""

import argparse
import os
import sys
import threading
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

WEBSHOP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webshop')
sys.path.insert(0, WEBSHOP_DIR)

DATA_DIR = os.path.join(WEBSHOP_DIR, 'data')
MODES = {
    'small': dict(
        file_path=os.path.join(DATA_DIR, 'items_shuffle_1000.json'),
        attr_path=os.path.join(DATA_DIR, 'items_ins_v2_1000.json'),
        index_dir='indexes',
        human_goals=False,
    ),
    'all': dict(
        file_path=os.path.join(DATA_DIR, 'items_shuffle.json'),
        attr_path=os.path.join(DATA_DIR, 'items_ins_v2.json'),
        index_dir='indexes_full',
        human_goals=True,
    ),
}

SESSION_TTL_S = 2 * 3600

app = FastAPI(title='webshop-env-server')
_sim_servers = {}
_sessions = {}          # sid -> dict(env=, mode=, last=)
_build_lock = threading.Lock()
_session_lock = threading.Lock()
_token = 'psgrpo'


def _get_sim_server(mode: str):
    """构建 (一次) 并返回模式级共享 SimServer。all 模式检索重定向到 indexes_full。"""
    if mode in _sim_servers:
        return _sim_servers[mode]
    with _build_lock:
        if mode in _sim_servers:
            return _sim_servers[mode]
        cfg = MODES[mode]
        import web_agent_site.envs.web_agent_text_env as wste
        from pyserini.search.lucene import LuceneSearcher
        index_path = os.path.join(WEBSHOP_DIR, 'search_engine', cfg['index_dir'])
        orig = wste.init_search_engine
        wste.init_search_engine = lambda num_products=None: LuceneSearcher(index_path)
        try:
            t0 = time.time()
            sim = wste.SimServer(
                0,                        # base seed (目标选择由 reset(session=idx) 确定, 与此无关)
                'http://127.0.0.1:3000',
                cfg['file_path'],
                cfg['attr_path'],
                None,                     # filter_goals
                -1,                       # limit_goals
                None,                     # num_products (数据文件本身已决定规模)
                cfg['human_goals'],
                False,                    # show_attrs
            )
            print(f'[server] mode={mode} loaded in {time.time()-t0:.0f}s, goals={len(sim.goals)}')
        finally:
            wste.init_search_engine = orig
        _sim_servers[mode] = sim
        return sim


def _make_env(mode: str, seed: int, session_prefix: str):
    import web_agent_site.envs.web_agent_text_env as wste
    sim = _get_sim_server(mode)
    # session_prefix 唯一化共享 SimServer 里的 user_sessions key:
    # GRPO 组内多个 env 用同一 goal idx reset —— 无前缀时它们会共用一个
    # 服务端会话字典, 互相覆盖 keywords/page/options (500 + 静默数据污染);
    # 加前缀后 session_id 互异, 而 goal 仍由 reset(session=idx) 的
    # session_int 精确指定 goals[idx], 组语义无损。
    return wste.WebAgentTextEnv(
        observation_mode='text',
        file_path=MODES[mode]['file_path'],
        attr_path=MODES[mode]['attr_path'],
        server=sim,
        seed=seed,
        session_prefix=session_prefix,
    )


def _check(token):
    if token != _token:
        raise HTTPException(status_code=401, detail='bad token')


def _get_session(sid: str):
    s = _sessions.get(sid)
    if s is None:
        raise HTTPException(status_code=404, detail='unknown session')
    s['last'] = time.time()
    return s


class CreateReq(BaseModel):
    mode: str = 'small'
    seed: int = 0


class ResetReq(BaseModel):
    sid: str
    idx: int


class StepReq(BaseModel):
    sid: str
    action: str


@app.get('/health')
def health():
    return dict(status='ok', modes_loaded=list(_sim_servers.keys()),
                sessions=len(_sessions), time=time.time())


@app.post('/session')
def create_session(req: CreateReq, x_token: str = Header(default='')):
    _check(x_token)
    if req.mode not in MODES:
        raise HTTPException(status_code=400, detail=f'mode must be one of {list(MODES)}')
    sid = uuid.uuid4().hex[:16]
    env = _make_env(req.mode, req.seed, session_prefix=f'{sid}-')
    with _session_lock:
        _sessions[sid] = dict(env=env, mode=req.mode, last=time.time())
    return dict(sid=sid)


@app.post('/reset')
def reset(req: ResetReq, x_token: str = Header(default='')):
    _check(x_token)
    s = _get_session(req.sid)
    obs, info = s['env'].reset(session=req.idx)
    info = dict(info or {})
    info['available_actions'] = s['env'].get_available_actions()
    info['won'] = False
    return dict(obs=obs, info=info)


@app.post('/step')
def step(req: StepReq, x_token: str = Header(default='')):
    _check(x_token)
    s = _get_session(req.sid)
    env = s['env']
    try:
        obs, reward, done, info = env.step(req.action)
    except Exception as e:  # noqa: BLE001 — env 契约: 无效动作 = no-op; 内部异常同样降级
        print(f'[server] step error (no-op fallback) action={req.action[:200]!r}: {e!r}')
        obs, reward, done, info = env.observation, 0.0, False, None
    info = dict(info or {})
    info['available_actions'] = env.get_available_actions()
    info['task_score'] = reward
    # WebshopWorker.step 的逐行等价 reward 重映射
    if done and reward == 1.0:
        info['won'] = True
        reward = 10.0
    else:
        info['won'] = False
        reward = 0
    return dict(obs=obs, reward=reward, done=done, info=info)


@app.get('/goals')
def goals(mode: str = 'small', x_token: str = Header(default='')):
    _check(x_token)
    if mode not in MODES:
        raise HTTPException(status_code=400, detail=f'mode must be one of {list(MODES)}')
    return dict(goals=_get_sim_server(mode).goals)


@app.delete('/session/{sid}')
def close_session(sid: str, x_token: str = Header(default='')):
    _check(x_token)
    with _session_lock:
        _sessions.pop(sid, None)
    return dict(ok=True)


def _ttl_reaper():
    while True:
        time.sleep(600)
        cutoff = time.time() - SESSION_TTL_S
        with _session_lock:
            dead = [sid for sid, s in _sessions.items() if s['last'] < cutoff]
            for sid in dead:
                _sessions.pop(sid, None)
        if dead:
            print(f'[server] reaped {len(dead)} idle sessions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--preload', default='small', help='comma list of modes to load at startup')
    parser.add_argument('--token', default='psgrpo')
    args = parser.parse_args()
    _token = args.token

    for m in [m for m in args.preload.split(',') if m]:
        _get_sim_server(m)
    threading.Thread(target=_ttl_reaper, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
