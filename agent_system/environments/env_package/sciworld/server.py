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

"""ScienceWorld HTTP 环境服务 (集中式 CPU 宿主, 架构镜像 webshop/server.py).

与 WebShop 的关键差异: 无共享 SimServer —— 每个会话独占一个 ScienceWorldEnv
(即一个 JVM, 实测启动 ~1.2s, E 机 RAM 富余覆盖 64 会话)。因此天然无
W01 事故⑤ 的共享会话字典污染问题; 会话隔离由进程模型保证。

组语义: /reset 传 variation idx —— ScienceWorld 的 variation 确定性决定初始
世界, GRPO 组内同 idx = 同初始状态 (与 webshop reset(session=idx) 同构)。

奖励协议 (prereg 决策 2026-07-30): 终局二值化 —— done 且 score==100 → 10.0,
否则 0; 原始 0-100 密集分数进 info['task_score'] 作旁路指标, 不进奖励通道。

观测增强: step 原始 obs 很简短 ("You move to the kitchen."), 服务端把
look/inv/valid(合法动作)/taskDesc 一并返回, GPU 侧薄客户端零语义加工。

用法:
    python server.py --port 6006 [--token psgrpo] [--step-limit 50]
"""

import argparse
import threading
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

SESSION_TTL_S = 2 * 3600

app = FastAPI(title='sciworld-env-server')
_sessions = {}          # sid -> dict(env=, task=, simplification=, last=)
_session_lock = threading.Lock()
_meta_lock = threading.Lock()
_meta_env = None        # 只读元数据查询 (task 列表/variation 划分) 用的常驻 env
_token = 'psgrpo'
_step_limit = 50


def _get_meta_env():
    global _meta_env
    with _meta_lock:
        if _meta_env is None:
            from scienceworld import ScienceWorldEnv
            _meta_env = ScienceWorldEnv("", envStepLimit=_step_limit)
        return _meta_env


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
    task: str = 'boil'
    simplification: str = ''


class ResetReq(BaseModel):
    sid: str
    idx: int


class StepReq(BaseModel):
    sid: str
    action: str


_ACTION_TEMPLATES_CACHE = None


def _pack_info(env, info, done: bool):
    """统一 info 载荷: look/inv/动作模板/物体表/taskDesc/score + won。

    刻意不回传完整合法组合动作 (数百条 × 每步 × 64 会话 = 带宽爆炸);
    回传 26 个动作模板 (任务无关常量, 进程级缓存) + 当前可指称物体表,
    prompt 侧按 "模板 + OBJ 槽位" 呈现 (SwiftSage 等 SW 智能体惯例)。
    """
    global _ACTION_TEMPLATES_CACHE
    if _ACTION_TEMPLATES_CACHE is None:
        _ACTION_TEMPLATES_CACHE = sorted(env.get_possible_actions())
    info = dict(info or {})
    out = dict(
        look=info.get('look') or env.look(),
        inventory=info.get('inv') or env.inventory(),
        available_actions=_ACTION_TEMPLATES_CACHE,
        possible_objects=sorted(env.get_possible_objects()),
        task_description=info.get('taskDesc', ''),
        task_score=float(info.get('score', 0)),
        variation_idx=info.get('variationIdx', -1),
        won=bool(done and float(info.get('score', 0)) >= 100.0),
    )
    return out


@app.get('/health')
def health():
    return dict(status='ok', sessions=len(_sessions), time=time.time())


@app.get('/tasks')
def tasks(x_token: str = Header(default='')):
    _check(x_token)
    return dict(tasks=_get_meta_env().get_task_names())


@app.get('/variations')
def variations(task: str, x_token: str = Header(default='')):
    _check(x_token)
    env = _get_meta_env()
    with _meta_lock:
        env.load(task, 0, '')
        return dict(train=list(env.get_variations_train()),
                    dev=list(env.get_variations_dev()),
                    test=list(env.get_variations_test()))


@app.post('/session')
def create_session(req: CreateReq, x_token: str = Header(default='')):
    _check(x_token)
    from scienceworld import ScienceWorldEnv
    sid = uuid.uuid4().hex[:16]
    env = ScienceWorldEnv("", envStepLimit=_step_limit)
    env.load(req.task, 0, req.simplification)
    with _session_lock:
        _sessions[sid] = dict(env=env, task=req.task,
                              simplification=req.simplification, last=time.time())
    return dict(sid=sid)


@app.post('/reset')
def reset(req: ResetReq, x_token: str = Header(default='')):
    _check(x_token)
    s = _get_session(req.sid)
    env = s['env']
    env.load(s['task'], int(req.idx), s['simplification'])
    obs, info = env.reset()
    packed = _pack_info(env, info, done=False)
    return dict(obs=obs, info=packed)


@app.post('/step')
def step(req: StepReq, x_token: str = Header(default='')):
    _check(x_token)
    s = _get_session(req.sid)
    env = s['env']
    try:
        obs, _reward, done, info = env.step(req.action)
    except Exception as e:  # noqa: BLE001 — env 契约: 无效动作 = no-op; 内部异常降级
        print(f'[server] step error (no-op fallback) action={req.action[:200]!r}: {e!r}')
        return dict(obs='No known action matches that input.', reward=0.0, done=False,
                    info=_pack_info(env, None, done=False))
    packed = _pack_info(env, info, done=bool(done))
    # prereg 协议: 终局二值化 (score==100 → 10), 密集分数只走旁路 task_score
    reward = 10.0 if packed['won'] else 0.0
    return dict(obs=obs, reward=reward, done=bool(done), info=packed)


@app.delete('/session/{sid}')
def close_session(sid: str, x_token: str = Header(default='')):
    _check(x_token)
    with _session_lock:
        s = _sessions.pop(sid, None)
    if s is not None:
        try:
            s['env'].close()
        except Exception:  # noqa: BLE001
            pass
    return dict(ok=True)


def _ttl_reaper():
    while True:
        time.sleep(600)
        cutoff = time.time() - SESSION_TTL_S
        with _session_lock:
            dead = [sid for sid, s in _sessions.items() if s['last'] < cutoff]
            popped = [_sessions.pop(sid) for sid in dead]
        for s in popped:
            try:
                s['env'].close()
            except Exception:  # noqa: BLE001
                pass
        if dead:
            print(f'[server] reaped {len(dead)} idle sessions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--token', default='psgrpo')
    parser.add_argument('--step-limit', type=int, default=50)
    args = parser.parse_args()
    _token = args.token
    _step_limit = args.step_limit

    _get_meta_env()  # 预热 JVM 一枚, 校验 scienceworld 可用
    threading.Thread(target=_ttl_reaper, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
