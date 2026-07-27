"""WebshopHTTPEnv 薄客户端单元测试: 本地假服务器, 不依赖外网/gym/web_agent_site."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_system.environments.env_package.webshop.http_envs import build_webshop_http_envs

TOKEN = 'testtok'
N_GOALS = 620


class _FakeWebshopHandler(BaseHTTPRequestHandler):
    sessions = {}          # sid -> dict(seed=, last_idx=)
    lock = threading.Lock()
    counter = [0]

    def log_message(self, *args):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _authed(self):
        if self.headers.get('X-Token') != TOKEN:
            self._json(401, dict(detail='bad token'))
            return False
        return True

    def do_GET(self):
        if self.path.startswith('/goals'):
            if not self._authed():
                return
            self._json(200, dict(goals=[dict(idx=i) for i in range(N_GOALS)]))
        else:
            self._json(404, dict(detail='nope'))

    def do_POST(self):
        if not self._authed():
            return
        req = self._body()
        if self.path == '/session':
            with self.lock:
                self.counter[0] += 1
                sid = f'sid{self.counter[0]}'
                self.sessions[sid] = dict(seed=req['seed'])
            self._json(200, dict(sid=sid))
        elif self.path == '/reset':
            s = self.sessions[req['sid']]
            s['last_idx'] = req['idx']
            # obs 编码 idx, 供组语义断言 (同 idx -> 同 obs)
            self._json(200, dict(obs=f"obs-goal-{req['idx']}",
                                 info=dict(available_actions={'has_search_bar': True}, won=False)))
        elif self.path == '/step':
            s = self.sessions[req['sid']]
            done = req['action'] == 'click[buy now]'
            self._json(200, dict(
                obs=f"after-{req['action']}-goal-{s['last_idx']}",
                reward=10.0 if done else 0,
                done=done,
                info=dict(available_actions={'has_search_bar': False}, task_score=1.0 if done else 0.0, won=done),
            ))
        else:
            self._json(404, dict(detail='nope'))

    def do_DELETE(self):
        if not self._authed():
            return
        sid = self.path.rsplit('/', 1)[-1]
        with self.lock:
            self.sessions.pop(sid, None)
        self._json(200, dict(ok=True))


@pytest.fixture(scope='module')
def fake_server():
    srv = ThreadingHTTPServer(('127.0.0.1', 0), _FakeWebshopHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f'http://127.0.0.1:{srv.server_address[1]}'
    srv.shutdown()


def test_group_semantics_and_goal_split(fake_server):
    env = build_webshop_http_envs(seed=3, env_num=2, group_n=2, server_url=fake_server,
                                  token=TOKEN, mode='small', is_train=True)
    assert env.goal_idxs == range(500, N_GOALS)
    obs, infos = env.reset()
    assert len(obs) == 4
    assert obs[0] == obs[1] and obs[2] == obs[3] and obs[0] != obs[2]
    assert all('available_actions' in i for i in infos)
    env.close()


def test_val_split_and_group1(fake_server):
    env = build_webshop_http_envs(seed=1003, env_num=3, group_n=1, server_url=fake_server,
                                  token=TOKEN, mode='small', is_train=False)
    assert env.goal_idxs == range(500)
    obs, _ = env.reset()
    assert len(obs) == 3
    env.close()


def test_step_passthrough(fake_server):
    env = build_webshop_http_envs(seed=5, env_num=1, group_n=2, server_url=fake_server,
                                  token=TOKEN, mode='small', is_train=True)
    env.reset()
    obs, rewards, dones, infos = env.step(['search[shoes]', 'click[buy now]'])
    assert rewards == [0, 10.0] and dones == [False, True]
    assert infos[1]['won'] is True and infos[0]['won'] is False
    assert obs[0].startswith('after-search[shoes]')
    env.close()


def test_bad_token_raises(fake_server):
    with pytest.raises(RuntimeError, match='failed after'):
        build_webshop_http_envs(seed=0, env_num=1, group_n=1, server_url=fake_server,
                                token='wrong', mode='small', is_train=True)


def test_action_count_mismatch(fake_server):
    env = build_webshop_http_envs(seed=9, env_num=1, group_n=2, server_url=fake_server,
                                  token=TOKEN, mode='small', is_train=True)
    with pytest.raises(ValueError):
        env.step(['only-one'])
    env.close()
