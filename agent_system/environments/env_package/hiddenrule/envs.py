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

"""
HiddenRule-Gym 的 ray 并行环境 (docs/hiddenrule_gym_design.md §4)

组环境语义: 同组 (连续 group_n 个 worker) 在 reset 时使用同一 seed
→ 同一布局与同一条隐藏规则 (GRPO/GiGPO 组内对比的前提)。
"""

import numpy as np
import ray

from agent_system.environments.env_package.hiddenrule.hiddenrule import HiddenRuleEnv, HRGConfig


class HiddenRuleWorker:
    """每个 ray actor 持有一个独立的 HiddenRuleEnv 实例"""

    def __init__(self, env_kwargs):
        self.env = HiddenRuleEnv(HRGConfig(**env_kwargs))

    def reset(self, seed):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def admissible_actions(self):
        return self.env.admissible_actions()


class HiddenRuleMultiProcessEnv:
    def __init__(self, seed=0, env_num=1, group_n=1,
                 resources_per_worker=None,
                 is_train=True, env_kwargs=None):
        resources_per_worker = resources_per_worker or {"num_cpus": 0.1}
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self._rng = np.random.default_rng(seed)

        env_kwargs = env_kwargs or {}
        worker_cls = ray.remote(**resources_per_worker)(HiddenRuleWorker)
        self.workers = [worker_cls.remote(env_kwargs) for _ in range(self.num_processes)]

    def reset(self):
        # 训练/评测 seed 段隔离 (与 sokoban 包同约定)
        if self.is_train:
            seeds = self._rng.integers(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = self._rng.integers(2**16, 2**32 - 1, size=self.env_num)
        seeds = np.repeat(seeds, self.group_n).tolist()  # 组内同 seed = 同隐藏规则

        results = ray.get([w.reset.remote(int(s)) for w, s in zip(self.workers, seeds)])
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        return obs_list, info_list

    def step(self, actions):
        assert len(actions) == self.num_processes
        results = ray.get([w.step.remote(a) for w, a in zip(self.workers, actions)])
        obs_list = [r[0] for r in results]
        reward_list = [r[1] for r in results]
        done_list = [r[2] for r in results]
        info_list = [r[3] for r in results]
        return obs_list, reward_list, done_list, info_list

    @property
    def get_admissible_commands(self):
        return ray.get([w.admissible_actions.remote() for w in self.workers])

    def close(self):
        for worker in self.workers:
            ray.kill(worker)


def build_hiddenrule_envs(seed=0, env_num=1, group_n=1,
                          resources_per_worker=None,
                          is_train=True, env_kwargs=None):
    return HiddenRuleMultiProcessEnv(seed, env_num, group_n,
                                     resources_per_worker, is_train, env_kwargs)
