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
HRG-e 离线探针 harness (P01): 对训好的 ckpt 回放 episode 并注入信念探针。

用法 (需 GPU; verl FSDP 分片 ckpt 先合并成 HF 目录, 见 verl 的 model_merger):
    python -m examples.hiddenrule.probe_checkpoint \
        --model_path /path/to/hf_model_dir \
        --n_episodes 16 --probe_every 5 --seed_base 9000 \
        --out probe_results.json

流程: 用被测策略生成动作驱动 HiddenRuleEnv; 每 probe_every 步暂停, 以同一
历史上下文向策略提出 generate_probes 的问题 (独立查询, 不进 episode),
score_answer 判分。输出按 (probe kind × rule family × 相对因果点时序) 聚合
—— "预测/信念得分骤降点 vs causal_turns" 即归因偏差图的数据源。

`--policy random` 提供无模型的下界基线 (探针答案随机 yes/no / 随机状态词)。
"""

import argparse
import json
import random
from collections import defaultdict

from agent_system.environments.env_package.hiddenrule.hiddenrule.env import HiddenRuleEnv
from agent_system.environments.env_package.hiddenrule.hiddenrule.probes import (
    audit_no_leakage,
    generate_probes,
    score_answer,
)
from agent_system.environments.env_package.hiddenrule.hiddenrule.world import HRGConfig


def _build_history_prompt(history, obs, question):
    lines = [
        "You are an expert agent operating in the HiddenRule Environment: several rooms "
        "contain levers, dials, buttons and notes. A hidden mechanism controls the vault.",
        "Here is what you have observed and done so far:",
    ]
    for t, (o, a) in enumerate(history, start=1):
        lines.append(f"Observation {t}: {o}")
        lines.append(f"Action {t}: {a}")
    lines.append(f"Current observation: {obs}")
    lines.append(f"Question: {question}")
    lines.append("Answer briefly.")
    return "\n".join(lines)


def make_model_generate_fn(model_path: str, max_new_tokens: int = 32):
    """transformers 后端 (延迟导入; verl 分片 ckpt 需先合并为 HF 目录)"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                 device_map='auto')

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors='pt').to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

    return generate


def run_probing(generate_fn, action_fn, n_episodes=16, probe_every=5,
                seed_base=9000, config=None, rng_seed=0):
    """
    generate_fn(prompt)->text: 探针问答后端;
    action_fn(obs, admissible, history)->action: 驱动 episode 的策略
        (可以同为模型, 也可以是随机策略)。
    返回逐探针记录列表 (含 episode/turn/因果标注), 供聚合分析。
    """
    config = config or HRGConfig()
    rng = random.Random(rng_seed)
    records = []
    for ep in range(n_episodes):
        env = HiddenRuleEnv(config)
        obs, info = env.reset(seed=seed_base + ep)
        audit_no_leakage(generate_probes(env.world, env.state), env.world)
        history = []
        done = False
        turn = 0
        while not done and turn < config.max_steps:
            if turn % probe_every == 0:
                for probe in generate_probes(env.world, env.state):
                    prompt = _build_history_prompt(history, obs, probe['question'])
                    reply = generate_fn(prompt)
                    records.append({
                        'episode': ep,
                        'turn': turn,
                        'kind': probe['kind'],
                        'family': env.world.rule.family,
                        'score': score_answer(probe['kind'], reply, probe['answer']),
                        'causal_turns': dict(info['causal_turns']),
                    })
            admissible = env.admissible_actions()
            action = action_fn(obs, admissible, history)
            new_obs, reward, done, info = env.step(action)
            history.append((obs, action))
            obs = new_obs
            turn += 1
    return records


def aggregate(records):
    agg = defaultdict(lambda: {'n': 0, 'sum': 0.0})
    for r in records:
        key = f"{r['family']}/{r['kind']}"
        agg[key]['n'] += 1
        agg[key]['sum'] += r['score']
    return {k: round(v['sum'] / v['n'], 4) for k, v in agg.items() if v['n'] > 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None,
                        help='HF 模型目录 (verl 分片 ckpt 先合并); 缺省 = random 基线')
    parser.add_argument('--n_episodes', type=int, default=16)
    parser.add_argument('--probe_every', type=int, default=5)
    parser.add_argument('--seed_base', type=int, default=9000)
    parser.add_argument('--out', default='probe_results.json')
    args = parser.parse_args()

    rng = random.Random(0)
    if args.model_path:
        generate_fn = make_model_generate_fn(args.model_path)
        action_fn = lambda obs, admissible, history: generate_fn(
            _build_history_prompt(history, obs,
                                  f"Choose one admissible action from: {admissible}. "
                                  "Reply with the action only.")).strip().split('\n')[0]
    else:
        generate_fn = lambda prompt: rng.choice(['yes', 'no', 'up', 'down', '0', '1'])
        action_fn = lambda obs, admissible, history: rng.choice(admissible)

    records = run_probing(generate_fn, action_fn, n_episodes=args.n_episodes,
                          probe_every=args.probe_every, seed_base=args.seed_base)
    summary = aggregate(records)
    with open(args.out, 'w') as f:
        json.dump({'summary': summary, 'records': records}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
