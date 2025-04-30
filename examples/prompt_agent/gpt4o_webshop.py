import os
import numpy as np
import time
import logging
from datetime import datetime
from agent_system.environments.env_manager import *
from openai import OpenAI

def build_env(env_name, env_num=1):
    group_n = 1
    if env_name == "webshop":
        # Test WebshopEnvironmentManager
        from agent_system.environments.env_package.webshop import webshop_projection
        from agent_system.environments.env_package.webshop import build_webshop_envs
        envs = build_webshop_envs(seed=1, env_num=env_num, group_n=group_n, is_train=False)
        env_manager = WebshopEnvironmentManager(envs, webshop_projection, 'webshop')
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
    
    return env_manager

class Agent:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
        )
        
    def get_action_from_gpt(self, obs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user", 
                    "content": obs
                }
            ],
            temperature=0.4,
            n=1,
            stop=None
        )
        action = response.choices[0].message.content.strip()
        return action

if __name__ == "__main__":
    os.makedirs("logs/webshop", exist_ok=True)
    log_fp = os.path.join(
        "logs/webshop", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    max_steps = 15
    env_num = 128
    test_times = 3
    env_name = "webshop"

    # Accumulated metrics
    overall_success_rates = []
    avg_task_scores = []

    env_manager = build_env(env_name, env_num)
    agent = Agent()

    for test_idx in range(test_times):
        logging.info(f"\n========== Start test {test_idx} ==========")
        start_time = time.time()

        obs, infos = env_manager.reset()
        env_dones = [False] * env_num

        overall_success = np.zeros(env_num, dtype=bool)
        task_scores = np.zeros(env_num, dtype=float)

        for step_idx in range(max_steps):
            logging.info(f"Step {step_idx}")
            actions = [
                "None" if env_dones[i] else agent.get_action_from_gpt(obs["text"][i])
                for i in range(env_num)
            ]

            obs, rewards, dones, infos = env_manager.step(actions)

            for i, done in enumerate(dones):
                if not env_dones[i] and not done:
                    assert rewards[i] == 0
                if not env_dones[i] and done:
                    env_dones[i] = True
                    overall_success[i] = bool(infos[i]["won"])
                    task_scores[i] = float(infos[i]["task_score"])

            if all(env_dones):
                logging.info("All environments finished early!")
                break

        # Round metrics
        success_rate = overall_success.mean()
        mean_score = task_scores.mean()
        overall_success_rates.append(success_rate)
        avg_task_scores.append(mean_score)

        logging.info(f"Test {test_idx} overall success rate: {success_rate:.4f}")
        logging.info(f"Test {test_idx} average task score: {mean_score:.4f}")
        logging.info(f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n")

    # Final Summary
    logging.info("=============== Final Summary ===============")
    logging.info(
        f"Total tests: {test_times} | Envs per test: {env_num} | Total episodes: {env_num * test_times}"
    )
    logging.info(
        f"Overall success avg ± std: {np.mean(overall_success_rates):.4f} ± {np.std(overall_success_rates):.4f}"
    )
    logging.info(
        f"Task score avg ± std: {np.mean(avg_task_scores):.4f} ± {np.std(avg_task_scores):.4f}"
    )

