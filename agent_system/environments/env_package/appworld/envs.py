import os
import numpy as np
import ray
import sys

from appworld import AppWorld, load_task_ids, update_root

update_root(os.path.join(os.path.dirname(__file__), "appworld"))

@ray.remote(num_cpus=0.25)
class AppWorldWorker:
    """
    Ray Actor that holds an instance of AppWorld and operates the environment
    based on method calls from the main process.
    """
    def __init__(self, worker_id, max_interactions):
        self.env = None
        self.current_step_count = 0
        self.max_interactions = max_interactions
        self.worker_id = worker_id
        
        self.url_id = 8000 + worker_id
        self.url = f"http://0.0.0.0:{self.url_id}"

    def reset(self, task_id):
        """Reset the environment with a new task."""
        if self.env is not None:
            self.env.close()

        self.current_step_count = 0

        self.env = AppWorld(
            task_id=task_id,
            experiment_name=f'default_{self.worker_id}',
            remote_environment_url=self.url,
        )

        obs = self.env.task.instruction
        info = {
            "task_id": task_id,
            "supervisor": dict(self.env.task.supervisor),
        }
        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        if self.env is None:
            raise RuntimeError("Environment not reset before step. Please call reset() first.")

        self.current_step_count += 1

        obs = self.env.execute(action)

        done = self.env.task_completed() or (self.current_step_count >= self.max_interactions)

        reward = 10.0 if self.env.task_completed() else 0.0

        info = {
            "won": self.env.task_completed(),
            "step_count": self.current_step_count
        }

        return obs, reward, done, info

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()


class AppWorldEnvs:
    """
    A Ray-based distributed wrapper for AppWorld.
    - Creates multiple Ray actors, each holding a separate AppWorld instance.
    - Implements Gym-style interfaces such as step() / reset() / close().
    """
    def __init__(self, 
                 dataset_name,
                 max_interactions,
                 seed,
                 env_num,
                 group_n,
                 start_server_id,
                 ):
        super().__init__()

        self.dataset_name = dataset_name
        self.max_interactions = max_interactions
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.task_ids = load_task_ids(dataset_name)

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create Ray actors (workers)
        self.workers = []
        for i in range(self.num_processes):
            worker = AppWorldWorker.remote(
                worker_id=start_server_id + i,
                max_interactions=self.max_interactions
            )
            self.workers.append(worker)

    def step(self, actions):
        """
        actions: Must be a list with length equal to self.num_processes, 
        each sent to the corresponding worker.
        
        Return format follows Gym's step() convention:
            observations, rewards, dones, infos
        """
        assert len(actions) == self.num_processes, "The length of actions must match the number of processes."

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Reset all worker environments simultaneously, 
        returning each environment's initial observation and info.
        """
        # randomly select self.env_num task_id from self.task_ids
        task_id = np.random.choice(self.task_ids, self.env_num, replace=False)
        # repeat task_id group_n times
        task_id = np.repeat(task_id, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(task_id[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        
        obs_list = []
        info_list = []

        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def close(self):
        """Close all workers."""
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)
        
        # Wait for all workers to close
        ray.get(futures)
        
        # Shutdown Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def render(self):
        """Implement this if visualization is needed."""
        pass

def build_appworld_envs(dataset_name="train",
                        max_interactions=50,
                        seed=0,
                        env_num=1, 
                        group_n=1,
                        start_server_id=0,
                        ):

    return AppWorldEnvs(
        dataset_name=dataset_name,
        max_interactions=max_interactions,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        start_server_id=start_server_id,
    )