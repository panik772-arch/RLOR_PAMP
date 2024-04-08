from datetime import time

import numpy as np

from RLOR.envs.cvrp_vehfleet_env import CVRPFleetEnv
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from RLOR.models.attention_model_wrapper import Agent
from RLOR.wrappers.recordWrapper import RecordEpisodeStatistics
from RLOR.wrappers.syncVectorEnvPomo import SyncVectorEnv


def make_env(env_id, seed, cfg={}):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__" :

    gym.envs.register(
        id="cvrp_v1",
        entry_point= "RLOR.envs.cvrp_vehfleet_env:CVRPFleetEnv",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training env setup
    envs = SyncVectorEnv([make_env("cvrp_v1", 1234 + i) for i in range(1000)])
    agent = Agent(device=device, name="cvrp_fleet").to(device)
    # agent.backbone.load_state_dict(torch.load('./vrp50.pt'))
    optimizer = optim.Adam(
        agent.parameters(), lr=0.01, eps=1e-5, weight_decay=0
    )

    #######################
    # Algorithm defintion #
    #######################
    num_steps = 100
    num_envs = 1024
    n_traj = 35
    total_timesteps = 600000
    learning_rate = 0.01

    # ALGO Logic: Storage setup
    obs = [None] * 100
    actions = torch.zeros((100, 1024) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((num_steps, num_envs, n_traj)).to(device)
    rewards = torch.zeros((num_steps, num_envs, n_traj)).to(device)
    dones = torch.zeros((num_steps, num_envs, n_traj)).to(device)
    values = torch.zeros((num_steps, num_envs, n_traj)).to(device)

    batch_size = int(num_envs * num_steps)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time()
    next_obs = envs.reset()
    next_done = torch.zeros(num_envs, n_traj).to(device)
    num_updates = total_timesteps // batch_size
    for update in range(1, num_updates + 1):
        agent.train()

        next_obs = envs.reset()
        encoder_state = agent.backbone.encode(next_obs)
        next_done = torch.zeros(num_envs, n_traj).to(device)
        r = []
        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, _ = agent.get_action_and_value_cached(
                    next_obs, state=encoder_state
                )
                action = action.view(num_envs, n_traj)
                values[step] = value.view(num_envs, n_traj)
            actions[step] = action
            logprobs[step] = logprob.view(num_envs, n_traj)
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device)
            next_obs, next_done = next_obs, torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    r.append(item)