import numpy as np
import torch
import gym
from RLOR.models.attention_model_wrapper import Agent
from RLOR.wrappers.syncVectorEnvPomo import SyncVectorEnv
from RLOR.wrappers.recordWrapper import RecordEpisodeStatistics
from plot_rlor import plot

if __name__ == "__main__":
    device = 'cpu'
    ckpt_path = './runs/cvrp-v1__exp17_colabT4_50_steps___1__1711303112/ckpt/390.pt'
    agent = Agent(device=device, name='cvrp_fleet').to(device)
    agent.load_state_dict(torch.load(ckpt_path))

    env_id = 'cvrp-v1'
    env_entry_point = 'envs.cvrp_vehfleet_env:CVRPFleetEnv'
    seed = 0

    gym.envs.register(
        id=env_id,
        entry_point=env_entry_point,
    )


    def make_env(env_id, seed, cfg={}):
        def thunk():
            env = gym.make(env_id, **cfg)
            env = RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk


    envs = SyncVectorEnv([make_env(env_id, seed, dict(n_traj=50))])

    # Inference

    trajectories = []
    agent.eval()
    obs = envs.reset()
    done = np.array([False])
    while not done.all():
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logits = agent(obs)
        if trajectories == []:  # Multi-greedy inference
            action = torch.arange(1, envs.n_traj + 1).repeat(1, 1)

        obs, reward, done, info = envs.step(action.cpu().numpy())
        trajectories.append(action.cpu().numpy())

    nodes_coordinates = np.vstack([obs['depot'], obs['observations'][0]])
    final_return = info[0]['episode']['r']
    best_traj = np.argmax(final_return)
    resulting_traj = np.array(trajectories)[:, 0, best_traj]
    resulting_traj_with_depot = np.hstack([np.zeros(1, dtype=int), resulting_traj])

    print(f'A route of length {final_return[best_traj]}')
    print('The route is:\n', resulting_traj_with_depot)

    plot(nodes_coordinates[resulting_traj_with_depot], obs)
