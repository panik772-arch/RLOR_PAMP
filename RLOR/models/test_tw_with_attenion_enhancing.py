from datetime import time

import numpy as np

from RLOR.envs.cvrp_vehfleet_env import CVRPFleetEnv
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from RLOR.models.attention_model_wrapper import Agent, stateWrapper
from RLOR.wrappers.recordWrapper import RecordEpisodeStatistics
from RLOR.wrappers.syncVectorEnvPomo import SyncVectorEnv
import matplotlib.pyplot as plt

'''
def plot_observations_and_distances(nodes, time_difference, observation_from_decoder, deadlines,
                                    current_node_index):
    """
    Plots a figure with 4 subplots:
    1. Original observations of nodes
    2. Distance matrix for all nodes
    3. Observations from the decoder
    4. Distance vector from the current node

    Args:
    - nodes (np.array): Original observations of nodes with shape (N, 2).
    - dist_matrix_all (np.array): Distance matrix for all nodes with shape (N, N).
    - observation_from_decoder (np.array): Observations from the decoder with shape (N, 2).
    - dist_vector_current_node (np.array): Distance vector from the current node with shape (N,).
    - current_node_index (int): Index of the current node.
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 16))  # Create a figure with 4 subplots

    # Plot 1: Original observations
    axs[0, 0].scatter(nodes[:, 0], nodes[:, 1], c='blue')
    for i, (x, y) in enumerate(nodes):
        axs[0, 0].text(x, y, str(i), color='red', fontsize=9)
    #axs[0, 0].scatter(nodes[current_node_index, 0], nodes[current_node_index, 1], color='red', s=20, marker='X',
    #                  label='Current Node')
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 0].set_title('Original Observations')

    # Plot 2: times matrix for all nodes time_windows
    # now calculate the difference between required times and . for small values high relevance.
    axs[0, 1].bar(np.arange(len(time_difference)), time_difference, color='orange')
    axs[0, 1].set_title(f'difference between required times and actual. Small values should have high relevance  {current_node_index}')
    axs[0, 1].set_xlabel('Node Index')
    axs[0, 1].set_ylabel('times')
    axs[0, 1].set_xticks(np.arange(len(time_difference)))
    axs[0, 1].set_xticklabels(np.arange(len(time_difference)), rotation=90)  # Rotate for better readabil

    # Plot 3: Observations from the decoder
    axs[1, 0].scatter(observation_from_decoder[:, 0], observation_from_decoder[:, 1], c='green')
    for i, (x, y) in enumerate(observation_from_decoder):
        axs[1, 0].text(x, y, str(i), color='purple', fontsize=9)
    axs[1, 0].scatter(observation_from_decoder[current_node_index, 0], observation_from_decoder[current_node_index, 1],
                      color='red', s=20, marker='X', label='Current Node')
    axs[1, 0].set_aspect('equal', adjustable='box')
    axs[1, 0].set_title('Observations from Decoder')

    # Plot 4: Bar plot of the distance vector from the current node
    axs[1, 1].bar(np.arange(len(deadlines)), deadlines, color='orange')
    axs[1, 1].set_title(f'deadlines from all nodes ')
    axs[1, 1].set_xlabel('Node Index')
    axs[1, 1].set_ylabel('ratio time/dist')
    axs[1, 1].set_xticks(np.arange(len(deadlines)))
    axs[1, 1].set_xticklabels(np.arange(len(deadlines)), rotation=90)  # Rotate for better readability

    plt.tight_layout()
    plt.show()
'''

def plot_observations_and_logits(observation_from_decoder,logits, time_matrix, deadlines,
                                 current_node_index, already_traveled_time):
    """
    Plots a figure with 4 subplots:
    1. observations from decoder
    2. logits from decoder (attention score without tanh function)
    3. time windows
    4. Distance vector from the current node

    Args:
    - nodes (np.array): Original observations of nodes with shape (N, 2).
    - dist_matrix_all (np.array): Distance matrix for all nodes with shape (N, N).
    - observation_from_decoder (np.array): Observations from the decoder with shape (N, 2).
    - dist_vector_current_node (np.array): Distance vector from the current node with shape (N,).
    - current_node_index (int): Index of the current node.
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 16))  # Create a figure with 4 subplots

    # Plot 1: Observations from the decoder
    axs[0, 0].scatter(observation_from_decoder[:, 0], observation_from_decoder[:, 1], c='green')
    for i, (x, y) in enumerate(observation_from_decoder):
        axs[0, 0].text(x, y, str(i), color='purple', fontsize=9)
    axs[0, 0].scatter(observation_from_decoder[current_node_index, 0], observation_from_decoder[current_node_index, 1],
                      color='red', s=20, marker='X', label='Current Node')
    # Add text label next to the current node
    offset = 0.02  # Adjust offset as needed for clear visibility
    axs[0,0].text(observation_from_decoder[current_node_index, 0] + offset, observation_from_decoder[current_node_index, 1],
                  f'traveled time {already_traveled_time}', color='black',
             fontsize=9)

    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 0].set_title('Observations from Decoder')

    # Plot 2: logit distribution from decoder
    axs[1, 0].bar(np.arange(len(logits)), logits, color='orange')
    axs[1, 0].set_title(f'Logits {current_node_index} computed based on time_ratio')
    axs[1, 0].set_xlabel('Node Index')
    axs[1, 0].set_ylabel('logits from decoder')
    axs[1, 0].set_xticks(np.arange(len(logits)))
    axs[1, 0].set_xticklabels(np.arange(len(logits)), rotation=90)  # Rotate for better readability

    # Plot 3: time windows
    axs[0, 1].bar(np.arange(len(deadlines)), deadlines, color='orange')
    axs[0, 1].set_title(f' deadlines for all nodes')
    axs[0, 1].set_xlabel('Node Index')
    axs[0, 1].set_ylabel('times')
    axs[0, 1].set_xticks(np.arange(len(deadlines)))
    axs[0, 1].set_xticklabels(np.arange(len(deadlines)), rotation=90)  # Rotate for better readability

    # Plot 4:
    axs[1, 1].bar(np.arange(len(time_matrix)), time_matrix, color='orange')
    axs[1, 1].set_title(f' anticipated travel times from this node: {current_node_index}')
    axs[1, 1].set_xlabel('Node Index')
    axs[1, 1].set_ylabel('times')
    axs[1, 1].set_xticks(np.arange(len(time_matrix)))
    axs[1, 1].set_xticklabels(np.arange(len(time_matrix)), rotation=90)  # Rotate for better readability

    plt.tight_layout()
    plt.show()


def plot_dist(nodes, dist_matrix, fig, axs):
    axs[0].scatter(nodes[:, 0], nodes[:, 1], c='blue', label='Nodes')
    for i, (x, y) in enumerate(nodes):
        axs[0].text(x, y, str(i), color='red', fontsize=12)
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].set_title('Node Coordinates with Indexes')
    axs[0].legend()

    # Plotting the distance matrix
    img = axs[1].imshow(dist_matrix, cmap='viridis')
    fig.colorbar(img, ax=axs[1], label='Distance')
    axs[1].set_xticks(np.arange(len(nodes)))
    axs[1].set_yticks(np.arange(len(nodes)))
    axs[1].set_xticklabels(np.arange(len(nodes)))
    axs[1].set_yticklabels(np.arange(len(nodes)))

    axs[1].set_title('Distance Matrix in the original observation')
    axs[1].set_xlabel('Node Index')
    axs[1].set_ylabel('Node Index')


def make_env(env_id, seed, cfg={}):
    def thunk():
        env = gym.make(env_id, **cfg)
        env = RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":

    gym.envs.register(
        id="cvrp_v1",
        entry_point="RLOR.envs.cvrp_vehfleet_tw_env:CVRPFleetTWEnv",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = 50
    num_envs = 10
    n_traj = 1
    total_timesteps = int(num_steps * num_envs)
    learning_rate = 0.01
    K_factor = V = 50
    max_nodes = 10
    time_scale = 10000
    problem = "cvrp_fleet_tw"
    # training env setup
    envs = SyncVectorEnv(
        [make_env("cvrp_v1", 1234 + i, cfg={"n_traj": n_traj, "max_nodes": max_nodes, "region_scale": time_scale}) for i in range(num_envs)])

    agent = Agent(device=device, name=problem, k=K_factor).to(device)
    # agent.backbone.load_state_dict(torch.load('./vrp50.pt'))
    optimizer = optim.Adam(
        agent.parameters(), lr=0.01, eps=1e-5, weight_decay=0
    )

    #######################
    # Algorithm defintion #
    #######################

    # ALGO Logic: Storage setup
    obs = [None] * num_steps
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(
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

        ' TESTE THE state-> Attention Score Enhancement here, because encoder state is basically the stateWrapper Object!'
        nodes = next_obs["observations"][0]
        print(f'one original observation from one trajectory ->  {nodes}')

        # Calculate the distance matrix
        diff = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        ''' 
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        plot_dist(nodes, dist_matrix, fig, axs)
        plt.tight_layout()
        plt.show()
        '''

        next_done = torch.zeros(num_envs, n_traj).to(device)
        r = []
        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, entropy_, value, state_, logits_ = agent.get_action_and_value_for_DAR(
                    next_obs, state=encoder_state
                )  # HERE! Decoder comes into play. State = encoder_state is here!

                ''' 
                This comt out from logits and then categorical (similar as softmax?) 
                 here: class Agent(nn.Module): .........logits = self.actor(x) 
                        probs = torch.distributions.Categorical(logits=logits)
                        Output: probs.probs[0]-> 
                        tensor([[0.0000e+00, 4.5114e-08, 1.6692e-06, 4.5698e-01, 7.8714e-08, 1.2622e-07,
                 1.6670e-07, 5.4301e-01, 2.0428e-07, 3.4246e-06, 6.7524e-08, 4.1635e-08,
                 3.5177e-07, 1.8298e-07, 9.9285e-08, 1.7272e-06, 9.0904e-08, 9.4693e-08,
                 1.2289e-06, 8.3905e-08, 3.7102e-08]])

                 So you see, nevertheless we scaled attention values by sometimes huge distance factors, in the final probs the probs for nodes 3 and 7 (here we set K ==3)
                  to node 3 and 7..
                        '''

                action = action.view(num_envs, n_traj)
                values[step] = value.view(num_envs, n_traj)

            actions[step] = action
            logprobs[step] = logprob.view(num_envs, n_traj)
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())

            '''
            check here within the second batch
            '''
            state = stateWrapper(next_obs, device=device, problem=problem)
            current_node = state.get_current_node()
            observation_from_decoder = state.get_observations_with_depot()

            #TW
            time_windows = state.get_tw()

            ratio, deadlines_requested, already_traveled_time, time_matrix_from_this_node= state.tw_ratio()
            print(f'traveled time so far {already_traveled_time}')

            #dist_matrix = state.calculate_time_distance_matrix()
            ratio_zero = ratio[0][0]

            logits_from_decoder = logits_


            #plot_observations_and_distances(nodes, difference_factor[0][0], observation_from_decoder[0],
            #                                time_windows[0],
            #                                current_node[0])

            plot_observations_and_logits(observation_from_decoder[0],logits_from_decoder[0][0],
                                         time_matrix_from_this_node[0][0],
                                         deadlines_requested[0][0],
                                         current_node[0][0],
                                         already_traveled_time[0][0]
                                         )

            rewards[step] = torch.tensor(reward).to(device)
            next_obs, next_done = next_obs, torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    r.append(item)