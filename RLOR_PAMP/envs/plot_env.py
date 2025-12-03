import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

from RLOR_PAMP.envs.cvrp_vector_env import CVRPVectorEnv
from RLOR_PAMP.envs.cvrp_vehfleet_env import CVRPFleetEnv


def render(obs,last_nodes, action, color_line):
    action = action[0]
    # Plot nodes
    if obs["action_mask"][0][action] == False: #visited
        plt.scatter(obs["observations"][action-1, 0], obs["observations"][action-1, 1], color='red', s=100,
                    label='Visited' if obs["action_mask"][0][action] == False else "")
        # plt.text(x=obs["nodes"][action, 0], y=obs["nodes"][action, 1], #s=f'{obs["nodes"][action, 2]:.2f}',
        #         color='black', fontsize=14)
    else:
        plt.scatter(obs["observations"][action-1, 0], obs["observations"][action-1, 1], color='green', s=100,
                    label='not visited' if obs["action_mask"][0][action] == True else "")

    # Draw routes
    last_pos = last_nodes[0]
    curr_pos = action
    # if obs["action_mask"][action] == 1:  # If node is visited
    start_pos = obs["depot"] if last_pos == 0 else obs["observations"][last_pos-1]
    end_pos = obs["depot"] if curr_pos == 0 else obs["observations"][curr_pos-1]
    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color=color_line, lw=2)
    # Create an arrow patch
    arrow = FancyArrowPatch(posA=(start_pos[0], start_pos[1]), posB=(end_pos[0], end_pos[1]),
                            arrowstyle='-|>', color=color_line, linewidth=2,
                            mutation_scale=20)  # Adjust mutation_scale for the size of the arrow head
    # Add the arrow patch to the plot
    plt.gca().add_patch(arrow)

    plt.text(x=obs["observations"][action-1, 0] - 0.03, y=obs["observations"][action-1, 1] - 0.03,
             s=f'{obs["current_load"][0]:.2f}', color='green', fontsize=12)

    ''' 
    else:
        plt.scatter(obs["nodes"][action, 0], obs["nodes"][action, 1], color='green', s=100,
                    label='Not Visited' if action == 0 else "")
        plt.text(x=obs["nodes"][action, 0], y=obs["nodes"][action, 1], s=f'{obs["nodes"][action, 2]:.2f}',
                 color='black', fontsize=12)
    '''

    plt.grid(True)
    plt.draw()
    plt.pause(0.2)

def cast_state_to_tensor(state):

    nodes = state["observations"]
    dep = state["depot"]
    demand = state["demand"]

    dep_array = np.expand_dims(dep, axis=0)  # Makes it (1,2)
    dep_demand = np.array([0])
    dep_combined = np.hstack((dep_array, dep_demand.reshape(-1, 1)))  # Reshape for concatenation
    nodes_combined = np.hstack((nodes, demand.reshape(-1, 1)))  # Reshape demand to (50,1) for hstack
    all_nodes = np.vstack((dep_combined, nodes_combined))
    return all_nodes


if __name__ == "__main__":
    env = CVRPFleetEnv()
    init_state = env.reset()

    #nr_veh = env.max_num_vehicles
    all_nodes = cast_state_to_tensor(init_state)

    print("Combined array shape:", all_nodes.shape)
    print("Combined array:\n", all_nodes)


    episode_reward = 0
    terminated = truncated = False

    observations = []
    actions = []
    rewards = []
    state = init_state
    # state = state[0]

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.clear()

    for i in range(1,len(all_nodes)):
        plt.scatter(all_nodes[i, 0], all_nodes[i, 1], color='blue', s=100,)
        plt.text(x=all_nodes[i, 0], y=all_nodes[i, 1], s=f'{all_nodes[i, 2]:.2f}', color='black',
                 fontsize=12)

    plt.scatter(all_nodes[0, 0], all_nodes[0, 1], color='red', s=100)
    colors = ['k', 'b', 'r', 'm', 'y', 'c', 'g']
    color = 'k'
    i = 1
    terminated = False

    while not terminated:
        #action = env.action_space.sample()

        valid_action_indices = [np.where(state["action_mask"][traj] == 1)[0] for traj in range(env.n_traj)]
        random_valid_actions = np.array([np.random.choice(indices) if len(indices) > 0 else None for indices in
                                  valid_action_indices])
        action = random_valid_actions
        print(action)

        last_nodes = state["last_node_idx"].copy()

        print(f'action {action}')
        state, reward, done, info = env.step(action)

        terminated = done[0]
        print(terminated)

        observations.append(state)  # Store observations
        actions.append(action[0])  # Store actions if necessary
        rewards.append(reward[0])

        print(f'remain. vehicle capa: {state["current_load"][0]:.2f} and reward {reward[0]}')
        this_nodes_with_depot_and_demand = cast_state_to_tensor(state)

        # start new color
        render(state,last_nodes, action,  color_line=color)
        if state["current_load"][0] == 1:
            color = colors[i]
            i += 1

        last_nodes = state["last_node_idx"]
        print(f"Action: {action}")

    plt.show(block=True)
    print(f' sum reward {sum(rewards)}')