from typing import List

import numpy as np
import pyvrp
import torch
import gym
from RLOR.models.attention_model_wrapper import Agent
from RLOR.wrappers.syncVectorEnvPomo import SyncVectorEnv
from RLOR.wrappers.recordWrapper import RecordEpisodeStatistics
from plot_rlor import plot
from RLOR.envs.cvrp_vehfleet_env import CVRPFleetEnv
from matplotlib import pyplot as plt
from pyvrp import Model, Solution
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution

def calculate_demand(routes, observations):
    total_demand = 0.0
    if isinstance(routes, List):
        for idx, route in enumerate(solution.get_routes(), 1):
            if len(route) != 0:  # Ensure the route is not empty
                customer_demands = np.sum(observations["demand"][0][route]) # Current customer
                total_demand += customer_demands

    else:
        for route in routes.values():
            # Add distance from depot to the first customer
            if len(route) != 0:  # Ensure the route is not empty
                for i in range(len(route)):
                    customer_demands = observations["demand"][0][route[i] - 1]  # Current customer
                    total_demand += customer_demands

    return total_demand


def calculate_route_dist(routes, depot, customers):
    """
        Calculates the total distance of all routes, starting and ending at the depot.

        Parameters:
        - routes (dict): A dictionary where keys are route names (e.g., "Route 1") and values are lists of customer indexes.
        - depot (ndarray): A NumPy array of shape (2,) representing the x and y coordinates of the depot.
        - customers (ndarray): A NumPy array of shape (N, 2) where N is the number of customers,
                               each with their x and y coordinates.

        Returns:
        - float: The total distance covered for all routes.
        """
    total_distance = 0.0
    for route in routes.values():
        # Add distance from depot to the first customer
        if len(route) != 0:  # Ensure the route is not empty
            first_customer_coords = customers[route[0] - 1]  # Adjust for 0-based indexing
            total_distance += np.linalg.norm(depot - first_customer_coords)

            # Sum distances between consecutive customers in the route
            for i in range(1, len(route)):
                customer_coords = customers[route[i] - 1]  # Current customer
                prev_customer_coords = customers[route[i - 1] - 1]  # Previous customer
                total_distance += np.linalg.norm(customer_coords - prev_customer_coords)

            # Add distance from the last customer back to the depot
            last_customer_coords = customers[route[-1] - 1]
            total_distance += np.linalg.norm(depot - last_customer_coords)

    return total_distance



def make_routes(traj):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    # Initialize variables
    routes_dict = {}
    start_index = -1  # To handle the first segment

    # Iterate through the list to find routes
    for i, value in enumerate(traj):
        if value == 0:  # Depot found
            if start_index != -1:  # Not the first depot
                # Save the previous route using 1-based indexing for readability
                route_key = f"Route {len(routes_dict) + 1}"
                routes_dict[route_key] = traj[start_index + 1:i]
            start_index = i

    # Handle the case where there is no trailing zero
    if traj[-1] != 0:
        route_key = f"Route {len(routes_dict) + 1}"
        routes_dict[route_key] = traj[start_index + 1:]

    return routes_dict

def plot_agent_solution(res_traj_with_depot, customers, depot, ax ):

    #customers = obs['observations'][0]
    # traj = [0,1,2,3,0...]

    #num_locs = len() + len(depot.shape)
    x_coords, y_coords = customers.T

    # These are the depots
    kwargs = dict(c="tab:red", marker="*", zorder=3, s=500)
    ax.scatter(
        depot[0],
        depot[1],
        label="Depot",
        **kwargs,
    )
    routes = make_routes(res_traj_with_depot)

    distance = calculate_route_dist(routes, depot, customers)

    for idx, (route_idx, route) in enumerate(routes.items(),1):

        if len(route) == 0:
            continue

        x = x_coords[route-1]
        y = y_coords[route-1]
        # Coordinates of clients served by this route.
        ax.scatter(x, y, label=f"Route {idx}", zorder=3, s=75)
        ax.plot(x, y)


        # Edges from and to the depot, very thinly dashed.
        kwargs = dict(ls=(1, (8, 10)), linewidth=1.0, color="black")
        ax.plot([depot[0], x[0]], [depot[1], y[0]], **kwargs)
        ax.plot([x[-1], depot[0]], [y[-1], depot[1]], **kwargs)


    ax.grid(color="grey", linestyle="solid", linewidth=0.2)
    ax.set_title("Solution RL-Agent")
    ax.set_aspect("equal", "datalim")

    print(f' Total Distance from RL-Solution: {distance}')
    ax.text(0.1, 0.98, f'Total Distance: {distance:.2f}', transform=ax.transAxes,  fontsize=12, color='blue')

    ax.legend(frameon=False, ncol=2)


if __name__ == "__main__":
    device = 'cpu'

    # './runs/cvrp-v1__exp17_colabT4_50_steps___1__1711303112/ckpt/390.pt' -> not working. returns to depot everytime

    env_ = CVRPFleetEnv()

    ckpt_path = "runs/argos_exp3.2/cvrp-v1__exp3.2_vf-argos_cluster_local_runtime__1__1711632522/ckpt/4200.pt"

    agent = Agent(device=device, name='cvrp_fleet').to(device)
    agent.load_state_dict(torch.load(ckpt_path))

    env_id = 'cvrp-v1'
    env_entry_point = 'RLOR.envs.cvrp_vehfleet_env:CVRPFleetEnv'
    seed = 1234

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

    envs = SyncVectorEnv([make_env(env_id, seed, dict(n_traj=100, max_nodes = 100, max_num_vehicles = 10))])
    obs = envs.reset()

    '''
    Find best solution with heuristics
    '''

    scale = 1000000

    m = Model()
    nodes = obs["observations"][0]
    dep = obs["depot"][0]
    demand = obs["demand"][0]
    nr_veh = env_.max_num_vehicles
    print(demand * scale)

    m.add_vehicle_type(nr_veh, capacity=1 * scale)

    print(f'{dep[0]} and {dep[1]}')

    depot = m.add_depot(x=int(dep[0] * scale), y=int(dep[1] * scale))

    clients = [
        m.add_client(x=int(nodes[idx][0] * scale), y=int(nodes[idx][1] * scale), demand=int(demand[idx] * scale),
                     prize=scale, required=False
                     )
        for idx in range(1, len(nodes))
    ]

    locations = [depot] + clients

    for frm in locations:
        for to in locations:
            distance = ((frm.x - to.x) ** 2 + (frm.y - to.y) ** 2) ** 0.5
            m.add_edge(frm, to, distance=distance)

    res = m.solve(stop=MaxRuntime(1))

    print(res)
    print(f'BEST SOLUTION FROM HEURISTIC: {res.cost()/scale}')


    '''
    Do inference
    '''

    # Inference

    trajectories = []
    agent.eval()
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

    x, y = nodes_coordinates[resulting_traj_with_depot].T
    customers = obs['observations'][0]
    depot = obs['depot'][0]

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    plot_solution(res.best, m.data(), plot_clients=True, ax=axs[0],scale = scale)
    plot_agent_solution(resulting_traj_with_depot, customers, depot, ax=axs[1])

    solution = res.best
    demand_in_OR_solution = calculate_demand(res.best.get_routes(), obs)
    routes_ = make_routes(resulting_traj_with_depot)
    demand_in_RL_solution = calculate_demand(routes_, obs)

    print(f' demand collected in OR solution {demand_in_OR_solution}, demand collected in RL solution {demand_in_RL_solution}')
    plt.tight_layout()
    plt.show()



    


