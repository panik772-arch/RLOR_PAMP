# Validate vrp with tw
# you need first generate a small problem with 20 customers
# try to manuelle draw the routes our apply som small script for optimization (pyvrp with tw from 0 to max)
# then plot this
import numpy as np
from matplotlib import pyplot as plt
from pyvrp import Model
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution

from RLOR.envs.cvrp_vehfleet_tw_env import CVRPFleetTWEnv
from RLOR.envs.vrp_data_with_tw import VRPDatasetTW
import torch
import gym
from RLOR.models.attention_model_wrapper import Agent
from RLOR.wrappers.syncVectorEnvPomo import SyncVectorEnv
from RLOR.wrappers.recordWrapper import RecordEpisodeStatistics
from test_and_plot_trained_model import plot_agent_solution


def genrate_duration_matrix(nodes_with_depo):
    num_nodes = nodes_with_depo.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(nodes_with_depo[i] - nodes_with_depo[j])
            else:
                distance_matrix[i, j] = 0  # Distance from node to itself is zero

    return distance_matrix

def make_env():
    env_ = CVRPFleetTWEnv()

    env_id = 'cvrp-v1'
    env_entry_point = 'RLOR.envs.cvrp_vehfleet_tw_env:CVRPFleetTWEnv'

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

    vehicles = 5
    seed = 3214
    # if its 2000 (min) we visit customers occasionally. Only min distance is matter. So, the vehicle trys to minimize the toures and results in shorter distance and less tours
    # if 10 000 is chosen, the vehicle trys to collect as many as possible and so, results in longer tours
    envs = SyncVectorEnv(
        [make_env(env_id, seed, dict(n_traj=50, max_nodes=50, max_num_vehicles=vehicles, penalty=10))])
    obs = envs.reset()

    return obs, envs


if __name__ == "__main__":
    max_nodes = 50
    eval_data = False
    eval_partition = "eval"
    eval_data_idx = 0
    region_scale = 10000
    vehicles = 5
    v = 50 #speed in km/h
    prize_for_visiting = 5000 #2000 min and 10000 max

    obs, envs = make_env()
    nodes = obs["observations"][0]
    dep = obs["depot"][0]
    demand = obs["demand"][0]
    tw = obs["tw"][0]
    print(demand * region_scale)

    #data = VRPDatasetTW[eval_partition, max_nodes, eval_data_idx, region_scale]
    #nodes =  data["loc"]
    #dep = data["depot"]
    #demand = data["demand"]
    #tw = data["tw"]
    print(tw)

    nodes_with_depo = np.concatenate((dep[None, ...], nodes))

    dist_matrix = genrate_duration_matrix(nodes_with_depo)
    DURATION_MATRIX = ( (dist_matrix * region_scale) / (v * region_scale) ) * 3600 #convert to duration in seconds according to tw
    print(f'{dep[0]} and {dep[1]}')

    m = Model()
    m.add_vehicle_type(vehicles, capacity=1 * region_scale)


    TIME_WINDOWS = [(0, int(tw[idx])) for idx in range(nodes.shape[0])]
    TIME_WINDOWS_depot = [0, 10000]

    depot = m.add_depot(x=int(dep[0] * region_scale),
                        y=int(dep[1] * region_scale),
                        tw_early=TIME_WINDOWS_depot[0],
                        tw_late=TIME_WINDOWS_depot[1],
                        )

    clients = [
        m.add_client(x=int(nodes[idx][0] * region_scale),
                     y=int(nodes[idx][1] * region_scale),
                     demand=int(demand[idx] * region_scale),
                     tw_early=TIME_WINDOWS[idx][0],
                     tw_late=TIME_WINDOWS[idx][1],
                     prize= prize_for_visiting,
                     required=False
                     )
        for idx in range(0, len(nodes))
    ]

    locations = [depot] + clients
    for frm_idx, frm in enumerate(locations):
        for to_idx, to in enumerate(locations):
            distance = ((frm.x - to.x) ** 2 + (frm.y - to.y) ** 2) ** 0.5
            duration = DURATION_MATRIX[frm_idx][to_idx]
            m.add_edge(frm, to, distance=distance,duration=duration)

    res = m.solve(stop=MaxRuntime(1))

    print(res)
    print(f'BEST SOLUTION FROM HEURISTIC: {res.cost() / region_scale}')

    total_demand = 0.0
    solution = res.best

    demand_with_depot = np.insert(demand, 0, 0)
    for idx, route in enumerate(solution.get_routes(), 1):
        if len(route) != 0:  # Ensure the route is not empty
            customer_demands = np.sum(demand_with_depot[route])  # Current customer
            total_demand += customer_demands

    #######################################################
    # RL OR AGENT

    ckpt_path = "RLOR/runs/cvrp-v1__ppo_or__1__1714136363/ckpt/4.pt"  # "runs/argos_exp3.2/cvrp-v1__exp3.2_vf-argos_cluster_local_runtime__1__1711632522/ckpt/8000.pt" #"runs/cvrp-v1__exp4.1_with_AttentionScore_Enhancing__1__1712436040/ckpt/390.pt" #"runs/cvrp-v1__exp4.1_with_AttentionScore_Enhancing__1__1712436040/ckpt/390.pt" #"runs/cvrp-v1__exp4.0_with_AttentionScore_Enhancing__1__1712328992/ckpt/200.pt"
    # "runs/argos_exp3.2/cvrp-v1__exp3.2_vf-argos_cluster_local_runtime__1__1711632522/ckpt/5000.pt"#"runs/athene_exp3.3/cvrp-v1__exp3.3_vf-athena_cluster_local_runtime_2__1__1712077050/ckpt/1000.pt" #
    device = 'cpu'
    agent = Agent(device=device, name='cvrp_fleet_tw', k=3).to(device)
    agent.load_state_dict(torch.load(ckpt_path))



    trajectories = []
    agent.eval()
    done = np.array([False])
    logits = []
    entropies = []
    values = []
    while not done.all():
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logit = agent(obs)
        if trajectories == []:  # Multi-greedy inference
            action = torch.arange(1, envs.n_traj + 1).repeat(1, 1)

        obs, reward, done, info = envs.step(action.cpu().numpy())
        trajectories.append(action.cpu().numpy())
        logits.append(logits)
        # entropies.append(entropy)
        # values.append(value)

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
    plot_agent_solution(resulting_traj_with_depot, customers, depot, ax=axs[1])
    plot_solution(res.best, m.data(), plot_clients=True, ax=axs[0], scale=region_scale)

    print(
        f' demand collected in OR solution {total_demand}')
    plt.tight_layout()
    plt.show()
