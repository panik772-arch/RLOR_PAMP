# Validate vrp with tw
# you need first generate a small problem with 20 customers
# try to manuelle draw the routes our apply som small script for optimization (pyvrp with tw from 0 to max)
# then plot this
import time
import csv
import datetime

import numpy as np
from matplotlib import pyplot as plt
from pyvrp import Model
from pyvrp.stop import MaxRuntime, NoImprovement, MaxIterations
from pyvrp.plotting import plot_solution

from RLOR.envs.cvrp_vehfleet_tw_env import CVRPFleetTWEnv
from RLOR.envs.vrp_data_with_tw import VRPDatasetTW
import torch
import gym
from RLOR.models.attention_model_wrapper import Agent
from RLOR.wrappers.syncVectorEnvPomo import SyncVectorEnv
from RLOR.wrappers.recordWrapper import RecordEpisodeStatistics
from test_and_plot_trained_model import plot_agent_solution, calculate_demand, make_routes

def plot_logits(logits, tw, max_tw):
    # Sample data
    num_steps = len(logits)
    num_nodes = len(logits[0])
    x = np.arange(num_steps)
    y = np.arange(num_nodes)
    X, Y = np.meshgrid(x, y)
    tw = np.insert(tw, 0, max_tw, axis=0)

    # Prepare Z data (logit values)
    Z = np.zeros((num_nodes, num_steps))
    for i in range(num_nodes):
        for j in range(num_steps):
            Z[i, j] = logits[j][i]

    # Plot
    plt.figure(figsize=(18, 18))
    plt.imshow(Z, cmap='viridis', aspect='auto', origin='lower', interpolation='nearest')

    # Set labels
    plt.xlabel('Steps')
    plt.ylabel(f'Nodes')
    plt.xticks(np.arange(num_steps), np.arange(1, num_steps + 1))
    plt.yticks(np.arange(num_nodes), [f"N{i}_tw:{tw[i]}" for i in range(num_nodes)])

    plt.colorbar(label='Logit Value')
    plt.title('Logit Values Over Steps and Nodes')
    plt.show()



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

def make_env(seed,n_traj, vehicles, max_nodes, min_tw, max_tw):
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

    vehicles = vehicles
    seed = seed
    # if its 2000 (min) we visit customers occasionally. Only min distance is matter. So, the vehicle trys to minimize the toures and results in shorter distance and less tours
    # if 10 000 is chosen, the vehicle trys to collect as many as possible and so, results in longer tours
    envs = SyncVectorEnv(
        [make_env(env_id, seed, dict(n_traj=n_traj, max_nodes=max_nodes, max_num_vehicles=vehicles, penalty=10, min_tw = min_tw, max_tw=max_tw))])
    obs = envs.reset()

    return obs, envs


def save_instance_to_csv(obs, vehicles, vehicle_capacity, region_scale,
                         max_tw, filename=None):
    """
    Save VRP instance in C101 format to CSV file.

    Parameters:
    -----------
    obs : dict
        Observation dictionary containing depot, observations (nodes), demand, and tw
    vehicles : int
        Number of vehicles
    vehicle_capacity : float
        Vehicle capacity (will be multiplied by region_scale)
    region_scale : int
        Scaling factor for coordinates and demands
    max_tw : int
        Maximum time window (due date for depot)
    filename : str, optional
        Output filename. If None, generates timestamp-based name
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vrp_instance_{timestamp}.csv"

    nodes = obs["observations"][0]
    depot = obs["depot"][0]
    demand = obs["demand"][0]
    tw = obs["tw"][0]

    num_customers = len(nodes)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header information
        writer.writerow(['VEHICLE'])
        writer.writerow(['NUMBER', 'CAPACITY'])
        writer.writerow([vehicles, int(vehicle_capacity)])
        writer.writerow([])

        # Write customer header
        writer.writerow(['CUSTOMER'])
        writer.writerow(['CUST NO.', 'XCOORD.', 'YCOORD.', 'DEMAND',
                         'READY TIME', 'DUE DATE', 'SERVICE TIME'])
        writer.writerow([])

        # Write depot (customer 0)
        writer.writerow([
            0,
            int(depot[0] * region_scale),
            int(depot[1] * region_scale),
            0,
            0,
            max_tw,
            0
        ])

        # Write all customers
        for idx in range(num_customers):
            writer.writerow([
                idx + 1,
                int(nodes[idx][0] * region_scale),
                int(nodes[idx][1] * region_scale),
                int(demand[idx] * region_scale),
                0,  # READY TIME (all customers ready from beginning)
                int(tw[idx]),  # DUE DATE
                90  # SERVICE TIME (default value, adjust as needed)
            ])

    print(f"Instance saved to: {filename}")
    return filename


def save_solutions_to_csv(heuristic_cost, rl_cost,
                          heuristic_time, rl_time,
                          heuristic_customers_served, rl_customers_served,
                          total_customers, region_scale,
                          filename=None):
    """
    Save solution comparison to CSV file.

    Parameters:
    -----------
    heuristic_cost : float
        Total distance from heuristic solution
    rl_cost : float
        Total cost from RL solution
    heuristic_time : float
        Time taken by heuristic solver (seconds)
    rl_time : float
        Time taken by RL solver (seconds)
    heuristic_customers_served : int
        Number of customers served by heuristic
    rl_customers_served : int
        Number of customers served by RL
    total_customers : int
        Total number of customers in instance
    region_scale : int
        Scaling factor used
    filename : str, optional
        Output filename. If None, generates timestamp-based name
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vrp_solutions_{timestamp}.csv"

    # Calculate percentages
    heuristic_pct = (heuristic_customers_served / total_customers) * 100
    rl_pct = (rl_customers_served / total_customers) * 100

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers
        writer.writerow(['Metric', 'Heuristic (PyVRP)', 'RL (DAR_RL)'])
        writer.writerow([])

        # Write solution costs
        writer.writerow(['Solution Cost',
                         f'{heuristic_cost / region_scale:.4f}',
                         f'{-1* rl_cost:.4f}'])

        # Write solving times
        writer.writerow(['Solving Time (seconds)',
                         f'{heuristic_time:.4f}',
                         f'{rl_time:.4f}'])

        # Write customers served
        writer.writerow(['Customers Served',
                         heuristic_customers_served,
                         rl_customers_served])

        # Write percentages
        writer.writerow(['% Customers Served',
                         f'{heuristic_pct:.2f}%',
                         f'{rl_pct:.2f}%'])

        # Write total customers
        writer.writerow([])
        writer.writerow(['Total Customers', total_customers, total_customers])

        # Write comparison
        writer.writerow([])
        writer.writerow(['Performance Comparison'])
        cost_diff_pct = ((rl_cost - (heuristic_cost / region_scale)) /
                         (heuristic_cost / region_scale)) * 100
        writer.writerow(['Cost Difference (%)',
                         f'{cost_diff_pct:.2f}% {"(RL better)" if cost_diff_pct < 0 else "(Heuristic better)"}'])

        time_speedup = heuristic_time / rl_time if rl_time > 0 else float('inf')
        writer.writerow(['Time Speedup (Heuristic/RL)', f'{time_speedup:.2f}x'])

    print(f"Solutions saved to: {filename}")
    return filename


def count_customers_served_heuristic(solution, demand_with_depot):
    """
    Count number of customers served in heuristic solution.

    Parameters:
    -----------
    solution : Solution object
        PyVRP solution object
    demand_with_depot : numpy array
        Demand array with depot at index 0

    Returns:
    --------
    int : Number of unique customers served
    """
    customers_served = set()
    for route in solution.routes():
        for customer_idx in route:
            if customer_idx > 0:  # Exclude depot
                customers_served.add(customer_idx)
    return len(customers_served)


def count_customers_served_rl(routes):
    """
    Count number of customers served in RL solution.

    Parameters:
    -----------
    routes : dict or list
        Dictionary with route names as keys and numpy arrays as values,
        OR list of routes from RL solution

    Returns:
    --------
    int : Number of unique customers served
    """
    customers_served = set()

    # Handle dictionary format (e.g., {'Route 1': array([...]), ...})
    if isinstance(routes, dict):
        routes = routes.values()

    for route in routes:
        # Handle numpy arrays and lists
        for customer_idx in route:
            # Skip non-numeric values (e.g., 'R' for return to depot)
            try:
                #Convert to int if it's a string or other type
                customer_idx = int(customer_idx) if not isinstance(customer_idx, int) else customer_idx
                if customer_idx > 0:  # Exclude depot (0)
                    customers_served.add(customer_idx)
            except (ValueError, TypeError):
                # Skip invalid entries like 'R', 'D', etc.
                continue
    return len(customers_served)

if __name__ == "__main__":
    max_nodes = 1000
    n_traj = 1000
    eval_data = False
    eval_partition = "eval"
    eval_data_idx = 0
    region_scale = 10000
    min_tw = 50
    max_tw = 10000
    vehicles = 5
    v = 0.0014 #50 #speed in km/h
    prize_for_visiting = 750 #10000 #2000 min and 10000 max
    seed = 1234
    MaxRunTime = 10000




    #######################################################
    # RL OR AGENT
    # 16000pt-> 6.63
    # 13000 -> 6.76
    # 10000-> 6.45
    # 8000 -> 6.33
    # 4000 -> 6.90 (or is 6.98
    # with 500 customers, 2000.pt -> 7.95 and 20000 5.03 and 8000.pt -> 3.07 and 10000.pt 3.11
    ckpt_path = "C:\\rlor_pamp_trained_models\\exp8.3_herakles_noEMbeddingsAndContext\\8000.pt"  # "runs/cvrp-v2__exp5_PAMP_localrun__1__1714518031/ckpt/100.pt" #"RLOR/runs/cvrp-v1__ppo_or__1__1714136363/ckpt/4.pt" #"runs/cvrp-v2__exp5_PAMP_localrun__1__1714518031/ckpt/100.pt"  # "runs/argos_exp3.2/cvrp-v1__exp3.2_vf-argos_cluster_local_runtime__1__1711632522/ckpt/8000.pt" #"runs/cvrp-v1__exp4.1_with_AttentionScore_Enhancing__1__1712436040/ckpt/390.pt" #"runs/cvrp-v1__exp4.1_with_AttentionScore_Enhancing__1__1712436040/ckpt/390.pt" #"runs/cvrp-v1__exp4.0_with_AttentionScore_Enhancing__1__1712328992/ckpt/200.pt"
    # "runs/argos_exp3.2/cvrp-v1__exp3.2_vf-argos_cluster_local_runtime__1__1711632522/ckpt/5000.pt"#"runs/athene_exp3.3/cvrp-v1__exp3.3_vf-athena_cluster_local_runtime_2__1__1712077050/ckpt/1000.pt" #

    obs, envs = make_env(seed= seed, n_traj=n_traj, vehicles = vehicles, max_nodes=max_nodes,min_tw=min_tw, max_tw=max_tw)
    nodes = obs["observations"][0]
    dep = obs["depot"][0]
    demand = obs["demand"][0]
    tw = obs["tw"][0]
    print(demand * region_scale)

    #data = VRPDatasetTW[eval_partition, max_nodes, eval_data_idx, region_scale]
    #nodes =  data["loc"]
    #dep = data["depot"]
    #demand = data["demand"]
    #w = data["tw"]
    print(tw)

    nodes_with_depo = np.concatenate((dep[None, ...], nodes))

    dist_matrix = genrate_duration_matrix(nodes_with_depo)
    DURATION_MATRIX = ((dist_matrix * region_scale) / (v * region_scale) ) #* 3600 #convert to duration in seconds according to tw
    print(f'{dep[0]} and {dep[1]}')

    m = Model()
    m.add_vehicle_type(vehicles, capacity=1 * region_scale)


    TIME_WINDOWS = [(0, int(tw[idx])) for idx in range(nodes.shape[0])]
    TIME_WINDOWS_depot = [0, max_tw]

    depot = m.add_depot(x=int(dep[0] * region_scale),
                        y=int(dep[1] * region_scale),
                        tw_early=TIME_WINDOWS_depot[0],
                        tw_late=TIME_WINDOWS_depot[1],
                        )

    clients = [
        m.add_client(x=int(nodes[idx][0] * region_scale),
                     y=int(nodes[idx][1] * region_scale),
                     delivery=int(demand[idx] * region_scale),
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
            dur = DURATION_MATRIX[frm_idx][to_idx]
            m.add_edge(frm, to, distance=distance,duration=dur)
    start_time_pyvrp = time.time()
    res = m.solve(stop=MaxIterations(MaxRunTime)) #stop=NoImprovement(MaxRunTime)
    end_time_pyvrp = time.time()

    print('pyvrp calculation time', end_time_pyvrp - start_time_pyvrp)



    print(res)
    print(f'BEST SOLUTION FROM HEURISTIC: {res.cost() / region_scale}')

    total_demand = 0.0
    solution = res.best

    demand_with_depot = np.insert(demand, 0, 0)
    for idx, route in enumerate(solution.routes(), 1):
        if len(route) != 0:  # Ensure the route is not empty
            customer_demands = np.sum(demand_with_depot[route])  # Current customer
            total_demand += customer_demands

    device = 'cpu'
    agent = Agent(device=device, name='cvrp_fleet_tw', k=50).to(device)
    agent.load_state_dict(torch.load(ckpt_path))

    trajectories = []
    agent.eval()
    done = np.array([False])
    i = 0
    # in each step (n_traj, max_nodes+depot)
    logits = {f'step_{i}':torch.zeros(n_traj, max_nodes+1) }
    entropies = []
    values = []

    start_time_nco = time.time()
    while not done.all():
        i+=1
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logit = agent(obs)
        if trajectories == []:  # Multi-greedy inference
            action = torch.arange(1, envs.n_traj + 1).repeat(1, 1)

        obs, reward, done, info = envs.step(action.cpu().numpy())
        trajectories.append(action.cpu().numpy())
        logits[f'step_{i}']= logit[0]
        # entropies.append(entropy)
        # values.append(value)

    nodes_coordinates = np.vstack([obs['depot'], obs['observations'][0]])
    final_return = info[0]['episode']['r']
    best_traj = np.argmax(final_return)

    end_time_nco = time.time()


    print('calculation time nco', end_time_nco - start_time_nco)

    resulting_traj = np.array(trajectories)[:, 0, best_traj]
    resulting_traj_with_depot = np.hstack([np.zeros(1, dtype=int), resulting_traj])

    print(f'A route of length {final_return[best_traj]}')
    print('The route is:\n', resulting_traj_with_depot)

    x, y = nodes_coordinates[resulting_traj_with_depot].T
    customers = obs['observations'][0]
    depot = obs['depot'][0]


    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    plot_agent_solution(resulting_traj_with_depot, customers, depot, ax=axs[1], demand = demand)
    plot_solution(res.best, m.data(), plot_clients=True, ax=axs[0],scale=region_scale) #
    routes_ = make_routes(resulting_traj_with_depot)

    logits_best = []
    for (traj, step_logits) in logits.items():
        # Choose the best trajectory
        logits_best.append(step_logits[best_traj])
    demand_in_RL_solution = calculate_demand(routes_, obs)

    print(
        f' demand collected in OR solution {total_demand}')
    print(
        f' demand collected in RL solution {demand_in_RL_solution}')

    # Save instance
    instance_file = save_instance_to_csv(
        obs=obs,
        vehicles=vehicles,
        vehicle_capacity=1 * region_scale,  # Your capacity value
        region_scale=region_scale,
        max_tw=max_tw,
        filename="vrp_instance.csv"
    )

    # Count customers served
    demand_with_depot = np.insert(demand, 0, 0)
    heuristic_customers = count_customers_served_heuristic(res.best, demand_with_depot)
    routes_rl = make_routes(resulting_traj_with_depot)
    rl_customers = count_customers_served_rl(routes_rl)

    # Save solutions
    solutions_file = save_solutions_to_csv(
        heuristic_cost=res.best.distance(),
        rl_cost=final_return[best_traj],
        heuristic_time=end_time_pyvrp - start_time_pyvrp,
        rl_time=end_time_nco - start_time_nco,
        heuristic_customers_served=heuristic_customers,
        rl_customers_served=rl_customers,
        total_customers=len(nodes),
        region_scale=region_scale,
        filename="vrp_solutions.csv"
    )

    print(f"\nFiles saved:")
    print(f"  Instance: {instance_file}")
    print(f"  Solutions: {solutions_file}")


    plot_logits(logits_best, tw, max_tw)


    plt.tight_layout()
    plt.show()
