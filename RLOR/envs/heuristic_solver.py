from matplotlib import pyplot as plt
from pyvrp import Model

from RLOR.envs.cvrp_vehfleet_env import CVRPFleetEnv
from pyvrp.plotting import plot_coordinates
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution

if __name__ == "__main__":
    m = Model()

    env = CVRPFleetEnv()
    initi_state = env.reset()
    #print(f'reset {initi_state}')
    scale = 10000

    nodes = initi_state["observations"]
    dep = initi_state["depot"]
    demand = initi_state["demand"]
    nr_veh = env.max_num_vehicles
    print(demand*scale)



    m.add_vehicle_type(nr_veh, capacity=1*scale)

    print(f'{dep[0]} and {dep[1]}')

    depot = m.add_depot(x=int(dep[0]*scale), y=int(dep[1]*scale))

    clients = [
        m.add_client(x=int(nodes[idx][0]*scale), y=int(nodes[idx][1]*scale), demand=int(demand[idx]*scale),
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

    _, ax = plt.subplots(figsize=(8, 8))
    plot_solution(res.best, m.data(),plot_clients=True, ax=ax)

    plt.show()





