import numpy as np
from matplotlib import pyplot as plt

from RLOR.envs.cvrp_vehfleet_env import CVRPFleetEnv


class DataVisualizer:
    def __init__(self, distance, rewards, num_veh, load):
        self.distance = distance
        self.rewards = rewards
        self.num_veh = num_veh
        self.load = load

    def plot_line_chart(self):
        # Extract the first item from each sublist for plotting
        first_elements_distance = [sublist[0] for sublist in self.distance]
        first_elements_rewards = [sublist[0] for sublist in self.rewards]
        first_elements_num_veh = [sublist[0] for sublist in self.num_veh]
        first_elements_load = [sublist[0] for sublist in self.load]

        plt.figure(figsize=(10, 6))
        plt.plot(first_elements_distance, label='Distance')
        plt.plot(first_elements_rewards, label='Rewards')
        plt.plot(first_elements_num_veh, label='Number of Vehicles')
        plt.plot(first_elements_load, label='Load')
        plt.title('Line Plot Comparisons')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(np.arange(len(first_elements_distance)))
        plt.grid(True)
        plt.show()

    def plot_scatter_charts(self):
        # Extract the first item from each sublist for plotting
        first_elements_load = [sublist[0] for sublist in self.load]
        first_elements_rewards = [sublist[0] for sublist in self.rewards]
        first_elements_distance = [sublist[0] for sublist in self.distance]

        # Creating subplots for scatter plots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot for Load vs Rewards
        axs[0].scatter(first_elements_load, first_elements_rewards)
        axs[0].set_title('Load vs Rewards')
        axs[0].set_xlabel('Load')
        axs[0].set_ylabel('Rewards')
        axs[0].grid(True)

        # Scatter plot for Distance vs Rewards
        axs[1].scatter(first_elements_distance, first_elements_rewards)
        axs[1].set_title('Distance vs Rewards')
        axs[1].set_xlabel('Distance')
        axs[1].set_ylabel('Rewards')
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = CVRPFleetEnv()
    init_state = env.reset()
    print(f'reset {init_state}')
    n_traj = env.n_traj
    rewards = [] # (20,35)
    done = np.full(n_traj, False)
    nr_of_steps = 0
    loads = []
    vehicles = []
    distances = []
    steps_per_vehicle = []
    #distances.append(env.distance)
    while not done[0]==True:
        action = env.action_space.sample()
        print(f'action {action}')
        state, reward, done, info =  env.step(action)

        distances.append(env.distance)
        rewards.append(reward)
        vehicles.append(state["num_veh"].copy())
        loads.append(state["current_load"].copy())
        #print(f'state {state}; \n reward {reward}; \n done {done};')
        nr_of_steps +=1
        print("______________________________________________")

    visualizer = DataVisualizer(distances, rewards, vehicles, loads)
    visualizer.plot_line_chart()  # This will open the first plot window
    visualizer.plot_scatter_charts()  # This will open



    sums = np.sum(rewards, axis = 0)

    print(f' size {sums.size} , sum the rewards {sums}')

    substract_fix_costs = sums + 5 #plus, becasue the distance is negative and I want to substract
    print(f' size {substract_fix_costs.size} ,  substract_fix_costs  {substract_fix_costs} \n')

    build_average_only_distance = substract_fix_costs / nr_of_steps

    print(f'build_average_only_distance {build_average_only_distance} \n \n')
    print(f' total average in all episodes without fixcosts {np.mean(build_average_only_distance)}')

    avg_episodic_return = np.mean(rewards, axis = 0)
    print(f'avg_episodic_return size {avg_episodic_return.size}, nr_of_steps {nr_of_steps} and values {avg_episodic_return}')

    print(f' total average in all episodes with fixcosts {np.mean(avg_episodic_return)}')

    print("difference is 0.400961 when number of stes is 20")




