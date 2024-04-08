import gym as gym
import numpy as np
from gym.spaces import Discrete, Box, Dict


class GraphEnvironment(gym.Env):
    def __init__(self):
        self.MAX_STEPS = 100
        self.max_episode_steps = 1000
        self.num_nodes = 10 + 1  # Including depot as the first node
        self.vehicle_capacity = 1  # because demand is normalized, else 20
        self.num_vehicles = 3
        self.max_distance = 1000  # Assuming max distance for reward calculation

        # Action space: node selection, including depot to return
        self.action_space = Discrete(self.num_nodes)

        # observations
        node_features = Box(low=0, high=100, shape=(self.num_nodes, 3), dtype=np.float32)  # x,y and demand
        #0 visited, 1 - to visit
        visited_nodes = Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.int8)
        last_position = Box(low=0, high=self.num_nodes, shape=(1,), dtype=np.int32)
        curr_position = Box(low=0, high=self.num_nodes, shape=(1,), dtype=np.int32)
        remaining_capacity_of_vehicle = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        num_veh = Discrete(self.num_vehicles+1)

        veh_features = Dict(
            {"last_pos": last_position, "curr_pos": curr_position, "remaining_capa": remaining_capacity_of_vehicle, "fleet": num_veh})

        # Observation: Demand, Visited status, Vehicle positions, Remaining capacities
        self.observation_space = Dict(
            {"nodes": node_features, "action_mask": visited_nodes, "veh_features": veh_features})

        self.reset(seed=None)
        self.valid_actions = None

    def reset(self, seed=None, options=None):
        node_demand = np.random.uniform(low=0, high=0.5, size=(self.num_nodes, 1))
        node_coords = np.random.uniform(low=0, high=1, size=(self.num_nodes, 2))
        # Concatenate node_demand and node_coords
        node_features = np.hstack((node_coords, node_demand))

        visited_nodes = np.zeros((self.num_nodes,), dtype=np.int8)  ## all nodes are initially zeros -> to visit
        visited_nodes[0] = np.array(1, dtype=np.int8 )  # Depot is always considered as visited
        node_features[0, 2] = np.array(0, dtype=np.int32 )  # demand in depot is 0

        remaining_capacity_of_vehicle = np.array([self.vehicle_capacity], dtype = np.float32)
        last_position = np.array([0], dtype=np.int32 )  # last pos is depot
        curr_position = np.array([0], dtype=np.int32 ) # current pos is depot at depot
        num_v = 3

        self.state = {"nodes": node_features, "action_mask": visited_nodes,
                      "veh_features": {"last_pos": last_position,"curr_pos": curr_position, "remaining_capa": remaining_capacity_of_vehicle,
                                       "fleet": num_v}}
        self.t = 0
        self.veh = 3
        info = {}

        #print("print state", self.state)
        return self.state, info

    def step(self, action):
        print(f'action from env {action} and  mask {self.state["action_mask"]}')
        self.t += 1  # the number of steps in an episode can not exceed the nuber of vehicles * num_of_nodes

        done = truncated = False
        self.state["action_mask"][0] = np.array(1, dtype=np.int8) # mask the depot at the beginning
        idx_last_pos = self.state["veh_features"]["last_pos"][0]
        idx_action = action
        coord_las_pos = self.state["nodes"][idx_last_pos, :2]
        coord_this_pos = self.state["nodes"][idx_action, :2]
        distance = self._calculate_distance(coord_las_pos,coord_this_pos)  # Implement this method based on your distance metric


        # vehicle is not allowed to select the last job and to select inactive job:
        # implemented by reward now

        ## if capacity of the vehicle exceed, return to depot

        ## TODO: Here attention.
        ## basically, here the agent only tries ones, and if the node demand exceed the capacity, it returns to depot.
        # this is not correct, since it would be better to try somehow another node, isn't?
        # but then I basically say here, we perform a neighborhood search, what is also not our intention
        if self.state["veh_features"]["remaining_capa"] < self.state["nodes"][action][2]:
            # return to depot, reduce number of vehicles and set the veh_capacity to max again
            self.state["action_mask"][0] = np.array(1, dtype=np.int8)
            self.state["veh_features"]["last_pos"] = np.array([action], dtype=np.int32)
            self.state["veh_features"]["curr_pos"] = np.array([0], dtype=np.int32) # move vehicle to th depot
            self.veh -= 1
            self.state["veh_features"]["fleet"] -= 1  ## additionally reduce the num of vehicles in the state dict
            self.state["veh_features"]["remaining_capa"] = np.array([self.vehicle_capacity], dtype= np.int32)  ## reset veh capacity
            depotLoc = self.state["nodes"][0][0:2]
            distance = self._calculate_distance(depotLoc,coord_las_pos) # distance between depot and last position of vehicle

            # penalize the agent, when there were other nodes with less demand, but he chose to return
            penalty = 0
            if (np.all(self.state["nodes"][:,2] <= self.state["veh_features"]["remaining_capa"])):
                penalty = distance*10
            print(f'penalty {penalty}')
            reward = -(distance + penalty)

        elif (action != 0 and self.state["nodes"][action][2]>0 and self.state["action_mask"][action] != 1 ):  # if not a depot, demand is > zero and not masked
            assert self.state["action_mask"][action] == np.array(0, dtype=np.int8), " never visit the same job twice"
            assert self.state["veh_features"]["remaining_capa"][0] > self.state["nodes"][action][2], "demand capacity cant exceed remaining load of vehicle"

            self.state["action_mask"][action] = np.array(1, dtype=np.int8)  #mask this node
            #self.state["action_mask"][0] =  np.array(1, dtype=np.int8)  # unmask depot
            self.state["veh_features"]["remaining_capa"][0] -= self.state["nodes"][action][2]  # reduce load of vehicle
            self.state["nodes"][action][2] =  np.array(0, dtype=np.int32)  # set demand to 0
            self.state["veh_features"]["last_pos"] = self.state["veh_features"]["curr_pos"]
            self.state["veh_features"]["curr_pos"] = np.array([action], dtype=np.int32) # move to this node
            print(f'last position { self.state["veh_features"]["last_pos"]} and current position {self.state["veh_features"]["curr_pos"]}')
            reward = -distance
        else:
            # Handle invalid action (soft constrain?) or forbid (hard constrain)
            reward = -self.max_distance  # Penalize invalid actions to discourage them
            print(f'max reward{reward}')

        # Check if all nodes are visited or no more vehicles available
        if (np.all(self.state["action_mask"] == 1) or np.all(self.state["nodes"][:,2] <= 0) or self.veh == 0 or
            self.state["veh_features"]["fleet"] == 0) :
            print("DONE")
            done = truncated = True

        info = {"idx_last_pos": idx_last_pos, "idx_action": idx_action, "coord_las_pos": coord_las_pos,
                "coord_this_pos": coord_this_pos, "distance": distance}

        return self.state, reward, done, info

    def render(self, obs, action, figure):
        '''
        plt.cla()
        plt.title('Vehicle Routing Problem State')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)

        # Plot nodes
        for i in range(self.num_nodes):
            if obs[0]["action_mask"][i] == 1:  # If node is visited
                plt.scatter(obs[0]["nodes"][i, 0], obs[0]["nodes"][i, 1], color='red', s=100,
                            label='Visited' if i == 1 else "")
            else:
                plt.scatter(obs[0]["nodes"][i, 0], obs[0]["nodes"][i, 1], color='blue', s=100,
                            label='Not Visited' if i == 1 else "")
            plt.text(obs[0]["nodes"][i, 0], obs[0]["nodes"][i, 1], str(i), color='black', fontsize=12)

        # Draw routes
        last_pos = obs[0]["veh_features"]["last_pos"][0]
        for i in range(1, self.num_nodes):
            if obs[0]["action_mask"][i] == 1:  # If node is visited
                start_pos = obs[0]["nodes"][last_pos, :2]
                end_pos = obs[0]["nodes"][i, :2]
                plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-', lw=2)

        # re-drawing the figure
        plt.draw()

        # to flush the GUI events
        #figure.flush_events()
        plt.pause(1)

    '''


    def _calculate_distance(self, loc1, loc2):
        # Placeholder: Implement actual distance calculation
        return np.sqrt(pow(loc1[0]-loc2[0], 2) + pow(loc1[1]-loc2[1], 2)) # Random dis

'''
see discssion:
I had exaclty the same problem. The big Problem is the dependency between your two actions(e.g. can't take the same ball twice). 
So one thing you can do is multiply them, so you have one big action space of 200x200=40000.
Then you are able to create the full mask in the env and pass it to the forward function for the masking. 
Other wise you need to work with dependent action sampling and distributions.

For me the multiplication was not an option as it would be to large. So I make it the following way:

Env creates a mask for Action 1 and XXX Masks for the depending Action 2.
In the model you will sample the action 1 (with tf.random.categorical) with your action 1 mask
Depending on the action 1 you select a mask for action 2 (tf.where) and sample action 2.
The output of the model should be logits and the sampled action.
You need to implement your own MultiCategorical action distribution to use your already sampled actions.

https://stackoverflow.com/questions/66405687/complex-action-mask-in-rllib

'''


if __name__ == "__main__":
    env = GraphEnvironment

    #collect batch manually
    observations = []
    actions = []
    done = False
    state = env.reset()


    while not done:
        mask = state[0]["node_mask"]  # returns list

        # observations contain original observations and the action mask
        # reward is random and irrelevant here and therefore not printed
        action = env.action_space.sample()  # Assuming the environment knows how to handle the mask internally
        next_obs, reward, done, truncated, _ = env.step(action)

        observations.append(next_obs)  # Store observations
        actions.append(action)  # Store actions if necessary

        print(f"Obs: {next_obs}, Action: {action}")
        state = next_obs

