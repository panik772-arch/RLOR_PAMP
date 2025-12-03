import gym
import numpy as np
from gym import spaces
from .vrp_data_with_tw import VRPDatasetTW


def assign_env_config(self, kwargs):
    """
    Set self.key = value, for each key in kwargs
    """
    for key, value in kwargs.items():
        setattr(self, key, value)


def dist(loc1, loc2):
    return ((loc1[:, 0] - loc2[:, 0]) ** 2 + (loc1[:, 1] - loc2[:, 1]) ** 2) ** 0.5


class CVRPFleetTWEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.max_nodes = 50
        self.capacity_limit = 40
        self.max_num_vehicles = 5
        self.n_traj = 50
        # if eval_data==True, load from 'test' set, the '0'th data
        self.eval_data = False
        self.eval_partition = "evaluation"
        self.eval_data_idx = 0
        self.demand_limit = 10
        self.penalty = 10
        self.min_tw = 50
        self.max_tw = 10000
        self.region_scale = 10000 # 10kmx10km region for time-windows. We need this to distribute the tw for further calculation in attentionModelWrapper
        assign_env_config(self, kwargs)

        obs_dict = {"observations": spaces.Box(low=0, high=1, shape=(self.max_nodes, 2))}
        obs_dict["depot"] = spaces.Box(low=0, high=1, shape=(2,))
        obs_dict["demand"] = spaces.Box(low=0, high=1, shape=(self.max_nodes,))
        obs_dict["tw"] = spaces.Box(low=0, high=self.max_tw+1, shape=(self.max_nodes,))
        obs_dict["action_mask"] = spaces.MultiBinary([self.n_traj, self.max_nodes + 1])  # 1: OK, 0: cannot go
        obs_dict["last_node_idx"] = spaces.MultiDiscrete([self.max_nodes + 1] * self.n_traj)
        obs_dict["current_load"] = spaces.Box(low=0, high=1, shape=(self.n_traj,))
        obs_dict["traveled_dist"] = spaces.Box(low=0, high=10, shape=(self.n_traj,))
        obs_dict["num_veh"] =  spaces.MultiDiscrete([self.max_num_vehicles] * self.n_traj)

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.MultiDiscrete([self.max_nodes + 1] * self.n_traj)
        self.reward_space = None

        self.reset()

    def seed(self, seed):
        np.random.seed(seed)

    def _STEP(self, action):
        '''
        CHECK THIS!
        Maybe It will rise a problem,  the Agent will try to return to depot, because then it has less to drive, less milage and hence, bigger reward!
        CHECK THIS!
        '''

        # if action == 0, return to depot, reset capacity and reduce number of vehicles

        self._go_to(action)  # Go to node 'action', modify the reward
        self.num_steps += 1
        self.state = self._update_state()

        # try this too. here is extended approach...
        is_all_visited = self.is_all_visited()
        no_vehicles_left = (self.num_veh == 0)
        traj_with_all_visited_or_no_vehicles = np.logical_or(is_all_visited, no_vehicles_left)
        #action = np.where(traj_with_all_visited_or_no_vehicles, 0, action)
        # need to revisit the first node after visited all other nodes
        self.done = is_all_visited #traj_with_all_visited_or_no_vehicles

        #self.done = (action == 0) & self.is_all_visited()

        #print(f' dones? {self.done}')
        #print(f' nr_of_vehicles {self.num_veh}')
        #print(f' is_all_visited {is_all_visited}')
        return self.state, self.reward, self.done, self.info

    # Euclidean cost function
    def cost(self, loc1, loc2):
        return dist(loc1, loc2)

    def is_all_visited(self):
        # assumes no repetition in the first `max_nodes` steps
        return self.visited[:, 1:].all(axis=1)

    def is_demand_exceed_capacity(self):
        pass

    def no_other_vehicles(self):
        # Find indices where the number of vehicles is 0
        # AND load is 1, meaning, that the load was reseted in go_to, because the destination was zero there.
        # This step occurs befor load is updated in def _update_state(self): -> def _update_mask(self)
        condition = (self.num_veh == 0) #& (self.load == 1)
        indices_no_vehicles = np.where(condition)[0]
        self.visited[indices_no_vehicles, 1:] = True

    def reward_func(self):

        #the idea is to use vehicle load as penalty. when 1 (nothing was delivered, it is bad. and when 0, its good, so it is linear dependency
        visited_traj= self.visited.sum(axis=1)
        return self.penalty * self.load #+ np.log(self.max_nodes/visited_traj)

    def _update_state(self):
        obs = {"observations": self.nodes[1:]}  # n x 2 array
        obs["depot"] = self.nodes[0]
        obs["tw"] = self.tw
        obs["traveled_dist"] = self.traveled_dist
        obs["action_mask"] = self._update_mask()
        obs["demand"] = self.demands
        obs["last_node_idx"] = self.last
        obs["current_load"] = self.load
        obs["num_veh"] = self.num_veh

        return obs

    def _update_mask(self):

        # set the whole row to True in self.visited
        self.no_other_vehicles()

        # Only allow to visit unvisited nodes
        action_mask = ~self.visited

        # can only visit depot when last node is not depot
        action_mask[:, 0] |= self.last != 0

        # or all visited
        action_mask[:, 0] |= self.is_all_visited()

        # Not allow to visit nodes when the load is zero
        action_mask[self.load <= 0, 1:] = False

        # not allow visit nodes with higher demand than capacity
        action_mask[:, 1:] &= self.demands <= (
            self.load.reshape(-1, 1) + 1e-5
        )  # to handle the floating point subtraction precision


        return action_mask

    def _RESET(self):
        self.visited = np.zeros((self.n_traj, self.max_nodes + 1), dtype=bool)
        self.visited[:, 0] = True
        self.num_steps = 0
        self.last = np.zeros(self.n_traj, dtype=int)  # idx of the last elem
        self.load = np.ones(self.n_traj, dtype=float)  # current load
        self.traveled_dist = np.zeros(self.n_traj, dtype=float)  # idx of the last elem

        if self.eval_data:
            self._load_orders()
        else:
            self._generate_orders()

        self.num_veh = np.array([self.max_num_vehicles] * self.n_traj)
        self.state = self._update_state()
        self.info = {}
        self.done = np.array([False] * self.n_traj)
        #optional for plotting
        self.distance = np.zeros(self.n_traj, dtype=float)

        return self.state

    def _load_orders(self):
        data = VRPDatasetTW[self.eval_partition, self.max_nodes, self.eval_data_idx, self.min_tw, self.max_tw]
        self.nodes = np.concatenate((data["depot"][None, ...], data["loc"]))
        self.demands = data["demand"]
        self.tw = data["tw"]

        self.demands_with_depot = self.demands.copy()

    def _generate_orders(self):
        self.nodes = np.random.rand(self.max_nodes + 1, 2)
        self.demands = (
                np.random.randint(low=1, high=self.demand_limit, size=self.max_nodes)
                / self.capacity_limit
        )

        self.tw = (
                np.random.randint(low=self.min_tw, high=self.max_tw, size=self.max_nodes) ##self.region_scale
                #/ self.region_scale #nomalize this
        )

        self.demands_with_depot = self.demands.copy()

    def _go_to(self, destination):

        #print(f'destinations {destination}')
        # if capacity is below the demand, return to depot, restart veh capacity and reduce the number of veh

        ### Ok, I guess I got it..action 21 for instance, this is basically node[destination-1] index in observation arrays. because 0-49 indexing  of arrays.
        # This is due to the dimension mismatch. We have indicies from 1 to 50. and the indexes 0-49 in the demands. We need to substract 1 in order to match arrays..
        any_depot_destinations = destination > 0
        # because actions are calculated from 51 array with depot. and here we wantaccess the nodes f the 50 size, without depot
        demands_indices = destination[destination > 0] - 1 # the same as list comprehension. Substract 1 from all destinations idx, when they are not zero (deopt).

        # TODO go only to nodes, where demand is < capacity. How to implement it and force it?

        demand_exceed_capacity = self.load[any_depot_destinations] < self.demands[demands_indices] # if there are any depot in trajectories, this becomes array.size<n_traj

        # Indices for operations
        destination_zero_condition = destination == 0
        true_indices_ = np.where(demand_exceed_capacity )[0]   # Indices where condition is true
        destination_zero_ = np.where(destination == 0)[0]

        # trajectories here demand exceed capacity or destination is 0
        true_indices = np.unique(np.concatenate((true_indices_, destination_zero_)))

        #check, if the num_veh is not already zero, and if so, delete the index from here.
        # because I don't want to reduce vehicles or give the penalty, when the number of vehicles already zero
        true_indices = true_indices[self.num_veh[true_indices] != 0]

        #if no depot destination and no exceeding capacity. just revert the indices of true array.
        false_indices = np.setdiff1d(np.arange(len(destination)),true_indices)

        reward = np.zeros(len(destination))

        if len(true_indices)>0:
            dest_not_reached = destination[true_indices] #all not reached actions
            self.num_veh[true_indices] -= 1 #if we are already here, decrease the num of vehicles

            # only when  demand exceed capacity, we assign previous node
            last_node = self.nodes[self.last[true_indices]] #get all previous nodes in the same trajectories
            dest_depot = np.zeros_like(dest_not_reached)
            depot_node = self.nodes[dest_depot] #depot are the same in each batch instance.
            dist = self.cost(depot_node, last_node)

            self.last[true_indices] = dest_depot #assign last visited node. in this case this is allways 0, because we return. dest_not_reached

            #set depot to true
            self.visited[true_indices, dest_depot] = True #changed it from dest_not_reached. because only depots are visited in this run, right?

            penalty_ = self.reward_func()
            #+ penalty_[true_indices[return_to_depot_idx]]
            reward[true_indices] = -(dist + penalty_[true_indices]  ) # 1 is a fix costs for vehicles. CHECK. To derive only the distance costs, we need to substract 5 from the total costs. right?
            destination[true_indices] = 0
            # At the moment, I model the case, where the LSP has only one vehicle during the day.
            # I need to reset the traveled distance here if I want to model the case where the service provider operates n vehicles in parallel.
            #
            self.traveled_dist[true_indices] = 0 # in case we model the vehicles one by one, -> self.traveled_dist[true_indices]  += dist
        if len(false_indices)>0:
            dest_next = destination[false_indices]
            dest_node = self.nodes[dest_next]
            dest_depot = np.zeros_like(dest_next)

            dist = self.cost(dest_node, self.nodes[self.last[false_indices]])

            self.load[false_indices] -= self.demands[dest_next-1]
            self.demands_with_depot[dest_next-1] = 0

            #set destination to True, aka reached
            self.visited[false_indices, dest_next] = True

            # unmask the depot if the previous node is not a depo
            self.visited[false_indices, dest_depot] = False

            self.last[false_indices] = dest_next
            reward[false_indices] = -(dist)
            self.traveled_dist[false_indices] += dist

        #need this condition, because otherwise sometimes it assigns 1 to load when we have already used all vehicles
        # ..this can cause some erroneuos behaviour, because we reset the load without changing the vehicle
        condition = (destination == 0)
        self.load[condition] = 1
        #* TODO: check,  it is <0 than it means vehicle has served demand, which has exceed its capacity, and it it not allowed!
        # SO it will be a bug in the future! '
        self.load[self.load<0] = 0
        self.num_veh[self.num_veh<=0] = 0
        self.reward = reward

        #Delete for training
        self.distance = dist

        #print(f'rewards from go_to {reward} \n ')
        #print(f'self.load:  {self.load} \n ')




    def step(self, action):
        # return last state after done,
        # for the sake of PPO's abuse of ff on done observation
        # see https://github.com/opendilab/DI-engine/issues/497
        # Not needed for CleanRL
        # if self.done.all():
        # return self.state, self.reward, self.done, self.info

        return self._STEP(action)

    def reset(self):
        return self._RESET()
