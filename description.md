## What has been done and what was the purpose here!

Date: 13.03.2024
+ here I implement the additional environment with finite vehicle fleet.
the issue was to adapt the environment for #ppo, so I can learn the policy with CleanRL framework.

The authors of RLOR https://arxiv.org/pdf/2303.13117.pdf state, that CleanRL was the best RL library for them. They tried RLLIB and other libraries, but CleanRL won
Why I need this? Well, first I want to compare the result of original RLOR code with my experiment in RLLIB.
Here the authors also stated, the RL platforms handle end-2-end NCO different. RL platforms make the total loop training encoder AND decoder, while train.
While original KOOL implementation only adjust the decoder wights with REINFORCE.
However, when we use PPO, especially in CleanRL, we work with Trajectories. And this is a trick.
CleanRL-> https://docs.cleanrl.dev/
For PPO best practices, tricks and implementation read carefully: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
In CleanRL these trajectories are directly implemented in the environment, whereby in RLLIB for example, we can use standart, simple environment without trajectories view.
It SIMPLIFIES a lot! because the trajectories in the environment is a big pain and costs me 3 days to understand and adjust only one function -> go_to(self) in the cvrp_vehfleet_env
For comparison, when using the RLLib, I need only define my simple problem environment, and the rest, i.e. parallellisation and vectorizing the environment  does RLLib under the hood

for RLLib documentation read also this https://openreview.net/pdf?id=trNDfee72NQ

Lets take a look on the environment. There are many issues with that!
First, you need to understand the concept of batches (like many parallel environment) and trajectories (like simultaneous actions in one timestep in the same environment)
First, we have to recall, that the PPO need trajectories, and they are somehow tricky to understand.
For example

 `![img.png](img.png "trajectori views")

so, as you see, in ppo_or.py algorithm, we sample in each 

    '''
     ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, _ = agent.get_action_and_value_cached(
                        next_obs, state=encoder_state
                    )
                    action = action.view(args.num_envs, args.n_traj)"
    '''

so the action becomes an 1D array of the shape (35,) when nr_trajectories in env is 35

next, we neet to adapt the environment respectively. For instance THE cvrp_vehfleet_env.py class:
we have some data_objects, like #nodes, #action_mask, #depot and #demand, which are the same for each batch.
(However, the demand is altered in each move! Still struggling to understand, why we do not copy it with each trajectory)

So, some data objects are of the shape [50] (nodes for example), the mask has the shape (35,51) where 35 are trajectories and 51 are nodes with depot
demand has size 50 (nr of nodes) but no trajectories...

However, self.done are trajectories (size 35), as well as actions (35 actions in each time step)
load.size() -> (35,);
last.size() -> (35,)
AND hence, => nr_veh (35,)

    '''
        self.num_veh = np.array([self.max_num_vehicles] * self.n_traj)
        self.done = np.array([False] * self.n_traj)
    '''

This basically says, that in each step, we  proceed different trajectories, some of them can end in dones soner, other later. 
The batch start to learn, when all trajectories dones are TRUE! 

____

Now, lets take a look on the go_to() function (causes the most pain!)

    '''
    def _go_to(self, destination):
        '''

The theory behind this was, to first identify the indexes for allowable actions, nodes that can be reached. 
And the not reachable actions, like when destination is already 0 or the demand exceed the vehicle load! 

    ''' 
        print(f'destinations {destination}')
        # if capacity is below the demand, return to depot, restart veh capacity and reduce the number of veh

        ### Ok, I guess I got it..action 21 for instance, this is basically node[destination-1] index in observation arrays. because 0-49 indexing  of arrays.
        # This is due to the dimension mismatch. We have indicies from 1 to 50. and the indexes 0-49 in the demands. We need to substract 1 in order to match arrays..
        any_depot_destinations = destination > 0
        # because actions are calculated from 51 array with depot. and here we wantaccess the nodes f the 50 size, without depot
        demands_indices = destination[destination > 0] - 1 # the same as list comprehension. Substract 1 from all destinations idx, when they are not zero (deopt).

        demand_exceed_capacity = self.load[any_depot_destinations] < self.demands[demands_indices] # if there any depot in trajectories, this becomes array.size<n_traj
        # Indices for operations
        destination_zero_condition = destination == 0
        true_indices_ = np.where(demand_exceed_capacity )[0]   # Indices where condition is true
        false_indices_ = np.where(~demand_exceed_capacity)[0]  # Indices where condition is false

        destination_zero_ = np.where(destination == 0)[0]
        destination_not_zero_ = np.where(destination != 0)[0]

        true_indices = np.unique(np.concatenate((true_indices_, destination_zero_)))

        #if no depot destination and no exceeding capacity. just revert the indices array where these conditions are true.
        false_indices = np.setdiff1d(np.arange(len(destination)),true_indices)

        reward = np.zeros(len(destination))
    ''' 


when this condition is true, then return the vehicle back to the depot: 

    '''
        if len(true_indices)>0:
            dest_not_reached = destination[true_indices]
            self.num_veh[true_indices] -= 1

            last_node = self.nodes[self.last[true_indices]]
            dest_depot = np.zeros_like(dest_not_reached)
            depot_node = self.nodes[dest_depot]
            dist = self.cost(depot_node, last_node)

            self.last[true_indices] = dest_not_reached

            #set depot to true
            self.visited[true_indices, dest_not_reached] = True

            #set the destination to False aka not reached
            #self.visited[true_indices, dest_not_reached] = True

            destination[true_indices] = 0
            reward[true_indices] = -dist
    '''

Note, how we indexing and selecting the trajectories!
For example, 
+ **dest_not_reached = destination[true_indices]** selects all actions *destination* where the nodes can not be reached

* then, we reduce the num_veh `self.num_veh[true_indices] -= 1` 

* we assign the array of last nodes of the shape (35,) th same last nodes from previous step
`last_node = self.nodes[self.last[true_indices]]`. Why? because we either didn't reach this node because of capacity constrains, OR the action is already 0

* then we calculate the distance between last nodes and the depot `dest_depot = np.zeros_like(dest_not_reached)
            ; depot_node = self.nodes[dest_depot]; 
            dist = self.cost(depot_node, last_node)` 
* and assign destinations only to the not_reached nodes , `self.last[true_indices] = dest_not_reached` (note, I changed this line to `self.last[true_indices] = dest_depot` because if the destination is zero, so the last node is also zero. and if it was not reached and returned to depot, then the last node is also zero. so in both cases we have zeros)
* the next line sets all visited nodes to True. In this case, only depot nodes `self.visited[true_indices, dest_depot] = True`
* at the end, when we return to depot, we set the actions to zero manually. And assign reward only to processed trajectories!


Next we do similarly to the trajectories, which were reached and where the vehicle moved from A to B. 
 
    '''
        if len(false_indices)>0:
            dest_next = destination[false_indices]
            dest_node = self.nodes[dest_next]
            dest_depot = np.zeros_like(dest_next)

            dist = self.cost(dest_node, self.nodes[self.last[false_indices]])

            self.last[false_indices] = dest_next

            self.load[false_indices] -= self.demands[dest_next-1]

            self.demands_with_depot[dest_next-1] = 0

            #set destination to True, aka reached
            self.visited[false_indices, dest_next] = True

            # unmask the depot
            self.visited[false_indices, dest_depot] = False

            reward[false_indices] = -dist
    '''

* the steps are similar, only that we assign to the last node the destinations `self.last[false_indices] = dest_next`
* and substract the demand from the load `self.load[false_indices] -= self.demands[dest_next-1]` Here is little bit unclear for me, WHY we substract all trajectories from the same load array. SO if we have different actions at the same time, can we substract from the same load different nodes many times from different trajectories? and when so, how to handle it, is it correct? IDK
* also we unmask the depot. 
* the line `self.demands_with_depot[dest_next-1] = 0` looks unnecessary to me. but however  


At the very end I do.

    '''
        self.load[destination == 0] = 1
        self.load[self.load<0] = 0
        self.num_veh[self.num_veh<=0] = 0
        self.reward = reward
    '''
..for all indices, for consistency

_______________________

I also added `self.no_other_vehicles()` in `def _update_mask(self):` in order to MASK the whole row of all nodes, when the number of vehicles in specific trajectory is 0! so it is additional check for this trajectory

CHECK THIS!
Maybe I will rise the problem, that the Agent will try to force the full usage of vehicles, because then it has less tours to drive, less milage and hence, bigger reward!
CHECK THIS!

Finally, because the all_visited function checks all visited nodes in trajectories and nodes (visited array has the shape (35,51)), the episode is done, where all nodes are visited. And they are visited, when there are no vehicles in trajectory!

_____

The last major adjustment was context class.
I also added 
    
    '''
    def _state_embedding(self, embeddings, state):
        state_embedding = -state.used_capacity[:, :, None]
        vehicles = state.get_num_veh()
        state_embedding = torch.cat(( state_embedding,vehicles[:,:, None]),-1) #(1024,35,2)
        return state_embedding
    '''
in context.py to include the vehicle constrain in the context state for decoder..

AND the class 
**class CVRPFleetEmbeddings(nn.Module):**

    '''
    def __init__(self, embedding_dim):
        super(CVRPFleetEmbeddings, self).__init__()
        node_dim = 3  # x, y, demand
        scalar = 1

        self.context_dim = embedding_dim + 2  # Embedding of last node + remaining_capacity AND num_vehicles
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # depot embedding PLUS num of vehicles as the 3 feature

    def forward(self, input):  # dict of 'loc', 'demand', 'depot', num_vehicles
        # batch, 1, 2 -> batch, 1, embedding_dim
        depot_embedding = self.init_embed_depot(input["depot"])[:, None, :]

        node_embeddings = self.init_embed(
            torch.cat((input["loc"], input["demand"][:, :, None]), -1)
        )
        out = torch.cat((depot_embedding, node_embeddings), 1)
        return out
    '''
____


NEXT STEPS:

TODO:
* Write and check the environment! Write a small class and check and visualize how the environment is working only with 1 trajectory!
* Try to deploy and train the model on cluster! Use some parallel computing and so on! 
* Check the model and compare with the RLLIB
* Check, if the agent behaves properly. If not, look in cvrp_flet_env class. I suppose, that the agent will pursuit to reduce the number of vehicles as quick as possible, to achieve minim reward! 
* 
compare the vrp solution with heuristic in python using this lib:
https://pyvrp.readthedocs.io/en/latest/examples/basic_vrps.html

Date: 14.03.2024

some improvements. Added the fixcosts when the vehicles return to depot in go_to().
`reward[true_indices[return_to_depot_idx]] = -(dist[return_to_depot_idx] + 1)`
- 1 is a fix costs for vehicles. CHECK. 
- implemented additional filter, if the vehicle is staying in depot, do not assign reward!! 
-
Realized validation i.e. Heuristic approach and plotting of the problem!
[heuristic_solver.py](RLOR%2Fenvs%2Fheuristic_solver.py)
[plot_env.py](RLOR%2Fenvs%2Fplot_env.py)
[test_env.py](RLOR%2Fenvs%2Ftest_env.py)