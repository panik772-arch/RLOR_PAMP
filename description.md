##### Note! The main working directory here is RLOR! 
## What has been done and what was the purpose he re!

Date: 13.03.2024
+ here I implement the additional environment with finite vehicle fleet.
the issue was to adapt the environment for #ppo, so I can learn the policy with CleanRL framework.

The authors of RLOR https://arxiv.org/pdf/2303.13117.pdf state, that CleanRL was the best RL library for them. They tried RLLIB and other libraries, but CleanRL won
Why I need this? Well, first I want to compare the result of original RLOR code with my experiment in RLLIB.
Here the authors also stated, the RL platforms handle end-2-end NCO different. RL platforms make the total loop training encoder AND decoder, while train.
While original KOOL implementation only adjust the decoder wights with REINFORCE.
However, when we use PPO, especially in CleanRL, we work with Trajectories. And this is a trick.
CleanRL-> https://docs.cleanrl.dev/
but the idea of learning on trajectories in POMO is from here: https://arxiv.org/pdf/2010.16011.pdf
For PPO best practices, tricks and implementation read carefully: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
In CleanRL these trajectories are directly implemented in the environment, whereby in RLLIB for example, we can use standart, simple environment without trajectories view.
It SIMPLIFIES a lot! because the trajectories in the environment is a big pain and costs me 3 days to understand and adjust only one function -> go_to(self) in the cvrp_vehfleet_env
For comparison, when using the RLLib, I need only define my simple problem environment, and the rest, i.e. parallellisation and vectorizing the environment  does RLLib under the hood

for RLLib documentation read also this https://openreview.net/pdf?id=trNDfee72NQ

Lets take a look at the environment. There are many issues with that!
First, you need to understand the concept of batches (like many parallel environment) and trajectories (like simultaneous actions in one timestep in the same environment)
First, we have to recall, that the PPO need trajectories, and they are somehow tricky to understand.
For example

![img.png](img.png "trajectori views")

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
*[Done] Write and check the environment! Write a small class and check and visualize how the environment is working only with 1 trajectory!
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

Date: 28.03.2024
The last updates and experiments:

- I have trained the model in google colab (the most expensive A100 runtime with gpu. It took around 5 hours. see exp17 in wandb https://wandb.ai/dl4log/rlor_finite_vehicle_fleet_env?nw=nwuserdeinelija ) the run i copypasted here:  C:\Users\dein_el\PycharmProjects\rlor_vrp\runs\cvrp-v1__exp17_colabT4_50_steps___1__1711303112- 
- This model behaved not correctly. The trained Agent visited only one customer and returned to the depot.
- The idea was to adjust the reward and to add the penalty as a function of a load of the vehicle. This was inspired by the realization of Prize Collection VRP in RL4OR e.g. However, I do not added prizes or additional embeddings here with the hope, that the reward adjustment will be sufficient
- After several days of experiments I finally got the desired behaviour. The Agent makes tours. What did I do?
  * first I tried the function like penalty_factor * (Sum(all possible demand)/Sum(collected demand so far)) -> It doesn't work
  * Then I just added the load*penalty_factor to the trajectories when the vehicles returns to depot (see indicies_true in _go_to() function in cvrp_vehfleet_env) and it was ok, so far
  * The Idea was, that if the load of the vehicle is not used (vehicle is still full (load = 1) and it returns to depot) then it will be penalized by factor
  * penalty factor 10 was ok, so far. Check the test_env plots in RLOR/envs/test_env.py. You can clearly see, the length of the tour and the assigned reward as a function of distance and load. See also figures : C:\Users\dein_el\PycharmProjects\rlor_vrp\test_env_figures

![example of a bad route](test_env_figures/bat route_check_rewards.png  "example of a bad route")


![example of a good route](test_env_figures/good_root_check_rewards.png  "example of a good route")

What did I change in code?
(commit 5d0ad22 Revision Number->  5d0ad2250aa7fcd25cf1e35ba3a4601a0cba48d5) This commit was also pushed to gitlab 
* basically, I only changed 'env class' . Because in the 'context' for instance I already consider Load, so I hope it will work somehow. If not, maybe I would have to add Price collection or loads to embeddings idk...

#### This was added to the previous `cvrp_fleet_env` version so far:

  * self.penalty = 10 
  * I commented some redundant logics like `traj_with_all_visited_or_no_vehicles' in _Step because is_all_visited works too
  * added this line  `# Not allow to visit nodes when the load is zero
        action_mask[self.load <= 0, 1:] = False` in _update_mask() (not, that in update_mask we smhow set everything to False where "True" was in original self.visited tensor)
 
  * e.g. self.distance = np.zeros(self.n_traj, dtype=float) in RESET is only needed, for test_env class 
###### in the _go_to():
  *  deleted the index selection: `return_to_depot_idx = self.last[true_indices] > 0` because it was somehow not useful, I think
  * added `penalty_ = self.reward_func()` ONLY to true indx meaning, only when the vehicles return to depot not empty
  * BUG: check, `self.load[self.load<0] = 0` because if it is <0 than it means vehicle has served demand, which has exceed its capacity, and it it not allowed! 
  ....SO the BUG can occur in the future! 


####  How to train model in **Google Colab**:
*  Just run the script from your dein_el account and modify if neccessary  `modified_rlor_fleetvrp_env.ipynb` in "Google Drive-Ordner/Colab Notebooks/RLOR/"
#### How to train model on local machine:
* run the command 
  * ` python RLOR/ppo_or.py --num-steps 50 --total-timesteps 20_000_000 --exp-name exp2.0_local_machine_after_refactoring_reward_shaping_ --env-id cvrp-v1 --env-entry-point envs.cvrp_vehfleet_env:CVRPFleetEnv --problem cvrp_fleet --track True --wandb-project-name rlor_finite_vehicle_fleet_env`

#### How to train model on cluster machines e.g. athen or argos:
* so far I started training on two cluster machines. Hów did I do that?
  * first, follow the instruction here `C:\tmp\env_commands_ubuntu.txt`
    * install conda env, install and create virtual_conda env
    * install ipykernel and jupyter in conda env
    * open colab and connect to the local runtime on these machines. 
    * login with you google account on the local machines. Go to "systemwekzeuge/einstellungen/kontos" and add your deinelija@gmail.com konto! It creates and connect your drive automatically. Google Drive will appear in Dateien!
    * now, you can store and retreat the files, like data and so on from there. So, copy the `data` folder from Drive-Ordner/Colab Notebooks/RLOR/ to the cluster machine
    * you can now clone the git repo as shown in the notebook and start it. modify data and files if neccesary 
    * that's all. enjoy. if the port like 8888 for jupyter was used, the wandb will track the learning rates and so on (only argos cluster). in other cases, when oter port was utilised, it will not track directly. 
    * saved checkpoints are also on the local machine in the run_folder

#### Test trained model
* use either the ...\rlor_vrp\model_inference.py or colab inference
* or inference for trained_rlor_cvrp_fleet model.ipynb in google colab. 
  * This is useful, when you trained it in colab for example, on gpu and th model should also be inferenced on gpu. 

#### How to push to gitlab.
* Unfortunately I produced some mess in my git and don't know how to heal this anymore. I tried to delete and to create gits and remotes again and again, but it didn't help. So only this solution works
  * commit and than force push to the remote ONLY the RLOR folder in IDE. Click on Git->right click on the main branch under the Local RLOR and then push, select force push. 
  * it will commit only RLOR folder. the rest (fior examle this file, or checkpoints or data) you have to upload there manually.

#### TODO for 01.04-07.04:
* first of all check the trained cluster model. Take the checkpoints from the clustermachines and transfer it through google_drive and test on your local machine!
* check,analyze. If it works, start to compose and document for the paper. 
* compare this model with he RLLib! plan and execute perfomance test, track and analyze the efficency, solutions and perfomance. Make simple tests...
* start to integrate this model into the global Vehicle Fleet model from RLLIB-> see project RL4Log-> HCVRP

Date: 02.04.2024 after Easter
+ I trained the model on clusters and it seems, it works properly. Look at runs/argos_exp3.2 for the checkpoints
+ I wrote a test class, which can plot both a OR solution and the RL solution, calculates true ditances and also print the collected demand -> `test_and_plot_trained_model.py`
+ also I added the `generate_datasets.py` file to produce the train and valid data. This was done for the new training run on athene machine with more train and valid data. (I geeraed 1000 instances of the train and valid with 20,50,75 and 100 as a graph size)
+ So far, the RL solutions look better than the OR. Not in terms of plots, here it is ot so good. But the distances or demands collected seem to be in a good range. 
![Results.png](runs%2Fargos_exp3.2%2FResults.png)

Date: 03.04.2024

+ I managed to add variable parameters to the environment: ` envs = SyncVectorEnv([make_env(env_id, seed, dict(n_traj=100, max_nodes = 100, max_num_vehicles = 10))])`
  + here I added `self.max_nodes = self.envs[0].max_nodes
        self.max_num_vehicles = self.envs[0].max_num_vehicles`  to the class ` class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments. `  and it overwrites the arguments in the env somehow automatically.

  + ##TODO: Try to adjust the penalty in the environmnt to controll the agent behaviour. Learn policy with penalty  = 1 and penalty= 30

DATE: 04.04.2024
* Idea. Maybe I need to integrate num_customers into the reward_function and not only penalize the agent on Load-basis. Instead, the number of served customers counts! Because in VRP each customer is equivalent, and this how bg is his order. Or Just add +(num_cust/total_demand)/penalty
* integrate different reward functions. (Num_of_collected_customers + )*penalty + 
###### TODO:     ## look https://arxiv.org/pdf/2401.06979.pdf the authors propose to scale this score with log(d_i,j) in order to mitigate the attention score dispersion!
        ## The DAR method amplifies the distance scores of the nearest
        ## neighbors to the current node using a logarithmic function,
        ## which guides the NS to prioritize nodes that are closer, even
        ## when the attention scores are similar.
        ## (2) Identification of random high-score nodes. DAR assists NS in excluding distant and irrelevant high-score nodes
        ## by penalizing faraway nodes.

Date: 05.04.2024

* added the functions for node amplifying from Wang2024. currently in attention_model_wrapper.py in `state`
* -> new branch RLOR -> distance_matrix_for_decoder
* added this implementation in the Attention Score. TEST it
* to integrate state in the attention, added the state variable to function_input ->  `def forward(self, query, key, state ,mask=torch.zeros([], dtype=torch.bool)): `
+ added state variable also to the upper function pointer() ` logits = self.pointer(glimpse, state, logit_K, mask) # in cvrp_env -> with mask in decoder e receive attentScrore of the shape (1025,50,51) as same as mask shape! with values like [-9,89, -inf, -inf...]. So depot is not -inf and all other -inf,,why?
`and `logits, glimpse = self.calc_logits(query,state, glimpse_K, glimpse_V, logit_K, mask)` to the Decoder-> advance()
+ started to implement for testing `models/test_models_with_attention_distance_enhancement.py`
+ ###### TODO: Test it. Maybe implement the 10 nodes problem and track the distribution of the attention scores to nodes
+ ###### MAYBE NOT MINUS LOG DIST BUT PLUS IN MY CASE

Date: 06.04.2024
###### Check, Im not sure, that a current node is a last_nd index wat I have. Check this...
because I provde in attention stateWrapper in decoder basically previous node..Is this a current load? Im not so sure..I have to use embedding and identify the index better...

-> now it is also possible to set args paraeter k in training. Try to train with k = 5, k=20, and k = 50 for example

##### Think about the A* algorithm. By implementing state and distance directly into the decoder we have the opportunity to directly scale decoder weights, like we do this in AStar heuristics. Multply the istance by someA factor. Cool, we so implement the RLOR- heuristics and manipulate the decoder weights.

Date: 08.04.2024
* added penalty for the num of unassigned customers. When only depot is visited (1) then it adds 3.9 to the penalty. when 10 customers visited -> log() is 1.5. 1 is by ca. 20 customers. 
and when more it decreases logarithm self.penalty * self.load + np.log(self.max_nodes/visited_traj)
* train with: python RLOR/ppo_or.py --num-steps 50 --k-neighbors = 3 --total-timesteps 20_000_000 --exp-name exp4.1_with_AttentionScore_Enhancing --env-id cvrp-v1 --env-entry-point envs.cvrp_vehfleet_env:CVRPFleetEnv --problem cvrp_fleet --track False --wandb-project-name rlor_finite_vehicle_fleet_env
* started experiments exp 5.2 and 5.3 on 08.04 at 17:50 with DAR method on argos with "k = 10, penalty = 10" and athene with k = 20 and penalty = 5. 

Date: 09.04.2024

It makes basically no sense to add DAR method to POMO, because POMO utilize the idea of multiple optimas at the same time and explore the entire solution space at th same time. 
Whereby DAR method is better for step-by-step solution construction. It would be better for RL, so it can select only neighbor nodes...
TODO: Try to implemen DAR Method with RLLib and compare to POMO CleanRL... 


18.04.2024

I implemented the DAR method for Time sensitive (time preferences) constraints. 
Assume the case, where the vehicle has limited fleet, cvrp AND some time preferences for customers. 
First, in that case I adjusted the environment: cvrp_vehfleet_tw_env.py
added  
` obs["tw"] = self.tw` and 
        `obs["traveled_dist"] = self.traveled_dist` to the observation space!
- in te environment I also added **traveled_time** to the observation, to be able to track the time the vehicle traveled so far. If it returns to depot, it is set to 0 in these trajectories!!
- Why? because I want to be able to add this time later in "time_to_reach" function to make a prediction, if the customers are reachable or not, and if not. I apply -10 to attenion enhancement factor there. for example 
- refactored the problem added "cvrp_fleet_tw" extension to encoder and context
- added --vehicle-speed args in the problem set up in `ppo_or_.py` class 

At the moment I generate the **TW** randomly in **RESET**
but it would be better, to create or take some benchmark problems for evaluation and training. 

Further refactoring steps

- in `attention_model_wrapper.py` I added following function:
- `def times_to_reach(self, v = 10): ###` v = vehicle speed! with 10 the traveled distances are small, in a range: 0.01 and 0.1

- where basicly the attention factors are calculated. The main idea is to calculate some attention enhancement factor as 
- **5/(5x+1)**  for example, where 5 is a scalar to scale/adjust the ratio function. It means this function is almost linear relation within 0 and 5 max. If the x ->0 is small (time preference. means sth like !!hurry up, its urgent!!), the factor is ratio->5
- if x is -> 5, (so no strong time pressure), the ratio -> 0.BUT if it is smaller than 0, then we already missed the customer, and we do not care about it anymore. so, the attention factor becomes -10
`torch.where(difference_factor > 0, 5 / (5 * difference_factor + 1), -10)`
- I also wrote a function to test this and check this, `RLOR/models/test_tw_with_attenion_enhancing.py`
- additionally, I added "cvrp_fleet_tw" problem and if this is given, we set `use_tanh = False` in MHA attention Score, because I suppose that tanh distribute u values between 0 and 1. like this 
![img_1.png](img_1.png) or 
![img_2.png](img_2.png)

so, if the attention scores are between -5 and 5, it makes no difference for attention enhancement here. right?

- added self.v and ATTENTION_SCALING scalars to the stateWrapper. Now its the only one place that control the speed and attention scaling and should be set up before training, for example. 
- TODO: check the attention factor dispersion! With and WIthout DAR! 
- for publication: why we do not set the penalty for missing cutomers? Because the customers are optional. for dyamical problems, we do not want to penalize the agent, if it missed today, because the parcel can be delivered tomorrow! 

25.04.2024
- I checked the DAR models ( `runs/argos_exp5.5_DAR_k50_` and `runs/Athene_exp5.4_DAR_k3_noCustomerPenalty` )
-  first observation: - it seems that the no_customer_penalty works better. So, get rif of this function with ln sth....
- second observation: - the DAR method performs somehow slightly better than normal network, with the same number of iteration...

25.04.2024
- TODO: plot distribution for U with tanh and without tanh function with ratio and without ratio! 
- - train the model!
- 
02.05.2024
- I add the PAMP method. So it is big t describe, but the main idea is to add a time factor, which is calculated as a travel time to the customer from the current node divided by the deadline (deadline = fixed time constraint - already traveled time so far)
- so the method utilize the DAR Idea, enhancing the attention score in pointer. If the time-factor is greater 1 or less than 0 (0<tf<1) the nodes will be masked. It means, the time deadline it expired and the customer can not be visited. 
- than we add this time factor to the TANH(u) + tf and THAN do clipping -> (tanh(u) + tf)+Clipping ! Originally, it was tanh(u+tf) * CLipping. I want just to ensure, that u and tf are approximately in the same range!
  - check both. maybe the first option will work, maybe second, maybe it makes no difference. How to check?
  - first train with the normal original setup and check: tanh(u+tf) *Clipping
- I also added TW to the embeddings! 
- to change the TW range we need to do following:
  - -> create a new evaluation set with new range for evaluation
  - -> change the range in the environment in reset load and radom generate
  - -> change the range in the stateWrapper! 
- I started 2 simulations /training on server: check the file how to start it on: C:\tmp\CLUSTER THROUGH PUTTY(train nco model).txt
- one is on argos from the repo rlor_fleet:
https://cluster:n79CvMEKFC22Esoys_D_@gitlab.dlr.de/StratGut/rl4log/end2end-nco-for-vrp-with-finite-fleet/rlor_vrp.git
  - here is a current code with 100-15000 time scale and standart Tanh(u+timeratio) *Clipp for ppo! 
- another one runs on herakles from repo  clone https://colab:M7fg9mq5aGJapfw5U_P6@gitlab.dlr.de/StratGut/rl4log/end2end-nco-for-vrp-with-finite-fleet/rlor_vrp_DAR_method.git
  - here is also a PAMP method but with 50-10000 scale and standrt tanh as well. Wait for 5 days ad look what happens..
  
03.05.2024
try to use only nodes in dynamic embeddings and in context only trajectories. will it work?
Context:

ratio, deadlines_requested, traveled_time, calculate_times_from_this_to_all = state.tw_ratio()
        ratio_trajectories = ratio.squeeze(-1)

Dynamic Embeddings: ratio, deadlines_requested, traveled_time, calculate_times_from_this_to_all = state.tw_ratio()
        ratio_nodes = ratio.squeeze(-2)

I Changed:  self.context_dim = embedding_dim + 3  in Embeddings, added to context the travel times and to the dynamic embeddings TW

06.05.2024

- my insight from multiple train runs is...The tw and context do not work if you train it with PAMP. 
- the method could converge ONLY if I set up context and embeddings the same as in normal CVRPfleetEnv and do not provide additional informations..smhw..
- so, its important, now I ONLY train with pure attention score enhancing in pointer, without any additional features. WHy the network can not capture time information? Idk...maybe we need to provide any feedback in environment as well, so this information is somehow reflected in training and there is some signal from the environment...
- BUT: take this to you paper and describe: 

Soft Constraints CVRP: 
![Figure_check_the_PAMP_logits.png](Figure_check_the_PAMP_logits.png)

![Figure_check_the_PAMP_routes.png](Figure_check_the_PAMP_routes.png)

for validation_vrp_with_tw and this settings:
 max_nodes = 10
    n_traj = 10
    eval_data = False
    eval_partition = "eval"
    eval_data_idx = 0
    region_scale = 10000
    min_tw = 0
    max_tw = 100
    vehicles = 5
    v = 50 #speed in km/h
    prize_for_visiting = 10000 #10000 #2000 min and 10000 max
    seed = 1234
    ckpt_path =  "C:\\rlor_pamp_trained_models\\exp8.3_herakles_noEMbeddingsAndContext\\800.pt"

- the heatmap fits very good to the routes. it selects best logits..the n4 node is than -10 because the tw is expired...so its more soft constraints?

- if you want to implement HARD constraint, then use:
- 
-         if self.use_tanh:
            logits = (torch.tanh(u)+time_ratio) * self.C
Tha we have hard constrained VRP like this: 
![Figure_check_the_PAMP_logits_HARD_Constraints.png](Figure_check_the_PAMP_logits_HARD_Constraints.png)
![Figure_check_the_PAMP_logits_HARD_Constraints_Routes.png](Figure_check_the_PAMP_logits_HARD_Constraints_Routes.png)

08.05.2024
- I suppose, that my PAMP method with context and embeddings, where I add the time windows to embeddings and traveled time to context, didn't work because I didn't normalize these values....can it be true?