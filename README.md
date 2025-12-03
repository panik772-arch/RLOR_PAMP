## PAMP-VRP: Solving Time-Constrained VRPs with Finite Fleet Using End-to-End Neural Combinatorial Optimization
- End-to-End NCO Model for Vehicle Routing Problem with Time Windows and limited Vehicle Fleet
- The model is based on RLOR framework https://github.com/cpwan/RLOR with enhanced penalty factor for fleet utilization and integrates the DAR method for attention rescalling introduced by Wang et al.2024
(arXiv preprint arXiv:2401.06979)

- To train the model run RLOR_PAMP/ppo_or.py with following parameters: 
-> python /content/RL_PAMP/RLOR/ppo_or.py --num-steps 50 --k-neighbors 3 --total-timesteps 6_000_000_000_000 --env-entry-point envs.cvrp_vehfleet_tw_env:CVRPFleetTWEnv --env-id cvrp-v2 --problem cvrp_fleet_tw --track True --wandb-project-name rlor_finite_vehicle_fleet_env 

- To test and inference run -> validate_vrp_with_tw.py from root

- For the environment see -> cvrp_vehfleet_tw_env.py 
- For the trained model see -> data/pretrained_models
- For the Attention Amplifier or any other policy manipulations see -> RLOR_PAMP/models/attention_model_wrapper.py; RLOR_PAMP/models/nets/ and especially RLOR_PAMP/models/nets/multi_head_attention.py  
