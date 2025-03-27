# Two-timescale-DGL-DDPG
This is the code of the paper:"Joint Service Caching, Communication and Computing Resource Allocation in Collaborative MEC Systems: A DRL-based Two-timescale Approach", which realizes the two-timescale joint optimization of multi-dimensional resources
DGL-DDPG contains 3 tree parts
the first part is the two-timescale multi-edge offloading environment. (core.py and environment.py ) there three other environmrnt for 3.4.5 edge cooperation env eg: environment3 the second part is the small-timescale inproved GA (GA-improve-onf.py) and there three other GA for 3.4.5 edge cooperation env eg: GA_co3 ) the final part is lstm-ddpg(main-lstm-ddpg and model lstm-ddpg and Replay-buffer)
we also add the plt.py and some simulation results.
Deep reinforcemnt learning for resource allocation!

# Paper:  Q. Liu, H. Zhang, X. Zhang and D. Yuan, "Joint Service Caching, Communication and Computing Resource Allocation in Collaborative MEC Systems: A DRL-Based Two-Timescale Approach," in IEEE Transactions on Wireless Communications, vol. 23, no. 10, pp. 15493-15506, Oct. 2024, 
# doi: 10.1109/TWC.2024.3430486.
keywords: {Resource management;Task analysis;Collaboration;Quality of service;Optimization;Energy consumption;Delays;Multi-dimensional resources allocation;collaborative offloading;service caching;two-timescale;long short-term memory (LSTM) network;deep deterministic policy gradient},

# Code structure
# model : the orginal DDPG (model_ddpg.py),  Improved-DDPG (model_twin_ddpg),and we proposed (model-lstmpy)
# envrionment: environment.py ,it has two edge servr and some user decives..... since we claim this algorithm could be used for multiple servers cooperation, we also generated three, four and five edge server environment like environment3/4/5.py
# hence different envrionment setting need different trainning model, so  main_lsm_ddpg for our claim, amd  mian_twin_ddpg for model_twin_ddpg , simarly, for different enveronment seting.
# in this paper we combine the GA algorithm with DDPG, so we design different GA algorthms for different seting, such as the GA-OBF.py (detail you can check the paper we mentioned before)
# core.py : for entity class defination
# Replay_buffer.py for memory the {s,a,s',r}
# we also put some results here

