# Two-timescale-DGL-DDPG
This is the code of the paper:"Joint Service Caching, Communication and Computing Resource Allocation in Collaborative MEC Systems: A DRL-based Two-timescale Approach", which realizes the two-timescale joint optimization of multi-dimensional resources
DGL-DDPG contains 3 tree parts
the first part is the two-timescale multi-edge offloading environment. (core.py and environment.py ) there three other environmrnt for 3.4.5 edge cooperation env eg: environment3 the second part is the small-timescale inproved GA (GA-improve-onf.py) and there three other GA for 3.4.5 edge cooperation env eg: GA_co3 ) the final part is lstm-ddpg(main-lstm-ddpg and model lstm-ddpg and Replay-buffer)
we also add the plt.py and some simulation results.
