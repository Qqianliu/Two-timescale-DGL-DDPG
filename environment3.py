# 20240603
# by qian
# 建一个智能体学习的环境，然后这个是一个完全自主的开始，因为开始简化以前的描述，
# 使得整个方法只有一个类和一个环境和一个网络部分

import gym
from gym import spaces
import numpy as np
from Two_MEC_env.core import *



# environment for all agents in the multi-agent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    def __init__(self):
        self.episode = 0
        # how many steps, do one caching decision
        self.cache_decision_fre  = 5
        # how many steps, update the  task which user request
        self.task_update_fre = 1
        # list of agents and entities (can change at execution-time!)
        self.num_User = 30
        self.num_Server =3
        self.num_task  = 10
        self.alpha = 0.5
        self.beta = 0.5
        self.xi = 1
        self.white_noise = -114  # dbm
        self.User = []
        #  this only one server
        self.Server = []
        self.Tasks = []
        # position dimensionality
        self.dim_p = 2
        self.tStoC = 0.02  # 100-200ms
        # large state for caching
        self.state = []
        self.service_offload_pop  = None
        self.service_gain = None
        self.cache_state = None
        # add
        # edge server id from 1: location :two dimension ; com_cap: 20GHZ; pow : 1w
        # bandwidth_cap: 5MHZ; cache and next cache = [0];
        # cache_cap :300GB ;backhual_rate : 100Mbps（100mpbs 云端很远；）;
        # 在初始化定义时，所有的单位都是基本单位。bits\
        # def __init__(self, index, type, loc, power, com_cap, bandwidth_cap, cache, next_cache, cache_cap,
        #            backhaul_rate):
        self.Server = [Server((i + 1), 'server%d' % (i + 1), np.zeros(2), 1, 20 * (10 ** 9),
                               20 * (10 ** 6), np.zeros(self.num_task), np.zeros(self.num_task),
                               200 * (1024 * 1024 * 1024 * 8), 200 * (1024 * 1024 * 8)) for i in
                        range(self.num_Server)]
        # user id from 0; location :fixed two dimension; com_cap : 1GHZ; power:20 dbm ; energy consumption: 5*10**-27
        # def __init__(self, index, type, loc, power, compfre):
        self.User = [User(i, 'user %d' % i, np.zeros(2), 20,1 * (10 ** 9), 5 * (10 ** -27))
                      for i in range(self.num_User)]

        # def __init__(self, index, cache_size 800MB-4000MB , com_fre 400cycle/1000cycle, size, 50kB- 500KB)
        #__init__(self, index, cache_size, com_fre, size):
        self.Tasks = [Task((i + 1), 2000, 400, 500) for i in range(self.num_task)]

        with open("task_characteristic.txt", "w") as f:
            total_size = 0
            for task in self.Tasks:
                f.write("size" + str(task.get_size/(1024 * 8)) + "\t")
                f.write("cycle" + str(task.get_cycle/ (1024 * 8)) + "\t")
                f.write("cache" + str(task.get_cache_size / (1024 *1024 * 8)) + "\n")
                total_size += task.get_cache_size
            f.write("total_cache_size" + str(total_size/ (1024*1024 *1024 * 8)))

        # agent_action and state space
        self.action_space_server_cache = []
        self.observation_space_server_cache = []

        self.ac_dim = self.num_task * self.num_Server
        self.action_space_server_cache.append(
            spaces.Box(low=-1, high=1, shape=(self.ac_dim,),
                       dtype=np.float32))
        self.obs_dim = self.num_task * (1+2*self.num_Server)
        # cache state of number of server
        # question: why not define obs_dim directly
        self.observation_space_server_cache.append(
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32))
        # initialize the location of user and server
        # edge server is 5G BS ,the coverage diameter is 250 m



    # obtain the observation from the env
    def _get_obs(self):
        state = []
        cache_state = []
        for server in self.Server:
            for c in server.cache:
                cache_state.append(c)
        # set the initial state :[0]
        self.cache_state = cache_state
        for i in self.cache_state:
            state.append(i)
        for i in self.service_gain:
            state.append(i)
        for i in self.service_offload_pop:
            state.append(i)
        self.state = np.array(state)
        # print("len",len(self.state))
        return self.state

    # initialize the environment
    def reset_env(self):
        print("reset_env!!!")
        np.random.seed(int(self.episode))
        #the communication range of the ES is 200*200 m
        # initialize the location of ES
        for i in range(self.num_Server):
         self.Server[i].loc = [(2*i+1)*100,0]
        # initialize the location of TDs
        ave_user= self.num_User/self.num_Server
        for i in range(self.num_User):
            server_id  =  int(i/ave_user)
            loc_x = np.random.randint(low=-100, high=100) + self.Server[server_id].loc[0]
            loc_y = np.random.randint(low=-100, high=100)+ self.Server[server_id].loc[1]
            loc = []
            loc.append(loc_x)
            loc.append(loc_y)
            self.User[i].loc = loc
        # clear the caching configration of ES
        cache = np.zeros(self.num_task)
        cache_state = []
        for server in self.Server:
            server.cache = cache
            server.next_cache = cache
            for c in server.cache:
                cache_state.append(c)
        # set the initial state :[0]
        self.cache_state = cache_state
        self.service_gain = np.zeros(self.num_task)
        self.service_offload_pop = np.zeros(self.num_task*self.num_Server)
        # reset the user's requset
        self.updat_user_requset()
        print("rest env end ")
        return self._get_obs()
    # Zipf distribution of user's request
    def number_of_certain_probability(self, sequence, probability):
        x = np.random.random()
        # print("x", x)
        cumulative_probability = 0.0
        # print("sequence", sequence)
        for item, item_probability in zip(sequence, probability):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        # print(item)
        return item

    def updat_user_requset(self):
        for user in self.User:
            p = [0.21229198890532444, 0.1219297292172975, 0.08815289960395892, 0.07003023968857178,
                 0.05858111079665425,
                 0.050630545381643195, 0.04475640280685803, 0.04022181056515523, 0.036604931484490705,
                 0.033646012803007885,
                 0.031175932876210673, 0.029079612096248132, 0.02727588960294407, 0.02570580313991125,
                 0.02432543406547894,
                 0.023101363815598092, 0.02200769044526022, 0.02102401229050682, 0.02013403063536803,
                 0.019324559779512168]
            a = [self.num_task - i for i in range(self.num_task)]
            # print("a",a,p[0:self.num_task])
            user.request = self.number_of_certain_probability(a, p[0:self.num_task])
            user.task_size = self.Tasks[user.request - 1].get_size
            user.task_cycle = self.Tasks[user.request - 1].get_cycle

    def _set_action_cache_popular(self,action):
        for i, server in enumerate(self.Server):
            server.cache =action
            print("after the maping :server%d.cache"%i,server.cache)
    def _set_action_cache_ga(self,action):
        cache = []
        # print("action", action)
        for i in action:
            if i > 0:
                cache.append(1)
            else:
                cache.append(0)
        for i, server in enumerate(self.Server):
            # print(cache[i * self.num_task:(i + 1) * self.num_task])
            server.cache = cache[i * self.num_task:(i + 1) * self.num_task]
            # print("after the maping :server%d.cache" % i, server.cache)
    def _set_action_cache(self, action):
        #print("action",action)
        cache = []

        for i in action:
            if i > 0:
                cache.append(1)
            else:
                cache.append(0)
        for i, server in enumerate(self.Server):
            # print(cache[i*self.num_task:(i+1)*self.num_task])
            server.cache = cache[i*self.num_task:(i+1)*self.num_task]
            # print("after the maping :server%d.cache"%i,server.cache)

    def _set_action_bandwidth(self,action):
        for i, band in enumerate(action):
            # 这里是[0,1] 的连续变量
            if band ==0:
                self.User[i].bandwidth = 0.01
                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            else:
                self.User[i].bandwidth = band

    def _set_action_frequency(self, action):
        for i, fre in enumerate(action):
            # 这里是[0,1] 的连续变量
            self.User[i].frequency = fre

    def _set_action_offload(self,action):
        # 这里是[0,1,2] 的离散的卸载
        for i, off in enumerate(action):
            self.User[i].offload = off





    def get_gain(self, server, user):
        ch_gains = 0
        server_loc = server.get_loc
        user_loc = user.get_loc
        # print('获取MBS与对应用户的增益', bs_loc, user_loc, bs_loc[0], user_loc[0], bs_loc[2], user_loc[1])
        distance = pow(pow((server_loc[0] - user_loc[0]), 2) + pow((server_loc[1] - user_loc[1]), 2), 0.5)
        ch_gains = 128.1 + 37.6 * np.log10(distance / 1000)  # m 转 distance KM
        # print("ch_gains:***************************ch_gains:")
        # print(ch_gains)
        # shadow = np.random.lognormal(0, 8, 1)
        # print(ch_gains, shadow)
        # shadow = 0
        ch_gains = pow(10, -ch_gains / 10)  # db  to w
        rayleigh = np.random.rayleigh(1)
        #rayleigh = 1
        gains = ch_gains * rayleigh  # w
        # print("gain",gains)
        return gains

    # GA算法解决offload &computing and  bandwidth 的主函数！
    def get_ob_data(self,ga, generations):
        qoe, t, e,off,band,fre = [],[],[],[],[],[]
        for generation in range(generations):
            # print("ga _pop",ga.pop)
            pop_off, pop_band,pop_fre = ga.translateDNA(ga.pop)
            # print("adter translate",ga.pop)
            # print("pop_off,pop_band",pop_band,len(pop_off))
            fitness, time, energy = ga.get_fitness(pop_off, pop_band,pop_fre)
            ga.evolve(fitness)
            best_idx = np.argmax(fitness)
            # print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
            #qoe.append(np.mean(fitness))
            qoe.append(fitness[best_idx])
            t.append(time[best_idx])
            e.append(energy[best_idx])
            off.append(pop_off[best_idx])
            band.append(pop_band[best_idx])
            fre.append(pop_fre[best_idx])
            #print(t[-1])
        # return qoe[-1], t[-1], e[-1],off[-1],band[-1],fre[-1]
        return qoe, t[-1], e[-1], off[-1], band[-1], fre[-1]
    # GA 解决一步的缓存，计算，通信资源分配
    def get_ob_data_3c(self,ga, generations):
        qoe, t, e, off, band, fre, cache = [], [], [], [], [], [], []
        for generation in range(generations):
            pop_off, pop_band, pop_fre, pop_cache = ga.translateDNA(ga.pop)
            # print("pop_off,pop_band",len(pop_band),len(pop_off))
            fitness, time, energy = ga.get_fitness(pop_off, pop_band, pop_fre, pop_cache)
            ga.evolve(fitness)
            best_idx = np.argmax(fitness)
            # print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
            qoe.append(fitness[best_idx])
            t.append(time[best_idx])
            e.append(energy[best_idx])
            off.append(pop_off[best_idx])
            band.append(pop_band[best_idx])
            fre.append(pop_fre[best_idx])
            cache.append(pop_cache[best_idx])
        return qoe[-1], t[-1], e[-1], off[-1], band[-1], fre[-1], cache[-1]

    def ger_reward_time_energy(self,user, env):
        server_id = int(user.offload) - 1
        # print("user.offload",user.offload,"server_id",server_id)
        server = env.Server[server_id]
        num_server_user = int(env.num_User / env.num_Server)
        if user.get_id < num_server_user:
            server_associate = env.Server[0]
            # print("user_id", user.get_id, "server", env.server[0].get_type)
        elif user.get_id >= num_server_user and user.get_id < 2 * num_server_user:
            server_associate = env.Server[1]
            # print("user_id", user.get_id, "server", env.server[1].get_type)
        else:
            server_associate = env.Server[2]
            # print("user_id",user.get_id,"server",env.server[2].get_type)
        task_size = user.get_task_size
        task_cycle = user.get_task_cycle
        # print("task_size and task cycle",task_size,task_cycle)
        com_fre = server.get_com_cap * user.get_frequency
        # print("user frequence", user.get_frequency, )

        # edge computing time
        if com_fre == 0:
            print(user.get_type, user.get_offload)
            print(user.get_frequency)
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!wrong")
        t_exe = task_cycle / com_fre
        # print("t_exe",t_exe)
        # user uplink bandwidth
        bandwidth = user.get_bandwidth * server.get_bandwidth_cap
        channel_gain = env.get_gain(server_associate, user)  # w
        power = pow(10, (user.get_power - 30) / 10)  # 20dbm to w - 0.1w
        self_gain = channel_gain * power
        white_noise_pow = pow(10, (-114 - 30) / 10)  # -144 dbm to w
        # uplink rate
        # print(np.log2(1 + self_gain / white_noise_pow))
        rate = bandwidth * np.log2(1 + self_gain / white_noise_pow)
        # from user to es  : transmission time
        if rate == 0:
            print(user.get_type, user.get_offload)
            # print("userbandwidth", user.get_bandwidth)
            # print("bandwidth!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!wrong")
        # print("bandwidth and rate",user.bandwidth, rate)
        t_trans = task_size / rate
        # print("transmission time", t_trans)

        # 这里是以瓦为单位还是以DBM 为单位呢？需要和本地计算的功率做一下比较
        e_trans = user.get_power * t_trans
        # print("e_trans", e_trans)
        if server.cache[user.get_request - 1] == 1:
            # print("cached all the service")
            t_total = t_exe + t_trans
            # print("edge computing time", t_total)
        else:
            #  cloud computing frequency：4GHZ
            t_total = t_trans + task_size / (4 * 1024 * 1024) + task_cycle / (4 * 10 ** 9) + 0.1
            # print("cloud computing time", t_total)
        return t_total, e_trans
