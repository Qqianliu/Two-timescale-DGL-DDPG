# time 20221115日
# by qian
# 利用ga 算法解决用户的卸载和带宽分配 和计算资源分配的问题
# 带宽分配和计算资源的分配采用连续的变量求解
import numpy as np
from Two_MEC_env.core import *
import cvxpy as cvx
import time as time




# GA算法解决offload & bandwidth & computing frequency的主函数！
def get_ob_data(ga, generations):
    qoe, t, e,off,band,fre = [],[],[],[],[],[]
    for generation in range(generations):
        pop_off, pop_band,pop_fre = ga.translateDNA(ga.pop)
        #print("pop_off,pop_band",len(pop_band),len(pop_off))
        fitness, time, energy = ga.get_fitness(pop_off, pop_band,pop_fre)
        ga.evolve(fitness)
        best_idx = np.argmax(fitness)
        # print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
        qoe.append(fitness[best_idx])
        t.append(time[best_idx])
        e.append(energy[best_idx])
        off.append(pop_off[best_idx])
        band.append(pop_band[best_idx])
        fre.append(pop_fre[best_idx])
    return qoe, t, e,off,band,fre




def get_gain( server, user):
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
    gains = ch_gains * rayleigh  # w
    # print("gain",gains)
    return gains


def ger_reward_time_energy(user,env):
    server_id = int(user.offload) - 1
    # print("user.offload",user.offload,"server_id",server_id)
    server = env.Server[server_id]
    if user.get_id >= (env.num_User/2):
        server_associate =env.Server[1]
    else:
        server_associate = env.Server[0]

    task_size = user.get_task_size
    task_cycle = user.get_task_cycle
    # print("task_size and task cycle",task_size,task_cycle)
    com_fre = server.get_com_cap * user.get_frequency
    #print("user frequence", user.get_frequency, )

    # edge computing time
    if com_fre == 0:
        print(user.get_type,user.get_offload)
        print(user.get_frequency)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!wrong")
    t_exe = task_cycle / com_fre
    #print("t_exe",t_exe)
    # user uplink bandwidth
    bandwidth = user.get_bandwidth * server.get_bandwidth_cap
    channel_gain = get_gain(server_associate, user)  # w
    power = pow(10, (user.get_power - 30) / 10)  # 20dbm to w - 0.1w
    self_gain = channel_gain * power
    white_noise_pow = pow(10, (-114 - 30) / 10)  # -144 dbm to w
    # uplink rate
    # print(np.log2(1 + self_gain / white_noise_pow))
    rate = bandwidth * np.log2(1 + self_gain / white_noise_pow)
    # from user to es  : transmission time
    if rate == 0:
        print(user.get_type, user.get_offload)
        print("userbandwidth",user.get_bandwidth)
        print("bandwidth!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!wrong")
    #print("bandwidth and rate",user.bandwidth, rate)
    t_trans = task_size / rate
    #print("transmission time", t_trans)

    # 这里是以瓦为单位还是以DBM 为单位呢？需要和本地计算的功率做一下比较
    e_trans = user.get_power * t_trans
    # print("e_trans",e_trans)
    if server.cache[user.get_request - 1] == 1:
        # print("cached all the service")
        t_total = t_exe + t_trans
        # print("edge computing time",t_total)
    else:
        #  cloud computing frequency：4GHZ
        t_total = t_trans + task_size / (4 * 1024 * 1024) + task_cycle / (4 * 10 ** 9) + 0.1
        # print("cloud computing time",t_total)
    return t_total, e_trans


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, DAN_bound,env ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.DNA_bound = DAN_bound
        # DNA_BOUND = [0,2] ,POP_SIZE = 2,DNA_SIZE = 3
        # [[0 1 1]
        #  [1 1 0]]
        single_pop = []
        for i in range(DNA_size):
            if i < DNA_size/3:
                single_pop.append(np.random.randint(*self.DNA_bound))
            else:
                single_pop.append(np.random.random())
        pop =[]
        for i in range(pop_size):
            pop.append(single_pop)

        #self.pop = np.random.randint(*self.DNA_bound, size=(pop_size, DNA_size))
        self.pop = np.array(pop)
        self.env = env



    # 实现和环境的映射 就是得出卸载动作和带宽分配的动作
    # this is a important function for scalability of edge server
    def translateDNA(self, DNA):
        # 这里的DNA 是一个种群的DNA
        bandwidth = []
        offload = []
        frequency = []
        env = self.env
        num_server = self.env.num_Server
        DAN1 = DNA.copy()
        for i in range(len(DNA)):
            #print("action_ob",len(action),action)
            o = DAN1[i][0:self.env.num_User]
            b = DAN1[i][self.env.num_User:2 * self.env.num_User]
            f = DAN1[i][2 * self.env.num_User:]
            # print("1offload",o)
            # print("1bandwidth",b)
            # print("1frequency",f)
            # 先从种群中挑选一个个体，对他进行一个策略输出
            b_count_1 = 0
            b_count_2 = 0
            f_count_1 = 0
            f_count_2 = 0
            for i in range(len(o)):
                if i >= len(b) / 2:
                    if o[i] == 0:
                        pass
                    else:
                        b_count_2 += b[i]
                else:
                    if o[i] == 0:
                        pass
                    else:
                        b_count_1 += b[i]
                if o[i] ==1:
                    f_count_1 +=f[i]
                if o[i] ==2:
                    f_count_2 +=f[i]
            # print("f_count_1,f_count_2", f_count_1, f_count_2)
            # print("b_count_1,b_count_2", b_count_1, b_count_2)
            b_new = []
            f_new = []
            for i in range(len(b)):
                if i >= len(b) / 2:
                    if o[i] == 0:
                        b_new.append(0)
                    else:
                        b_new.append(np.round(b[i] / b_count_2, 3))
                else:
                    if o[i] == 0:
                        b_new.append(0)
                    else:
                        b_new.append(np.round(b[i] / b_count_1, 3))
                if o[i] == 0:
                    f_new.append(0)
                if o[i] == 1:
                    # if f[i] == 0:
                    #     # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    f_new.append(np.round(f[i] / f_count_1, 3))
                if o[i] == 2:
                    # if f[i] == 0:
                    #     # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    f_new.append( np.round(f[i]/f_count_2,3))
            # print("offload",o)
            # print("f_nwe", f_new)
            # print("f_lod",f)
            for i in range(len(o)):
                if o[i] !=0:
                    if f_new[i] == 0:
                        f_new[i] =0.1
                        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%wrong! ",i)

                        # print("offload:",o)
                        # print("1bandwidth", b)
                        # print("1frequency", f)
                        # print("bandwidth",b_new)
                        # print("frequency",f_new)
            offload.append(o)
            bandwidth.append(b_new)
            frequency.append(f_new)
        #print("bandwidth",len(bandwidth))
        return offload, bandwidth,frequency

    # 得出在当前卸载和带宽分配下的目标函数，也要是一种qoe 的形式
    def get_fitness(self, pop_off, pop_band, pop_fre):
        fitness = []
        local_qoe = []
        time = []
        energy = []
        total_time = 0
        total_energy = 0
        total_qoe = 0
        for i in range(len(pop_off)):
            # 计算在所有种群内的时延和能耗，qoe
            # print("step:", i)
            env = self.env
            # set_action
            off = pop_off[i]
            band = pop_band[i]
            fre = pop_fre[i]
            # print("off",off)
            # print("band",band)
            # print("fre",fre)

            env._set_action_offload(off)
            # print("band",len(band),band)
            env._set_action_bandwidth(band)
            env._set_action_frequency(fre)
            total_time = 0
            total_energy = 0
            total_qoe = 0
            # 开始计算奖励函数的所有用户的QOS 部分
            for user in env.User:
                # cloud time /and energy
                # time_th = user.get_task_size / (100 * (1024 * 1024)) + user.get_task_cycle / (4 * 10 ** 9)
                if user.get_id >= (env.num_User / 2):
                    server_associate = env.Server[1]
                else:
                    server_associate = env.Server[0]
                bandwidth = (env.num_Server / env.num_User) * server_associate.get_bandwidth_cap
                channel_gain = get_gain(server_associate, user)  # w
                power = pow(10, (user.get_power - 30) / 10)  # 20dbm to w - 0.1w
                self_gain = channel_gain * power
                white_noise_pow = pow(10, (-114 - 30) / 10)  # -144 dbm to w
                rate = bandwidth * np.log2(1 + self_gain / white_noise_pow)
                # from user to es  : transmission time
                # print("@@@@@@@@@@@@@@@@bandwidth and rate", user.get_bandwidth, rate)
                t_trans = user.get_task_size / rate
                # print("transmission time", t_trans)
                energy_th = user.get_power * t_trans
                t_local = user.get_task_cycle / user.get_com_cap
                e_local = user.get_energy * user.get_com_cap ** 2 * user.get_task_cycle
                time_th = t_local
                energy_th = max(energy_th, e_local)
                # print("time_th", time_th)
                # print("energy_th",energy_th)
                # local computing
                if user.offload == 0:
                    # t_local = user.get_task_cycle / user.get_com_cap
                    # e_local = user.get_energy * user.get_com_cap ** 2 * user.get_task_cycle
                    qoe = env.alpha * ((time_th - t_local) / time_th) + env.beta * ((energy_th - e_local) / energy_th)
                    # print("local_qoe",qoe)
                    total_time += t_local
                    total_energy += e_local
                    total_qoe += qoe
                else:

                    t_edge, e_edge = ger_reward_time_energy(user, env)
                    qoe = env.alpha * ((time_th - t_edge) / time_th) + env.beta * ((energy_th - e_edge) / energy_th)
                    # print("edge_qoe",qoe)
                    total_time += t_edge
                    total_energy += e_edge
                    total_qoe += qoe
            fitness.append(total_qoe)
            time.append(total_time)
            energy.append(total_energy)
        return fitness, time, energy

    def select(self, fitness):
        # 采用的优生劣汰的方式，根据自己的效用函数，选择出合适的种群中的个体，
        # 因为在choice 这里允许重复，qoe大的个体会被多次选择
        #print("fitness:",fitness,type(fitness),len(fitness))
        # 确保所有的适应值都是正值
        # min_ = min(fitness)
        # print(min_)
        fitness_true = [np.exp(i) for i in fitness]
        # print("min",min(fitness_true))
        #print("fitness_true:", fitness_true, type(fitness_true), len(fitness_true))
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
                               p = fitness_true / sum(fitness_true))
        #print(idx)
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            # cross_rate = 0.5 选择一母亲个体的坐标，进行交叉 # select another individual from pop
            i_ = np.random.randint(0, self.pop_size, size=1)[0]
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            # 在相应的交叉的点上和母亲节点交叉
            parent[cross_points] = pop[i_][cross_points]
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                if point < self.DNA_size/3:
                    child[point] = np.random.randint(*self.DNA_bound)
                else:
                    child[point] = np.random.random()
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        # print("pop after envolve",pop)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


