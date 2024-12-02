
# time 20221115
# by qian
# 利用ddpg 做出缓存，遗传算法带宽、计算资源分配和计算卸载的决策！！
# 单步做出缓存和下面的决策，用于训练，和方案的对比
# 这里有五个智能体edge server
import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
#from GA_OBF import GA
from GA_co5 import GA
from arguments import parse_args
from Replay_buffer import ReplayBuffer
# import joint_3c_scenario.scenario as scenarios
from model_lstm import DDPG
#from model_ddpg import DDPG
from Two_MEC_env.environment5 import MultiAgentEnv




def train(arglist):


    # train the agents in the offload environment
    # input arglist: env parameters
    # out the reward of agents , the model needed to save
    print("""step1: create the environment """)
    env =  MultiAgentEnv()
    print('step 1 Env {} is right ...'.format(arglist.scenario_name))

    print("""step2: create agents""")
    # 首先是获取到 服务器智能体的状态和动作的维度，用来产生智能体：
    #print("env.observation_space_server_cache",env.observation_space_server_cache[0])
    obs_dim = env.observation_space_server_cache[0].shape[0]
    print('obs_dim',obs_dim)
    action_dim = env.action_space_server_cache[0].shape[0]
    print('action_dim', action_dim)
    # ddpg
    agents_server_cache = DDPG(obs_dim,action_dim, arglist)
    # 利用启发是的遗传算法 求解带宽资源分配和用户卸载
    CROSS_RATE = 0.450
    MUTATE_RATE = 0.1
    POP_SIZE = 30
    N_GENERATIONS = 100
    ga_ob = GA(DNA_size=env.num_User*3, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE,
               pop_size=POP_SIZE, DAN_bound=[0, env.num_Server + 1], env=env)

    print('step 2 The {} agents are inited ...'.format(1))

    print('step 3 starting iterations ...')

    game_step = 0
    # 每一步用户和服务器的效用
    users_time = []
    users_energy = []
    users_qoe = []
    server_switch_cost = []
    rewards_server_cache = []
    ep_count_cache = 0

    # 刚开始的时候。是上下两层都进行重置！！！只用reset_high 既可
    # 获得到初始化的状态
    # max_epsode 500
    # each_episode: 200 step  做200 此的卸载和带宽分配决策d
    var_cache = 1
    for episode_gone in range(arglist.max_episode):
        env.episode = episode_gone
        obs_n_server_cache = env.reset_env()
        print("obs_n_server_cache",np.shape(obs_n_server_cache))
        action_n_server_cache = 0
        reward_server_n_cache = 0
        ep_qoe = 0
        ep_cache_cost = 0
        ep_reward = 0
        ep_time =0
        ep_energy = 0
        task_list1 = np.zeros(env.num_task)
        task_list2 = np.zeros(env.num_task)
        task_list3 = np.zeros(env.num_task)
        task_list4 = np.zeros(env.num_task)
        task_list5 = np.zeros(env.num_task)
        service_gain_list = np.zeros(env.num_task)
        for episode_cnt in range(arglist.per_episode_max_len):  # 每一回合里面的step 20*20数目
            game_step += 1
            print("this is the step " + str(episode_cnt) + " of episode " + str(episode_gone))
            var_cache *= 0.9882
            # 500 步0.988
            current_cache = []
            for server in env.Server:
                current_cache.append(server.cache)
                # print("current_cache",current_cache)
            # caching decisionsM FORlstm-ddpg
            action_n_server_cache = agents_server_cache.select_action(obs_n_server_cache,None)
            #action_n_server_cache = agents_server_cache.select_action(obs_n_server_cache)
            action_n_server_cache = action_n_server_cache[0].tolist()
            action_n_server_cache = np.clip(np.random.normal(action_n_server_cache, var_cache), -1, 1)
            #print("noise: action_n_server_cache", action_n_server_cache)
            # print("NETWORK: action_n_server_cache", action_n_server_cache)
            env._set_action_cache(action_n_server_cache)
            # 进行小尺度的资源分配决策
            step_qoe = []
            step_time = []
            step_energy = []

            for step in range(env.cache_decision_fre):
                #  启发式的算法获取当前时刻缓存结果情况下的卸载和带宽决策，以及用户的qoe，奖励函数中系统qoe 时延和能耗
                ## 迭代多次之后，收敛的动作和奖励
                # 把动作映射到环境中去，或者相应的qoe
                qoe, delay, energy,off,band,fre = env.get_ob_data(ga_ob,N_GENERATIONS)
                # with open("simulation/QoSga3edge%d.txt"%step, "w") as f:
                #     for r in qoe:
                #         f.write(str(r) + '\n')
                # # 现在把这个动作施加到环境中去
                # print("off ",off)
                # print("bandwidth",band)
                # print(fre)
                env._set_action_offload(off)
                env._set_action_bandwidth(band)
                env._set_action_frequency(fre)


                #状态更新

                # 计算在这个动作下的时延和能耗
                total_time = 0
                total_energy = 0
                total_qoe = 0
                num_server_user = int(env.num_User/env.num_Server)
                # 开始计算奖励函数的所有用户的QOS 部分
                for i,user in enumerate(env.User):
                    # cloud time /and energy
                    # time_th = user.get_task_size / (100 * (1024 * 1024)) + user.get_task_cycle / (4 * 10 ** 9)
                    # confirm associate edge server
                    if user.get_id < num_server_user:
                        server_associate = env.Server[0]
                        # print("user_id", user.get_id, "server", env.server[0].get_type)
                    elif user.get_id >= num_server_user and user.get_id < 2 * num_server_user:
                        server_associate = env.Server[1]
                        # print("user_id", user.get_id, "server", env.server[1].get_type)
                    elif user.get_id >= 2 * num_server_user and user.get_id < 3 * num_server_user:
                        server_associate = env.Server[2]
                    elif user.get_id >= 3 * num_server_user and user.get_id < 4 * num_server_user:
                        server_associate = env.Server[3]
                    else:
                        server_associate = env.Server[4]


                    bandwidth = (env.num_Server / env.num_User) * server_associate.get_bandwidth_cap
                    channel_gain = env.get_gain(server_associate, user)  # w
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
                    print("time_th", time_th)
                    print("energy_th", energy_th)
                    #为了计算缓存的增益，和本地计算相比
                    qoe_local = env.alpha * ((time_th - t_local) / time_th) + env.beta * (
                            (energy_th - e_local) / energy_th)
                    # local computing
                    if user.offload == 0:
                        # t_local = user.get_task_cycle / user.get_com_cap
                        # e_local = user.get_energy * user.get_com_cap ** 2 * user.get_task_cycle
                        qoe = env.alpha * ((time_th - t_local) / time_th) + env.beta * (
                                    (energy_th - e_local) / energy_th)
                        print("local_qoe", qoe)
                        total_time += t_local
                        total_energy += e_local
                        total_qoe += qoe
                    else:

                        t_edge, e_edge = env.ger_reward_time_energy(user, env)
                        qoe = env.alpha * ((time_th - t_edge) / time_th) + env.beta * ((energy_th - e_edge) / energy_th)
                        print("edge_qoe", qoe)
                        total_time += t_edge
                        total_energy += e_edge
                        total_qoe += qoe
                        # 计算一下选择卸载的用户的缓存增益
                        service_gain_list[user.get_request-1] += qoe-qoe_local
                # 将这段动作不变时间内用户的服务质量求和
                # 因此需要加上这个时隙
                # update the off popular
                off_state = []
                for user in env.User:
                    if user.offload ==1:
                        if env.Server[0].cache[user.request-1] ==1:
                            task_list1[user.request-1] += 1
                    if user.offload ==2:
                        if env.Server[1].cache[user.request-1] ==1:
                            task_list2[user.request-1] += 1
                    if user.offload ==3:
                        if env.Server[2].cache[user.request-1] ==1:
                            task_list3[user.request-1] += 1
                    if user.offload ==4:
                        if env.Server[3].cache[user.request-1] ==1:
                            task_list4[user.request-1] += 1
                    if user.offload ==5:
                        if env.Server[4].cache[user.request-1] ==1:
                            task_list5[user.request-1] += 1
                # print("!!!!!!!",task_list1)
                # print("**************",task_list2)
                for i in task_list1:
                    off_state.append(i)
                for i in task_list2:
                    off_state.append(i)
                for i in task_list3:
                    off_state.append(i)
                for i in task_list4:
                    off_state.append(i)
                for i in task_list5:
                    off_state.append(i)
                # normalization
                max_s = np.max(off_state)
                if np.min(off_state) ==0:
                    min_s = 0.1
                else:
                    min_s =  np.min(off_state)
                for i in range(len(off_state)):
                    off_state[i] = (max_s-off_state[i])/(max_s-min_s)
                env.service_offload_pop = off_state
                # update the caching_gain

               # 归一化这个服务缓存增益的变量
                max_s = np.max(service_gain_list)
                if np.min(service_gain_list) == 0:
                    min_s = 0.1
                else:
                    min_s = np.min(service_gain_list)
                for i in range(len(service_gain_list)):
                    service_gain_list[i] = (max_s - service_gain_list[i]) / (max_s - min_s)
                env.service_gain = service_gain_list
                ## update user request
                env.updat_user_requset()
                """
                
                ## 更新一下环境中的状态，缓存增益和卸载的决策
                ## env 更新一下用户的请求，然后再重新求解一下问题就好了
              """
                step_qoe.append(total_qoe)
                step_time.append(total_time)
                step_energy.append(total_energy)
            # with open("simulation_low/qoe.txt", "w") as f:
            #     for r in step_qoe:
            #         f.write(str(r) + '\n')
            # with open("simulation_low/delay.txt", "w") as f:
            #     for r in step_time:
            #         f.write(str(r) + '\n')
            # with open("simulation_low/energy.txt", "w") as f:
            #     for r in step_energy:
            #         f.write(str(r) + '\n')
            print("the task offloading and bandwidth allocation is ok ")
            swithc_cache_value = 0
            for i,server in enumerate( env.Server):
                cache = server.cache
                past_cache = current_cache[i]
                #print("past_cache",past_cache)
                print("cache",cache)
                for i in range(len(cache)):
                    if cache[i]==1 and past_cache[i] ==0:
                        # 因为我们设定cache_size = [0,50]
                        print("task %d"%i,env.Tasks[i].get_cache_size/server.backhaul_rate)
                        swithc_cache_value += env.Tasks[i].get_cache_size/server.backhaul_rate
            print("swithc_cache_value",swithc_cache_value)
            constraint_over_cache_cap = 0
            for server in env.Server:
                cache_size = 0
                for i, task in enumerate(server.cache):
                    if task == 1:
                        cache_size += env.Tasks[i].get_cache_size
                if cache_size > server.get_cache_cap:
                    print("cache_size",cache_size/ (1024 * 1024 * 8))
                    #print("server.get_cache_cap",server.get_cache_cap/ (1024 * 1024 * 1024 * 8))
                    constraint_over_cache_cap += (cache_size - server.get_cache_cap) / (1024 * 1024 * 1024 * 8)
                else:
                    constraint_over_cache_cap+=0
            print("constraint_over_cache_cap", constraint_over_cache_cap)

            reward_server_n_cache = np.mean(step_qoe)/10 - swithc_cache_value/1000 - constraint_over_cache_cap/100
            print("np.mean(step_qoe)/10",np.mean(step_qoe)/10)
            print("swithc_cache_value/1000 ",swithc_cache_value/1000 )

            print("the cache reward :", reward_server_n_cache)
            next_obs_server_n_cache = env._get_obs()

            # print("next_obs_server_n_cache",next_obs_server_n_cache)
            # # save experience
            agents_server_cache.replay_buffer.push((obs_n_server_cache,action_n_server_cache,
                                         next_obs_server_n_cache,reward_server_n_cache))

            # save the data of each episode
            # 将每一步的奖励函数中的每个部分都存一下
            # ep_cache_cost.append(swithc_cache_value)
            # #ep_reward += reward_server_n_cache
            # # 将每一步的时延和能耗加到这一回合中去
            # ep_time.append(delay)
            # ep_energy.append(energy)
            # ep_qoe.append(qoe)
            # ep_reward.append(reward_server_n_cache)
            ep_reward += reward_server_n_cache
            ep_cache_cost += swithc_cache_value
            ep_qoe += np.mean(step_qoe)
            obs_n_server_cache = next_obs_server_n_cache

            if game_step >= arglist.learning_start_step:
                if game_step % arglist.learning_fre == 0:
                    print("learning !!!!!")
                    agents_server_cache.update(arglist)



        users_qoe.append(ep_qoe)
        rewards_server_cache.append(ep_reward)
        server_switch_cost.append(ep_cache_cost)

    return rewards_server_cache,users_qoe,server_switch_cost


if __name__ == '__main__':
    arglist = parse_args()
    num_server = 5
    num_user = 50
    # agent1, agent2, agent_user,a1,a2,a6,a10,a17,a19,all_a = train(arglist)
    rewards_server_cache, users_qoe, server_switch_cost = train(arglist)
    with open("simulation/reward_lstm_5EDGE500.txt", "w") as f:
        for r in rewards_server_cache:
            f.write(str(r) + '\n')
    # with open("simulation/server_switch_cost_c5_C60.txt", "w") as f:
    #     for cost in server_switch_cost:
    #         f.write(str(cost) + '\n')
    #
    # # with open("simulation_lstm/time_10.txt", "w") as f:
    # #     for ep_time in users_time:
    # #         f.write(str(ep_time) + '\n')
    # # with open("simulation_lstm/energy_10.txt", "w") as f:
    # #     for ep_energy in users_energy:
    # #         f.write(str(ep_energy) + '\n')
    #
    # with open("simulation/qoe_user_c5_C60.txt", "w") as f:
    #     for qoe in users_qoe:
    #         f.write(str(qoe) + '\n')
    print('end')

