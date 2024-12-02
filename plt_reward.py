# time 2022110320230703
# by qian
# 奖励函数.
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

x = []
y = []
with open('simulation/reward_lstm_3EDGE.txt' , 'r') as f:
    line = f.readline()
    while line:
        line_data = float(line.strip('\n'))
        y.append(line_data)
        line = f.readline()
x = []
x_mean = []
y_1 = []
y_mean = []
ave = 1
count = 0
for i in range(len(y)):
    if y[i] < -1000:
        print("i and index", i, y[i])
        count += 1
    else:
        x.append(i)
        y_1.append(y[i])
for i in range(len(y_1)):
    if i % ave == 0:
        x_mean.append(i)
        y_mean.append(np.mean(y_1[i:i + ave]))

x_ddpg = []
y_ddpg = []
with open('simulation/reward_lstm_user30.txt' , 'r') as f:
    line = f.readline()
    while line:
        line_data = float(line.strip('\n'))
        y_ddpg.append(line_data)
        line = f.readline()
x = []
x_mean_ddpg = []
y_1ddpg = []
y_mean_ddpg = []
ave = 1
count = 0
for i in range(len(y_ddpg)):
    if y_ddpg[i] < -500:
        # print("i and index", i, y[i])
        count += 1
    else:
        x.append(i)
        y_1ddpg.append(y_ddpg[i])
for i in range(len(y_1ddpg)):
    if i % ave == 0:
        x_mean_ddpg.append(i)
        y_mean_ddpg.append(np.mean(y_1ddpg[i:i + ave]))
# # plt.plot(x, y_1, color='lightsteelblue', label="offload ")



plt.plot(x_mean, y_mean,label="LSTM-DDPG-M3-N30",color="r", marker="^")
#plt.plot(x_mean_ddpg, y_mean_ddpg, color='steelblue',label="DDPG")
plt.legend()
plt.grid()
# plt.ylabel("Cumulative reward function" )
# plt.xlabel("Episodes")
# plt.savefig("LstmDDPG.pdf")
plt.show()