import numpy as np
from  matplotlib import pyplot as plt
plt.rcParams['font.family'] = ['Simsun']

#

QoS_20= []
QoS_30 = []
QoS_40 = []
QoS_50 = []

with open('simulation/reward_lstm_2EDGE.txt' , 'r') as f:
    line = f.readline()
    while line:
        line_data = float(line.strip('\n'))
        QoS_20.append(line_data)
        line = f.readline()
with open('simulation/reward_lstm_user30.txt' , 'r') as f:
    line = f.readline()
    while line:
        line_data = float(line.strip('\n'))
        QoS_30.append(line_data)
        line = f.readline()
with open('simulation/reward_lstm_user40.txt' , 'r') as f:
    line = f.readline()
    while line:
        line_data = float(line.strip('\n'))
        QoS_40.append(line_data)
        line = f.readline()
with open('simulation/reward_lstm_user50.txt' , 'r') as f:
    line = f.readline()
    while line:
        line_data = float(line.strip('\n'))
        QoS_50.append(line_data)
        line = f.readline()

print(np.mean(QoS_20[150:200]))
plt.plot(range(len(QoS_20[0:200])),QoS_20[0:200],label="LSTM-DDPG-M2-N20",color="r", marker="^")
# plt.plot(range(len(QoS_30[0:200])),QoS_30[0:200],label="LSTM-DDPG-M2-N30",color="blue",marker= "o")
# plt.plot(range(len(QoS_40)),QoS_40,label="LSTM-DDPG-M2-N40",color="green",marker="*")
# plt.plot(range(len(QoS_50)),QoS_50,label="LSTM-DDPG-M2-N50",color="orange", marker="x")
# plt.plot(x_mean_I,y_mean,label="Improve-GA",color="r")
# plt.plot(x_mean_I,y_mean_I,label="GA",color="blue")
plt.ylabel("Cumulative reward function" )
plt.xlabel("Episodes")
plt.legend(ncol= 2,fontsize=11)
plt.grid(linestyle='-.')
plt.savefig("LSTM-DDPG-EDGE2.pdf")
plt.savefig("LSTM-DDPG-EDGE2")
# plt.savefig("TWIN-LSTM-DDPG.pdf")
# plt.savefig("TWIN-LSTM-DDPG")
plt.show()