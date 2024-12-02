import numpy as np
from  matplotlib import pyplot as plt
plt.rcParams['font.family'] = ['Simsun']

#
ave = 5
QoS_20= np.zeros(100)
QoS_30 = np.zeros(100)
QoS_40 = np.zeros(100)
QoS_50 = np.zeros(100)
for i in range(ave):
    QoS_temp = []
    with open('simulation/QoS_user20%d.txt'%i, 'r') as f:
        line = f.readline()
        while line:
            line_data = float(line.strip('\n'))
            QoS_temp.append(line_data)
            line = f.readline()
    for i in range(len(QoS_20)):
        QoS_20[i] += QoS_temp[i]

for i in range(ave):
    QoS_temp = []
    with open('simulation/QoS_user30%d.txt'%i, 'r') as f:
        line = f.readline()
        while line:
            line_data = float(line.strip('\n'))
            QoS_temp.append(line_data)
            line = f.readline()
    for i in range(len(QoS_30)):
        QoS_30[i] += QoS_temp[i]

for i in range(ave):
    QoS_temp = []
    with open('simulation/QoS_user40%d.txt'%i, 'r') as f:
        line = f.readline()
        while line:
            line_data = float(line.strip('\n'))
            QoS_temp.append(line_data)
            line = f.readline()
    for i in range(len(QoS_40)):
        QoS_40[i] += QoS_temp[i]

for i in range(ave):
    QoS_temp = []
    with open('simulation/QoS_user50%d.txt'%i, 'r') as f:
        line = f.readline()
        while line:
            line_data = float(line.strip('\n'))
            QoS_temp.append(line_data)
            line = f.readline()
    for i in range(len(QoS_50)):
        QoS_50[i] += QoS_temp[i]
#
plt.plot(range(len(QoS_20)),QoS_20/5,label="Small-M2-N20",color="r", marker="^")
plt.plot(range(len(QoS_30)),QoS_30/5,label="Small-M2-N30",color="blue",marker= "o")
plt.plot(range(len(QoS_40)),QoS_40/5,label="Small-M2-N40",color="green",marker="*")
plt.plot(range(len(QoS_50)),QoS_50/5,label="Small-M2-N50",color="orange", marker="x")
# plt.plot(x_mean_I,y_mean,label="Improve-GA",color="r")
# plt.plot(x_mean_I,y_mean_I,label="GA",color="blue")
plt.xlabel("Number of steps")
plt.ylabel("The QoS of all TDs")

plt.legend(ncol= 2,fontsize=12)
plt.grid(linestyle='-.')
#plt.legend()
plt.savefig("GA_convence_response")
plt.savefig("GA_convence_response.pdf")
# plt.savefig("Ch4-GA.pdf")
plt.show()