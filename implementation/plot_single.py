import matplotlib.pyplot as plt
plt.style.use('seaborn')
import csv
import numpy as np
import math
from pandas import read_csv

'''with open('time_step.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(row[0])'''

#plt.close()

#data = numpy.memmap('time_step.csv', mode='r')
#y = [i for i in range(len(data))]

#fig, ax = plt.subplots()
#ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#1001686
'''num = 500
data = read_csv('time_step.csv')
data = data.rename(columns={'0.0':'reward'})
data['smooth_path'] = data['reward'].rolling(num, min_periods=1).mean()
data['path_deviation'] = data['reward'].rolling(num, min_periods=1).std()
data = data.fillna(0)
print(len(data['reward']))

plt.plot(data['smooth_path'], linewidth=0.1, linestyle='-', marker=',', label="Reward per Time Step")'''
plt.close()
individual = [0.000053*1e6, 0.000099*1e6, 0.000126*1e6]
brute = [0.000161*1e6, 0.000901*1e6, 0.00505*1e6]
MP = [0.039639*1e6, 0.069292*1e6, 0.106364*1e6]

x = [4, 6, 8]

#additional = [103.5, 110.25, 118.3, 137, 176.95, 338.86, 1065.14]
#y = [i for i in range(len(additional))]

plt.plot(x,individual, color='k', linewidth=0.7, marker='*', label='Individual')
plt.plot(x,brute, color='g', linewidth=0.7, marker='*', label='Brute')
plt.plot(x,MP, color='b', linewidth=0.7, marker='*',label='Max-Plus')
plt.ylabel('Time(microsecond)')
plt.xlabel('Number of Intersections')
plt.legend(loc='best')
#plt.yscale("log")
#plt.xticks(np.linspace(0,3, ), [0, 4, 6, 8, 10])
plt.grid(color='k', alpha=.1)
#plt.yticks([0, -1500, -2000, -2500, -3000, -3500, -4000, -4500])
#plt.fill_between(data['path_deviation'].index, (data['smooth_path'] -2*data['path_deviation']/math.sqrt(num)),(data['smooth_path'] +2*data['path_deviation']/math.sqrt(num)), color='b', alpha=.1)

plt.show()



#data.plot(kind='line', linestyle='-', linewidth=0.01, marker=',')
# plt.show()
#plt.switch_backend('TkAgg')
#plt.get_backend()
#plt.plot(data, y, label='Reward/time_step', linewidth = 0.001, linestyle='-', marker=",")
#mng = plt.get_current_fig_manager()
#mng.resize(*mng.window.maxsize())

