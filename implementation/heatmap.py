#import seaborn as sb
import numpy as np
import os
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sb
import pandas as pd
BC_tt = np.array([[189.69, 237.219, 282.14],
                [330.75, 488, 701.78],
                [373.5, 1869.7, 773.96]])

ind_tt = np.array([[329.162, 424.136, 325.93],
                  [161.06, 521.32, 343.8],
                  [670.26, 2216.619, 1040.208]])

mp_tt = np.array([[174.5, 263.98, 280.6],
                  [274.3, 526.4, 706.9],
                  [378.3, 1423.8, 786.294]])

BC_reward = np.array([[-6.32, -6.38, -7.54],
                      [-45.86, -42.8, -51.16],
                      [-96.97, -163.61, -98.9]])

ind_reward = np.array([[-8.5, -11.5, -8.6],
                       [-40.27, -39.2, -38.36],
                       [-101.1, -140.17, -119.18]])


plt.subplots(figsize=(15, 15))
sb.palplot(sb.color_palette("ch:2.5,-.2,dark=.3"))
heatmap1 = sb.heatmap(mp_tt, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average Travel Time'})
#heatmap2 = sb.heatmap(data1, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average Travel Time'})
#heatmap3 = sb.heatmap(data2, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average Travel Time'})
plt.xticks(np.arange(3)+0.5, [4, 6, 8], va="center")
plt.yticks(np.arange(3)+0.5, [0.05, 0.2, 0.4], va="center")
plt.xlabel('Number of Traffic Intersections')
plt.ylabel('Different Traffic Demand')
sb.set(font_scale=1)
plt.show()
plt.savefig('trained_MP_HM', dpi=300)
#heatmap1.set_ylim(20,20)

