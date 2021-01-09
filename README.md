# Multi-Agent DQRN For Different Traffic Demands

## Introduction:
In today’s world due to rapid urbanisation there has been a shift of population from rural to urban areas especially in developing countries in search of better opportunities. This has lead to unplanned urbanisation leading to a particularly important issue of increased traffic congestion. This has in turn led to environmental degradation and health issues among people. With the current advancement in Artificial Intelligence, especially in the field of Deep Neural Networks various attempts have been made to apply it in the field of Traffic Light Control. This thesis was an attempt to take forward the problem of solving traffic congestion thereby reducing the total travel time. One of the contributions of the thesis was to study the performance of Deep Recurrent Q-network models in different traffic demands for Multi-Agent systems. Another contribution was to apply different coordination algorithms along with Transfer Learning in Multi-Agent Systems or multiple traffic intersections and study their behaviour. Lastly, the performance of these algorithms were also studied when the number of intersections and the demand increase.

## Research Questions:

* How does the Transfer Planning approach combined with the Max-Plus or Brute Coordination algorithm perform when compared to Individual Coordination?
This can studied by training agents for single as well as multiple traffic intersections. The multiple traffic intersections can be trained for a small source problem which then can be extended to a bigger problem by using Transfer Planning. Finally, comparison can be done on the basis of the
reward function and average travel time.

* How does the TLC agent perform for different traffic demands?
Again the above approach can be followed, but first the agents need to be trained for different traffic congestion scenarios. In order to define congestion, various literature can be studied and conclusions can be drawn on how to choose low, medium or high traffic congestion.

* How does the different coordination algorithms perform computationally in case of Traffic Light-Control problem?
This can be studied by measuring the time complexity and the actual runtime for the different algorithms as the number of intersections increase. There can be several factors influencing the performance as the number of agents are scaled up.

## Methodology:
The state is defined in terms of a binary matrix representing a traffic intersection, where 1 corresponds to a car at that position. This is inputted into the deep Convolutional Recurrent Neural Network which outputs the Q-value. For inferential purpose, the action corresponding to the maximum Q-value is implemented. 

## Single Agent Implementation:
![Methodology](https://github.com/azlaanmsamad/MultiAgentDQRNForDifferentTrafficDemands/blob/main/presentation/methodology.png)

## Reward Function:
Reward functions were modelled in terms of average delay and average waiting time at each time step where delay accounts for slow moving cars and waiting time for stationary cars. The reward function is defined as shown below:
![rewardFucn](https://github.com/azlaanmsamad/MultiAgentDQRNForDifferentTrafficDemands/blob/main/presentation/rewardFunction.png)

## Simulation of Urban MObility (SUMO):
An open source software [Simulation of Urban MObility (SUMO)](https://www.eclipse.org/sumo/) was used for simulating traffic scenarios. A python script is implemented in order to communicate with the software. A generic way of implementation is shown below:
![SUMO](https://github.com/azlaanmsamad/MultiAgentDQRNForDifferentTrafficDemands/blob/main/presentation/sumo.png)

## Multi Agent Coordination Algorithms with Transfer Learning:
Three Algorithms used in are:

* Individual Coordination:
![IC](https://github.com/azlaanmsamad/MultiAgentDQRNForDifferentTrafficDemands/blob/main/presentation/IndividualCoordinationAlgo.png)

* Brute Coordination:
![BC](https://github.com/azlaanmsamad/MultiAgentDQRNForDifferentTrafficDemands/blob/main/presentation/TP_BruteCoordination.png)

* Maxplus Algorithm:
![MP](https://github.com/azlaanmsamad/MultiAgentDQRNForDifferentTrafficDemands/blob/main/presentation/maxplusAlgo.png)

## Results:
The trained network is tested at every 10k time step for different traffic demand for different number of agents using the above mentioned Coordination Algorithms. The results are displayed in terms of heatmaps as shown below:

* Individual Coordination:
![ICResult](https://github.com/azlaanmsamad/MultiAgentDQRNForDifferentTrafficDemands/blob/main/finalplot/heatmap/TravelTimeHM/IND_HM_TT.png)

* Brute Coordination:
![BCResult](https://github.com/azlaanmsamad/MultiAgentDQRNForDifferentTrafficDemands/blob/main/finalplot/heatmap/TravelTimeHM/BC_HM_TT.png)

* Maxplus:
![MPResult](https://github.com/azlaanmsamad/MultiAgentDQRNForDifferentTrafficDemands/blob/main/finalplot/heatmap/TravelTimeHM/MP_HM_TT.png)

## Conclusion:

* How does the Transfer Planning approach combined with the Max-Plus or the Brute Coordination algorithm perform when compared to the Individual Coordination?
The Transfer Planning approach significantly saves computational cost and time. It is fruitful when combined with an optimal coordination algorithm. However Individual Coordination lacks observability of the global environment. It acts independently and there is no sort of communication between the direct or indirect neighbours. For most cases towards the end of the training period its performance improves. However its performance is still unpredictable. The performance of Brute Coordination and Max-Plus are similar to each other and they also possess a consistent behaviour unlike IC. The Max-Plus algorithm has better observability than IC due to message passing mechanism. While the BC selects the joint action corresponding to the maximum global payoff.

* How does the TLC agent perform for different traffic demands?
For a given number of intersections, the increase in the traffic demand causes an introduction of fluctuations. There is an increase in the average travel time and the dip in rewards as the traffic demand increases. The increase in fluctuations when the demand increases varies for the different number of intersections.

* How does the different coordination algorithms perform computationally in case of Traffic Light Control problem?
The Individual Coordination performance is exceptionally fast, followed by Brute Coordination and then Max-Plus. Despite being fast, IC has unpredictable behaviour and sometimes it can perform better than any other algorithms. While in other cases its performance is compromised. The Max-Plus algorithm performance is dependent on the total number of iterations performed. However for low iterations, the MP can outperform BC, however the resulting actions may or may not be optimal.
