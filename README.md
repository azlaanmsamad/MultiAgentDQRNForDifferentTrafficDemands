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
An open source software [Simulation of Urban MObility (SUMO)](https://www.eclipse.org/sumo/) was used for simulating traffic scenarios. 
Reward functions were modelled in terms of average delay and average waiting time at each time step where delay accounts for slow moving cars and waiting time for stationary cars.

![Methodology](https://github.com/azlaanmsamad/MultiAgentDQRNForDifferentTrafficDemands/blob/main/presentation/methodology.png)
## Results:

## Conclusion:
