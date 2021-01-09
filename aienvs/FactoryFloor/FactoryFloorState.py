import numpy as np
from aienvs.FactoryFloor.Map import Map


class FactoryFloorState():

    def __init__(self, robotList, taskList, map:Map):
        """
        @param map the floor map, immutable
        """
        self.robots = robotList
        self.tasks = taskList
        self._map = map
        self.step = 0

    def addRobot(self, robot):
        self.robots.append(robot)

    def addTask(self, task):
        self.tasks.append(task)

    def getMap(self):
        return self._map
    
    def __str__(self):
        """
        for hashing
        """
        return "Robots: " + str([str(robot) for robot in self.robots ]) + "Tasks: " + str([str(task) for task in self.tasks ])

    def __eq__(self, other):
        return str(self)==str(other)

    def __hash__(self):
        return str(self)
 

def encodeStateAsArray(state:FactoryFloorState):
    width = state.getMap().getWidth()
    height = state.getMap().getHeight()
    # first one for the tasks, then each layer for one robot in lexicographic order
    result = np.zeros([width, height,2+len(state.robots)])

    robots = {}
    for robot in state.robots:
        robots[robot.getId()]=robot.getPosition()

    sortedIds = sorted(robots.keys(), key=str.lower)
    
    idx=0
    while idx < len(sortedIds):
        pos=robots[sortedIds[idx]]
        result[pos[0], pos[1], idx+2] += 1
        if sum(sum(result[:,:,idx+2])) != 1:
            breakpoint()
            raise "Something went wrong"
        idx+=1

    for task in state.tasks:
        pos = task.getPosition()
        result[pos[0], pos[1], 1] += 1

    result[:,:,0] += state.step*np.ones([width,height])
   
    return result

def toTuple(a):
    try:
        return tuple(toTuple(i) for i in a)
    except TypeError:
        return a

# TODO: make this snippet a test
# obs=env.reset()
# from aienvs.FactoryFloor.FactoryFloorTask import FactoryFloorTask
# import numpy as np
# obs.addTask(FactoryFloorTask(np.array([2,2])))
# obs.addTask(FactoryFloorTask(np.array([0,0])))
# encoded=encodeStateAsArray(obs, env.observation_space.nvec[1],env.observation_space.nvec[2], "robot1")

