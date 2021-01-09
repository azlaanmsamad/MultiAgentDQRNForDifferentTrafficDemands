from random import Random
import random
from numpy import array, ndarray
import copy


class Map():
    """
    Contains a static (immutable) discrete square map
    Each position in the map can have various values, 
    eg a free space or a wall.
    This map is like a paper map of the environment,
    so without moving parts.
    immutable means nobody should access private variables
    and there are no setters so this object will never change.
    """

    def __init__(self, map:list, ptask:float):
        """
        @param map: list of strings. Each string represents one line (x direction) 
        of the map. All lines must have same length.  There must be at least one line.
        The first line represents y=0, the second line y=1, etc.
        The first column of each like represents x=0, the second x=1, etc.
        There are a number of agreed-on characters on the map:
        - * indicates a wall
        - '.' indicates a free floor tile
        - digit 1..9 indicates a location where a task can appear. The number indicates
        the weight of the task: the task location is picked by weighted random choice.
        @param ptask: probability that a task is added in a time step (this is assuming
        a stepped simulation, rather than a timed one). Ignored if there are no digits
         on the floor
        """
        self._map = map
        self._taskProbability = ptask
        width = self.getWidth()
        for line in map:
            if width != len(line):
                raise ValueError("Map must be square")
        self._cachedTaskPositions = tuple(Map._getTasksList(map))
        
        weights = Map._getWeightsList(map)
        totalweight = sum(weights)
        if totalweight == 0:
            self._cachedTaskWeights = tuple([])
        else:
            self._cachedTaskWeights = tuple([w / totalweight for w in weights])

    def getWidth(self) -> int:
        return len(self._map[0])
    
    def getHeight(self) -> int:
        return len(self._map)
    
    def getFullMap(self):
        """
        @return: a copy of the original map provided to the constructor
        """
        return copy.deepcopy(self._map)
    
    def getTaskProbability(self):
        """
        @return: the probability that a task will appear on the map 
        during a time step
        """
        return self._taskProbability
    
    def get(self, pos:ndarray) -> str:
        """
        @param pos the map position as (x,y) tuple 
        @return character at given pos 
        """
        return self._map[pos[1]][pos[0]]
    
    def getTaskPositions(self) -> tuple:
        """
        @return: the list of task positions ( (x,y) tuples )
        """
        return self._cachedTaskPositions
    
    def getTaskWeights(self) -> tuple:
        """
        @return: list of task weights, ordered to match getTaskPositions
        """
        return self._cachedTaskWeights
    
    @staticmethod
    def _getTasksList(map:list):
        """
        @param map the map , see __init__
        Get list of all task positions on the map, in order of colums / rows.
        Positions are numpy arrays [x,y].
        """
        poslist = []
        for y in range(len(map)):
            for x in range(len(map[0])):
                if map[y] [x] in "123456789":
                    poslist += [ array([x, y]) ]
        return poslist
    
    @staticmethod
    def _getWeightsList(map:list):
        """
        @param map the map , see __init__
        Get list of all weights on the map, in order of colums / rows
        """
        weightlist = []
        for row in map:
            for value in row:
                if value in "123456789":
                    weightlist += [ int(value) ]
        return weightlist
    
    def getRandomPosition(self, random:Random) -> array:
        """
        @param random number generator,  instance of Random()
        @return: numpy array : random position on the map. The returned position 
        will be #isInside but may be on a wall.
        """
        return array([random.randint(0, self.getWidth() - 1), random.randint(0, self.getHeight() - 1)])
    
    def isInside(self, pos:ndarray) -> bool:
        """
        @return: true iff the position is within the bounds of this map. The top left is [0,0]
        """
        return pos[0] >= 0 and pos[0] < self.getWidth() and pos[1] >= 0 and pos[1] < self.getHeight()
    
    def getPart(self, area:ndarray):  # -> Map
        """
        @param area a numpy array of the form [[xmin,ymin],[xmax,ymax]]. 
        @return: A copy of a part of this map, spanning from [xmin,ymin] to [xmax, ymax]
        (both ends inclusive). 
        """
        newmap = []
        for y in range(area[0, 1], area[1, 1] + 1):
            newmap = newmap + [self._map[y][area[0, 0]:area[1, 0] + 1]]
        
        # use raw original values to compute scalings
        oldweight = sum(Map._getWeightsList(self._map))
        if oldweight == 0:
            newtaskp = 0
        else:
            newtaskp = self._taskProbability * sum(Map._getWeightsList(newmap)) / oldweight
        return Map(newmap, newtaskp)
    
    # A MAP IS IMMUTABLE
    def __deepcopy__(self, memo):
        # create a copy with self.linked_to *not copied*, just referenced.
        return self
