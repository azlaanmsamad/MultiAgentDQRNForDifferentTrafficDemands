from numpy import array, ndarray


class FactoryFloorRobot():
    """
    A robot on the factory floor
    """
    def __init__(self, robotId, pos:ndarray):
        """
        @param pos tuple (x,y) with initial robot position.
        Initializes the robot
        """
        if not isinstance(pos, ndarray):
            raise ValueError("pos must be numpy array but got " + str(type(pos)))
        self._id = robotId
        self._pos = pos

    def getId(self):
        """
        returns the robot identifier
        """
        return self._id

    def getPosition(self):
        """
        @return: (x,y) array with current robot position
        """
        return self._pos
    
    def setPosition(self, newpos):
        """
        @param newpos: an array (x,y) with the new robot position
        """
        self._pos = newpos
    
    def __str__(self):
     """
     for hashing
     """
     return "Id: " + self._id + " Pos: " + str(self._pos)
        
