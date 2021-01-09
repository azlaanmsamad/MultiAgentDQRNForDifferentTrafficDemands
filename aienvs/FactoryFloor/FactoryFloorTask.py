from numpy import ndarray


class FactoryFloorTask():
    """
    A task on the factory floor
    """
    _taskIdCounter = 1  # to generate new task ids

    def __init__(self, newpos:ndarray):
        """
        Initializes the task, places it at (0,0)
        """
        if not isinstance(newpos, ndarray):
            raise ValueError("newpos must be numpy array but got " + str(type(newpos)))
        self.pos = newpos
        self._id = FactoryFloorTask._taskIdCounter
        FactoryFloorTask._taskIdCounter += 1

    def getId(self):
        """
        returns the task identifier
        """
        return self._id

    def getPosition(self):
        """
        @return: (x,y) tuple with task position
        """
        return self.pos
    
    def __str__(self):
     return "Task " + str(self._id) + " pos " + str(self.pos)
