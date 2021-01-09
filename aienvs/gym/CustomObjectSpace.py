
from gym.spaces import Space


class CustomObjectSpace(Space):
    """
    This is a space just like the Gym Box, Dict, Discrete etc;
    but represents 'general object of type T'. 
    Sampling this space gives a python object of a type
    provided in the constructor (so generally, a python object).
    This is typically used as observation space.
    """

    def __init__(self, typicalobject):
        """
        @param typicalobject object of type T.
        This object is returned from sample.
        all objects in this space must be of type T.
        """
        self._typicalObject = typicalobject
        
    def sample(self):
        return self._typicalObject
    
    def contains(self, x):
        return type(x) == type(self._typicalObject)

