from gym.spaces import Space, Dict, Discrete, MultiDiscrete, Box, Tuple, MultiBinary
from aienvs.gym.ModifiedActionSpace import ModifiedActionSpace
import math
from math import floor
from abc import ABC, abstractmethod
from collections import OrderedDict
from numpy import array


class DecoratedSpace(ABC):
    '''
    Decorates gym spaces with extra functionality to make it possible
    to handle all spaces in a generic way.
    '''

    def __init__(self, space:Dict):
        '''
        DO NOT CALL THIS DIRECTLY. Use create.
        @param space a gym dictionary space. 
        @param mergeInfo a list containing lists of keys to be merged. 
        For instance [['a','b'],['c','d','e']] indicates
        that the keys a and b are to be merged into a 
        new key a_b, and c,d,e are to be merged into new key c_d_e
        The remaining keys in the original space are left untouched.
        '''
        self._gymspace = space
        
    @staticmethod
    def create(space: Space):  # -> DecoratedSpace:
        '''
        factory method, creating the correct instance of 
        DecoratedSpace according to the type given
        '''
        if (isinstance(space, Dict)):
            return DictSpaceDecorator(space)
        if (isinstance(space, Discrete)):
            return DiscreteSpaceDecorator(space)
        if (isinstance(space, Box)):
            return BoxSpaceDecorator(space) 
        if (isinstance(space, Tuple)):
            return TupleSpaceDecorator(space) 
        if (isinstance(space, MultiDiscrete)):
            return MultiDiscreteSpaceDecorator(space)
        if (isinstance(space, MultiBinary)):
            return MultiBinarySpaceDecorator(space)
        if (isinstance(space, DecoratedSpace)):
            return space

        raise Exception("Unsupported space type " + str(space))  

    def getOriginalSpace(self):
        return self.getSpace()

    def unpack(self, actions):
        return actions

    def getSpace(self):
        '''
        the gym space that is decorated
        '''
        return self._gymspace

    @property
    def n(self):
        return self.getSize()
        
    @abstractmethod
    def getSize(self):
        '''
        @return:  the total number of possible discrete values in this space.
        This has to be determined dynamically as 
        the space can change over time. May return math.inf
        to indicate infinite number of  discrete values are possible.
        '''
    
    def getSubSpaces(self) -> list:  # list<DecoratedSpace>
        '''
        @return: (possibly empty) list of DecoratedSpaces that are child of this space.
        '''
        return []

    @abstractmethod
    def getById(self, n:int):
        '''
        @return: the nth element of this space.
        n>=0, n < getSize().
        Returned object is similar to what you
        would get with sample, but more controlled. 
        '''

    def numberToList(self, n:int, subsizes:list):
        '''
        convert an int into a list of numbers.
        subsizes contains the maximum value of each number of the return list
        @param n the number to convert. Can be at most the product of numbers in subsizes
        @param subsizes the maximum size of each number in the returned list, eg [3,2,2]
        means first digit of returned list must be <3, etc.
        @return: a list of numbers that uniquely match with n, 
        where each number in the list is smaller than the number in subsizes.
        The returned list always has same length as subsizes.
        '''
        selection = []
        for max in subsizes:
            selection.append(n % max)
            n = floor(n / max)
        return selection

    # calling other methods from self._gymspace
    def __getattr__(self, attr):
        #avoid recursion
        gymspace = self.__getattribute__('_gymspace')
        return getattr(gymspace, attr)
   

class DictSpaceDecorator(DecoratedSpace):
    '''
    Decorates a spaces.Dict

    '''

    def getSubSpaces(self):
        return [self.getSubSpace(id) for id in self.getIds()]
    
    def getIds(self):
        '''
        the keys of this Dict, in proper order
        '''
        return self.getSpace().spaces.keys()
    
    def getSubSpace(self, id:str) -> DecoratedSpace:
        '''
        @param id the id of the space
        @return: space that has given id
        '''
        return DecoratedSpace.create(self.getSpace().spaces[id])
    
    # Override
    def getSize(self):
        size = 1
        for space in self.getSubSpaces():
            size = size * space.getSize()
        return size
    
    def getById(self, n:int):
        nrList = self.numberToList(n, [space.getSize() for space in self.getSubSpaces()])
        nrList = list(zip(self.getIds(), nrList))
        return OrderedDict([(id, self.getSubSpace(id).getById(m)) for id, m in nrList])

    def get(self, id:str):
        return self.getSubSpace(id)


class DiscreteSpaceDecorator(DecoratedSpace):
    '''
    Decorates a spaces.Discrete
    '''

    # Override
    def getSize(self):
        return self.getSpace().n

    def getById(self, n:int):
        return n


class MultiDiscreteSpaceDecorator(DecoratedSpace):
    '''
    Decorates a spaces.MultipleDiscrete
    '''

    # Override
    def getSize(self):
        size = 1
        for n in self.getSpace().nvec:
            size = size * n
        return size

    def getById(self, n:int):
        return array(self.numberToList(n, self.getSpace().nvec))


class TupleSpaceDecorator(DecoratedSpace):
    '''
    Decorates a spaces.Tuple
    '''

    # Override
    def getSize(self):
        size = 1
        for space in self.getSubSpaces():
            size = size * space.getSize()
        return size

    def getSubSpaces(self):
        return [DecoratedSpace.create(space) for space in self.getSpace().spaces]

    def getById(self, n:int):
        subspaces = self.getSubSpaces()
        nrList = self.numberToList(n, [space.getSize() for space in subspaces])
        res = []
        for i in range(0, len(subspaces)):
            res.append(subspaces[i].getById(nrList[i]))
        return tuple(res)


class BoxSpaceDecorator(DecoratedSpace):
    '''
    Decorates a spaces.Box. Special case with partial support,
    since this space has infinite number of elements.
    '''

    # Override
    def getSize(self):
        return math.inf

    def getById(self, n:int):
        raise Exception("Box space can not be sampled discretely")


class MultiBinarySpaceDecorator(DecoratedSpace):
    '''
    Decorates a spaces.MultiBinary
    '''

    # Override
    def getSize(self):
        return 2 ** self.getSpace().n

    def getById(self, n:int):
        return array(self.numberToList(n, [2] * self.getSpace().n))
