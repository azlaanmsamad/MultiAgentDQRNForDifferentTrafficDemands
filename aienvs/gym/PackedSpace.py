from gym.spaces import Space, Dict, Discrete
from aienvs.gym.DecoratedSpace import DecoratedSpace, DictSpaceDecorator
from aienvs.gym.ModifiedActionSpace import ModifiedActionSpace
from aienvs.gym.DecoratedSpace import DictSpaceDecorator
from _collections import OrderedDict


class PackedSpace(ModifiedActionSpace):
    '''
    combines several keys from the original space into 
    a single key. This appears to the outside
    as a Dict with the keys merged.
    But you can call unpack to convert an action
    in this packed space back into an action
    for the original space.
    '''

    def __init__(self, actionspace: Dict, packing: dict):
        '''
        @actionspace the original actionspace
        @packing a dict that maps new dict keys 
        into a list of keys from actionspace.
        The keys in the list are to be removed from the 
        actionspace and the new dict keys are added. 
        IMPORTANT packing keys must NOT be in actionspace.
        example. Say your actionspace has keys a,b,c.
        Then packing could be {'a_b':['a','b']}. The new 
        space will then have keys 'a_b' and 'c'
        
        '''
        self._originalspace = DecoratedSpace.create(actionspace)
        self._subdicts = {}
        newdict = actionspace.spaces.copy()
        # now replace keys according to packing instructions.
        for id in packing:
            subdict = self._createSubspace(packing[id])
            self._subdicts[id] = subdict
            newdict[id] = subdict.getSpace()
            for oldkey in packing[id]:
                if not oldkey in newdict:
                    raise Exception("Packing instruction " + str(packing) + " refers unknown key " + oldkey)
                newdict.pop(oldkey)
        # we set this up as if it is a dict
        # NOTE    super(Dict, self).__init__(newdict) does NOT work as intended
        Dict.__init__(self, newdict)

    def _createSubspace(self, subids: list) -> DictSpaceDecorator:
        '''
        @param subids a list of key/ids in our Dict that 
        have to be merged into a Discrete.
        @return DictSpaceDecorator that contains only the subid's
        from the original space. 
        '''
        newdict = { id:space \
                   for id, space in self._originalspace.getSpace().spaces.items() \
                   if id in subids }
        return DictSpaceDecorator(Dict(newdict))
    
    def unpack(self, action: OrderedDict) -> OrderedDict:
        newactions = {}
        for actid, value in action.items():
            if actid in self._subdicts:
                origactions = self._subdicts[actid].getById(value)
                for origid, origact in origactions.items():
                    newactions[origid] = origact
            else:
                newactions[actid] = value
        return OrderedDict(newactions)
    
    def getOriginalSpace(self) -> OrderedDict: 
        return self._originalspace.getSpace()
    
