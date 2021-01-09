from aienvs.FactoryFloor.Map import Map
from networkx.classes.graph import Graph
from numpy import array, ndarray


class FactoryGraph(Graph):

    def __init__(self, map:Map):
        """
        constructs a graph with possible steps
        @param the map to make a graph for
        """
        super().__init__()
        
        for y in range(0, map.getHeight()):
            for x in range(0, map.getWidth()):
                pos = array([x, y])
                if map.get(pos) == '*':
                    continue
                self.add_node(str(pos))
                for neighbour in self._neighbours(pos):
                    if map.isInside(neighbour) and map.get(neighbour) != '*':
                        self.add_edge(str(pos), str(neighbour))

    def _neighbours(self, pos:array):
        return [ pos + [0, 1], pos + [0, -1], pos + [1, 0], pos + [-1, 0]]
    
