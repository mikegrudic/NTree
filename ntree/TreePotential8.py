from numba import jitclass, prange
from numba import int32, deferred_type, optional, float64, boolean, int64, njit, jit, void
import numpy as np
from .helpers import *

node_type = deferred_type()

spec = [
    #('bounds', float64[:]),
    ('size', float64),
    #('points', float64[:,:]),
    #('masses', float64[:]),
    ('Npoints', int64),
    ('mass', float64),
    ('COM', float64[:]),
    ('center', float64[:]),
    ('IsLeaf', boolean),
    ('HasChild', boolean[:]),
    #('children', list)
    ('child0', optional(node_type)),
    ('child1', optional(node_type)),
    ('child2', optional(node_type)),
    ('child3', optional(node_type)),
    ('child4', optional(node_type)),
    ('child5', optional(node_type)),
    ('child6', optional(node_type)),
    ('child7', optional(node_type))
]

@jitclass(spec)
class BHTree(object):
    def __init__(self, center, size):

        self.center = center
        self.size = size
        self.IsLeaf = True
        self.HasChild = np.zeros(8,dtype=np.bool_)
#        self.COM = np.empty(3)
#        self.COM[:] = np.nan
        #self.COM[0] = np.nan #np.repeat(np.nan,3)
        self.child0 = None        
        self.child1 = None
        self.child2 = None
        self.child3 = None
        self.child4 = None
        self.child5 = None
        self.child6 = None
        self.child7 = None

    def InsertPoint(self, x, m):
#    """Inserts a point of position x and mass m into the tree."""
    
        if not self.COM.size: # no point already lives here, so let's make a leaf self and store the point there
            self.COM  = x
            self.mass = m
            self.IsLeaf = True
            return
        #otherwise we gotta split this up
        if self.IsLeaf:
            SpawnChildWithPoint(self, self.COM, self.mass)
            self.IsLeaf = False    
        SpawnChildWithPoint(self, x, m)

node_type.define(BHTree.class_type.instance_type)

        


@njit#(void(node_type, float64[:], float64))
def SpawnChildWithPoint(node, x, m):
    """Spawns a child node for a point at position x and mass m to live in."""
    signs = (x > node.center)
    sector = SignsToSector(signs) # number from 0 to 7 deciding which octant

    child_size = node.size/2
    child_center =  node.center + child_size*(signs-0.5)
    if signs[0]:
        if signs[1]:
            if signs[2]:
                if not node.HasChild[0]:
                    node.child0 = BHTree(child_center, child_size)
                    node.HasChild[0] = True
                node.child0.InsertPoint(x, m)
            else:
                if not node.HasChild[1]:
                    node.child1 = BHTree(child_center, child_size)
                    node.HasChild[1] = True
                node.child1.InsertPoint(x, m)
        else:
            if signs[2]:
                if not node.HasChild[2]:
                    node.child2 = BHTree(child_center, child_size)
                    node.HasChild[2] = True
                node.child2.InsertPoint(x, m)
            else:
                if not node.HasChild[3]:
                    node.child3 = BHTree(child_center, child_size)
                    node.HasChild[3] = True
                node.child3.InsertPoint(x, m)
    else:
        if signs[1]:
            if signs[2]:
                if not node.HasChild[4]:
                    node.child4 = BHTree(child_center, child_size)
                    node.HasChild[4] = True
                node.child4.InsertPoint(x, m)
            else:
                if not node.HasChild[5]:
                    node.child5 = BHTree(child_center, child_size)
                    node.HasChild[5] = True
                node.child5.InsertPoint(x, m)
        else:
            if signs[2]:
                if not node.HasChild[6]:
                    node.child6 = BHTree(child_center, child_size)
                    node.HasChild[6] = True
                node.child6.InsertPoint(x, m)
            else:
                if not node.HasChild[7]:
                    node.child7 = BHTree(child_center, child_size)
                    node.HasChild[7] = True
                node.child7.InsertPoint(x, m)

@jit
def ConstructTree(points, masses):
    mins = np.min(points,axis=0)
    maxes = np.max(points,axis=0)
    center = (maxes+mins)/2
    size = np.max(maxes-mins)
    root = BHTree(center, size)
    for i in range(len(points)):
        InsertPoint(root, points[i], masses[i])
#    root.GetMoments()
    return root

        
