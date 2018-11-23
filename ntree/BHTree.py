from .helpers import *
import numpy as np

class BHTree:
    """Manifestly 3D Barnes-hut tree that stores the total mass and center of mass of particles in each node."""
    def __init__(self, center, size):
        self.COM = None
        self.center = center
        self.size = size
        self.mass = None
        self.IsLeaf = False
        
    def InsertPoint(self, x, m):
        """Inserts a point of position x and mass m into the tree."""
        if self.COM is None: # no point already lives here, so let's make a leaf node and store the point there
            self.COM  = x
            self.mass = m
            self.IsLeaf = True
            return # we're done
        #otherwise we gotta split this up
        if self.IsLeaf:
            self.children = 8 * [None,] # set up the children
            self.SpawnChildWithPoint(self.COM, self.mass) #  insert the point that already lived here
            self.IsLeaf = False   # no longer a leaf anymore
        self.SpawnChildWithPoint(x, m)   # insert the point that we originally wanted to
    
    def SpawnChildWithPoint(self, x, m):
         """Spawns a child node for a point at position x and mass m to live in."""
         signs = (x > self.center)
         sector = SignsToSector(signs) # number from 0 to 7 deciding which octant
         if not self.children[sector]:  # if the child we need does not exist, create it
             child_size = self.size/2
             child_center =  self.center + child_size*(signs-0.5)
             self.children[sector] = BHTree(child_center, child_size)
         self.children[sector].InsertPoint(x, m)   # 

    
    def GetMoments(self):
        """Computes the mass and center of mass of a node recursively."""
        if not self.IsLeaf: #: return self.mass, self.COM
            self.mass = 0.
            self.COM = np.zeros(3)
            for c in self.children:
                if c is None: continue
                mc, xc = c.GetMoments()
                self.mass += mc
                self.COM += mc*xc
            self.COM /= self.mass
        return self.mass, self.COM

    def ConstructTree(points, masses, mins=None, maxes=None):
        if not mins: mins = np.min(points,axis=0)
        if not maxes: maxes = np.max(points,axis=0)
        center = (maxes+mins)/2
        size = np.max(maxes-mins)
        root = BHTree(center, size)
        for i in range(len(points)):
            root.InsertPoint(points[i], masses[i])
        root.GetMoments()
        return root
        
    def TreeWalk(self, x, nodes, theta=0.7):
        """populates a list of nodes from a treewalk with opening angle theta"""
        r = Dist(x, self.COM)

        if self.size/r < theta or self.IsLeaf:
            nodes.append(self)
        else:
            for c in self.children:
                if c is None: continue
                if c.IsLeaf: nodes.append(c)
                else: c.TreeWalk(x, nodes, theta)

    def PotentialWalk(self, x, theta=0.7):
        r = Dist(x, self.COM)
        phi = 0.
        if self.size/r < theta or self.IsLeaf:
            phi = -self.mass / r
        else:
            for c in self.children:
                if c: phi += c.PotentialWalk(x, theta)
                
        return phi
