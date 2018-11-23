from .helpers import *
import numpy as np
from scipy.spatial.distance import cdist

class BHTree2:
    """Manifestly 3D Barnes-hut tree that stores the total mass and center of mass of particles in each node."""
    def __init__(self, center, size, leafsize=16):
        self.COM = None
        self.center = center
        self.size = size
        self.mass = None
        self.positions = None
        self.masses = None
        self.IsLeaf = False
        self.leafsize = leafsize
        
    def InsertPoint(self, x, m):
        """Inserts a point of position x and mass m into the tree."""
        if not self.mass: # no point already lives here, so let's make a leaf node and store the point there
            self.COM  = x
            self.mass = m
            self.IsLeaf = True
            self.positions = [x,]
            self.masses = [m,]
            return # we're done
        elif self.IsLeaf:  
            self.positions.append(x)   # add the particle to the list of particles in the leaf
            self.masses.append(m)
            # now check whether we've gone over the limit; if so, we gotta split it up and make it an internal node
            if len(self.masses) > self.leafsize:
                self.children = 8 * [None,] # set up the children
                signs = np.array(self.positions) > self.center
                sectors = SignsToSectors(signs)
                # create the children needed
                child_size = self.size/2
                for i in range(len(self.masses)):
#                    child = self.children[sectors[i]]
                    if not self.children[sectors[i]]:
                        self.children[sectors[i]] = BHTree2(self.center + child_size*(signs[i]-0.5), child_size, self.leafsize)
                    self.children[sectors[i]].InsertPoint(self.positions[i], self.masses[i])
                self.positions = None
                self.masses = None
                self.IsLeaf = False
        else: # we've got an internal node, so we gotta figure out which child to put it in and send it there
            signs = (x > self.center)
            sector = SignsToSector(signs) # number from 0 to 7 deciding which octant
            if not self.children[sector]:  # if the child we need does not exist, create it
                child_size = self.size/2
                child_center =  self.center + child_size*(signs-0.5)
                self.children[sector] = BHTree2(child_center, child_size, self.leafsize)
            self.children[sector].InsertPoint(x, m)   #
#            self.SendPointToChild(x,m)

    def SendPointToChild(self, x, m):
        """Figures out which child node to send the point to, and initiates point insertion for that node."""
        signs = (x > self.center)
        sector = SignsToSector(signs) # number from 0 to 7 deciding which octant
        if not self.children[sector]:  # if the child we need does not exist, create it
            child_size = self.size/2
            child_center =  self.center + child_size*(signs-0.5)
            self.children[sector] = BHTree2(child_center, child_size, self.leafsize)
        self.children[sector].InsertPoint(x, m)   # 

    def GetMoments(self):
        """Computes the mass and center of mass of a node recursively."""
        if self.IsLeaf:
            self.masses, self.positions = np.array(self.masses), np.array(self.positions)
            return np.sum(self.masses), np.average(self.positions, axis=0, weights=self.masses)
        else:
            self.mass = 0.
            self.COM = np.zeros(3)
            for c in self.children:
                if c is None: continue
                mc, xc = c.GetMoments()
                self.mass += mc
                self.COM += mc*xc
            self.COM /= self.mass
            return self.mass, self.COM

    def ConstructTree(points, masses, mins=None, maxes=None, leafsize=16):
        if not mins: mins = np.min(points,axis=0)
        if not maxes: maxes = np.max(points,axis=0)
        center = (maxes+mins)/2
        size = np.max(maxes-mins)
        root = BHTree2(center, size, leafsize=leafsize)
        for i in range(len(points)):
            root.InsertPoint(points[i], masses[i])
        root.GetMoments()
        return root

    def CountInternalNodes(self):
        if self.IsLeaf: return 0
        else:
            return sum([c.CountInternalNodes() if c is not None else 0 for c in self.children]) + 1

    def CountLeafNodes(self):
        if self.IsLeaf: return 1 #len(self.masses)
        else:
            return sum([c.CountLeafNodes() if c is not None else 0 for c in self.children])
        
    def TreeWalk(self, x, mass_list, x_list, n, theta=0.7):
        """populates a list of masses and positions from a treewalk with opening angle theta"""
        r = Dist(x, self.COM)
        if self.size/r < theta:
            mass_list[n] = self.mass
            x_list[n] = self.COM
            n += 1
        else:
            for c in self.children:
                if c is None: continue
                if c.IsLeaf:
                    N = len(c.masses)
                    mass_list[n:n+N] = c.masses
                    x_list[n:n+N] = c.positions
                    n += N
                else: c.TreeWalk(x, mass_list, x_list, n, theta)

    def Potential(self, target, G=1., theta=0.7):
        x, m = np.empty((0,3)), np.empty((0,))
        self.TreeWalk(target, m, x, theta)
#        x = np.array(x)
#        m = np.array(m)
#        r = cdist(target[np.newaxis,:], x)
        return -G*np.sum([M/Dist(target,X) for M, X in zip(m, x)])  #PotentialFromLists(target,m,x) #np.sum(np.array(m)/r)
                
    def PotentialWalk(self, x, theta=0.7):
        r = Dist(x, self.COM)
        phi = 0.
        if self.size/r < theta or self.IsLeaf:
            phi = -self.mass / r
        else:
            for c in self.children:
                if c: phi += c.PotentialWalk(x, theta)
                
        return phi
    
from numba import njit
@njit
def PotentialFromLists(target, m, x):
    N = len(m)
    tot = 0.
    for i in range(N):
        tot += m[i]/Dist(target, x[i])
    return tot
