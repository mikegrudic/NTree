from .helpers import *
import numpy as np
from scipy.spatial.distance import cdist
from pykdgrav.bruteforce import BruteForcePotential

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
        
    def InsertPoint(self, pos, m):
        """Inserts a point of position pos and mass m into the tree."""
        if not self.mass: # no point already lives here, so let's make a leaf node and store the point there
            self.COM  = pos
            self.mass = m
            self.IsLeaf = True
            self.positions = [pos,]
            self.masses = [m,]
            return # we're done
        elif self.IsLeaf:  
            self.positions.append(pos)   # add the particle to the list of particles in the leaf
            self.masses.append(m)
            # now check whether we've gone over the limit; if so, we gotta split it up and make it an internal node
            if len(self.masses) > self.leafsize:
                self.children = 8 * [None,] # set up the children
                signs = np.array(self.positions) > self.center
                sectors = SignsToSectors(signs)
                # create the children needed
                child_size = self.size/2
                for i in range(len(self.masses)):
                    if not self.children[sectors[i]]:
                        self.children[sectors[i]] = BHTree2(self.center + child_size*(signs[i]-0.5), child_size, self.leafsize)
                    self.children[sectors[i]].InsertPoint(self.positions[i], self.masses[i])
                self.positions = None
                self.masses = None
                self.IsLeaf = False
        else: # we've got an internal node, so we gotta figure out which child to put it in and send it there
            signs = (pos > self.center)
            sector = SignsToSector(signs) # number from 0 to 7 deciding which octant
            if not self.children[sector]:  # if the child we need does not eposist, create it
                child_size = self.size/2
               child_center =  self.center + child_size*(signs-0.5)
                self.children[sector] = BHTree2(child_center, child_size, self.leafsize)
            self.children[sector].InsertPoint(pos, m)   #
#            self.SendPointToChild(pos,m)

    def SendPointToChild(self, pos, m):
        """Figures out which child node to send the point to, and initiates point insertion for that node."""
        signs = (pos > self.center)
        sector = SignsToSector(signs) # number from 0 to 7 deciding which octant
        if not self.children[sector]:  # if the child we need does not eposist, create it
            child_size = self.size/2
            child_center =  self.center + child_size*(signs-0.5)
            self.children[sector] = BHTree2(child_center, child_size, self.leafsize)
        self.children[sector].InsertPoint(pos, m)   # 

    def GetMoments(self):
        """Computes the mass and center of mass of a node recursively."""
        if self.IsLeaf:
            self.masses, self.positions = np.array(self.masses), np.array(self.positions)
#            BruteForcePotential(self.positions, self.masses)
            return np.sum(self.masses), np.average(self.positions, axis=0, weights=self.masses)
        else:
            self.mass = 0.
            self.COM = np.zeros(3)
            for c in self.children:
                if c is None: continue
                mc, posc = c.GetMoments()
                self.mass += mc
                self.COM += mc*posc
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
        
    def TreeWalk(self, target, mass_list=None, pos_list=None, theta=0.7):
        """populates a list of node masses and positions from a treewalk with opening angle theta"""
        if mass_list is None:
            mass_list = []
        if pos_list is None:
            pos_list = []
        r = Dist(target, self.COM)
        if self.size/r < theta:
            mass_list.append(self.mass)
            pos_list.append(self.COM)
        else:
            for c in self.children:
                if c is None: continue
                if c.IsLeaf:
                    N = len(c.masses)
                    mass_list += c.masses.tolist()
                    pos_list += c.positions.tolist()
                else:
                    c.TreeWalk(target, mass_list, pos_list, theta)
                    
        return mass_list, pos_list

    def Potential(self, target, G=1., theta=0.7):
        #np.empty((0,3)), np.empty((0,))
        m, pos = self.TreeWalk(target, theta=theta)
#        pos = np.array(pos)
#        m = np.array(m)
#        r = cdist(target[np.newaxis,:], pos)
        return PotentialFromLists(target,m,pos) #-G*np.sum(np.array(m)/r) #p.sum([M/Dist(target,POS) for M, POS in zip(m, pos)])  # #np.sum(np.array(m)/r)
                
    def PotentialWalk(self, target, theta=0.7):
        r = Dist(target, self.COM)
        phi = 0.
        if self.size/r < theta:
            phi = self.mass / r
        else:
            for c in self.children:
                if c is None: continue
                if c.IsLeaf:                    
                    phi += PotentialFromLists(target, c.masses, c.positions)
                else:
                    phi += c.PotentialWalk(target, theta=theta)
        return phi
#                    N = len(c.masses)
#                    mass_list = mass_list + c.masses.tolist()
#                    pos_list = pos_list + c.positions.tolist()
#                    mass_list[n:n+N] = c.masses
#                    pos_list[n:n+N] = c.positions
#                    n += N
#                else:
                    #print(target, mass_list, pos_list)
#                    c.TreeWalk(target, mass_list, pos_list, theta)            for c in self.children:
#                if c: phi += c.PotentialWalk(pos, theta)
                
       # return phi
    
from numba import jit
@jit
def PotentialFromLists(target, m, pos):
    N = len(m)
    tot = 0.
    for i in range(N):
        r = Dist(target, pos[i])
        if r>0: tot += m[i]/r
    return tot
