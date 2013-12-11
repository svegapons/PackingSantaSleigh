# -*- coding: utf-8 -*-
"""

"""
import os
import csv
import math
import time
import numpy as np
import skimage.morphology
import skimage.measure
import pdb


#class Sleigh:
#    """
#    """
#    def __init__(dim_x=1000, dim_y=1000):
#        """
#        dim_z = infinity
#        """
#        self.dim_x = dim_x
#        self.dim_y = dim_y
#    
#    #    
##    self.surface = np.zeros((self.dim_x, self.dim_y), dtype=int)
##    
##    self.subm = []
##    
##    self.height = 0
#    
##    def place_present(x, y, z, present):
##        """
##        Place the present in the sleigh at the given positions.
##        Parameters:
##            x, y, z: define the first point 
##            present: row from the present file: id, length_x, length_y, length_z
##            (It is assumed that the present was already rotated (if needed))
##        """        
##        
##        self.subm.append([present[0], x+1, y+1, z+1, x+1, y+present[2]+1, z+1, x+present[1]+1, y+1, z+1, x+present[1]+1, y+present[2]+1, z+1, x+1, y+1, z+present[3]+1, x+1, y+present[2]+1, z+present[3]+1, x+present[1]+1, y+1, z+present[3]+1, x+present[1]+1, y+present[2]+1, z+present[3]+1])
##        self.surface[x:x+present[1], y:y+present[2]] += present[3]
#                       
#                       
#    def place_new_present(present):
#        """
#        Finds the best place for a given present.
#        """
#                    
#        
#        #Important code
#        li = skimage.morphology.label(self.surface) + 1
#        
#        self.n_conn_comp = np.max(li)
#        #dict_info = skimage.measure.regionprops(li)
#                       
#                       
#            
#            
#class Surface:
#    """
#    Represents the top of the sleigh...gives the shape of the current position.
#    """ 
#    
#    def __init__(dim_x=1000, dim_y=1000):
#        """
#        Creates an initial surface...all zeros.
#        """  
#        self.data = np.zeros((dim_x, dim_y), dtype=np.int)
#        #Only one connected component at the begining.
#        self.n_cc = 0
#    
#    
#    def evaluate():
#        """
#        Evaluate the current state of the surface.
#        """
#        li = skimage.morphology.label(self.surface) + 1
#        self.n_cc = np.max(li)
#        
#        
#        
#    def evaluate_candidate(x1, y1, x2, y2, dim_z):
#        """
#        Provides the evaluation for the possible insertion of a present on the
#        rectangle defined by (x1,y1) (x2,y2) with a height dim_z.
#        Elements to take into account for evaluation:
#        - Number of connected components (- => better)
#        - Increase in z dimension (- => better)
#        - Size of connected components (+ => better)
#        - Maximum slope (- => better)
#        """
#        
#    pass    
#    ##IDEA
#    #Leer los presentes tal que la suma de sus dimensiones sea aproximadamente
#    #el area de la base del trineo...este batch de presentes tratar de resolverlo 
#    #con optimizacion. Asi sucesivamente pasar a otro batch.
#    #La idea es q los presentes en un mismo batch no hay por que ponerlos en orden,
#    #pq si estan a la misma altura la medida los organiza...solo hay q minimizar el 
#    #alto del trineo.
#    #El area promedio es 50*50=2500...es decir q harian falta unos 400 presentes
#    #para llenar los 1000*1000 casillas de un nivel..por tanto se podrian usar 
#    #batches de 400 objetos.
            
            

#
#def find_batches2(ord_pres, vol_pres, dim_x, dim_y):
#    """
#    """
#    batches = []
#    sums = 0
#    cont = 0 
#    while cont < len(ord_pres):
#        for i in range(cont + 1, len(ord_pres)):
#            tv = np.max(ord_pres[cont:i,0])
#            rv = np.sum(vol_pres[cont:i])
#            if(rv > tv*dim_x*dim_y):
#                batches.append([cont, i - 1, tv])
#                sums += tv
#                cont = i - 1
#                print batches[-1]                
#                break
#            if i == len(ord_pres)-1:
#                sums += tv
#                return batches, sums 
#            
#    return batches, sums 


           
    
    
def find_batches(ord_pres, vol_pres):
    """
    """
    batches = []
    sums = 0
    cont = 0 
    while cont < 1000000:
        for i in range(cont + 1, 1000000):
            tv = np.max(ord_pres[cont:i,0])
            rv = np.sum(vol_pres[cont:i])
            if(rv > tv*1000000.):
                batches.append([cont, i - 1, tv])
                sums += tv
                cont = i - 1
                print batches[-1]                
                break
            if i == 999999:
                sums += tv
                return batches, sums 
            
    return batches, sums 
        
#Batches 236,     



def solve_sub_problem(pres, id_pres, vol_pres, dim_z):
    """
    Solve the problem for a batch of presents.
    1000*1000*dim_z is the size of the cube inside which all presents in pres
    have to be placed.
    """
    #Sorting by the smaller-dimension.
    idx = ord_pres[:,0].argsort()
    pres = pres[idx]
    id_pres = id_pres[idx]
    vol_pres = vol_pres[idx]
    
    #This list will have the presents already placed in the sleigh
    res 
    
    #The sleigh for this subproblem.
    sleigh = np.array((1000, 1000, dim_z), dtype=np.int32)
    #Position where the next present is going to be placed.
    pos_x = 0
    pos_y = 0
    pos_z = 0
    #Dimensions of the part of the sleigh to be used. Now is the whole sleigh 
    #but in recursive calls dimensions will be decreased.
    d_x = 1000
    d_y = 1000
    d_z = dim_z
    #Cada vez q se pone un presente el problema se subdivide en 3 nuevos.
    
    
    
    
def _solve_sub_prob(sleigh, min_x, dim_x, min_y, dim_y, min_z, dim_z, pres, id_pres, flag_pres):
    """
    """
#    for i in range(len(pres)):
#        if 


def fit_inside_volume(dim_x, dim_y, dim_z, vol_x, vol_y, vol_z):
    """
    Determine wheather it is possible to place a present with dimensions
    dim_x, dim_y and dim_z inside a volume of dimensions vol_x, vol_y, vol_z.
    Tries all possible rotations and returns a boolean array denoting the 
    available rotations...
    the order of rotations are:
    0 -> dim_x, dim_y, dim_z
    1 -> dim_x, dim_z, dim_y
    2 -> dim_y, dim_x, dim_z
    3 -> dim_y, dim_z, dim_x
    4 -> dim_z, dim_x, dim_y
    5 -> dim_z, dim_y, dim_x
    """
    res = np.zeros(6)
    xx = vol_x - dim_x > 0
    yx = vol_y - dim_x > 0
    zx = vol_z - dim_x > 0
    
    xy = vol_x - dim_y > 0
    yy = vol_y - dim_y > 0
    zy = vol_z - dim_y > 0
    
    xz = vol_x - dim_z > 0
    yz = vol_y - dim_z > 0
    zz = vol_z - dim_z > 0
    
    res[0] = (xx and yy and zz)
    res[1] = (xx and yz and zy)
    res[2] = (xy and yx and zz)
    res[3] = (xy and yz and zx)
    res[4] = (xz and yx and zy)
    res[5] = (xz and yy and zx)
    
    return res
    

    

def run(path='presents.csv'):
    """
    """
    #Get id and values of the presents
    ld = np.loadtxt(path, dtype=np.int64, delimiter=',', skiprows=1)
    id_pres = ld[:,0] 
    pres = ld[:,1:]
    print type(pres[0,0])
    #Get a n ordered copy of the presents
    ord_pres = np.sort(pres, axis=1)
    #Volumes of the presents
    vol_pres = pres[:,0].astype(np.float64) * pres[:,1].astype(np.float64) * pres[:,2].astype(np.float64)
    
    res, sums = find_batches(ord_pres, vol_pres)
    print "Cota inferior: " + str(sums)
    print np.sum(vol_pres)
    
#    res, sums = find_batches2(ord_pres[:236], vol_pres[:236], 1000, 195)
#    return res, sums
    
#res, sums = run()    
    
    
###########################################
######## NEW CODE #########################
###########################################


class PackingSantaSleigh:
    """
    My solution to the Packing Santa's Sleigh problem.
    """
    def __init__(self, dim_x=1000, dim_y=1000, path_presents='presents.csv'):
        """
        dim_x: defines the dimension of the sleigh along the x coordinate
        dim_y: defines the dimension of the sleigh along the y coordinate
        path_presents: Path to the file with the presents to be placed inside the sleigh
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        
        #Get id and values of the presents
        print "Loading list of presents..."
        ld = np.loadtxt(path_presents, dtype=np.int64, delimiter=',', skiprows=1)
        self.id_pres = ld[:,0]
        self.pres = ld[:,1:]
        #Present with index i is located at position i-1 in self.pres (id of presents starts in 1)
        
        #Number of presents
        self.n_pres = len(self.pres)
        
        #Get an ordered copy of the presents
        print "Computing ordered version of the presents..."
        self.ord_pres = np.sort(self.pres, axis=1)
        #Volumes of the presents
        print "Computing volumes of the presents..."
        self.vol_pres = self.pres[:,0].astype(np.float64) * self.pres[:,1].astype(np.float64) * self.pres[:,2].astype(np.float64)
        
        #Array to be used for the solution to the problem. For each present
        #the first 3 values are x,y,z of first point and the other three values
        #the x,y,z of the other point needed to define its location.
        self.solution = np.zeros((self.n_pres, 6), dtype=np.int32)
        
        #Mask to be used for the subproblems
        self.batch_mask = np.zeros(self.n_pres, dtype=bool)        
        self.cum_vol_sol = 0.
        
        
        
    def _find_batch(self, id_presents):
        """
        Starting from initial_position get a batch of consecutive presents
        that fits into a box with dim_z equal to the maximum z value of the
        objects in the batch.
        ini_position: Position to start the search.
        id_presents: The id of presents from which the batch is going to be selected.
        """
        batch = []
        max_d = 0.
        rv = 0.
        ini_position = np.min(id_presents)
        for i in id_presents:
            tv = self.ord_pres[i - 1, 0]
            if tv > max_d:
                max_d = tv
            rv += self.vol_pres[i - 1]
            if rv > max_d * float(self.dim_x) * float(self.dim_y):  
                #pdb.set_trace()
                print str(rv)
                return batch, max_d 
            batch.append(i)
                
        return id_presents, max_d 
        
            
    def solve(self):
        """
        Solve the Packing problem. Return the assignation of presents.
        """
        id_presents = self.id_pres
        #Main loop. Problem is splited in different subproblems (batches)
        cont = 0
        while len(id_presents) > 0:
            print "Analyzing batch number %s" %(cont)
            batch, max_d = self._find_batch(id_presents)
            #Main code here
            self.batch_mask[np.array(batch) - 1] = True
            print "Batch: %s, %s, %s" %(np.min(batch) , np.max(batch), max_d)
            self.solve_batch(0, max_d + max_d/5)  
            
            pdb.set_trace()
            ##################
            ##ENCONTRAR EL MEJOR SPLIT BASADO EN LOS RESULTADOS DE LA UBICACION REAL OBTENIDA

            id_presents = self.id_pres[np.max(batch) :]
#            id_presents = list(set(id_presents) - set(batch))
#            id_presents.sort()
            cont += 1
            
    
    def place_present(self, sleigh):
        """
        Place one present at the begining of the sleigh
        The present with a dimension closer to one of the dimensions of the 
        sleigh is selected.
        If no present can be placed in the sleigh, idp = -1
        sleigh: Position of the current sleigh
        """
        #index of the present to be placed (by default = -1, meaning that no present was placed)
        idp = -1
        
        #Getting the objects
        id_pres = self.id_pres[self.batch_mask]
        ord_pres = self.ord_pres[self.batch_mask]
        
        sx1, sy1, sz1, sx2, sy2, sz2 = sleigh
        #Position to be returned
        r_x = sx1
        r_y = sy1
        r_z = sz1
        
        #Size of the sleigh
        dx = sx2 - sx1
        dy = sy2 - sy1
        dz = sz2 - sz1
        
        best_fit = []
        aux = 0
        min_val = 100000000000.
        vol_threshold = 500000
        
        for i in range(len(ord_pres)):
            x, y, z = ord_pres[i, 0], ord_pres[i, 1], ord_pres[i, 2]
            
#            the order of rotations are:
#            0 -> dim_x, dim_y, dim_z
#            1 -> dim_x, dim_z, dim_y
#            2 -> dim_y, dim_x, dim_z
#            3 -> dim_y, dim_z, dim_x
#            4 -> dim_z, dim_x, dim_y
#            5 -> dim_z, dim_y, dim_x
            
            res = np.zeros(6)
            xx = dx - x
            yx = dy - x
            zx = dz - x
            
            xy = dx - y
            yy = dy - y
            zy = dz - y
            
            xz = dx - z
            yz = dy - z
            zz = dz - z
            

#            Coding of the possible rotations...
#            (xx and yy and zz)
            if xx>=0 and yy>=0 and zz>=0:
                #better fit in one dimension
                #aux = min(xx, yy, zz)
                #minimum empty volume
                if dx*dy*dz < vol_threshold:
                    aux = min(xx, yy, zz)
                else:
                    aux = xx*yy*zz
                if aux <= min_val:
                    best_fit = [i, x, y, z]
                    min_val = aux
#            (xx and yz and zy)                    
            if xx>=0 and yz>=0 and zy>=0:
                #aux = min(xx, yz, zy)
                #minimum empty volume
                if dx*dy*dz < vol_threshold:
                    aux = min(xx, yz, zy)
                else:
                    aux = xx*yz*zy
                if aux <= min_val:
                    best_fit = [i, x, z, y]
                    min_val = aux
#            (xy and yx and zz)                    
            if xy>=0 and yx>=0 and zz>=0:
                #aux = min(xy, yx, zz)
                #minimum empty volume
                if dx*dy*dz < vol_threshold:
                    aux = min(xy, yx, zz)
                else:                
                    aux = xy*yx*zz
                if aux <= min_val:
                    best_fit = [i, y, x, z]
                    min_val = aux
#            (xy and yz and zx)                    
            if xx>=0 and yz>=0 and zy>=0:
                #aux = min(xx, yz, zy)
                #minimum empty volume
                if dx*dy*dz < vol_threshold:
                    aux = min(xx, yz, zy)
                else:
                    aux = xx*yz*zy
                if aux <= min_val:
                    best_fit = [i, x, z, y]
                    min_val = aux                
#            (xz and yx and zy)  
            if xz>=0 and yx>=0 and zy>=0:
                #aux = min(xz, yx, zy)
                #minimum empty volume
                if dx*dy*dz < vol_threshold:
                    aux = min(xz, yx, zy)
                else:
                    aux = xz*yx*zy
                if aux <= min_val:
                    best_fit = [i, z, x, y]
                    min_val = aux                     
#            (xz and yy and zx)                    
            if xz>=0 and yy>=0 and zx>=0:
                #aux = min(xz, yy, zx)
                #minimum empty volume
                if dx*dy*dz < vol_threshold:
                    aux = min(xz, yy, zx)
                else:
                    aux = xz*yy*zx
                if aux <= min_val:
                    best_fit = [i, z, y, x]
                    min_val = aux             
        
        if best_fit != []:
            #Place the best present..
            idp = id_pres[best_fit[0]]
            self.solution[idp - 1] = np.array([sx1, sy1, sz1, sx1 + best_fit[1], sy1 + best_fit[2], sz1 + best_fit[3]])
            r_x = sx1 + best_fit[1]
            r_y = sy1 + best_fit[2]
            r_z = sz1 + best_fit[3]
            self.batch_mask[idp - 1] = False
            print "Object with id %s placed at [%s, %s, %s], [%s, %s, %s]" %(idp, sx1, sy1, sz1, r_x, r_y, r_z)
            self.cum_vol_sol += (r_x - sx1)*(r_y - sy1)*(r_z - sz1)
                
        return idp, r_x, r_y, r_z
        

    def solve_batch(self, base_z, dim_z):
        """
        Solve the problem for a batch of presents.
        base_z: z coordinate of the begining of the sleigh for the subproblem.
        dim_z: The size of the sleigh is (self.dim_x * self.dim_y * dim_z).
        """
        #The sleigh for this subproblem.
        sleigh = [0, 0, base_z, self.dim_x, self.dim_y, dim_z]        
        self._solve_sub_batch(sleigh)
                
    
        
    def _solve_sub_batch(self, sleigh):
        """
        """
        sx1, sy1, sz1, sx2, sy2, sz2 = sleigh
        print "Sleigh: [%s, %s, %s], [%s, %s, %s]" %(sx1, sy1, sz1, sx2, sy2, sz2)
        idp, r_x, r_y, r_z = self.place_present(sleigh) 
        #If no present can be placed inside the sleigh...return!
        if idp < 0:
            return 
            
        #Delta values
        dx = sx2 - r_x
        dy = sy2 - r_y
        dz = sz2 - r_z
        
#        if dx<=0 or dy<=0 or dz<=0:
#            return
        
        #Computation of the three new subproblems depending on the size of 
        #each dimension.
        if dz == np.min([dx, dy, dz]):
            sleigh_1 = [sx1, sy1, r_z + 1, r_x, r_y, sz2]
            if dx <= dy:
                sleigh_2 = [r_x + 1, sy1, sz1, sx2, r_y, sz2]
                sleigh_3 = [sx1, r_y + 1, sz1, sx2, sy2, sz2]
            else:
                sleigh_2 = [sx1, r_y + 1, sz1, r_x, sy2, sz2]
                sleigh_3 = [r_x + 1, sy1, sz1, sx2, sy2, sz2]
        elif dy == np.min([dx, dy, dz]):
            sleigh_1 = [sx1, r_y + 1, sz1, r_x, sy2, r_z]
            if dx <= dz:
                sleigh_2 = [r_x + 1, sy1, sz1, sx2, sy2, r_z]
                sleigh_3 = [sx1, sy1, r_z + 1, sx2, sy2, sz2]
            else:
                sleigh_2 = [sx1, sy1, r_z + 1, r_x, sy2, sz2]
                sleigh_3 = [r_x + 1, sy1, sz1, sx2, sy2, sz2]
        else:
            sleigh_1 = [r_x + 1, sy1, sz1, sx2, r_y, r_z]
            if dy <= dz:
                sleigh_2 = [sx1, r_y + 1, sz1, sx2, sy2, r_z]
                sleigh_3 = [sx1, sy1, r_z + 1, sx2, sy2, sz2]
            else:
                sleigh_2 = [sx1, sy1, r_z + 1, sx2, r_y, sz2]
                sleigh_3 = [sx1, r_y + 1, sz1, sx2, sy2, sz2]
        
        #Recursive call for the three new subproblems...
        self._solve_sub_batch(sleigh_1)
        self._solve_sub_batch(sleigh_2)
        self._solve_sub_batch(sleigh_3)
            
            
if __name__ == '__main__':

    psl = PackingSantaSleigh()
    psl.solve()




       