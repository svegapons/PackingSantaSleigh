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
        self.id_pres = ld[700000:,0] - 700000
        self.pres = ld[700000:,1:]
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
        
        self.id_batch = None
        self.pres_batch = None
        
        
        
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
        base_z = 0
        while len(id_presents) > 0:
            print "Analyzing batch number: !!!!!!!!!!!!!!!!!!!!!!!!! %s !!!!!!!!!!!!!" %(cont)
            batch, max_d = self._find_batch(id_presents)
            #Main code here
            self.batch_mask[np.array(batch) - 1] = True
            
            #Getting the presents for the current batch.
            self.id_batch = self.id_pres[self.batch_mask]
            print "Batch of %s presents with Ids in range [%s - %s]" %(len(self.id_batch), self.id_batch[0], self.id_batch[-1])
            self.pres_batch = self.ord_pres[self.batch_mask]
            #pdb.set_trace()
            
            #print "Batch: %s, %s, %s" %(np.min(batch) , np.max(batch), max_d + max_d/10)
            self.solve_batch(base_z, base_z + max_d + max_d/20)
            
            #pdb.set_trace()
            ##################
            ##ENCONTRAR EL MEJOR SPLIT BASADO EN LOS RESULTADOS DE LA UBICACION REAL OBTENIDA
            id_presents = np.concatenate((self.id_pres[self.batch_mask], self.id_pres[np.max(batch) :]))
            base_z = np.max(self.solution[:,-1]) + 1
            #pdb.set_trace()
#            id_presents = list(set(id_presents) - set(batch))
#            id_presents.sort()
            cont += 1 
            
        np.savetxt('raw_solution.csv', self.solution, delimiter = ',')
        
        #inverting the z-coordinates in order to have a valid submission.
        max_z = np.max(self.solution[-1])
        submission = np.zeros((self.n_pres, 25), dtype=np.int64)
        for i in self.solution:
#                Vertex convention: x1 y1 z1
#                       x1 y2 z1
#                       x2 y1 z1
#                       x2 y2 z1
#                       x1 y1 z2
#                       x1 y2 z2
#                       x2 y1 z2
#                       x2 y2 z2
            self.solution[2] = max_z - self.solution[2]
            self.solution[5] = max_z - self.solution[5]
            x1 = self.solution[0] + 1
            y1 = self.solution[1] + 1
            z1 = self.solution[2] + 1
            x2 = self.solution[3] + 1
            y2 = self.solution[4] + 1
            z2 = self.solution[5] + 1
            submission[i] = np.array([x1, y1, z1, x1, y2, z1, x2, y1, z1, x2, y2, z1, x1, y1, z2, x1, y2, z2, x2, y1, z2, x2, y2, z2], dtype=np.int64)
            np.savetxt('my_submission.csv', submission, delimiter = ',')
               
    
    
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
        
        sx1, sy1, sz1, sx2, sy2, sz2 = sleigh
        #Position to be returned
        r_x = sx1
        r_y = sy1
        r_z = sz1
        
        #Size of the sleigh
        dx = sx2 - sx1 + 1
        dy = sy2 - sy1 + 1
        dz = sz2 - sz1 + 1
        
        best_fit = []
        aux = 0
        min_val = 100000000000.
        vol_threshold = 1000000
        
        for i in range(len(self.pres_batch)):
            if self.batch_mask[self.id_batch[i]-1] == False:
                continue
            x, y, z = self.pres_batch[i, 0], self.pres_batch[i, 1], self.pres_batch[i, 2]
            
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
            idp = self.id_batch[best_fit[0]]
            self.solution[idp - 1] = np.array([sx1, sy1, sz1, sx1 + best_fit[1] - 1, sy1 + best_fit[2] - 1, sz1 + best_fit[3] - 1])
            r_x = sx1 + best_fit[1] - 1
            r_y = sy1 + best_fit[2] - 1
            r_z = sz1 + best_fit[3] - 1
            self.batch_mask[idp - 1] = False
            #print "Object with id %s placed at [%s, %s, %s], [%s, %s, %s]" %(idp, sx1, sy1, sz1, r_x, r_y, r_z)
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
        print "Sleigh: [%s, %s, %s], [%s, %s, %s]" %(0, 0, base_z, self.dim_x, self.dim_y, dim_z)
        self._solve_sub_batch(sleigh)
                
    
        
    def _solve_sub_batch(self, sleigh):
        """
        """
        sx1, sy1, sz1, sx2, sy2, sz2 = sleigh
        #print "Sleigh: [%s, %s, %s], [%s, %s, %s]" %(sx1, sy1, sz1, sx2, sy2, sz2)
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
        
        
#        #Computation of the three new subproblems depending on the size of 
#        #each dimension.
#        if dz == np.min([dx, dy, dz]):
#            sleigh_1 = [sx1, sy1, r_z + 1, r_x, r_y, sz2]
#            if dx <= dy:
#                sleigh_2 = [r_x + 1, sy1, sz1, sx2, r_y, sz2]
#                sleigh_3 = [sx1, r_y + 1, sz1, sx2, sy2, sz2]
#            else:
#                sleigh_2 = [sx1, r_y + 1, sz1, r_x, sy2, sz2]
#                sleigh_3 = [r_x + 1, sy1, sz1, sx2, sy2, sz2]
#        elif dy == np.min([dx, dy, dz]):
#            sleigh_1 = [sx1, r_y + 1, sz1, r_x, sy2, r_z]
#            if dx <= dz:
#                sleigh_2 = [r_x + 1, sy1, sz1, sx2, sy2, r_z]
#                sleigh_3 = [sx1, sy1, r_z + 1, sx2, sy2, sz2]
#            else:
#                sleigh_2 = [sx1, sy1, r_z + 1, r_x, sy2, sz2]
#                sleigh_3 = [r_x + 1, sy1, sz1, sx2, sy2, sz2]
#        else:
#            sleigh_1 = [r_x + 1, sy1, sz1, sx2, r_y, r_z]
#            if dy <= dz:
#                sleigh_2 = [sx1, r_y + 1, sz1, sx2, sy2, r_z]
#                sleigh_3 = [sx1, sy1, r_z + 1, sx2, sy2, sz2]
#            else:
#                sleigh_2 = [sx1, sy1, r_z + 1, sx2, r_y, sz2]
#                sleigh_3 = [sx1, r_y + 1, sz1, sx2, sy2, sz2]
        
        
       ####Dejando siempre z para el ultimo llamado => Debe ayudar a mantener el orden!!        
        if dy < dx:
            sleigh_1 = [sx1, r_y + 1, sz1, r_x, sy2, sz2]
            sleigh_2 = [r_x + 1, sy1, sz1, sx2, sy2, sz2]
            sleigh_3 = [sx1, sy1, r_z + 1, r_x, r_y, sz2]
        else:
            sleigh_1 = [r_x + 1, sy1, sz1, sx2, r_y, sz2]
            sleigh_2 = [sx1, r_y + 1, sz1, sx2, sy2, sz2]
            sleigh_3 = [sx1, sy1, r_z + 1, r_x, r_y, sz2]
        #########

        
        #Recursive call for the three new subproblems...
        self._solve_sub_batch(sleigh_1)
        self._solve_sub_batch(sleigh_2)
        self._solve_sub_batch(sleigh_3)
            
            
if __name__ == '__main__':

    psl = PackingSantaSleigh()
    psl.solve()
    np.savetxt('solution.csv', psl.solution, delimiter = ',')
    print "Finito"



       