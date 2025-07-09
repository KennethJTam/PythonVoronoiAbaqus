# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:54:04 2025

@author: Kenneth J. Tam (The University of Melbourne)
@date:   24/06/25
@notes:  Generate Voronoi SuperCell and Trim
         There is probably a better way, but I'm not a coder and I am dumb
"""

from scipy.spatial import Voronoi
import numpy as np
import matplotlib.pyplot as plt

##=========================================================================
# Function
##=========================================================================

# Determine whether the voronoi ridge is fully or partially within the domain
def in_or_out(p1,p2,p3,box):
    
    if (
        all(box[0] <= x <= box[1] for x in p1) and
        all(box[2] <= y <= box[3] for y in p2) and
        all(box[4] <= z <= box[5] for z in p3)
        ):
            return 1 # fully
    
    else:
        for coord in range(len(p1)):
            if (
                (box[0] <= p1[coord] <= box[1]) and
                (box[2] <= p2[coord] <= box[3]) and
                (box[4] <= p3[coord] <= box[5]) 
                ):
                    return 2 # partially
    
    return 3 # not within 

# Redefine ridge vertices to be within domain
def trim(p1,p2,p3,box):
    
    # initialise and set up coords
    x = np.zeros(2)
    y = np.zeros(2)
    z = np.zeros(2)
    
    if (
        (box[0] <= p1[0] <= box[1]) and 
        (box[2] <= p2[0] <= box[3]) and 
        (box[4] <= p3[0] <= box[5]) 
        ):     
        point_i = 0
    else:
        point_i = 1
    point_o = 0 if point_i == 1 else 1
    
    # define coords as tail = 0 and head = 1
    x = [p1[point_i],p1[point_o]]
    y = [p2[point_i],p2[point_o]]
    z = [p3[point_i],p3[point_o]]
    
    # find largest deviation to use as base point  
    grad_norm = [(x[1]-x[0])/(box[1]-box[0]),
                 (y[1]-y[0])/(box[3]-box[2]),
                 (z[1]-z[0])/(box[5]-box[4])]
    grad = [(x[1]-x[0]),(y[1]-y[0]),(z[1]-z[0])]
    grad_sgn = np.sign(grad)
    grad_max = np.argmax(abs(np.asarray(grad)))
    
    # check for if max direction is within, if yes, finds something outside
    if grad_max == 0:
        if (box[0] <= x[0] <= box[1]) and (box[0] <= x[1] <= box[1]):
            grad_max = 1 if not ((box[2] <= y[0] <= box[3]) and (box[2] <= y[1] <= box[3])) else 2
                
        
    elif grad_max == 1:
        if (box[2] <= y[0] <= box[3]) and (box[2] <= y[1] <= box[3]):
            grad_max = 0 if not ((box[0] <= x[0] <= box[1]) and (box[0] <= x[1] <= box[1])) else 2

    elif grad_max == 2:
        if (box[4] <= z[0] <= box[5]) and (box[4] <= z[1] <= box[5]):
            grad_max = 0 if not ((box[0] <= x[0] <= box[1]) and (box[0] <= x[1] <= box[1])) else 1
        
    
    
    # define trimmed points
    if grad_max == 0:
        x[1]=box[1] if (grad_sgn[grad_max] == 1) else box[0]
        ratio = (x[1]-x[0])/abs((p1[1]-p1[0]))
        y[1] = y[0]+grad_sgn[grad_max]*ratio*grad_norm[1]*(box[3]-box[2])
        z[1] = z[0]+grad_sgn[grad_max]*ratio*grad_norm[2]*(box[5]-box[4])
        
    elif grad_max == 1:
        y[1]=box[3] if (grad_sgn[grad_max] == 1) else box[2]
        ratio = (y[1]-y[0])/abs((p2[1]-p2[0]))
        x[1] = x[0]+grad_sgn[grad_max]*ratio*grad_norm[0]*(box[1]-box[0])
        z[1] = z[0]+grad_sgn[grad_max]*ratio*grad_norm[2]*(box[5]-box[4])

    elif grad_max == 2:
        z[1]=box[5] if (grad_sgn[grad_max] == 1) else box[4]
        ratio = (z[1]-z[0])/abs((p3[1]-p3[0]))
        x[1] = x[0]+grad_sgn[grad_max]*ratio*grad_norm[0]*(box[1]-box[0])
        y[1] = y[0]+grad_sgn[grad_max]*ratio*grad_norm[1]*(box[3]-box[2])

    return x,y,z # not within 
        
##=========================================================================
# Generate Random X,Y,Z Coord for Seeds in a Domain
##=========================================================================

# initialize random number generator
rng = np.random.default_rng(10)

# create a set of coord points in 3D
sheet_x = 9
sheet_y = 7.76
sheet_z = 2
seeds = 3
x = rng.uniform(0,sheet_x,seeds)
y = rng.uniform(0,sheet_y,seeds)
z = rng.uniform(0,sheet_z,seeds)
eps = np.finfo(float).eps

#define bounding box
bounding_box = np.array([sheet_x, 2*sheet_x, sheet_y, 2*sheet_y, 4*sheet_z, 5*sheet_z]) 

##=========================================================================
# Replicate Domain in a 3x3x3 Grid - Super Cell
##=========================================================================

# save original coords
x_temp = x     
y_temp = y
z_temp = z

# seeds for supercell generation
for i in range(3):
    for j in range(3):
        for k in range(9):
            if (i==0 and j==0 and k==0): # skip original coords
                continue
            x = np.append(x,x_temp+i*sheet_x)
            y = np.append(y,y_temp+j*sheet_y)
            z = np.append(z,z_temp+k*sheet_z)

points = np.vstack((x, y, z))
points = np.transpose(points) # total number of point: seeds*3^3  

##=========================================================================
# Generate Super Cell Voronoi Tesselation: Visualisation and Coords
##=========================================================================

# creation of Voronoi
vor = Voronoi(points)

# plot initialise 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(bounding_box[0],bounding_box[1]); ax.set_ylim(bounding_box[2],bounding_box[3]); ax.set_zlim(bounding_box[4],bounding_box[5])

# Determine location of ridges; inside domain, partially, or not
filter_ridges = []   
filter_regions = []   
modified_family = []  
check = 0   
for simplex in vor.ridge_vertices:
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):
        
        # pairs of simplex
        ordered = [] 
        for coord in range(len(simplex)):
            coord1=coord+1
            if coord1 == len(simplex): coord1=0 
            ordered.append([simplex[coord],simplex[coord1]])
        
        # extract pairs within or partially within domain
        for pair in ordered:
            pair = np.asarray(pair)
            a = in_or_out(np.asarray(vor.vertices[pair, 0]),np.asarray(vor.vertices[pair, 1]),
                          np.asarray(vor.vertices[pair, 2]),bounding_box)  
            if a == 1 or a==2:
                filter_ridges.append(pair)
                for ridge in pair:
                    modified_family.append(ridge)
                check = 1
                
        if check == 1:
            filter_regions.append(modified_family) 
            modified_family = [] 
            check = 0

# filter ridges per region
for simplex in filter_regions:
    test_value = 1
    test_simplex = []    
    for ridge in simplex:
        if ridge != test_value:
            test_simplex.append(ridge)
            test_value = ridge
    modified_family.append(test_simplex)
    

                
# Draw
modified_ridges = []  
for simplice in filter_ridges:
    simplice = np.asarray(simplice)  
    a = in_or_out(np.asarray(vor.vertices[simplice, 0]),np.asarray(vor.vertices[simplice, 1]),
                  np.asarray(vor.vertices[simplice, 2]),bounding_box)                    
    # draw ridges within domain
    if a == 1:
        plt.plot(vor.vertices[simplice, 0], vor.vertices[simplice, 1], vor.vertices[simplice, 2], 'y-')
        # test = 1
    #draw ridges partially within domain
    elif a == 2:
        trimmed = trim(np.asarray(vor.vertices[simplice, 0]),np.asarray(vor.vertices[simplice, 1]),
                       np.asarray(vor.vertices[simplice, 2]),bounding_box)
             
        aa = in_or_out(trimmed[0],trimmed[1],trimmed[2],bounding_box)
        while (aa == 2):
            # plt.plot(vor.vertices[simplice, 0], vor.vertices[simplice, 1], vor.vertices[simplice, 2], 'm-')
            trimmed1 = trim(trimmed[0],trimmed[1],trimmed[2],bounding_box)
                 
            aa = in_or_out(trimmed1[0],trimmed1[1],trimmed1[2],bounding_box)
            trimmed = trimmed1
            
            
        modified_ridges.append([trimmed[0][0],trimmed[1][0],trimmed[2][0]])
        modified_ridges.append([trimmed[0][1],trimmed[1][1],trimmed[2][1]])
        plt.plot(trimmed[0], trimmed[1], trimmed[2], 'c-')
modified_ridges=np.asarray(modified_ridges)


plt.show()


bound_plot = np.asarray( [[bounding_box[0],bounding_box[2],bounding_box[4]],[bounding_box[1],bounding_box[2],bounding_box[4]],
                          [bounding_box[0],bounding_box[3],bounding_box[4]],[bounding_box[0],bounding_box[2],bounding_box[5]],
                          [bounding_box[1],bounding_box[3],bounding_box[4]],[bounding_box[1],bounding_box[2],bounding_box[5]],
                          [bounding_box[0],bounding_box[3],bounding_box[5]],[bounding_box[1],bounding_box[3],bounding_box[5]]])

ax.plot(bound_plot[:, 0], bound_plot[:, 1], bound_plot[:, 2], 'b.')
