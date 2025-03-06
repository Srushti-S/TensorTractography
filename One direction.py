# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:33:40 2019

@author: Srushti

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


# create a function to evaluate a gaussian (normal distribution)
def gaussian(X, Y, mu_x, mu_y, sigma):
    e_numerator = (X - mu_x) ** 2
    e_denominator = 2 * (sigma ** 2)
    
    Gx = 1.0 / (np.sqrt(2 * np.pi * sigma * sigma)) * np.exp(-e_numerator / e_denominator)
    
    # calculate the gaussian along y
    e_numerator = (Y - mu_y) ** 2
    Gy = 1.0 / (np.sqrt(2 * np.pi * sigma * sigma)) * np.exp(-e_numerator / e_denominator)
    
    return Gx * Gy


D = 10                          # specify the size of the spatial domain
N = 50                         # specify the size of the image


x = np.linspace(-D, D, N)       # create a position vector for the x-axis
y = x  
                         # create a position vector for the y-axis
[X, Y] = np.meshgrid(x, y)      # create a meshgrid representing the spatial axis

mu_0 = [-5, -5]                 # set the mean of two 2D gaussian functions
mu_1 = [5, 5]

sigma_0 = 3                     # set the standard deviation of two 2D gaussian functions
sigma_1 = 3

G0 = gaussian(X, Y, mu_0[0], mu_0[1], sigma_0)
G1 = gaussian(X, Y, mu_1[0], mu_1[1], sigma_1)

# add both Gaussians together
I = G0 + G1
#plt.imshow(I)

# calculate the gradient
[Iy, Ix] = np.gradient(I)

# calculate the extent for displaying images
ex = [-D, D, D, -D]

#display both gradients
#plt.subplot(2, 3, 1)
#plt.imshow(Ix, extent=ex)
#plt.title("Gradient along X")
#plt.colorbar()
#plt.subplot(2, 3, 4)
#plt.imshow(Iy, extent=ex)
#plt.title("Gradient along Y")
#plt.colorbar()

# allocate space for the tensor field
T = np.zeros(Ix.shape + (2,2))

# calculate each component of the 2D structure tensor
T[:, :, 0, 0] = Ix * Ix
T[:, :, 0, 1] = Ix * Iy
T[:, :, 1, 0] = Iy * Ix
T[:, :, 1, 1] = Iy * Iy
#
#plt.subplot(2, 3, 2)
#plt.imshow(T[:, :, 0, 0], extent=ex)
#plt.title("Ix * Ix")
#plt.subplot(2, 3, 5)
#plt.imshow(T[:, :, 0, 1], extent=ex)
#plt.title("Iy * Ix")
#plt.subplot(2, 3, 6)
#plt.imshow(T[:, :, 1, 1], extent=ex)
#plt.title("Iy * Iy")

# calculate the vector field from the tensor field T
[evals, evecs] = np.linalg.eigh(T)
V = evecs[:, :, :, 0]


C = np.zeros(Ix.shape + (3,))   #dy = V[2,1,1]
                                    #dx = V[2,1,0]
C[:, :, 0:2] = np.abs(V)

plt.subplot(1, 2, 1)
plt.quiver(X, np.flipud(Y), np.flipud(V[:, :, 0]), np.flipud(V[:, :, 1]))

plt.title("Tensor field")
#initial point
pt = [0,1]

x_coordinate = pt[0]                #x co-ordinate of initial point
y_coordinate = pt[1]                #y co-ordinate of initial point

px_xy = (D+D)/N                         # 1 pixel value equivalent to co-ordinate value

step_size = 0.1             #step size

px =np.array([x_coordinate])
py =np.array([y_coordinate])

while x_coordinate<=1 or  x_coordinate>=-1 :  
   # for t in range(50):
        
    #for x_coordinate in range(10):    
        
    if type(x_coordinate)==int or type(y_coordinate)==int: 
        
        p_x = int( (x_coordinate+10) / 0.4 )       #initial p_x is 0   
        
        p_y = int( 49- (y_coordinate+10) / 0.4 )   #initial p_y is 49   
           
        if p_y > 49 or p_x > 49 :       #index 50 is out of bounds for axis 0 with size 50
            break
        dy = V[p_x,p_y,1]
        dx = V[p_x,p_y,0]
        
            
        #Calculate next (x,y)points on the graph
            
        x_coordinate = x_coordinate + (dx*step_size) 
        y_coordinate = y_coordinate + (dy*step_size)
        
        px = np.append(px,[x_coordinate])
        py = np.append(py,[y_coordinate])      

    else:
        
        #Claculation of (x1,y1), (x2,y2), (x3,y3), (x4,y4) are the surrounding co-ordinates
        
        n = 0
        
        
        while n < len(x):                   #x = np.linspace(-D, D, N)
            
            v = x[n]-x_coordinate
            n=n+1
            
            if -1<v<0:
                x0 = x_coordinate - v
        
       
        
        
        p_x1 = int( (x0+D) / px_xy )        #initial p_x is 0      
        p_x3 = p_x1
       
        
        x1 = x0 + px_xy
        p_x2 = int( (x1+D) / px_xy )        #initial p_x is 0  
        p_x4 = p_x2
        
        n = 0
       
        while n < len(y):                   #y = np.linspace(-D, D, N)
            
            val = y[n]-y_coordinate
            n=n+1
            
            if -1<val<0:
                y0 = y_coordinate - val
                
        p_y1 = int( 49- (y0+D) /px_xy )   #initial p_y is 49   
        p_y2 = p_y1
        
            
        y1 = y0 + px_xy       
        p_y3 = int( 49- (y1+D) / px_xy )   #initial p_y is 49  
        p_y4 = p_y3
        
        if p_y3 > 49 or p_x3 > 49 :       #index 50 is out of bounds for axis 0 with size 50
            break
        if p_y1 > 49 or p_x1 > 49 :       #index 50 is out of bounds for axis 0 with size 50
            break 
        if p_y2 > 49 or p_x2 > 49 :       #index 50 is out of bounds for axis 0 with size 50
            break
        if p_y4 > 49 or p_x4 > 49 :       #index 50 is out of bounds for axis 0 with size 50
            break
            
        z11 = T[p_x1, p_y1, 0, 0] 
        z12 = T[p_x1, p_y1, 0, 1] 
        z13 = T[p_x1, p_y1, 1, 0] 
        z14 = T[p_x1, p_y1, 1, 1]  
       
            
        z21 = T[p_x2, p_y2, 0, 0] 
        z22 = T[p_x2, p_y2, 0, 1] 
        z23 = T[p_x2, p_y2, 1, 0] 
        z24 = T[p_x2, p_y2, 1, 1]  
        
            
        z31 = T[p_x3, p_y3, 0, 0] 
        z32 = T[p_x3, p_y3, 0, 1] 
        z33 = T[p_x3, p_y3, 1, 0] 
        z34 = T[p_x3, p_y3, 1, 1]  
        
        z41 = T[p_x4, p_y4, 0, 0] 
        z42 = T[p_x4, p_y4, 0, 1] 
        z43 = T[p_x4, p_y4, 1, 0] 
        z44 = T[p_x4, p_y4, 1, 1]  
       
        z1 = [z11,z21,z31,z41]
        z2 = [z12,z22,z32,z42]
        z3 = [z13,z23,z33,z43]
        z4 = [z14,z24,z34,z44]
        
        x_c = [x0,x1]
        y_c = [y0,y1]      
        
        
        f1 = interpolate.interp2d(x_c,y_c,z1,kind="linear")
        f2 = interpolate.interp2d(x_c,y_c,z2,kind="linear")
     
        f3 = interpolate.interp2d(x_c,y_c,z3,kind="linear")
        f4 = interpolate.interp2d(x_c,y_c,z4,kind="linear")
        
        
        z1_new = f1(x_coordinate,y_coordinate)
        z2_new = f2(x_coordinate,y_coordinate)
        z3_new = f3(x_coordinate,y_coordinate)
        z4_new = f4(x_coordinate,y_coordinate)
        
        
        z_n = np.array([z1_new,z2_new,z3_new,z4_new])
        z_new = z_n.reshape(2,2)
        
        
        [evals, evecs] = np.linalg.eigh(z_new)
        ev = evecs[:, 0]   
        
              
        dx_dt = ev[0]
        dy_dt = ev[1]
               
        x_coordinate = x_coordinate + (dx_dt*step_size)
        y_coordinate = y_coordinate + (dy_dt*step_size)
        
       
        px = np.append(px,[x_coordinate])
        py = np.append(py,[y_coordinate])
        


#print(px,py)    
#plt.subplot(1,3,3)
#plt.plot(px,py)
#plt.title("Graph")
#
#
#
#
#
#plt.show()   


