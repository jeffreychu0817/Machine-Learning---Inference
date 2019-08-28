# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 22:27:20 2018

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:11:55 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def dist(x, u):
    return np.exp(x*u)/(np.exp(u) + np.exp(-u))


def neighbours(i,j,M,N,size=8):
    if size==4:
        if (i==0 and j==0):
            n=[(0,1), (1,0)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-1)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else:
            n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        return n
    if size==8:
        if (i==0 and j==0):
            n=[(0,1), (1,1), (1,0)]
        elif i==0 and j==N-1:
            n=[(0,N-2), (1,N-2), (1,N-1)]
        elif i==M-1 and j==0:
            n=[(M-1,1), (M-2,1), (M-2,0)]
        elif i==M-1 and j==N-1:
            n=[(M-1,N-2), (M-2,N-2), (M-2,N-1)]
        elif i==0:
            n=[(0,j-1), (0,j+1), (1,j-1), (1,j+1), (1,j)]
        elif i==M-1:
            n=[(M-1,j-1), (M-1,j+1), (M-2, j-1), (M-2, j+1), (M-2,j)]
        elif j==0:
            n=[(i-1,0), (i+1,0), (i+1,1), (i-1,1), (i,1)]
        elif j==N-1:
            n=[(i-1,N-1), (i+1,N-1), (i-1, N-2), (i+1, N-2), (i,N-2)]
        else:
            n=[(i-1,j), (i-1,j+1), (i-1, j-1), (i+1, j-1), (i+1,j), (i+1,j+1), (i,j-1), (i,j+1)]
        return n
    return -1


def calculateLikelihood(img, z, i, j):
    imgb = np.copy(img)
    imgf = np.copy(img)
    color = ('b','g','r')
    lb = 1
    lf = 1
    for channel,col in enumerate(color):
        imgb[z==-1,:] = 0
        histb = cv2.calcHist(imgb,[channel],None,[10],[0,256])
        imgf[z==1,:] = 0
        histf = cv2.calcHist(imgf,[channel],None,[10],[0,256])
        lb *= histb[(int)(img[i][j][channel]/25.6)] / np.sum(histb)
        lf *= histf[(int)(img[i][j][channel]/25.6)] / np.sum(histf)
    if lb > lf:
        return 1
    else:
        return -1

def calculateCorrelationNeighbour(img,i,j):
     neighbour_points = neighbours(i, j, img.shape[0], img.shape[1])
     value = 0
     for point in neighbour_points:
         value += img[point[0]][point[1]]
#     if(value > 0):
#         value = 1
#     else:
#         value = -1
     return value #/ len(neighbour_points)


# proportion of pixels to alter
#img = cv2.imread('pug.jpg', -1)
img = cv2.imread('images/8.png', -1)
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.imshow(img)
plt.title('original image')
plt.xticks([])
plt.yticks([])

z = np.sum(img, axis=2)
z = z.astype('float64')
mz = np.mean(z)

#z = np.random.rand(img.shape[0], img.shape[1])
#mz = 0.5
#
#z[z<=mz] = -1
#z[z>mz] = 1

z_store = np.size((10, z.shape[0], z.shape[1]))
w = 1000
for t in range(30):
    print(t)
    checkCoverge = True
    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            z_old = z[i][j]
            z[i][j] = 1
            p1 =  z[i][j]*calculateCorrelationNeighbour(z, i, j) + w*z[i][j]*calculateLikelihood(img,z,i,j)
            z[i][j] = -1
            p2 = z[i][j]*calculateCorrelationNeighbour(z, i, j) + w*z[i][j]*calculateLikelihood(img,z,i,j)
            if(p1 > p2):
                z[i][j] = 1
            else:
                z[i][j] = -1
            if(z_old != z[i][j]):
                checkCoverge = False
    if(checkCoverge == True):
        print(t)
        break
ax3 = fig.add_subplot(212)
ax3.imshow(z,cmap='gray')      
plt.title('segmented image')
plt.xticks([])
plt.yticks([])
            
            
            
            
            
            
            
            
            
            
            