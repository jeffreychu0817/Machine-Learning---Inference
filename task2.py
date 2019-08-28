# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:47:39 2018

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:30:11 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread


def dist(x, u):
    return np.exp(x*u)/(np.exp(u) + np.exp(-u))


def add_gaussian_noise(im,prop,varSigma):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im)
    im2[index] += e[index]
    return im2


def add_saltnpeppar_noise(im,prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    return im2


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


def imshowImage(z):
    z_temp = np.copy(z)
    for i in range(0, z.shape[0]):      
        for j in range(0, z.shape[1]):
            if z[i][j] == -1:
                z_temp[i][j] = 0
            else:
                z_temp[i][j] = 1
    return z_temp

# proportion of pixels to alter
prop = 0.7
varSigma = 0.1
im = imread('images/pug_gray.jpg')
im = im/255
fig = plt.figure()
ax = fig.add_subplot(221)
ax.imshow(im,cmap='gray')
plt.title('original image')
plt.xticks([])
plt.yticks([])
im2 = add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(222)
ax2.imshow(im2,cmap='gray')
plt.title('noised image')
plt.xticks([])
plt.yticks([])
#im2 = add_saltnpeppar_noise(im,prop)
#ax3 = fig.add_subplot(223)
mi = np.mean(im2)
im2[im2>mi] = 1
im2[im2<=mi] = -1

#ax3 = fig.add_subplot(132)
#ax3.imshow(im2,cmap='gray')
#plt.title('noised image')
#plt.xticks([])
#plt.yticks([])
z = np.copy(im2)
z_store = np.zeros((100, z.shape[0], z.shape[1]))
weight = 5
for t in range(0, 30000):
    z_old = z
    #z_store[t] = z
    i = np.random.randint(0, z.shape[0])
    j = np.random.randint(0, z.shape[1])
    p_numerator = (dist(im2[i][j], 1*weight)*dist(1, calculateCorrelationNeighbour(z_old,i,j)))
    p_denominator = p_numerator + (dist(im2[i][j], -1*weight)*dist(-1, calculateCorrelationNeighbour(z_old,i,j)))
    p = p_numerator / p_denominator
    t = np.random.rand()
    if(p > t):
         z[i][j] = 1
    else:
         z[i][j] = -1
#for t in range(0, 30):
#    z_old = z
#    z_store[t] = z
#    for i in range(0, z.shape[0]):
#        for j in range(0, z.shape[1]):
#            p_numerator = (dist(im2[i][j], 1*weight)*dist(1, calculateCorrelationNeighbour(z_old,i,j)))
#            p_denominator = p_numerator + (dist(im2[i][j], -1*weight)*dist(-1, calculateCorrelationNeighbour(z_old,i,j)))
#            p = p_numerator / p_denominator
#            np.random.seed(42)
#            t = np.random.rand()
#            #print(t)
#            if(p > t):
#                z[i][j] = 1
#            else:
#                z[i][j] = -1
ax4 = fig.add_subplot(223)
ax4.imshow(z,cmap='gray')
plt.title('denoised image')
plt.xticks([])
plt.yticks([])

#for i in range(0,30,1):
#    plt.imshow(z_store[i],cmap='gray')
#    plt.show()