# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:30:11 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread


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
     return value * img[i][j]


def calculateAffectedPoint(img, i, j):
    neighbour_points = neighbours(i, j, img.shape[0], img.shape[1])
    value = 0
    for point in neighbour_points:
        value += calculateCorrelationNeighbour(img, point[0], point[1])
    value += calculateCorrelationNeighbour(img, i, j)
    return value

# proportion of pixels to alter
prop = 0.7
varSigma = 0.1
im = imread('images/pug_gray.jpg')
im = im/255
fig = plt.figure()
ax = fig.add_subplot(311)
ax.imshow(im,cmap='gray')
plt.title('original image')
plt.xticks([])
plt.yticks([])

im2 = add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(312)
ax2.imshow(im2,cmap='gray')
plt.title('noised image')
plt.xticks([])
plt.yticks([])
#im2 = add_saltnpeppar_noise(im,prop)
#ax3 = fig.add_subplot(223)
#ax3.imshow(im2,cmap='gray')
h = 2#latent variable weight
b = 0.5#correlation between neighbor
n = 2.1#correlation between variable

#convert y to [-1,1]
#for i in range(0, im2.shape[0]):      
#    for j in range(0, im2.shape[1]):
#        if im2[i][j] < 0.1 and im2[i][j] > -0.1:
#            im2[i][j] = -1
#        else:
#            im2[i][j] = 1
mi = np.mean(im2)
im2[im2>mi] = 1
im2[im2<=mi] = -1
#im2 = (im-0.5)*2
z = np.copy(im2)
for t in range(0, 10):
    checkCoverge = True
    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            z_old = z[i][j]
            z[i][j] = 1
            #pe = h * z[i][j] - n * z[i][j] * im2[i][j] - b * calculateAffectedPoint(z, i, j)
            pe = h * z[i][j] - n * z[i][j] * im2[i][j] - 2 * b * calculateCorrelationNeighbour(z, i, j)
            z[i][j] = -1
            #ne = h * z[i][j] - n * z[i][j] * im2[i][j] - n * calculateAffectedPoint(z, i, j)
            ne = h * z[i][j] - n * z[i][j] * im2[i][j] - 2 * b * calculateCorrelationNeighbour(z, i, j)
            if(pe > ne):
                z[i][j] = -1
            else:
                z[i][j] = 1
            if(z_old != z[i][j]):
                checkCoverge = False
    if(checkCoverge == True):
        print(t)
        #break
#for i in range(0, z.shape[0]):      
#    for j in range(0, z.shape[1]):
#        if z[i][j] == -1:
#            z[i][j] = 0
#        else:
#            z[i][j] = 1
ax4 = fig.add_subplot(313)
ax4.imshow(z,cmap='gray')
plt.title('denoised image')
plt.xticks([])
plt.yticks([])