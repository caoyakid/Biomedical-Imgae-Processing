#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:36:04 2020

@author: yuetinghe
"""

import numpy as np
import matplotlib.pyplot as plt


with open("mri-a.raw","rb") as file:

    img = np.fromfile(file, dtype=np.ubyte)
    
img = img.reshape([256,256])

#2(a)
A = np.zeros([256,256])
for i in range(1,256):
    A[:,i] = abs(img[:,i] - img[:,i-1])
plt.figure()
plt.imshow(A, cmap="gray")
plt.show()
#2(b)
B = np.zeros([256,256])
for j in range(255):
    B[:,j] = abs(img[:,j+1] - img[:,j])
plt.figure()
plt.imshow(B, cmap="gray")
plt.show()
#2(c)
C = np.zeros([256,256])
for m in range(1,256):
    C[m,:] = abs(img[m,:] - img[m-1,:])
plt.figure()
plt.imshow(C, cmap="gray")
plt.show() 
#2(d)
D = np.zeros([256,256])
for n in range(255):
    D[n,:] = abs(img[n+1,:] - img[n,:])
plt.figure()
plt.imshow(D, cmap="gray")
plt.show()

#--------------------------------------------------

