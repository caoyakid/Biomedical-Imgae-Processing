#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:43:57 2020

@author: yuetinghe
"""

import numpy as np
import matplotlib.pyplot as plt


with open("mri-a.raw","rb") as file:

    img = np.fromfile(file, dtype=np.ubyte)
    
img = img.reshape([256,256])
plt.imshow(img , cmap = "gray")
#1(a)
total_sum = np.sum(img) 
avg_value = total_sum/(256**2)
print("S = ", total_sum)
print("A = ", avg_value)
#1(b)
jmg_1 = np.zeros([256,256])
for i in range (256):
    jmg_1[i,:] = 255 - img[i,:]
    
#1(c)
jmg_2 = img.T

#1(d)
jmg_3 = np.zeros(shape=(256,256))
for m in range(256):
    jmg_3[m,:] = img[255 - m,:]

#1(e)
jmg_4 = np.zeros(shape=(256,256))
for n in range(256):
    jmg_4[:,n] = img[:,255 - n]

# =============================================================================
#another solution for (d)(e)
#plt.imshow(np.fliplr(img) , cmap = "gray") #flip horizontally
#plt.imshow(np.flipud(img), cmap="gray" ) #flip vertically
# =============================================================================
plt.subplot(4, 2, 1)
plt.imshow(jmg_1 , cmap = "gray")
plt.subplot(4, 2, 2)
plt.imshow(jmg_2, cmap = "gray")
plt.subplot(4, 2, 3)
plt.imshow(jmg_3, cmap = "gray")
plt.subplot(4, 2, 4)
plt.imshow(jmg_4, cmap = "gray")
