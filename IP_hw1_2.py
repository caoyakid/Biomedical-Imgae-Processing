#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 19:41:05 2020

@author: yuetinghe
"""

import numpy as np
import matplotlib.pyplot as plt



with open("mri-a.raw","rb") as file:

    img = np.fromfile(file, dtype=np.ubyte)
    
img = img.reshape([256,256])

newimg_7bit = img // 2
newimg_4bit = img // 16
newimg_1bit = img // 128

plt.subplot(3, 1, 1)
plt.imshow(newimg_7bit , cmap = "gray")
plt.subplot(3, 1, 2)
plt.imshow(newimg_4bit , cmap = "gray")
plt.subplot(3, 1, 3)
plt.imshow(newimg_1bit , cmap = "gray")
