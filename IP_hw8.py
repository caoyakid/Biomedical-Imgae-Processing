# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:00:47 2020

@author: yuehtingho
"""


import numpy as np 
import cv2
import matplotlib
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# Open the image file
with open("mri-a.256","rb") as file:
    img1 = np.fromfile(file, dtype=np.ubyte)
img1 = img1.reshape([256,256])

with open("mri-gauss.256","rb") as file:
    img2 = np.fromfile(file, dtype=np.ubyte)
img2 = img2.reshape([256,256])

with open("mri-ps.256","rb") as file:
    img3 = np.fromfile(file, dtype=np.ubyte)
img3 = img3.reshape([256,256])

edges1 = cv2.Canny(img1,100,200)
edges2 = cv2.Canny(img2,100,200)
edges3 = cv2.Canny(img3,100,200)

plt.figure(1)
plt.subplot(321),plt.imshow(img1,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(322),plt.imshow(edges1,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(323),plt.imshow(img2,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(324),plt.imshow(edges2,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(325),plt.imshow(img3,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(326),plt.imshow(edges3,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# Otsu's thresholding
ret1,th1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure(2)
plt.subplot(221), plt.imshow(img1, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.hist(img1.ravel(), 256)
plt.title('Original Image Histogram'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(th1, cmap = 'gray')
plt.title('Otsu thresholding'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.hist(th1.ravel(), 256)
plt.title('Otsu thresholding Histogram'), plt.xticks([]), plt.yticks([])


# Setting the font size for all plots.
matplotlib.rcParams['font.size'] = 9

# The input image.
# =============================================================================
# image = data.camera()
# =============================================================================

# Applying multi-Otsu threshold for the default value, generating
# three classes.
thresholds = threshold_multiotsu(img1)

# Using the threshold values, we generate the three regions.
regions = np.digitize(img1, bins=thresholds)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

# Plotting the original image.
ax[0].imshow(img1, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')

# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
ax[1].hist(img1.ravel(), bins=255)
ax[1].set_title('Histogram')
for thresh in thresholds:
    ax[1].axvline(thresh, color='r')

# Plotting the Multi Otsu result.
ax[2].imshow(regions, cmap='gray')
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')
plt.figure(3)
plt.subplots_adjust()

plt.show()