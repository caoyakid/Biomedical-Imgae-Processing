# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:46:32 2020

@author: yuehtingho
"""


import numpy as np 
import matplotlib.pyplot as plt
import cv2

# Open the file
with open("mri-a.256","rb") as file :
    img1 = np.fromfile(file, dtype = np.ubyte)
img1 = img1.reshape([256,256])

with open("comb.256","rb") as file :
    img2 = np.fromfile(file, dtype = np.ubyte)
img2 = img2.reshape([256,256])

# reshape 前面設-1 將所有數字降為1D 後面數字是重新分配的行數
# 將一個像素點的RGB值作為一個單元處理
Z = img1.reshape((-1,2))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
# 使用center內的值取代原像素點的值
res = center[label.flatten()]
# 使用reshape調整取代後的影像
res2 = res.reshape((img1.shape))


cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()


C = img2.reshape((-1,1))
C = np.float32(C)
criteria_b = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 6.0)
k = 7
ret_b,label_b,center_b=cv2.kmeans(C,k,None,criteria_b,10,cv2.KMEANS_RANDOM_CENTERS)
center_b = np.uint8(center_b)
res_b = center_b[label_b.flatten()]
res2_b = res_b.reshape((img2.shape))

plt.figure(1)
plt.subplot(121),plt.imshow(img1 ,cmap = 'gray')
plt.title('mri-a.256 original image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(122),plt.imshow(res2 ,cmap = 'gray')
plt.title('mri-a.256 use k-mean algorithm'), plt.xticks([]), plt.yticks([])
plt.figure(2)
plt.hist(res2.ravel(), bins=128)
plt.title("Histogram of mri after k-mean")
plt.figure(3)
plt.subplot(121),plt.imshow(img2 ,cmap = 'gray')
plt.title('comb.256 original image'), plt.xticks([]), plt.yticks([]) 
plt.subplot(122),plt.imshow(res2_b ,cmap = 'gray')
plt.title('comb.256 use k-mean algorithm'), plt.xticks([]), plt.yticks([])
plt.figure(4)
plt.hist(res2_b.ravel(), bins=128)
plt.title("Histogram of mri after k-mean")
plt.show()