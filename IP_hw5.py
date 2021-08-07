# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:33:37 2020

@author: yuehtingho
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2

with open("mri-a.256","rb") as file:

    img1 = np.fromfile(file, dtype=np.ubyte)
    
with open("mri-gauss.256","rb") as file:
    
    img2 = np.fromfile(file, dtype=np.ubyte)

with open("mri-ps.256","rb") as file:
    
    img3 = np.fromfile(file, dtype=np.ubyte)
    
img = img1.reshape([256,256])
img_gauss = img2.reshape([256,256])
img_ps = img3.reshape([256,256])

sigma = np.std(img_gauss)

def MSE(f,g):
    nr, nc = f.shape[:2]
    MSE=0.0
    for x in range(nr):
        for y in range(nc):
            MSE+=(float(f[x,y])-float(g[x,y]))**2
    MSE/=(nr*nc)
    return MSE
    

blur3 = cv2.GaussianBlur(img_gauss,(3,3),1)
sigma3 = np.std(blur3)
blur5 = cv2.GaussianBlur(img_gauss,(5,5),1)
sigma5 = np.std(blur5)
blur9 = cv2.GaussianBlur(img_gauss,(9,9),1)
sigma9 = np.std(blur9)
blur15 = cv2.GaussianBlur(img_gauss,(15,15),1)
sigma15 = np.std(blur15)


u = [MSE(img, blur3),MSE(img, blur5),MSE(img, blur9),MSE(img, blur15)]
for i in range(len(u)):
    print("MSE=",u[i])

plt.figure(1)
plt.subplot(221),plt.imshow(blur3, cmap='gray'),plt.title('Gaussian Filtering_mask size 3')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur5, cmap='gray'),plt.title('Gaussian Filtering_mask size 5')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(blur9, cmap='gray'),plt.title('Gaussian Filtering_mask size 9')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(blur15, cmap='gray'),plt.title('Gaussian Filtering_mask size 15')
plt.xticks([]), plt.yticks([])
plt.show()

print('-------------------------------------------')

bblur3 = cv2.GaussianBlur(img_ps,(3,3),1)
ps_sigma3 = np.std(bblur3)
bblur5 = cv2.GaussianBlur(img_ps,(5,5),1)
ps_sigma5 = np.std(bblur5)
bblur9 = cv2.GaussianBlur(img_ps,(9,9),1)
ps_sigma9 = np.std(bblur9)
bblur15 = cv2.GaussianBlur(img_ps,(15,15),1)
ps_sigma15 = np.std(bblur15)

m = [MSE(img, bblur3), MSE(img, bblur5), MSE(img, bblur9), MSE(img, bblur15)]
for i in range(len(m)):
    print("MSE=" , m[i])

plt.figure(2)
plt.subplot(221),plt.imshow(bblur3, cmap='gray'),plt.title('Gaussian Filtering_mask size 3')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(bblur5, cmap='gray'),plt.title('Gaussian Filtering_mask size 5')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(bblur9, cmap='gray'),plt.title('Gaussian Filtering_mask size 9')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(bblur15, cmap='gray'),plt.title('Gaussian Filtering_mask size 15')
plt.xticks([]), plt.yticks([])
plt.show()

print('-------------------------------------------')
median3 = cv2.medianBlur(img_ps,3)
median5 = cv2.medianBlur(img_ps,5)
median7 = cv2.medianBlur(img_ps,7)
median9 = cv2.medianBlur(img_ps,9)

v = [MSE(img, median3),MSE(img, median5),MSE(img, median7),MSE(img, median9)]
for i in range(len(v)):
    print("MSE=",v[i])

plt.figure(3)
plt.subplot(221),plt.imshow(median3, cmap='gray'),plt.title('Median Filtering_mask size 3')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(median5, cmap='gray'),plt.title('Median Filtering_mask size 5')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(median7, cmap='gray'),plt.title('Median Filtering_mask size 9')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(median9, cmap='gray'),plt.title('Median Filtering_mask size 15')
plt.xticks([]), plt.yticks([])