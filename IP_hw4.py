# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:38:43 2020

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

def translate(img, x, y):
    # 定義平移矩陣
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return shifted
## 1
r_a=0
for i in range(0,255):
    for j in range(0,255):
        r_a = r_a +  int(img[i][j])*int(img_gauss[i][j])
        
print('The correlation between mri-a and mri-gauss is', r_a)

r_b=0
for i in range(0,255):
    for j in range(0,255):
        r_b = r_b +  int(img[i][j])*int(img_ps[i][j])
        
print('The correlation between mri-a and mri-ps is', r_b)

## 2 

for x in range(0,12,2):
    r_2=0
    for i in range(0,255):
        for j in range(0,255):
            r_2 = r_2 +  int(img[i][j])*int(translate(img, 0, -x)[i][j])
        
    print('The correlation between mri-a and translated mri-a is', r_2)
     
# =============================================================================
# r_10=0
# for i in range(0,255):
#     for j in range(0,255):
#         r_10 = r_10 + int(img[i][j])*int(translate(img, 0, -10)[i][j])
#         
# print('The correlation between mri-a and translated mri-a is', r_10)
# =============================================================================


pixel_sum=0
for i in range(0,255):
    for j in range(0,255):
        pixel_sum = pixel_sum + int(img[i][j])
    
kernel = img/pixel_sum
dst = cv2.filter2D(img,-1,kernel,borderType=cv2.BORDER_CONSTANT)
cv2.imshow("Spatial filtering", dst)

# =============================================================================
# kernel2 = np.ones((3,3),np.float32)/9
# dst2 = cv2.filter2D(img,-1,kernel2,borderType=cv2.BORDER_CONSTANT)
# cv2.imshow("Spatial filtering2", dst2)
# =============================================================================
# cv2.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])
# src: input image / ddepth:desired depth of the output image. If it is negative, it will be the same as that of the input image.
# borderType: pixel extrapolation method. (像素外推法)

# =============================================================================
# 濾波器也稱為遮罩(MASK) 核心(KERNEL) 模板(TEMPLATE) 視窗(WINDOW) 每一格都有對應的濾波器係數
# =============================================================================


# =============================================================================
# height, width = img_ps.shape
# 
# # Create ORB detector with 5000 features. 
# orb_detector = cv2.ORB_create(5000) 
#   
# # Find keypoints and descriptors. 
# # The first arg is the image, second arg is the mask 
# #  (which is not reqiured in this case). 
# kp1, d1 = orb_detector.detectAndCompute(img, None) 
# kp2, d2 = orb_detector.detectAndCompute(img_ps, None) 
#   
# # Match features between the two images. 
# # We create a Brute Force matcher with  
# # Hamming distance as measurement mode. 
# matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
#   
# # Match the two sets of descriptors. 
# matches = matcher.match(d1, d2) 
#   
# # Sort matches on the basis of their Hamming distance. 
# matches.sort(key = lambda x: x.distance) 
#   
# # Take the top 90 % matches forward. 
# matches = matches[:int(len(matches)*90)] 
# no_of_matches = len(matches) 
#   
# # Define empty matrices of shape no_of_matches * 2. 
# p1 = np.zeros((no_of_matches, 2)) 
# p2 = np.zeros((no_of_matches, 2)) 
#   
# for i in range(len(matches)): 
#   p1[i, :] = kp1[matches[i].queryIdx].pt 
#   p2[i, :] = kp2[matches[i].trainIdx].pt 
#   
# # Find the homography matrix. 
# homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
#   
# # Use this matrix to transform the 
# # colored image wrt the reference image. 
# transformed_img = cv2.warpPerspective(img, 
#                     homography, (width, height)) 
#   
# # Save the output. 
# cv2.imwrite('output123.jpg', transformed_img) 
# =============================================================================
