#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:24:46 2020

@author: yuetinghe
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


with open("mri-a.raw","rb") as file:

    img = np.fromfile(file, dtype=np.ubyte)
    
img = img.reshape([256,256])
cv2.imshow("Origin", img)
cv2.waitKey(0)
if cv2.waitKey(0) ==27:# 按'Esc'退出 在mac加fn
    
    cv2.destroyAllWindows()

(h,w) = img.shape[:2]
center = (w / 2,h / 2)
#1
#1(a)
M = cv2.getRotationMatrix2D(center,0,0.5)
shrink = cv2.warpAffine(img,M,(w,h))
cv2.imshow("Shrink by half",shrink)
cv2.waitKey(0)
if cv2.waitKey(0) ==27:# 按'Esc'退出 在mac加fn
    
    cv2.destroyAllWindows()
#1(b)
def rotatedegree(img, r):
    N = cv2.getRotationMatrix2D(center,r,1)
    rotate = cv2.warpAffine(img,N,(w,h))
    
    return rotate

u=[rotatedegree(img,35),rotatedegree(img,120),
   rotatedegree(img,210),rotatedegree(img,325)]
for i in range(len(u)):
    cv2.imshow("Rotate",u[i])
    cv2.waitKey(0)
    if cv2.waitKey(0) ==27:# 按'Esc'退出 在mac加fn
    
        cv2.destroyAllWindows()
    
#1(c)
def translate(img, x, y):
    # 定義平移矩陣
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return shifted

v=[translate(img, 10, 30), translate(img, -20, 31),
   translate(img, -15, -15), translate(img, 31, -28),]
for i in range(len(v)):
    cv2.imshow('Translate',v[i])
    cv2.waitKey(0)
    if cv2.waitKey(0) ==27:# 按'Esc'退出 在mac加fn
    
        cv2.destroyAllWindows()