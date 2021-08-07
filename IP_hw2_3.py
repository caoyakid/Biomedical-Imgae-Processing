#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:40:57 2020

@author: yuetinghe
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


with open("mri-a.raw","rb") as file:

    img = np.fromfile(file, dtype=np.ubyte)
    
img = img.reshape([256,256])

with open("baboon.raw","rb") as file:

    jmg = np.fromfile(file, dtype=np.ubyte)
    
jmg = jmg.reshape([256,256])


w1 = 1*img + 0*jmg
plt.figure()
plt.title('W1')
plt.imshow(w1, cmap="gray")
plt.show()

w2 = 0.75*img + 0.25*jmg
plt.figure()
plt.title('W2')
plt.imshow(w2, cmap="gray")
plt.show()

w3 = 0.5*img + 0.5*jmg
plt.figure()
plt.title('W3')
plt.imshow(w3, cmap="gray")
plt.show()

w4 = 0.25*img + 0.75*jmg
plt.figure()
plt.title('W4')
plt.imshow(w4, cmap="gray")
plt.show()

w5 = 0*img + 1*jmg
plt.figure()
plt.title('W5')
plt.imshow(w5, cmap="gray")
plt.show()

step_list = [0.01 * x for x in range(0, 101)]

cv2.imshow("Bonus", jmg)
for i in step_list:
    res = cv2.addWeighted(img, (1-i), jmg, i, 0)
    cv2.imshow("Bonus", res)
    cv2.waitKey(60)
if cv2.waitKey(0) ==27:# 按'Esc'退出 在mac加fn
    
    cv2.destroyAllWindows()

