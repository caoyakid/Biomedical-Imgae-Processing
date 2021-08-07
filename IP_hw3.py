# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:17:40 2020

@author: yuehtingho
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.special as special

#讀檔
with open("mri-bright.256","rb") as file:

    img1 = np.fromfile(file, dtype=np.ubyte)
    
with open("mri-dark.256","rb") as file:
    
    img2 = np.fromfile(file, dtype=np.ubyte)
    
img_bri = img1.reshape([256,256])
img_dark = img2.reshape([256,256])
## 1
#Beta矯正 (Contrast enhancement)

def beta_correction(f, a, b):
    g=f.copy()
    nr, nc = f.shape[:2]
    x = np.linspace(0, 1, 256)
    table = np.round(special.betainc(a, b, x)*255, 0)
    if f.ndim !=3:
        for x in range(nr):
            for y in range(nc):
                g[x,y] = table[f[x,y]]
    else:
        for x in range(nr):
            for y in range(nc):
                for k in range(3):
                    g[x,y,k]= table[f[x,y,k]]
                    
    return g

#gamma 矯正
def gamma_correction(f, gamma):
    g=f.copy()
    nr, nc = f.shape[:2]
    c= 255.0/(255.0**gamma)
    table= np.zeros(256)
    for i in range(256):
        table[i]= round(i**gamma*c, 0)
    if f.ndim !=3:
        for x in range(nr):
            for y in range (nc):
                g[x,y] = table[f[x,y]]
                
    else:
        for x in range(nr):
            for y in range (nc):
                for k in range(3):
                    g[x,y,k] = table[f[x,y,k]]
                
    return g


def main():
    img_bri_c=beta_correction(img_bri, a=2.0, b=2.0)
    img_dark_c=beta_correction(img_dark, a=2.0, b=2.0)
    img_bri_g = gamma_correction(img_bri, 2.0)
    img_dark_g = gamma_correction(img_dark, 0.2)
    cv2.imshow("Oringinal Image1", img_bri)
    cv2.imshow("Oringinal Image2", img_dark)
    cv2.imshow("Enhance contrast-mri.bright-beta", img_bri_c)
    cv2.imshow("Enhance contrast-mri.dark-beta", img_dark_c) 
    cv2.imshow("Enhance contrast-mri.bright-gamma", img_bri_g)  
    cv2.imshow("Enhance contrast-mri.dark-gamma", img_dark_g)
    cv2.imwrite('mri-bright.png', img_bri)
    cv2.imwrite('mri-dark.png', img_dark)
    cv2.imwrite('Enhance contrast-mri.bright-beta.png', img_bri_c)
    cv2.imwrite('Enhance contrast-mri.dark-beta.png', img_dark_c)
    cv2.imwrite('Enhance contrast-mri.bright-gamma.png', img_bri_g)
    cv2.imwrite('Enhance contrast-mri.dark-gamma.png', img_dark_g)
    cv2.waitKey(0)
    
main()


                

## 2
equ_bri = cv2.equalizeHist(img_bri)
res_bri = np.hstack((img_bri,equ_bri)) #stacking images side-by-side
cv2.imwrite('res_bright.png',res_bri)
cv2.imshow("Histogram Equalization_bright", res_bri)

equ_dark = cv2.equalizeHist(img_dark)
res_dark = np.hstack((img_dark,equ_dark)) #stacking images side-by-side
cv2.imwrite('res_dark.png',res_dark)
cv2.imshow("Histogram Equalization_dark", res_dark)
#before histogram
hist,bins = np.histogram(img_bri.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.figure(1)
plt.plot(cdf_normalized, color = 'b')
plt.hist(img_bri.flatten(),256,[0,256], color = 'r')
plt.title("mri-bright cdf before the equalization")
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

hist,bins = np.histogram(img_dark.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.figure(2)
plt.plot(cdf_normalized, color = 'b')
plt.hist(img_dark.flatten(),256,[0,256], color = 'r')
plt.title("mri-dark cdf before the equalization")
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

#after histogram
hist,bins = np.histogram(equ_bri.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.figure(3)
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ_bri.flatten(),256,[0,256], color = 'r')
plt.title("mri-bright cdf after the equalization")
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

hist,bins = np.histogram(equ_dark.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.figure(4)
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ_dark.flatten(),256,[0,256], color = 'r')
plt.title("mri-dark cdf after the equalization")
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()