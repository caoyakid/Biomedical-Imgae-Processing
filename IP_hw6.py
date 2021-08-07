# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:40:56 2020

@author: yuehtingho
"""


import numpy as np 
import cv2
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

##1(a)
#製造出一個長方形
a = np.zeros((256,256), dtype = np.uint8)
phantom = cv2.rectangle(a, (103,78), (153,178),(255,255,255),-1)
# =============================================================================
# 畫三角形
# # pt1 = (128,103)
# # pt2 = (78,153)
# # pt3 = (178,153)
# 
# # cv2.circle(phantom, pt1, 1, (255,255,255), -1)
# # cv2.circle(phantom, pt2, 1, (255,255,255), -1)
# # cv2.circle(phantom, pt3, 1, (255,255,255), -1)
# # triangle_cnt = np.array( [pt1, pt2, pt3] )
# # cv2.drawContours(phantom, [triangle_cnt], 0, (255,255,255), -1)
# =============================================================================

##1(b)
#取FFT
FFT_p = fft2(phantom)
magnitude_spectrum = 20*np.log(np.abs(FFT_p))
#定義一個phase的函式
def phase_spectrum(f):
    F = fft2(f)
    phase = np.angle(F, deg = True) #True 角度制 / False 弧度制
    nr, nc = phase.shape[:2]
    for x in range(nr):
        for y in range(nc):
            if phase[x,y]<0:
                phase[x,y] = phase[x,y] + 360
            phase[x,y] = int(phase[x,y]*255/360)
    g = np.uint8(np.clip(phase,0,255))  #clip 截取(超出的部分強制設為邊界)
    return g

pre_phase = phase_spectrum(FFT_p)

plt.figure(1)
plt.imshow(phantom, cmap = 'gray')
plt.title('Constructed phantom'), plt.xticks([]), plt.yticks([])
plt.figure(2)
plt.subplot(121), plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude of F'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(pre_phase, cmap='gray')
plt.title('Phase response of F'), plt.xticks([]), plt.yticks([])
plt.show()

##1(c)
enhance_magnitude = np.log(np.abs(1+FFT_p))
plt.figure(3)
plt.imshow(enhance_magnitude, cmap='gray')
plt.title('Enhance magnitude of f'), plt.xticks([]), plt.yticks([])
##1(d)
def spectrum(f):
    F = fft2(f)
    Fshift = fftshift(F)
    mag = np.log(np.abs(Fshift+1))
    mag = mag / mag.max()*255.0
    g = np.uint8(mag)
    return g 

magnitude_spectrum2 = spectrum(phantom)

#前處理(移到影像中心)
def frequency_filtering(f):
    nr, nc = f.shape[:2]
    
    fp = np.zeros([nr,nc])
    for x in range(nr):
        for y in range(nc):
            fp[x,y] = pow(-1,x+y)*f[x,y]
            
    return fp

img = frequency_filtering(phantom)
new_phase = phase_spectrum(img)
plt.figure(4)
plt.subplot(121), plt.imshow(magnitude_spectrum2, cmap = 'gray')
plt.title('Magnitude of G'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(new_phase, cmap='gray')
plt.title('Phase response of G'), plt.xticks([]), plt.yticks([])
plt.show()

##2
with open("mri-a.256","rb") as file:
    img1 = np.fromfile(file, dtype=np.ubyte)
img1 = img1.reshape([256,256])

FFT_mri = fft2(img1)
enhance_magnitude_mri = np.log(np.abs(1+FFT_mri))
magnitude_spectrum_mri = spectrum(img1)
img_mri = frequency_filtering(img1)
phase_mri = phase_spectrum(img_mri)
plt.figure(5)
plt.subplot(131), plt.imshow(enhance_magnitude_mri, cmap = 'gray')
plt.title('Enhance magnitude of mri-a.256'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum_mri, cmap = 'gray')
plt.title('Magnitude of mri-a.256'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(phase_mri, cmap='gray')
plt.title('Phase response of mri-a.256'), plt.xticks([]), plt.yticks([])
plt.show()


##3
combined = np.multiply(np.abs(FFT_mri), np.exp(1j*np.angle(FFT_mri)))
imgCombined = np.real(np.fft.ifft2(combined))

combined_a = np.multiply(np.abs(FFT_mri), np.exp(1j*np.angle(a)))
imgCombined_a = np.real(np.fft.ifft2(combined_a))

combined_b = np.multiply(np.abs(a), np.exp(1j*np.angle(FFT_mri)))
imgCombined_b = np.real(np.fft.ifft2(combined_b))

plt.figure(6)
plt.subplot(131), plt.imshow(imgCombined, 'gray')
plt.title('IFFT of origin mri256-a'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(imgCombined_a, 'gray')
plt.title('IFFT of  mri256-a reset the phase response'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(imgCombined_b, 'gray')
plt.title('IFFT of mri256-a reset the magnitude response'), plt.xticks([]), plt.yticks([])