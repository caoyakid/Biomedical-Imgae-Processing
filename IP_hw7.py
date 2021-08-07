# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:54:49 2020

@author: yuehtingho
"""


import numpy as np 
import cv2
import matplotlib.pyplot as plt

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

# calculate the standard deviation to decide the kernel size
print(np.std(img1))
print(np.std(img2))
print(np.std(img3))

# Implement the Gaussian Filter
dst1 = cv2.GaussianBlur(img1, (19,19), 3, 3)
dst2 = cv2.GaussianBlur(img2, (19,19), 3, 3)
dst3 = cv2.GaussianBlur(img3, (19,19), 3, 3)
dst1 = dst1.astype(np.int16)
dst2 = dst2.astype(np.int16)
dst3 = dst3.astype(np.int16)
# Laplacian kernel 3 x 3
Lap3x3 = np.array([[1, 1, 1], [1,-8,1],[1,1,1]],dtype=float)
# apply Laplacian to an image
lapimg1 = cv2.filter2D(dst1,-1,Lap3x3)
lapimg2 = cv2.filter2D(dst2,-1,Lap3x3)
lapimg3 = cv2.filter2D(dst3,-1,Lap3x3)

# =============================================================================
# 判別式
# minLoG = cv2.morphologyEx(lapimg1, cv2.MORPH_ERODE, np.ones((3,3)))
# maxLoG = cv2.morphologyEx(lapimg1, cv2.MORPH_DILATE, np.ones((3,3)))
# zeroCross = np.logical_or(np.logical_and(minLoG < 0,  lapimg1 > 0), np.logical_and(maxLoG > 0, lapimg1 < 0))
# =============================================================================

def Zero_crossing(pic):
    z_c_image = np.zeros(pic.shape)
    
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    
    for i in range(1, pic.shape[0] - 1):
        for j in range(1, pic.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [pic[i+1, j-1],pic[i+1, j],pic[i+1, j+1],pic[i, j-1],pic[i, j+1],pic[i-1, j-1],pic[i-1, j],pic[i-1, j+1]]
            d = np.max(neighbour)
            e = np.min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1
 
 
            # If both negative and positive values exist in 
            # the pixel neighborhood, then that pixel is a 
            # potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood
            # difference with the pixel
 
            if z_c:
                if pic[i,j]>0:
                    z_c_image[i, j] = pic[i,j] + np.abs(e)
                elif pic[i,j]<0:
                    z_c_image[i, j] = np.abs(pic[i,j]) + d
                
    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image/z_c_image.max()*255
    z_c_image = np.uint8(z_c_norm)
 
    return z_c_image


result1 = Zero_crossing(lapimg1)
cv2.imshow("Marr-Hildreth edge-detection", result1)
cv2.waitKey(0)
if cv2.waitKey(0) ==27:# 按'Esc'退出 在mac加fn
    
    cv2.destroyAllWindows()
    
result2 = Zero_crossing(lapimg2)
cv2.imshow("Marr-Hildreth edge-detection", result2)
cv2.waitKey(0)
if cv2.waitKey(0) ==27:# 按'Esc'退出 在mac加fn
    
    cv2.destroyAllWindows()
    
result3 = Zero_crossing(lapimg3)
cv2.imshow("Marr-Hildreth edge-detection", result3)
cv2.waitKey(0)
if cv2.waitKey(0) ==27:# 按'Esc'退出 在mac加fn
    
    cv2.destroyAllWindows()
    
    
## Method 2
def edgesMarrHildreth(img, sigma):
    """
        finds the edges using MarrHildreth edge detection method...
        :param im : input image
        :param sigma : sigma is the std-deviation and refers to the spread of gaussian
        :return:
        a binary edge image...
    """

    # 根据输入的高斯标准差确定窗口大小，3 * sigma占99.7%
    size = int(2 * (np.ceil(3 * sigma)) + 1)
    # 生成(-size / 2 + 1, size / 2 )的网格
    x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1), np.arange(-size / 2 + 1, size / 2 + 1))
    # 计算LoG核
    kernel = ((x ** 2 + y ** 2 - (2.0 * sigma ** 2)) / sigma ** 4) * np.exp(
        -(x ** 2 + y ** 2) / (2.0 * sigma ** 2))  # LoG filter

    kern_size = kernel.shape[0]
    # 生成与输入图像相同大小的全零矩阵log
    log = np.zeros_like(img, dtype=float)

    # 应用LoG核
    for i in range(img.shape[0] - (kern_size - 1)):
        for j in range(img.shape[1] - (kern_size - 1)):
            window = img[i:i + kern_size, j:j + kern_size] * kernel
            log[i, j] = np.sum(window)

    # 将log由float转换为int64
    log = log.astype(np.int64, copy=False)

    # 生成与log相同大小的全零矩阵zero_crossing
    zero_crossing = np.zeros_like(log)

    # 判断零交叉点
    for i in range(log.shape[0] - (kern_size - 1)):
        for j in range(log.shape[1] - (kern_size - 1)):
            if log[i][j] == 0:
                if (log[i][j - 1] < 0 and log[i][j + 1] > 0) or (log[i][j - 1] < 0 and log[i][j + 1] < 0) or (
                        log[i - 1][j] < 0 and log[i + 1][j] > 0) or (log[i - 1][j] > 0 and log[i + 1][j] < 0):
                    zero_crossing[i][j] = 255
            if log[i][j] < 0:
                if (log[i][j - 1] > 0) or (log[i][j + 1] > 0) or (log[i - 1][j] > 0) or (log[i + 1][j] > 0):
                    zero_crossing[i][j] = 255

    # 绘制输出图像
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(log, cmap='gray')
    plt.axis('off')
    a.set_title('Laplacian of Gaussian')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(zero_crossing, cmap='gray')
    plt.axis('off')
    string = 'Zero Crossing sigma = '
    string += (str(sigma))
    a.set_title(string)
    plt.show()

    return zero_crossing

edge_detection1 = edgesMarrHildreth(img1, 3)
edge_detection2 = edgesMarrHildreth(img2, 3)
edge_detection3 = edgesMarrHildreth(img3, 3)

Edge_detection1 = edgesMarrHildreth(img1, 1)
Edge_detection2 = edgesMarrHildreth(img2, 1)
Edge_detection3 = edgesMarrHildreth(img3, 1)