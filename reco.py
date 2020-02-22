import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import pytesseract

img = cv2.imread('araba/16.png',0)
img2 = img.copy()
template = cv2.imread('template/t0.png',0)

img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
template = cv2.adaptiveThreshold(template,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2) 

w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
       'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

img = img2.copy()
method = eval('cv2.TM_CCOEFF')

# Apply template Matching
res = cv2.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
plt.imshow(res,cmap = 'gray')

top_left = max_loc

bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img,top_left, bottom_right, 255, 2)
crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
"""cv2.imshow("cropped",crop_img) 
cv2.waitKey(0)"""

cv2.imwrite('crop_img.png',crop_img)

plt.subplot(121),plt.imshow(crop_img,cmap = 'gray')
plt.title('Plate Number= '), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle('matching')


plt.show()
 



