# Program to geometricaly transform a screw parallel to the x-axis
# Regardless to the position of this screw given in an image

import cv2
import numpy as np
from matplotlib import pyplot as plt

# BIGGER TO DO 
# Adaptive threshold verbessern/nutzen für die contours bestimmung bei Bildern mit verschiedenen Farben (nicht nur graue/silberne/dunkle Schrauben)

### STEP 1 ####
##############################################################################################


screw_img = cv2.imread('C:\Users\PC\Desktop\DeepLearning\TestImages\Schrauben\schraube27.jpg', 0)
# screw_img2 = cv2.imread('C:\Users\PC\Desktop\DeepLearning\TestImages\Schrauben\schraube7.jpg')
# maybe we need to switch the color of our object for the contour finding
# screw_img2 = cv2.cvtColor(screw_img2, cv2.COLOR_BGR2GRAY)
screw_img_blurred = cv2.medianBlur(screw_img, 5)

cv2.imshow("Median Blurred screw", screw_img_blurred)
cv2.imshow("Normal screw", screw_img)

screw_thr_adapt = cv2.adaptiveThreshold(screw_img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                         cv2.THRESH_BINARY_INV, 11, 2)

# HERE Starts strategy with normal threshold
# already working with screw number 7 (schraube7.jpg)

#############################################################################DID NOT WORK WITH SCREW NUMBER 53!!!
### USING TEST IMAGE SCCREW IMAGE NUMBER 37 FOR DEVELOPEMENT ! ! GOOD EXAMPLE ! 
screw_img2 = cv2.imread('C:\Users\PC\Desktop\DeepLearning\TestImages\Schrauben\schraube51.jpg')
screw_gray = cv2.cvtColor(screw_img2, cv2.COLOR_BGR2GRAY)
# We need the inverted binary array for a better contour finding
# hardcoded threshold value for down bound pixel value, which we accept

# Here we have a pixel matrix represantation of our picture 
retval, screw_thr = cv2.threshold(screw_gray, 207, 255, cv2.THRESH_BINARY_INV)
# ENDS 

cv2.imshow("Thresholded gaussian screw img", screw_thr_adapt)
cv2.imshow("Thresholded screw img", screw_thr)


### STEP 2 ####
###############################################################################################


# Do a morphologycal transformation using a 3x3 kernel/quadar, with either normal or gaussian threshold
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Pick the right thresholded img adaptive or normal: 1. screw_thr or 2. screw_thr_adapt
thr_screw = cv2.morphologyEx(screw_thr, cv2.MORPH_CLOSE, rect_kernel)

### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


cv2.imshow("Rect Thr Screw", thr_screw)

img_ , contours, heirarchy = cv2.findContours(thr_screw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# If we wanna build the hull for all contour points
# for cnt in contours:
#     hull = cv2.convexHull(cnt)
#     cv2.drawContours(screw_img2, [hull], -1, (0, 0, 255), 1) 

# cv2.imshow("Hull Screw", screw_img2)

### STEP 3 ####
###############################################################################################


# Display our area sizes
max_area = 0.0
n_idx = -1
for index, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    print('contour area is ' + str(area))
    if max_area < area:
        max_area = area
        n_idx = index

print('max area is ' + str(max_area))
print('realted index to the max area is ' + str(n_idx))

cnt = contours[2]
# Get the 3rd contour because the 1st is the screen boundarie box and the 2nd is a single dot currently...
cv2.drawContours(screw_img2, contours, n_idx, (0,255,0), 3)

# if we wanna draw a circle around the contour given, single contour
# (x,y), radius = cv2.minEnclosingCircle(cnt)
# center = ( int(x), int(y))
# radius = ( int(radius))
# cv2.circle(screw_img, center, radius, (0,207,0), 2)

cv2.imshow("Contoured screw", screw_img2)

### STEP 4 ####
###############################################################################################


#Calculate the orientation of our screw using center of mass, main rotation axis and the PCACompute function
#--------------------------------------------------------------------
# Generate a matrix of non black points, as we have an BINARY_INV pixel matrix represented by screw_thr
matrix = np.argwhere(screw_thr != 0)

# As we are accessing pixels in [row,col] order we gotta do a swapping of those !
matrix[: , [0, 1]] = matrix[: , [1, 0]]

# We gotta preserve a numpy array consisting of float values for the PCA function
matrix = np.array(matrix).astype(np.float32) 

meanv, eigenvec = cv2.PCACompute(matrix, mean = np.array([]))

center = tuple(meanv[0])
center_rm = tuple(( (int) (meanv[0][0] + 150), (int) (meanv[0][1])))
center_lm = tuple(( (int) (meanv[0][0] - 150), (int) (meanv[0][1])))

# Calculate the longst axis using the biggest eigenvector
axis1 = tuple(meanv[0] + eigenvec[1]*100)

print(meanv[0])
print(meanv[0][0])
print(meanv[0][1])
print(type(meanv[0]))
# print the eigenvec array
print(eigenvec)
print(eigenvec[0])
print(eigenvec[1])
print(type(eigenvec[0]))

cv2.circle(screw_img2, center, 5, (0, 0, 255))
cv2.line(screw_img2, center, axis1, (0, 0, 255))
cv2.line(screw_img2, center, center_rm, (0, 0, 255), 1)
cv2.line(screw_img2, center, center_lm, (0, 0, 255), 1)

cv2.imshow("Center with main axis of screw_img2 ", screw_img2)

### STEP 5 ####
###############################################################################################
#--------------------------------------------------------------------

pst1 = np.float32([[meanv[0][0] - 250, meanv[0][1]], [meanv[0][0] + eigenvec[0][0]*10, meanv[0][1] + eigenvec[0][1]*10], [meanv[0][0] + 250, meanv[0][1]]])

# Calculate the dst position 2 (pst2)
pst2 = np.float32([ [meanv[0][0] - 240, meanv[0][1] - 108 ], [meanv[0][0] + eigenvec[0][0]*10 + 1, meanv[0][1] ], [meanv[0][0] + 240, meanv[0][1] + 108] ])

print(pst1)
print(pst2)

M = cv2.getAffineTransform(pst1, pst2)
print(M)
print(type(M))

rows, cols, ch = screw_img2.shape
dst = cv2.warpAffine(screw_img2, M, (cols, rows)) 

print(screw_img2)
print(dst)

# Carefull print out dst ! 
cv2.imshow("Geom. Transf. screw_img2 ", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
