# rotate_screws
Screw-rotation program using the framework OpenCV. 

Steps done by program:
1. Threshholding the given image, to filter the needed pixels/information
2. Morphologycal transformation using a 3x3 kernel/quadar
3. Draw a contour around our transformed image
4. Calculate the longst axis using the biggest eigenvector (Calculate the orientation of our screw using center of mass, main rotation axis and the PCACompute function)
5. Rotate the image using the longest symmatrical axis in the screw image (So that this longest axis is parallel to the x-axis in our coordinate system)
