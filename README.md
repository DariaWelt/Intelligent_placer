# Intelligent placer
Given a picture of several objects on a light horizontal surface and a polygon as input, determine whether it 
is possible to place all these objects simultaneously on the plane so that they fit into the polygon.

## Task definition
Algorithm input: 
- The path to a file with tif, png or jpg extension containing a three-channel RGB image on which on the horizontal 
- light surface there are objects from a known set;
- List of pairs of numbers(float) describing the polygon.
- [Optional] Mode of polygon representation (string value). Two are supported: pixels(by default) and relative.  

Algorithm output:
- True if items can be placed in a polygon, otherwise false.

## Input image requirements
- Objects cannot intersect;
- Objects must fully fit into the area specified in the algorithm parameter;
- There must be a distance of at least 50 pixels between objects;
- The smaller of the two visible object dimensions must be at least 50 pixels;
- The larger of the two visible object dimensions must be at least 100 pixels;
- The angle of inclination of the chamber from the surface normal should be within 10 degrees;
- The same object can be present in the photo several times. Objects will be considered independently in solving the 
placement problem;
- The polygon must contain more than 2 corners;
- The large side of the image must be less than 4500 pixels.

## Items data and testing dataset

Photos of possible items and background image (for second argument of algorithm): 
https://drive.google.com/drive/folders/1gk90RoNKPqP_lwWpY8cYFQpLThSHtY2m?usp=sharing 

Dataset for testing: images with jpg extension and csv file of input/output data: 
https://drive.google.com/drive/folders/1XGFnvI2nywZ0e6qcOO4kyAmzYbnEYxoU?usp=sharing 

## Placer Algorithm
The algorithm for finding objects:
1. Get blurred image using `skimage.filters.gaussian(image, 5)` and convert it into grayscale
2. Get the contours of the image using the `skimage.filters.sobel`
3. Remove the noise from the obtained image of the contour by binarization
4. Represent contours as a polygon using `cv2.findContours`
5. Consider only contours with area no less than 50*50 pixels, because the minimum width of the object is 50 pixels 
according to the stipulation
6. Match each found contour with the object whose contour is most similar to the given one. If all the contours are not
 similar, remove the current one from consideration
7. Construct a descriptor (a vector of 128 numbers) for each point on the contour of the object to be identified
8. Construct a descriptor for each point on the contour of the matched object
9. Solve the assignment problem for the distance matrix between these points and map the contour points to each other
10. Find the homography matrix and project the mask of the object onto the mask of the size of the input image. Assign
 pixels of the mask to the label values of the matched object class
11. If the homography matrix is not found, or the points of the projected mask are out of bounds, or these points 
intersect with another segmented object, we consider that the object is not identified, otherwise we add the identified
contour to the set of "good" contours

The algorithm for place problem:
1. Sum areas of all "good" contours
2. Get the area of input polygon
3. If the sum less than the area of input polygon, then return true, otherwise - false

## Get started
- clone repository
- upload photos of possible items with masks from drive and put it into `intelligent_placer_lib/data/`. If you want to 
use yor own data, you could put it (images and masks) into this dir.

for testing:
- put dataset into `test/data`
- put dataset specification json into `test/data`. Example of such file already put here.

## Improvements todo

#### Time:
- reduce size of items images so theirs contours will be calculated faster
- read all images in memory once
- set maximum number of contour points (because we have two items with a circular shape)

#### Accuracy:
- Consider that pixel in item mask could be addressed with area (set of pixels) in result mask
- replace naive algorithm of placing with the arranging of objects by parallel transfer
- use physics solutions: place objects randomly to polygon and use potential field method