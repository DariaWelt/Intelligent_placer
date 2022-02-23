# Intelligent placer
Given a picture of several objects on a light horizontal surface and a polygon as input, determine whether it is possible to place all these objects simultaneously on the plane so that they fit into the polygon.

## Task definition
Algorithm input: 
- The path to a file with tif, png or jpg extension containing a three-channel RGB image on which on the horizontal light surface there is a sheet of A4 paper with a polygon drawn on the left and objects from a known set on the right;
- The path to the file with the same extension as in the previous paragraph, which contains an image of the surface on which a sheet of paper with a polygon and items will be placed;
- 4 integers describing the area within which the integer objects we are going to consider are contained. the first and second numbers represent the start and end pixel (inclusive) horizontally, and the second and third numbers represent the start and end pixel of the area vertically. The area has to contain objects in full, otherwise this object will not be counted and processed.

Algorithm output:
- True if items can be placed in a polygon, otherwise false.

## Input image requirements
- Objects cannot intersect;
- Objects must fully fit into the area specified in the algorithm parameter;
- There must be a distance of at least 10 pixels between objects;
- The border of a polygon must be at least 4 pixels thick at its narrowest point;
- The smaller of the two visible object dimensions must be at least 20 pixels;
- The larger of the two visible object dimensions must be at least 100 pixels;
- The angle of inclination of the chamber from the surface normal should be within 10 degrees;
- The same object can be present in the photo several times. Objects will be considered independently in solving the placement problem;
- The polygon must contain more than 2 corners (otherwise the algorithm will return false);
- The large side of the image must be less than 4500 pixels;
- The polygon must be drawn inside the sheet of paper so the border is no closer to the edge of the paper than 5 pixels;
- The sheet of paper with a drawn polygon has to fit into the picture together with its borders.

## Items data and testing dataset

Photos of possible items and background image (for second argument of algorithm): https://drive.google.com/drive/folders/1gk90RoNKPqP_lwWpY8cYFQpLThSHtY2m?usp=sharing 

Dataset for testing: images with jpg extension and csv file of input/output data: https://drive.google.com/drive/folders/1lkSI2wThZ_rEAMk5LxqLqtVBevjNOp2e?usp=sharing 
