# Images Description

## checkerboard_original.jpg
All the other checkerboard_*.jpg are derived from this original image through cropping using GIMP Image Editor
GIMP Image editor uses the same coordinate configuration as OpenCV:
(x, y) corresponds to (col, row). The top left corner is (0, 0) and
x increases as you go to the right, y increases as you go to the bottom

All the cropping needs 2 parameters:
1. Top-left corner coordinate => (x, y)
2. Size => (width, height)

## checkerboard_1.jpg
Top-left corner: (272, 328)
Size: (3576, 2480)

## checkerboard_2.jpg
Top-left corner: (274, 330)
Size: (3576, 2480)
The effect of this image relative to checkerboard_1.jpg is that the object
moved 2 pixels left and up

## checkerboard_3.jpg
Top-left corner: (276, 332)
Size: (3576, 2480)
The effect of this image relative to checkerboard_2.jpg is that the object
moved 2 pixels left and up

## checkerboard_4.jpg (Use for pyramid)
Top-left corner: (280, 336)
Size: (3576, 2480)
The effect of this image relative to checkerboard_1.jpg is that the object
moved 10 pixels left and up
