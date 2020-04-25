# import the necessary packages
import argparse
import cv2
import numpy as np
import json
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

radius = 5
points = None
grabbing = None
size = 500

def compute_homography(points):
	camera_points = np.array([
		[points[0][0], points[0][1]],
		[points[1][0], points[1][1]],
		[points[2][0], points[2][1]],
		[points[3][0], points[3][1]],
	])
	map_points = np.array([
		[0, 0], #top_left
		[0, 1], #top_right
		[1, 1], #bottom_right
		[1, 0], #bottom_left
	])

	h, _ = cv2.findHomography(camera_points, size/4 + map_points * size / 2)
	return h

def in_circle(point, circle, radius):
	return np.sqrt( (point[0] - circle[0]) ** 2 + (point[1] - circle[1]) ** 2 ) <= radius

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global points, grabbing, radius, homography

	# if the left mouse button was clicked and mouse is over a point
	# drag the point
	if grabbing is not None:
		points[grabbing] = (x, y)
		homography = compute_homography(points)

	if event == cv2.EVENT_LBUTTONDOWN:
		for idx in range (len(points)):
			point = points[idx]
			if in_circle((x, y), point, radius):
				grabbing = idx
	elif event == cv2.EVENT_LBUTTONUP:
		grabbing = None
		homography = compute_homography(points)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
filename = args["image"]

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(filename)

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

h, w = image.shape[0], image.shape[1]

points = [ 
	( int (w/2 - w/5), int(h/2 - h/5) ), 
	( int (w/2 - w/5), int(h/2 + h/5) ), 
	( int (w/2 + w/5), int(h/2 + h/5) ), 
	( int (w/2 + w/5), int(h/2 - h/5) ) 
]

def draw_points(image):
	color = (192, 55, 240)
	for idx in range( len(points) ):
		point = points[idx]
		r = radius
		if grabbing is not None and grabbing == idx:
			r = radius * 2
		cv2.circle(image, point, r, color, -1)
	cv2.line(image, points[0], points[1], color, 2)
	cv2.line(image, points[1], points[2], color, 2)
	cv2.line(image, points[2], points[3], color, 2)
	cv2.line(image, points[3], points[0], color, 2)

homography = compute_homography(points)

write_text = False
while True:
	# display the image and wait for a keypress
	clone = image.copy()
	draw_points(clone)

	warped = cv2.warpPerspective(image, homography, (size, size) )

	if write_text:
		cv2.putText(clone, 'Saved!', (int(w/2 - 50), int(h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) 
	
	cv2.imshow("image", clone)
	cv2.imshow("warp", warped)
	key = cv2.waitKey(1) & 0xFF

	# if the 'c' key is pressed, break from the loop
	if key == ord("q"):
		break
	if key == ord("s"):
		calib = { "image" : [
			[points[0][0], points[0][1]],
			[points[1][0], points[1][1]],
			[points[2][0], points[2][1]],
			[points[3][0], points[3][1]],
		], "map": [
			[0, 0], #top_left
			[0, 1], #top_right
			[1, 1], #bottom_right
			[1, 0], #bottom_left
		]}

		with open("calib.json", "w") as f:
			f.write(json.dumps(calib))
			write_text = True

# close all open windows
cv2.destroyAllWindows()