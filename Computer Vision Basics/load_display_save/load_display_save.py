# USAGE
# python load_display_save.py --image florida_trip.png

# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and show some basic information on it
image = cv2.imread(args["image"])
print("width: %d pixels" % (image.shape[1]))
print("height: %d pixels" % (image.shape[0]))
print("channels: %d" % (image.shape[2]))

# show the image and wait for a keypress
cv2.imshow("Image", image)
cv2.waitKey(0)

# save the image -- OpenCV handles converting filetypes
# automatically
cv2.imwrite("newimage.jpg", image)