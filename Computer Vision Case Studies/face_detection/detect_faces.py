# USAGE
# python detect_faces.py --face cascades/haarcascade_frontalface_default.xml --image images/obama.png

# import the necessary packages
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True, help="Path to where the face cascade resides")
ap.add_argument("-i", "--image", required=True, help="Path to where the image file resides")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face detector and detect faces in the image
detector = cv2.CascadeClassifier(args["face"])
faceRects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5,
	minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
print "I found %d face(s)" % (len(faceRects))

# loop over the faces and draw a rectangle around each
for (x, y, w, h) in faceRects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the detected faces
cv2.imshow("Faces", image)
cv2.waitKey(0)