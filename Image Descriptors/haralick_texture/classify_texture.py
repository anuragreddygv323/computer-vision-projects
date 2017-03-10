# USAGE
# python classify_texture.py --training training --test testing

# import the necessary packages
from sklearn.svm import LinearSVC
import argparse
import mahotas
import glob
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the dataset of textures")
ap.add_argument("-t", "--test", required=True, help="Path to the test images")
args = vars(ap.parse_args())

# initialize the data matrix and the list of labels
print "[INFO] extracting features..."
data = []
labels = []

# loop over the dataset of images
for imagePath in glob.glob(args["training"] + "/*.png"):
	# load the image, convert it to grayscale, and extract the texture
	# name from the filename
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	texture = imagePath[imagePath.rfind("/") + 1:].split("_")[0]

	# extract Haralick texture features in 4 directions, then take the
	# mean of each direction
	features = mahotas.features.haralick(image).mean(axis=0)

	# update the data and labels
	data.append(features)
	labels.append(texture)

# train the classifier
print "[INFO] training model..."
model = LinearSVC(C=10.0, random_state=42)
model.fit(data, labels)
print "[INFO] classifying..."

# loop over the test images
for imagePath in glob.glob(args["test"] + "/*.png"):
	# load the image, convert it to grayscale, and extract Haralick
	# texture from the test image
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	features = mahotas.features.haralick(gray).mean(axis=0)

	# classify the test image
	pred = model.predict(features.reshape(1, -1))[0]
	cv2.putText(image, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 3)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)