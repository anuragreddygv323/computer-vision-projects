# USAGE
# python train.py --dataset data/digits.csv --model models/svm.cpickle

# import the necessary packages
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from pyimagesearch.descriptors import HOG
from pyimagesearch.utils import dataset
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the dataset file")
ap.add_argument("-m", "--model", required=True, help="path to where the model will be stored")
args = vars(ap.parse_args())

# load the dataset and initialize the data matrix
(digits, target) = dataset.load_digits(args["dataset"])
data = []

# initialize the HOG descriptor
hog = HOG(orientations=18, pixelsPerCell=(10, 10), cellsPerBlock=(1, 1), normalize=True)

# loop over the images
for image in digits:
	# deskew the image, center it
	image = dataset.deskew(image, 20)
	image = dataset.center_extent(image, (20, 20))

	# describe the image and update the data matrix
	hist = hog.describe(image)
	data.append(hist)

# train the model
model = LinearSVC(random_state=42)
model.fit(data, target)

# dump the model to file
joblib.dump(model, args["model"])