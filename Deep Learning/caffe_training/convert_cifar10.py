# USAGE
# python convert_cifar10.py --dataset ~/cifar-10-batches-py \
#	--output output/cifar10/images --train output/cifar10/train.txt \
#	--test output/cifar10/val.txt

# import the necessary packages
from __future__ import print_function
from pyimagesearch.utils.dataset import build_cifar10
import argparse
import glob

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-t", "--train", required=True, help="path to output training file")
ap.add_argument("-v", "--test", required=True, help="path to output testing file")
args = vars(ap.parse_args())

# open the training file for writing and build the set of images
print("[INFO] gathering training data...")
f = open(args["train"], "w")
build_cifar10(glob.glob("{}/data_batch_*".format(args["dataset"])), args["output"], f)
f.close()

# open the testing file for writing and build the set of images
print("[INFO] gathering testing data...")
f = open(args["test"], "w")
build_cifar10(["{}/test_batch".format(args["dataset"])], args["output"], f)
f.close()