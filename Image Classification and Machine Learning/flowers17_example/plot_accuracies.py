# USAGE
# python plot_accuracies.py --conf conf/flowers17.json

# import the necessary packages
from pyimagesearch.utils import Conf
import matplotlib.pyplot as plt
import argparse
import cPickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
args = vars(ap.parse_args())

# load the configuration and the accuracies data
conf = Conf(args["conf"])
data = cPickle.loads(open(conf["accuracies_path"]).read())

# plot the accuracies
plt.style.use("ggplot")
plt.figure()
plt.title("Bag of Visual Words Accuracies")
plt.xlabel("# of visual words")
plt.ylabel("% accuracy")
plt.plot(data.keys(), data.values())
plt.show()