#!/usr/bin/env sh

EXAMPLE=/home/ubuntu/pyimagesearch-gurus/caffe_examples/output/cifar10/db
DATA=/home/ubuntu/pyimagesearch-gurus/caffe_examples/output/cifar10
TOOLS=$CAFFE_ROOT/build/tools

$TOOLS/compute_image_mean $EXAMPLE/training_lmdb \
  $DATA/dataset_mean.binaryproto

echo "Done."
