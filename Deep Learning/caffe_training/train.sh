#!/usr/bin/env sh

# train the model
TOOLS=$CAFFE_ROOT/build/tools
$TOOLS/caffe train -solver cifar10_solver.prototxt