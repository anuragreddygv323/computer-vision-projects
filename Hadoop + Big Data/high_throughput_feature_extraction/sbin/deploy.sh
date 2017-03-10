#!/bin/sh

# create a .zip of the module so it can be deployed to Hadoop
rm -f deploy/pyimagesearch.zip
zip -q -r deploy/pyimagesearch.zip pyimagesearch/