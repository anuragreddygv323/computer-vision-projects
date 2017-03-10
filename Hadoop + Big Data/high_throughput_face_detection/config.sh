# set the base directory
BASE=$(pwd)

# update the PATH to include the 'pyimagesearch' library
PYTHONPATH="${PYTHONPATH}:${BASE}"
export PYTHONPATH

# configure service paths
export HADOOP_HOME="/Users/adrianrosebrock/PyImageSearch/Services/hadoop"