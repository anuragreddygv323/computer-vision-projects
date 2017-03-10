#!/bin/sh

# grab the current working directory
BASE=$(pwd)

# create the latest deployable package
sbin/deploy.sh

# change directory to where Hadoop lives
cd $HADOOP_HOME

# (potentially optional): turn off safe mode
bin/hdfs dfsadmin -safemode leave

# remove the previous output directory
bin/hdfs dfs -rm -r /user/guru/ukbench/output

# define the set of local files that need to be present to run the Hadoop
# job -- comma separate each file path
FILES="${BASE}/feature_extractor_mapper.py,\
${BASE}/deploy/pyimagesearch.zip"

# run the job on Hadoop
bin/hadoop jar share/hadoop/tools/lib/hadoop-streaming-*.jar \
    -D mapreduce.job.reduces=0 \
    -files  ${FILES} \
    -mapper ${BASE}/feature_extractor_mapper.py \
    -input /user/guru/ukbench/input/ukbench_dataset.txt \
    -output /user/guru/ukbench/output