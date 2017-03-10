#!/bin/sh

# stop Hadoop
echo "[INFO] Stopping Hadoop"
cd $HADOOP_HOME
sbin/stop-dfs.sh