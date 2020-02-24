#!/bin/bash

# Kill python processes on server $1 

(echo "cd $2"
echo "put node.py"
echo "put head_node.py"
echo "put Parallelizer.py"
echo "put policy.py"
echo "exit") | sftp -v -i ../../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null
