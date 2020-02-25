#!/bin/bash

(echo "cd $3"
echo "put $2"
echo "exit") | sftp -v -i ../../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null
