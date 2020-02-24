#!/bin/bash

(echo "cd repos/husky-pac-fpv-nav"
echo "put $2"
echo "exit") | sftp -v -i ../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null
