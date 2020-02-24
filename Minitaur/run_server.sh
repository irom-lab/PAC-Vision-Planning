#!/bin/bash

#echo "Starting Server: $1"

#(echo "cd test"
#echo "put ../../shell_scripts/sftp.sh"
#echo "exit") | sftp -v -i ../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null

(echo "cd repos/husky-pac-fpv-nav"
echo "put mu_server.pt"
echo "put logvar_server.pt"
echo "exit") | sftp -v -i ../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null

(echo "cd repos/husky-pac-fpv-nav"
echo "python node.py --start_seed $2 --num_seeds $3 --num_cpu $4 --num_gpu $5 --server_ind $6 --reg_include $7"
echo "exit") | ssh -v -i ../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null

(echo "cd repos/husky-pac-fpv-nav"
echo "get output$6.pt"
echo "exit") | sftp -v -i ../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null