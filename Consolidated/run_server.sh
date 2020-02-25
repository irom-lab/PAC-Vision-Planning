#!/bin/bash

#echo "Server $1 got start_seed $2 and num_seeds $3"

(echo "cd $8"
echo "put mu_server.pt"
echo "put logvar_server.pt"
echo "exit") | sftp -v -i ../../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null

(echo "cd $8"
echo "python node.py --start_seed $2 --num_seeds $3 --num_cpu $4 --num_gpu $5 --server_ind $6 --itr $7"
echo "exit") | ssh -v -i ../../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null

(echo "cd $8"
echo "get output$6.pt"
echo "exit") | sftp -v -i ../../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null