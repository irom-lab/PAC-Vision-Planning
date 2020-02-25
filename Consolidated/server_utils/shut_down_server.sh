#!/bin/bash

# Kill python processes on server $1 

(echo "killall python"
echo "exit") | ssh -v -i ../../../AWS/Instances/AWS-ML.pem ubuntu@$1 &> /dev/null