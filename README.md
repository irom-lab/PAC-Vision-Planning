# PAC-Vision-Planning

This repository contains code for the results in: [Probably Approximately Correct Vision-Based Planning using Motion Primitives](https://irom-lab.princeton.edu/wp-content/uploads/2020/02/Veer.PACBayes.pdf)

### Examples in the code:
1. Quadrotor navigating an obstacle field using a depth map from an oboard RGB-D camera
2. Quadruped (Minitaur, Ghost Robotics) traversing rough terrain using proprioceptive and exteroceptive (depth map from onbaord RGB-D camera) feedback

### Dependencies:
1. PyBullet
2. PyTorch
3. Tensorboard

### Important details:
Relevant parameters for each example are provided in a config json file located in the configs folder. The environments for each training example are drawn from a distribution, hence they are generated by varying the random seed. 

### The training process has three main parts:
1. Train a Prior using Evolutionary Strategies:
   - Quadrotor: ```python train_ES.py --config_file configs/config_quadrotor.json```
   - Minitaur: ```python train_ES.py --config_file configs/config_minitaur.json```

2. Draw a m poilicies i.i.d. from the prior above and compute the cost for each policy on N new environments:
   - Quadrotor: ```python compute_policy_costs.py --config_file configs/quadrotor.json --start_seed 580 --num_envs N --num_policies m```
   - Minitaur: ```python compute_policy_costs.py --config_file configs/config_minitaur.json --start_seed 10 --num_envs N --num_policies m```

3. Perform PAC-Bayes optimization with the parametric REP in the paper on N environments and m policies using the costs computed above:
   - Quadrotor: ```python PAC_Bayes_opt.py --config_file configs/config_quadrotor.json --num_envs N --num_policies m```
   - Minitaur: ```python PAC_Bayes_opt.py --config_file configs/config_minitaur.json --num_envs N --num_policies m```

The trained posterior can be tested using the quad_test.py and minitaur_test.py scripts.



