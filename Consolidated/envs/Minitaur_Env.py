#!/usr/bin/env python3

import os

# SUPPRESS PRINTING
null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
save = os.dup(1), os.dup(2)
os.dup2(null_fds[0], 1)
os.dup2(null_fds[1], 2)

import numpy as np
from pybullet_envs.minitaur.envs import minitaur_gym_env
import math
from policy.minitaur_policy import Policy
import torch
import warnings
warnings.filterwarnings('ignore')

# Set the initial position
from pybullet_envs.minitaur.envs import minitaur
minitaur.INIT_POSITION = [0, 0, 0.2]

# ENABLE PRINTING
os.dup2(save[0], 1)
os.dup2(save[1], 2)
os.close(null_fds[0])
os.close(null_fds[1])

class Environment:

    def __init__(self, max_angle, time_step=0.02, gui=False):
        self.gui = gui
        self.max_angle = max_angle

        if self.gui:
            self.minitaur_env = minitaur_gym_env.MinitaurGymEnv(
                                urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
                                render=True,
                                motor_velocity_limit=np.inf,
                                pd_control_enabled=True,
                                hard_reset=False,
                                on_rack=False,
                                reflection=False)
        else:
            self.minitaur_env = minitaur_gym_env.MinitaurGymEnv(
                                urdf_version=minitaur_gym_env.DERPY_V0_URDF_VERSION,
                                render=False,
                                motor_velocity_limit=np.inf,
                                pd_control_enabled=True,
                                hard_reset=False,
                                on_rack=False)

        self.minitaur_env.minitaur.time_step = time_step
        self.p = self.minitaur_env._pybullet_client
        # self.minitaur_env.minitaur.SetFootFriction(1.0)
        # self.minitaur_env.minitaur.SetFootRestitution(0.1)
        # self.prim_lib = np.load('prim_lib.npy')
        # textureId = self.p.loadTexture("heightmaps/table.png")
        # self.p.changeVisualShape(self.minitaur_env.ground_id, -1, textureUniqueId=textureId)
        self.terraintextureId = self.p.loadTexture("heightmaps/oak-wood.png")

    def generate_htfield(self, num_rows=12):
        '''Generate a heightfield.
        Resource: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/heightfield.py'''
        p = self.p
        numHeightfieldRows = num_rows
        numHeightfieldColumns = num_rows
        heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns

        for i in range(numHeightfieldRows*numHeightfieldColumns):
            # heightfieldData[i] = 0.1
            if (i%2)==0:
                heightfieldData[i] = np.random.uniform(self.h_lim[0],self.h_lim[1])
            else:
                heightfieldData[i] = 0

        terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD,
                                              flags = p.GEOM_CONCAVE_INTERNAL_EDGE, # this flag ensures foot does not get stuck
                                              meshScale=[1,1,1],
                                              heightfieldTextureScaling=(numHeightfieldRows-1)/2,
                                              heightfieldData=heightfieldData,
                                              numHeightfieldRows=numHeightfieldRows,
                                              numHeightfieldColumns=numHeightfieldColumns)
        textureId = p.loadTexture("heightmaps/wm_height_out.png")

        terrain = p.createMultiBody(0, terrainShape)

        # p.changeVisualShape(terrain, -1, rgbaColor=[1,1,1,1])
        p.changeVisualShape(terrain, -1, textureUniqueId = textureId)
        
        # Remove the previous terrain and establish the new one
        # Note: first time this function is called, the default terrain of minitaur_env
        # which is plane ground is removed. Subsequent calls remove the previous heightfield
        self.minitaur_env._pybullet_client.removeBody(self.minitaur_env.ground_id)
        self.minitaur_env.ground_id = terrain

        return terrain
        
    def is_fallen(self):
        """Decide whether the minitaur has fallen.
    
        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.5), the minitaur is considered fallen.
    
        Returns:
          Boolean value that indicates whether the minitaur has fallen.
        """
        orientation = self.minitaur_env.minitaur.GetBaseOrientation()
        rot_mat = self.minitaur_env._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.3)

    def generate_steps(self, numObs=25):
        p = self.p
        numObs *= 2

        linkMasses = [None] * (numObs)
        colIdxs = [None] * (numObs)
        visIdxs = [None] * (numObs)
        posObs = [None] * (numObs)
        orientObs = [None] * (numObs)
        parentIdxs = [None] * (numObs)
        linkInertialFramePositions = [None] * (numObs)
        linkInertialFrameOrientations = [None] * (numObs)
        linkJointTypes = [None] * (numObs)
        linkJointAxis = [None] * (numObs)

        for obs in range(numObs):
            linkMasses[obs] = 0.0
            parentIdxs[obs] = 0
            linkInertialFramePositions[obs] = [0, 0, 0]
            linkInertialFrameOrientations[obs] = [0, 0, 0, 1]
            linkJointTypes[obs] = p.JOINT_FIXED
            linkJointAxis[obs] = np.array([0, 0, 1])
            orientObs[obs] = p.getQuaternionFromEuler([0., np.pi/4, 0])

        posObs, orientObs, colIdxs, visIdxs = self._generate_steps_sub(p, posObs, orientObs, colIdxs, numObs)
        
        obsUid = p.createMultiBody(baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, basePosition=[0, 0, 0],
                                    baseOrientation=[0, 0, 0, 1], baseInertialFramePosition=[0, 0, 0],
                                    baseInertialFrameOrientation=[0, 0, 0, 1], linkMasses=linkMasses,
                                    linkCollisionShapeIndices=colIdxs, linkVisualShapeIndices=visIdxs,
                                    linkPositions=posObs, linkOrientations=orientObs, linkParentIndices=parentIdxs,
                                    linkInertialFramePositions=linkInertialFramePositions,
                                    linkInertialFrameOrientations=linkInertialFrameOrientations,
                                    linkJointTypes=linkJointTypes, linkJointAxis=linkJointAxis)

        for obs in range(numObs):
            p.changeVisualShape(obsUid, visIdxs[obs], textureUniqueId=self.terraintextureId)

        x_goal = self.goal
        y_goal = 0
        posObs = np.array([None] * 3)
        posObs[0] = x_goal
        posObs[1] = y_goal
        posObs[2] = 0  # set z at ground level
        # colIdxs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1,5.0,0.1])
        colIdxs = -1
        visIdxs = p.createVisualShape(p.GEOM_BOX, 
                                             halfExtents=[0.05,5.0,0.15],
                                             rgbaColor=[0.7, 0, 0, 1])
        
        linkMasses = 0.0
        parentIdxs = 0
        linkInertialFramePositions = [0, 0, 0]
        linkInertialFrameOrientations = [0, 0, 0, 1]
        linkJointTypes = p.JOINT_FIXED
        linkJointAxis = np.array([0, 0, 1])
        orientObs = p.getQuaternionFromEuler([0., 0., 0.])

        p.createMultiBody(baseCollisionShapeIndex=colIdxs, baseVisualShapeIndex=visIdxs, basePosition=posObs)#,

        return obsUid

    def _generate_steps_sub(self, p, posObs, orientObs, colIdxs, numObs):
        visIdxs = [None]*numObs

        for obs in range(int(numObs/2)):  # Cylindrical obstacles
            posObs_obs1 = [None] * 3
            posObs_obs2 = [None] * 3
            theta = np.random.rand(1)*(self.max_angle*math.pi/180)*(2/3)
            l1 = 0.5
            l2 = l1
            theta_rotate = theta
            h = l1 * np.sin(np.pi/4 - theta) /(2**0.5)
            d = (2**0.5) * l1 * np.cos(np.pi/4 + theta)
            halfExtents = [l1/2,5.0,l2/2]
            x_temp = 0.5 + obs*l1
            y_temp = 0
            
            posObs_obs1[0] = x_temp
            posObs_obs1[1] = y_temp
            posObs_obs1[2] = -h # set z at ground level
            posObs[2*obs] = posObs_obs1
            colIdxs[2*obs] = p.createCollisionShape(p.GEOM_BOX, flags = p.GEOM_CONCAVE_INTERNAL_EDGE, halfExtents=halfExtents)
            orientObs[2*obs] = p.getQuaternionFromEuler([0., -theta_rotate, 0])
            visIdxs[2*obs] = p.createVisualShape(p.GEOM_BOX, 
                                               halfExtents=halfExtents,)
            # print(posObs_obs[0])
            posObs_obs2[0] = x_temp + d
            # print(posObs_obs[0])
            posObs_obs2[1] = y_temp
            posObs_obs2[2] = -h # set z at ground level
            posObs[2*obs+1] = posObs_obs2
            colIdxs[2*obs+1] = p.createCollisionShape(p.GEOM_BOX, flags = p.GEOM_CONCAVE_INTERNAL_EDGE, halfExtents=halfExtents)
            orientObs[2*obs+1] = p.getQuaternionFromEuler([0., theta_rotate, 0])
            visIdxs[2*obs+1] = p.createVisualShape(p.GEOM_BOX, 
                                               halfExtents=halfExtents,)
           
        return posObs, orientObs, colIdxs, visIdxs

    # def _get_bounding_amplitude(self, prim):
    #     return self.prim_lib[prim]


    def execute_policy(self, policy, goal, alpha, time_step=0.01, speed=40, comp_len=10, prim_horizon=50, 
                       image_size=50, device=torch.device('cuda'), record_vid=False, vid_num=0):
        
        if record_vid:
            import cv2
            # videoObj = cv2.VideoWriter('video'+str(vid_num)+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
            #                            50, (minitaur_gym_env.RENDER_WIDTH, minitaur_gym_env.RENDER_HEIGHT))
            videoObj = cv2.VideoWriter('video'+str(vid_num)+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                       50, (minitaur_gym_env.RENDER_WIDTH, minitaur_gym_env.RENDER_HEIGHT))

        t_flag = 0
        goal_cost = np.zeros(1)
        coll_cost = np.ones(1)
        cost = goal_cost + coll_cost
        total_time_steps = comp_len * prim_horizon
        
        
        #                             baseOrientation=[0, 0, 0, 1], baseInertialFramePosition=[0, 0, 0],
        #                             baseInertialFrameOrientation=[0, 0, 0, 1], linkMasses=linkMasses,
        #                             linkCollisionShapeIndices=colIdxs, linkVisualShapeIndices=visIdxs,
        #                             linkPositions=posObs)
        # , linkOrientations=orientObs, linkParentIndices=parentIdxs,
        #                             linkInertialFramePositions=linkInertialFramePositions,
        #                             linkInertialFrameOrientations=linkInertialFrameOrientations,
        #                             linkJointTypes=linkJointTypes, linkJointAxis=linkJointAxis)

        
        for i in range(5):
            action = [0,0,0,0,0,0,0,0]
            self.minitaur_env.step(action)

        for i in range(comp_len):
            # Get current depth map
            cam_pos = list(self.minitaur_env.minitaur.GetBasePosition())
            cam_orn = list(self.minitaur_env.minitaur.GetTrueBaseOrientation())
            _, depth = self._mount_cam(cam_pos, cam_orn)
            
            # Decide primitive from the policy
            depth = torch.Tensor(depth).view([1, 1, image_size, image_size])
            assert depth.nelement()!=0,"Tensor is empty."

            motor_angles = torch.Tensor(self.minitaur_env.minitaur.GetMotorAngles()).view([1,8]).detach()
            motor_velocities = torch.Tensor(self.minitaur_env.minitaur.GetMotorVelocities()).view([1,8]).detach()
            base_pos = torch.Tensor(self.minitaur_env.minitaur.GetBasePosition()).view([1,3]).detach()
            base_orn = torch.Tensor(self.minitaur_env.minitaur.GetBaseRollPitchYaw()).view([1,3]).detach()

            control_params = policy(depth.to(device),
                                    motor_angles.to(device), 
                                    motor_velocities.to(device),
                                    base_pos.to(device),
                                    base_orn.to(device))[0]
            amplitude1 = (control_params[0].item()*0.8)+0.2
            # amplitude1 = torch.clamp(control_params[0], min=0.2, max=1.0).item()
            amplitude2 = (control_params[1].item()*0.8)+0.2
            # amplitude2 = torch.clamp(control_params[0], min=0.2, max=1.0).item()
            steering_amplitude = torch.clamp(control_params[2], min=0.0, max=min(1-amplitude1, 1-amplitude2)).item()
            # phase1 = control_params[3].item() * math.pi
            # phase2 = control_params[4].item() * math.pi
            speed = control_params[3].item()*20 + 20

            for step_counter in range(prim_horizon):

                t = step_counter * time_step + t_flag
                
                # if t>4.1:
                #     import matplotlib.pyplot as plt

                #     cam_pos = list(self.minitaur_env.minitaur.GetBasePosition())
                #     cam_orn = list(self.minitaur_env.minitaur.GetTrueBaseOrientation())
                #     rgb, _ = self._mount_cam(cam_pos, cam_orn, w=500, h=500)
                    
                #     fig = plt.figure()
                #     ax = plt.subplot(111)
                #     ax.set_yticklabels([])
                #     ax.set_xticklabels([])

                #     plt.imshow(rgb, cmap='gray', interpolation='nearest')
                #     plt.savefig('minitaur_rgb_view.png')

                #     time.sleep(600)

                # amplitude1 = 0.5
                # amplitude2 = 0.5
                # steering_amplitude = 0.0
                # speed = 50

                if record_vid:
                    rgb = self.minitaur_env.render()
                    cv2.imshow('Vis_Vid_Rec', cv2.cvtColor(np.uint8(rgb), cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                    videoObj.write(cv2.cvtColor(np.uint8(rgb), cv2.COLOR_RGB2BGR))
                
                phase1 = math.pi
                phase2 = phase1
                # Applying asymmetrical sine gaits to different legs can steer the minitaur.
                a1 = math.sin(t * speed) * (amplitude1 + steering_amplitude)
                a2 = math.sin(t * speed + phase1) * (amplitude1 - steering_amplitude)
                a3 = math.sin(t * speed) * amplitude2
                a4 = math.sin(t * speed + phase2) * amplitude2
                action = [a1, a2, a2, a1, a3, a4, a4, a3]
                self.minitaur_env.step(action)
                
                # Compute costs
                # rob_pos =  np.array(cam_pos[0:2])
                rob_pos =  cam_pos[0]
                # goal_cost = np.linalg.norm(rob_pos-goal, ord=2)/10
                goal_cost = np.abs(rob_pos-goal)/goal
                fall_cost = 1 - (step_counter + i*prim_horizon)/(total_time_steps-1)
                cost = alpha * fall_cost + (1-alpha) * goal_cost
                end_position = self.minitaur_env.minitaur.GetBasePosition()
                end_position = end_position[0]
                if self.is_fallen():
                    if record_vid:
                        videoObj.release()
                    return cost, fall_cost, goal_cost, end_position
                if end_position>goal:
                    goal_cost = 0.
                    cost = alpha * fall_cost + (1-alpha) * goal_cost
                    if record_vid:
                        videoObj.release()
                    return cost, fall_cost, goal_cost, end_position
            
            end_position = self.minitaur_env.minitaur.GetBasePosition()
            end_position = end_position[0]
            # print(end_position)
            # print("Speed:", np.linalg.norm(start_position-end_position)/(prim_horizon*time_step))

            t_flag += prim_horizon*time_step

        if record_vid:
            videoObj.release()
        return cost, fall_cost, goal_cost, end_position


    def _mount_cam(self, base_p, base_o, w=50, h=50):
        '''
        Mounts an RGB-D camera on a robot in pybullet

        Parameters
        ----------
        w : Width
        h : Height
        base_p : Base position
        base_o : Base orientation as a quaternion

        Returns
        -------
        rgb : RGB image
        depth : Depth map
        '''

        p = self.p
        cam_pos = base_p

        # Rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(base_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (1, 0, -0.05) # x-axis
        init_up_vector = (0, 0, 1) # z-axis
        cam_translate = (0.1,0,0.1) # Shift camera from the base poitions w.r.t. to the link's local frame

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        cam_pos = cam_pos + rot_matrix.dot(cam_translate)
        view_matrix = p.computeViewMatrix(cam_pos, cam_pos + 0.1 * camera_vector, up_vector)
        # Get Image
        projection_matrix = p.computeProjectionMatrixFOV(fov=90.0, aspect=1., nearVal=0.01, farVal=1000.)

        # heightfield does not work with the tiny renderer, have to use the openGL renderer
        _, _, rgb, depth, _ = p.getCameraImage(w, h, view_matrix, projection_matrix)#, flags=p.ER_NO_SEGMENTATION_MASK)
        
        # Reshape rgb image and drop the alpha layer (#4)
        rgb = np.array(rgb, dtype=np.uint8)
        rgb = np.reshape(rgb, (w, h, 4))
        rgb = rgb[:, :, :3]

        # Reshape depth map
        depth = np.array(depth, dtype=np.float32)
        far=1000.0
        near=0.01
        depth = far*near/(far - (far - near)*depth)
        depth = np.reshape(depth, (w, h))

        return rgb, depth


if  __name__ == '__main__':
    env = Environment([0,0.1], gui=True)
    env.terrain = env.generate_htfield()
    env.minitaur_env.reset()
    policy = Policy()
    policy = policy.to('cuda')
    cost, coll_cost, goal_cost = env.execute_policy(policy, goal=np.ones(2), alpha=0.5)
    print(cost, coll_cost, goal_cost)