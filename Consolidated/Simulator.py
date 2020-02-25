#!/usr/bin/env python3

import numpy as np
import torch
import math
import time
from gen_prim import gen_path, gen_traj, gen_acc

RENDER_HEIGHT = 360*4
RENDER_WIDTH = 480*4

class Simulator:

    def __init__(self, comp_len, prim_horizon, dt=0.05, alpha=0.2, t_horizon=100, device="cuda"):
        self.t_horizon = t_horizon
        self.comp_len = comp_len
        self.prim_horizon = prim_horizon
        self.device = device
        self.alpha = alpha
        self.dt = dt
        self.prim_lib_x_traj = np.load('primitives/prim_lib_x_traj.npy')
        self.prim_lib_y_traj = np.load('primitives/prim_lib_y_traj.npy')
        self.prim_lib_z_traj = np.load('primitives/prim_lib_z_traj.npy')
        self.prim_lib_v = np.load('primitives/prim_lib_v.npy')
        self.prim_lib_x_acc = np.load('primitives/prim_lib_x_acc.npy')
        self.prim_lib_y_acc = np.load('primitives/prim_lib_y_acc.npy')
        self.prim_lib_z_acc = np.load('primitives/prim_lib_z_acc.npy')

    def simulate_controller(self, env, policy, DepthFilter, gen_new_env=True, 
                            rem_old_env=True, RealTimeSim=False, image_size=50, 
                            plot_line=False, record_vid=False, vid_num=0):
        ''' rem_old_env is not essential, but setting it to False helps in improving
            efficiency when we want to keep running on the same env. It avoids removing
            obstacles when the same ones have to be re-spawned.'''

        # Parameters
        self.plot_line = plot_line

        p = env.p
        p.resetDebugVisualizerCamera(cameraDistance=5., cameraYaw=0., cameraPitch=-25., cameraTargetPosition=[0, 5, 0])

        if RealTimeSim:
            p.setRealTimeSimulation(1)

        cost, collision_cost, goal_cost, traj = self.compute_environment_cost(policy, 
                                                                              DepthFilter, 
                                                                              env, 
                                                                              gen_new_env, 
                                                                              rem_old_env, 
                                                                              image_size,
                                                                              record_vid=record_vid, 
                                                                              vid_num=vid_num)
        
        return cost, collision_cost, goal_cost
    
    def plot_traj(self, p, traj):
        for i in range(len(traj)-1):
            lineFrom = traj[i]
            lineTo = traj[i+1]
            p.addUserDebugLine(lineFrom, lineTo, [1, 0, 0], lineWidth=5)
    
    def compute_primtive_traj(self, prim_id, x0, y0, z0):
        
        x_traj = self.prim_lib_x_traj[prim_id,:] + x0
        y_traj = self.prim_lib_y_traj[prim_id,:] + y0
        z_traj = self.prim_lib_z_traj[prim_id,:] + z0
        
        x_acc = self.prim_lib_x_acc[prim_id,:]
        y_acc = self.prim_lib_y_acc[prim_id,:]
        z_acc = self.prim_lib_z_acc[prim_id,:]
        
        return x_traj, y_traj, z_traj, x_acc, y_acc, z_acc
    
    def compute_environment_cost(self, policy, DepthFilter, env, gen_new_env, rem_old_env, image_size=50, record_vid=False, vid_num=0):
        '''Executes the neural network policy'''

        # Sample environment
        if gen_new_env:
            obs_uid = env.generate_obstacles()
        else:
            obs_uid = env.obsUid
            
        if record_vid:
            import cv2
            # videoObj = cv2.VideoWriter('video'+str(vid_num)+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
                                       # 20, (RENDER_WIDTH, RENDER_HEIGHT))
            videoObj = cv2.VideoWriter('video'+str(vid_num)+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                       20, (RENDER_WIDTH, RENDER_HEIGHT))

        # Initialize position of robot
        state = env.init_position
        quat = env.init_orientation

        env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)
        
        cost = 0.  # cost of image, if we've crashed, can adjust radius
        traj = [state, state]
        # time.sleep(5)

        for i in range(self.comp_len):
            
            # quat = env.init_orientation
            _, depth = self.mount_cam(env, state, quat, h=image_size, w=image_size)
            depth_tensor = torch.Tensor(depth).view([1, 1, image_size, image_size])

            assert depth_tensor.nelement()!=0,"Tensor is empty."
            
            res_out = policy(depth_tensor.to(self.device))[0]
            filtered_depth = DepthFilter(depth_tensor.to(self.device))[0]
            filtered_depth = filtered_depth.view(1,-1)
            score = res_out + filtered_depth 
            index = score.max(1).indices
            r = int(index/5)
            c = int(index%5)
            prim = int(np.abs(r-4)+5*c)

            x_traj, y_traj, z_traj, x_acc, y_acc, z_acc = self.compute_primtive_traj(prim_id=prim, 
                                                                                x0=state[0], 
                                                                                y0=state[1], 
                                                                                z0=state[2])
            
            for t in range(self.prim_horizon):
                
                traj[0] = traj[1]
                state = [x_traj[t], y_traj[t], z_traj[t]]
                traj[1] = state
                
                if t < self.prim_horizon-1:
                    '''
                    Using differential flatness of the quadrotor to compute the roll-pitch trajectory
                    reference: Mellinger, Daniel, and Vijay Kumar. "Minimum snap trajectory generation and control for quadrotors." 
                               IEEE International Conference on Robotics and Automation, 2011.
                               Equation (6)
                    '''
                    # print(x_acc)
                    # normal to the body frame in the global coordinates
                    normal = np.array([x_acc[t], y_acc[t], z_acc[t]+9.8])
                    quat1 = self.vecs2quat([0,0,1], normal)
                    # quat2 is for moving the robot to the X-configuration
                    quat2 = np.array(env.p.getQuaternionFromEuler([0., 0., np.pi/4]))
                    quat = self.quatMult(quat1,quat2)
                else:
                    quat = env.p.getQuaternionFromEuler([0., 0., np.pi/4])

                # Update position of pybullet object
                env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)
                
                if env.gui:
                    env.p.resetDebugVisualizerCamera(cameraDistance=3.0,
                                                      cameraYaw=0.0,
                                                      cameraPitch=-30.0,
                                                      cameraTargetPosition=[state[0], state[1], state[2]])
                    if self.plot_line:
                        self.plot_traj(env.p, traj)

                    if record_vid:
                        
                        # w, h, view_matrix, proj_matrix = env.p.getDebugVisualizerCamera()[0:4]
                        
                        view_matrix = env.p.computeViewMatrixFromYawPitchRoll(
                                                        cameraTargetPosition=[state[0], state[1], state[2]],
                                                        distance=3.0,
                                                        yaw=0.0,
                                                        pitch=-30.0,
                                                        roll=0,
                                                        upAxisIndex=2)
                        proj_matrix = env.p.computeProjectionMatrixFOV(fov=90,
                                                                    aspect=float(RENDER_WIDTH) /
                                                                    RENDER_HEIGHT,
                                                                    nearVal=0.1,
                                                                    farVal=100.0)
                        (_, _, px, _, _) = env.p.getCameraImage(
                            width=RENDER_WIDTH,
                            height=RENDER_HEIGHT,
                            renderer=env.p.ER_BULLET_HARDWARE_OPENGL,
                            viewMatrix=view_matrix,
                            projectionMatrix=proj_matrix)
                        rgb_array = np.array(px)
                        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
                        rgb = rgb_array[:, :, :3]
                        
                        cv2.imshow('Vis_Vid_Rec', cv2.cvtColor(np.uint8(rgb), cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                        videoObj.write(cv2.cvtColor(np.uint8(rgb), cv2.COLOR_RGB2BGR))
                        
                        # sphere = env.p.loadURDF("./URDFs/Sphere/sphere2red.urdf", 
                        #                         basePosition=state,
                        #                         baseOrientation=env.p.getQuaternionFromEuler([0., 0., np.pi/4]),
                        #                         useFixedBase=1,
                        #                         globalScaling=0.02)  # Load robot from URDF                    

                        
                goal_cost = 1. - np.abs(state[1]-1)/15.
               
                # Get closest points
                contact_points_obj = env.p.getClosestPoints(env.quadrotor, obs_uid, 0.0)
                contact_points_ground = env.p.getClosestPoints(env.quadrotor, env.ground, 0.0)
                collision_cost = (1-(i*self.prim_horizon + t)/(self.comp_len*self.prim_horizon-1))
                cost = self.alpha * collision_cost + (1-self.alpha) * goal_cost

                # If collision, then remove obstacles and return the cost
                if contact_points_obj or contact_points_ground:
                    if rem_old_env:
                        env.p.removeBody(obs_uid)
                    if record_vid:
                        videoObj.release()
                    return cost, collision_cost, goal_cost, traj

        # Remove obstacles
        if rem_old_env:
                env.p.removeBody(obs_uid)
        if record_vid:
            videoObj.release()

        return cost, collision_cost, goal_cost, traj
    
    @staticmethod
    def angle(a, b):
        return np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    
    # Find quaternion that rotates x vector to y , not unique since not aligning frames but just normals
    # https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    @staticmethod
    def vecs2quat(x,y):
        out = np.zeros(4)
        out[:3] = np.cross(x, y)
        out[3] = np.linalg.norm(x)*np.linalg.norm(y)+np.dot(x, y)
        if np.linalg.norm(out) < 1e-4:
            return np.append(-x, [0])  # 180 rotation
        return out/np.linalg.norm(out)
    
    # Multiply two quaternions (a,b,c,w)
    @staticmethod
    def quatMult(p, q):
        w = p[3]*q[3] - np.dot(p[:3], q[:3])
        abc = p[3]*q[:3] + q[3]*p[:3] + np.cross(p[:3], q[:3])
        return np.hstack((abc, w))

    def deterministic_prior(self, env, DepthFilter, image_size=50, plot_line=False):
        '''Executes the neural network policy'''

        # Sample environment
        obs_uid = env.obsUid

        # Initialize position of robot
        state = env.init_position
        quat = env.init_orientation

        env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)
        
        cost = 0.  # cost of image, if we've crashed, can adjust radius
        traj = [state, state]

        for i in range(self.comp_len):
            
            _, depth = self.mount_cam(env, state, quat, h=image_size, w=image_size)
            depth_tensor = torch.Tensor(depth).view([1, 1, image_size, image_size])

            assert depth_tensor.nelement()!=0,"Tensor is empty."
            filtered_depth = DepthFilter(depth_tensor.to(self.device))[0]
            filtered_depth = filtered_depth.view(1,-1)
            index = filtered_depth.max(1).indices
            r = int(index/5)
            c = int(index%5)
            prim = int(np.abs(r-4)+5*c)
            x_traj, y_traj, z_traj = self.compute_primtive_traj(prim_id=prim, 
                                                                x0=state[0], 
                                                                y0=state[1], 
                                                                z0=state[2])

            for t in range(self.prim_horizon):
                
                traj[0] = traj[1]
                state = [x_traj[t], y_traj[t], z_traj[t]]
                traj[1] = state
                
                # Update position of pybullet object
                quat = env.p.getQuaternionFromEuler([0., 0., np.pi/4])
                env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)
                
                if env.gui:
                    env.p.resetDebugVisualizerCamera(cameraDistance=3.0,
                                                      cameraYaw=0.0,
                                                      cameraPitch=-30.0,
                                                      cameraTargetPosition=[state[0], state[1], state[2]])
                    
                    if plot_line:
                        self.plot_traj(env.p, traj)

                        
                goal_cost = 1. - np.abs(state[1]-1)/15.
               
                # Get closest points
                contact_points_obj = env.p.getClosestPoints(env.quadrotor, obs_uid, 0.0)
                contact_points_ground = env.p.getClosestPoints(env.quadrotor, env.ground, 0.0)
                collision_cost = (1-(i*self.prim_horizon + t)/(self.comp_len*self.prim_horizon-1))
                # collision_cost = (1-i/(self.comp_len-1))
                cost = self.alpha * collision_cost + (1-self.alpha) * goal_cost

                # If collision, then remove obstacles and return the cost
                if contact_points_obj or contact_points_ground:
                    return cost, collision_cost, goal_cost

        return cost, collision_cost, goal_cost

    
    def generate_image_data(self, policy, env, image_size=50):
        '''Executes the neural network policy'''

        # Sample environment
        obs_uid = env.obsUid

        # Initialize position of robot
        state = env.init_position
        quat = env.init_orientation

        env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)
        
        cost = 0.  # cost of image, if we've crashed, can adjust radius
        depth_data = []

        for i in range(self.comp_len):
            
            _, depth = self.mount_cam(env, state, quat, h=image_size, w=image_size)
            depth_data.append(np.array(depth))
            depth_tensor = torch.Tensor(depth).view([1, 1, image_size, image_size])

            assert depth_tensor.nelement()!=0,"Tensor is empty."
            pd = policy(depth_tensor.to(self.device))[0]
            prim = pd.max(0)[1].to(self.device).item()
            x_traj, y_traj, z_traj = self.compute_primtive_traj(prim_id=prim, 
                                                                x0=state[0], 
                                                                y0=state[1], 
                                                                z0=state[2])

            for t in range(self.prim_horizon):
                
                state = [x_traj[t], y_traj[t], z_traj[t]]
                
                # Update position of pybullet object
                quat = env.p.getQuaternionFromEuler([0., 0., np.pi/4])
                env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)
                
                if env.gui:
                    env.p.resetDebugVisualizerCamera(cameraDistance=3.0,
                                                      cameraYaw=0.0,
                                                      cameraPitch=-30.0,
                                                      cameraTargetPosition=[state[0], state[1], state[2]])
                        
                goal_cost = 1. - np.abs(state[1]-1)/15.
               
                # Get closest points
                contact_points_obj = env.p.getClosestPoints(env.quadrotor, obs_uid, 0.0)
                contact_points_ground = env.p.getClosestPoints(env.quadrotor, env.ground, 0.0)
                collision_cost = (1-(i*self.prim_horizon + t)/(self.comp_len*self.prim_horizon-1))
                cost = self.alpha * collision_cost + (1-self.alpha) * goal_cost

                # If collision, then remove obstacles and return the cost
                if contact_points_obj or contact_points_ground:
                    return cost, collision_cost, goal_cost, depth_data

        return cost, collision_cost, goal_cost, depth_data
    
    def generate_image_data_filter(self, DepthFilter, env, image_size=50):
        '''Executes the neural network policy'''

        # Sample environment
        obs_uid = env.obsUid

        # Initialize position of robot
        state = env.init_position
        quat = env.init_orientation

        env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)
        
        cost = 0.  # cost of image, if we've crashed, can adjust radius
        depth_data = []
        prim_labels = []

        for i in range(self.comp_len):
            
            _, depth = self.mount_cam(env, state, quat, h=image_size, w=image_size)
            depth_data.append(np.array(depth))
            depth_tensor = torch.Tensor(depth).view([1, 1, image_size, image_size])

            assert depth_tensor.nelement()!=0,"Tensor is empty."
            filtered_depth = DepthFilter(depth_tensor.to(self.device))[0]
            filtered_depth = filtered_depth.view(1,-1)
            index = filtered_depth.max(1).indices
            r = int(index/5)
            c = int(index%5)
            prim = int(np.abs(r-4)+5*c)
            prim_labels.append(prim)
            x_traj, y_traj, z_traj = self.compute_primtive_traj(prim_id=prim, 
                                                                x0=state[0], 
                                                                y0=state[1], 
                                                                z0=state[2])

            for t in range(self.prim_horizon):
                
                state = [x_traj[t], y_traj[t], z_traj[t]]
                
                # Update position of pybullet object
                quat = env.p.getQuaternionFromEuler([0., 0., np.pi/4])
                env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)
                
                if env.gui:
                    env.p.resetDebugVisualizerCamera(cameraDistance=3.0,
                                                      cameraYaw=0.0,
                                                      cameraPitch=-30.0,
                                                      cameraTargetPosition=[state[0], state[1], state[2]])
                        
                goal_cost = 1. - np.abs(state[1]-1)/15.
               
                # Get closest points
                contact_points_obj = env.p.getClosestPoints(env.quadrotor, obs_uid, 0.0)
                contact_points_ground = env.p.getClosestPoints(env.quadrotor, env.ground, 0.0)
                collision_cost = (1-(i*self.prim_horizon + t)/(self.comp_len*self.prim_horizon-1))
                cost = self.alpha * collision_cost + (1-self.alpha) * goal_cost

                # If collision, then remove obstacles and return the cost
                # if contact_points_obj or contact_points_ground:
                #     return cost, collision_cost, goal_cost, depth_data, prim_labels

        return cost, collision_cost, goal_cost, depth_data, prim_labels
    
    def visualize_prims(self, prim, env, gen_new_env, rem_old_env, image_size=50):
        '''Executed the neural network policy'''

        # Initialize position of robot
        init_state = [0., 0., 2.]
        state = [0., 0., 2.]
        # initial_heading = [0., 1., -0.2]
        quat = env.p.getQuaternionFromEuler([0., 0., np.pi/4])
        

        # Rotated vectors
        # camera_vector = rot_matrix.dot(init_camera_vector)
        # up_vector = rot_matrix.dot(init_up_vector)


        # ############################################
        # Must match with mount_cam
        
        cam_translate = (0.12,0.12,-0.09) # Shift camera from the base poitions w.r.t. to the link's local frame
        
        rot_matrix = env.p.getMatrixFromQuaternion(quat)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial camera vector relative to the local body frame
        init_camera_vector = (1, 1, -0.11) # x-axis
        # Rotated vectors: initial_heading is the direction in which the camera points
        camera_heading = list(rot_matrix.dot(init_camera_vector))
        cam_pos = list(state + rot_matrix.dot(cam_translate))
        # ############################################
        
        env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)

        traj = [state, state]
        traj_fr = [state, state] #front right
        traj_fl = [state, state] #front left
        traj_br = [state, state] #back right
        traj_bl = [state, state] #back left
        
        translate_fr = [0.3, 0, 0]
        translate_fl = [0, 0.3, 0]
        translate_br = [-0.3, 0, 0]
        translate_bl = [0, -0.3, 0]

        if env.gui:
            env.p.resetDebugVisualizerCamera(cameraDistance=5.0,
                                              cameraYaw=0.0,
                                              cameraPitch=-25.0,
                                              cameraTargetPosition=[state[0], state[1], state[2]])



        x_traj, y_traj, z_traj, x_acc, y_acc, z_acc = self.compute_primtive_traj(prim_id=prim, 
                                                                            x0=state[0], 
                                                                            y0=state[1], 
                                                                            z0=state[2])


        def compute_angle(pos, cam_pos):
            angle = 0
            current_heading = np.array(pos) - np.array(cam_pos)
            current_angle = np.arccos(np.dot(current_heading, camera_heading)
                          / (np.linalg.norm(current_heading)*np.linalg.norm(camera_heading)))
            current_angle *= 180/math.pi
            if angle < current_angle:
                angle = current_angle
            return angle

        for t in range(self.prim_horizon):

            traj[0] = traj[1]
            state = [x_traj[t], y_traj[t], z_traj[t]]
            traj[1] = state
            
            traj_fr[0] = traj[0] + rot_matrix.dot(translate_fr)
            traj_fl[0] = traj[0] + rot_matrix.dot(translate_fl)
            traj_br[0] = traj[0] + rot_matrix.dot(translate_br)
            traj_bl[0] = traj[0] + rot_matrix.dot(translate_bl)

            traj_fr[1] = traj[1] + rot_matrix.dot(translate_fr)
            traj_fl[1] = traj[1] + rot_matrix.dot(translate_fl)
            traj_br[1] = traj[1] + rot_matrix.dot(translate_br)
            traj_bl[1] = traj[1] + rot_matrix.dot(translate_bl)
            
            angle_fr = compute_angle(traj_fr[1], cam_pos)
            angle_fl = compute_angle(traj_fl[1], cam_pos)
            angle_br = compute_angle(traj_br[1], cam_pos)
            angle_bl = compute_angle(traj_bl[1], cam_pos)
            
            angle = max([angle_fl, angle_fr, angle_bl, angle_br])
            
            # self.plot_traj(env.p, traj)
            # self.plot_traj(env.p, traj_fr)
            # self.plot_traj(env.p, traj_fl)
            # self.plot_traj(env.p, traj_br)
            # self.plot_traj(env.p, traj_bl)
            
         
            # Update position of pybullet object
            # quat = env.p.getQuaternionFromEuler([0., 0., np.pi/4])
            # env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)
            
        print("Maximum angle of primitive "+str(prim)+": ", angle)
        
        sphere = env.p.loadURDF("./URDFs/Sphere/sphere2red.urdf", 
                                basePosition=state,
                                baseOrientation=env.p.getQuaternionFromEuler([0., 0., np.pi/4]),
                                useFixedBase=1,
                                globalScaling=0.2)  # Load robot from URDF
        env.p.changeVisualShape(sphere, -1, rgbaColor=[0.54, 0.27, 0.07, 1])

        rgb, depth = self.mount_cam(env, init_state, quat, h=image_size*10, w=image_size*10)


        # visIdxs = env.p.createVisualShape(env.p.GEOM_SPHERE,
        #                                   radius=0.15, 
        #                                   rgbaColor=[0.7, 0, 0, 1])

        
        return angle, rgb, depth
            

    def mount_cam(self, env, base_p, base_o, w=50, h=50):
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

        p = env.p
        cam_pos = base_p

        # Rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(base_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (1, 1, -0.11) # x-axis
        init_up_vector = (0, 0, 1) # z-axis
        cam_translate = (0.12,0.12,-0.09) # Shift camera from the base poitions w.r.t. to the link's local frame

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        cam_pos = cam_pos + rot_matrix.dot(cam_translate)
        view_matrix = p.computeViewMatrix(cam_pos, cam_pos + 0.1 * camera_vector, up_vector)

        # Get Image
        projection_matrix = p.computeProjectionMatrixFOV(fov=130.0, aspect=1., nearVal=0.01, farVal=1000)
        _, _, rgb, depth, _ = p.getCameraImage(w, h, view_matrix, projection_matrix, flags=p.ER_NO_SEGMENTATION_MASK)

        # Reshape rgb image and drop the alpha layer (#4)
        rgb = np.array(rgb, dtype=np.uint8)
        rgb = np.reshape(rgb, (w, h, 4))
        rgb = rgb[:, :, :3]

        # Reshape depth map
        depth = np.array(depth, dtype=np.float32)
        far=1000.0
        near=0.01
        depth = far*near/(far - (far - near)*depth)
        '''
        depth = np.array(depth, dtype=np.float32)
        dmin = np.min(depth)
        dmax = np.max(depth)
        depth -= dmin + (dmax-dmin)/2
        depth /= np.max(depth)
        depth += 1
        depth *= 127
        depth = np.array(depth, dtype=np.uint8)
        '''
        depth = np.reshape(depth, (w, h))

        return rgb, depth

    def mount_360_cam(self, env, base_p, base_o, w=50, h=50):
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

        es = env.p.getEulerFromQuaternion(base_o)

        forward = base_o
        left = env.p.getQuaternionFromEuler([es[0], es[1], es[2]+np.pi/2])
        right = env.p.getQuaternionFromEuler([es[0], es[1], es[2]-np.pi/2])
        back = env.p.getQuaternionFromEuler([es[0], es[1], es[2]-np.pi])

        rbg_f, depth_f = self.mount_cam(env, base_p, forward, w, h)
        rbg_l, depth_l = self.mount_cam(env, base_p, left, w, h)
        rbg_r, depth_r = self.mount_cam(env, base_p, right, w, h)
        rbg_b, depth_b = self.mount_cam(env, base_p, back, w, h)

        return (rbg_f, rbg_l, rbg_b, rbg_r), (depth_f, depth_l, depth_b, depth_r)