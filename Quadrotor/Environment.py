#!/usr/bin/env python3

import os

# SUPPRESS PRINTING
null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
save = os.dup(1), os.dup(2)
os.dup2(null_fds[0], 1)
os.dup2(null_fds[1], 2)

import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np
import time

# ENABLE PRINTING
os.dup2(save[0], 1)
os.dup2(save[1], 2)
os.close(null_fds[0])
os.close(null_fds[1])

class Environment:

    def __init__(self, r_lim, num_obs, parallel=False, gui=False, x_min=-5.0, x_max=5.0, y_min=0.0, y_max=10.0):
        self.parallel = parallel
        self.gui = gui

        self.height_obs = 5
        self.robot_radius = 0.3
        self.r_lim = r_lim
        self.num_obs = num_obs

        # Set goal for this environment
        self.xG = np.array([[0.,15.]])
        # self.xG = (np.random.rand(1,2) -0.5)*[20,0]+ [0.,10.]

        self.x_lim = [x_min, x_max]
        self.y_lim = [y_min, y_max]

        self.p = None
        self.husky = None
        self.sphere = None
        self.setup_pybullet()

    def setup_pybullet(self):

        if self.parallel:
            if self.gui:
                print("Warning: Can only have one thread be a gui")
                p = bc.BulletClient(connection_mode=pybullet.GUI)
            else:
                p = bc.BulletClient(connection_mode=pybullet.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        else:
            if self.gui:
                pybullet.connect(pybullet.GUI)
                p = pybullet
                # This just makes sure that the sphere is not visible (we only use the sphere for collision checking)
            else:
                pybullet.connect(pybullet.DIRECT)
                p = pybullet

        p.setGravity(0, 0, 0)
        self.ground = p.loadURDF("./URDFs/plane.urdf")  # Ground plane
        # p.changeVisualShape(self.ground, -1, rgbaColor=[0.9,0.9,0.9,0.7])
        self.init_position = [0, 0, 2]
        self.init_orientation = p.getQuaternionFromEuler([0., 0., np.pi/4])
        quadrotor = p.loadURDF("./URDFs/Quadrotor/quadrotor.urdf", 
                               basePosition=self.init_position,
                               baseOrientation=self.init_orientation,
                               useFixedBase=1, 
                               globalScaling=1)  # Load robot from URDF
        p.changeVisualShape(quadrotor, -1, rgbaColor=[0.5,0.5,0.5,1])
        
        self.p = p
        self.quadrotor = quadrotor

    def set_gui(self, gui):
        self.p.disconnect()
        self.gui = gui
        self.setup_pybullet()

    def generate_obstacles(self):
        p = self.p
        numObs = self.num_obs

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
            visIdxs[obs] = -1
            parentIdxs[obs] = 0
            linkInertialFramePositions[obs] = [0, 0, 0]
            linkInertialFrameOrientations[obs] = [0, 0, 0, 1]
            linkJointTypes[obs] = p.JOINT_FIXED
            linkJointAxis[obs] = np.array([0, 0, 1])
            if obs < 3:
                orientObs[obs] = [0, 0, 0, 1]
            else:
                # orientObs[obs] = list(np.random.randn(4)*0.1)
                orientObs[obs] = [(np.random.randn(1)[0]), 0, (np.random.randn(1)[0]), 1]

        posObs, colIdxs, visIdxs = self.generate_obstacles_sub(p, posObs, colIdxs, visIdxs)

        obsUid = p.createMultiBody(baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1, basePosition=[0, 0, 0],
                                   baseOrientation=[0, 0, 0, 1], baseInertialFramePosition=[0, 0, 0],
                                   baseInertialFrameOrientation=[0, 0, 0, 1], linkMasses=linkMasses,
                                   linkCollisionShapeIndices=colIdxs, linkVisualShapeIndices=visIdxs,
                                   linkPositions=posObs, linkOrientations=orientObs, linkParentIndices=parentIdxs,
                                   linkInertialFramePositions=linkInertialFramePositions,
                                   linkInertialFrameOrientations=linkInertialFrameOrientations,
                                   linkJointTypes=linkJointTypes, linkJointAxis=linkJointAxis)
        
        return obsUid

    def generate_obstacles_sub(self, p, posObs, colIdxs):
        numObs = self.num_obs
        heightObs = self.height_obs
        rmin = self.r_lim[0]
        rmax = self.r_lim[1]
        xmin = self.x_lim[0]
        xmax = self.x_lim[1]
        ymin = self.y_lim[0]
        ymax = self.y_lim[1]

        for obs in range(numObs):  # Cylindrical obstacles
            posObs_obs = np.array([None] * 3)
            posObs_obs[0] = xmin + (xmax - xmin) * np.random.random_sample(1)
            posObs_obs[1] = ymin + (ymax - ymin) * np.random.random_sample(1)
            posObs_obs[2] = 0  # set z at ground level
            posObs[obs] = posObs_obs
            radiusObs = rmin + (rmax - rmin) * np.random.random_sample(1)
            colIdxs[obs] = p.createCollisionShape(p.GEOM_CYLINDER, radius=radiusObs, height=heightObs)

        return posObs, colIdxs
    
    def generate_safe_initial_env(self, min_dist=1.0):
        gen_obs_flag = True
        while gen_obs_flag:
            self.obsUid = self.generate_obstacles()
            closest_points = self.p.getClosestPoints(self.quadrotor, self.obsUid, min_dist)
            if not closest_points:
                gen_obs_flag = False
            else:
                self.p.removeBody(self.obsUid)

class TestEnvironment(Environment):

    def __init__(self, r_lim, num_obs, parallel=False, gui=False, x_min=-5.0, x_max=5.0, y_min=0.0, y_max=10.0, safecircle=2.):
        Environment.__init__(self, r_lim, num_obs, parallel, gui, x_min, x_max, y_min, y_max)
        self.safecircle = safecircle
        # self.obsUid = self.generate_obstacles()

    def generate_obstacles_sub(self, p, posObs, colIdxs, visIdxs):
        numObs = self.num_obs
        heightObs = self.height_obs
        rmin = self.r_lim[0]
        rmax = self.r_lim[1]
        xmin = self.x_lim[0]
        xmax = self.x_lim[1]
        ymin = self.y_lim[0]
        ymax = self.y_lim[1]
        
        color_lib = [[0.8,0.8,0,1], [0,0.7,0,1], [0,0.5,1,1], [0.7,0,0,1]]
        
        ht_wall = 4
        wall_thickness = 0.2

        x_wall = (xmin+xmax)/2
        y_wall = (ymin+ymax)/2
        posObs_obs = np.array([None] * 3)
        posObs_obs[0] = x_wall
        posObs_obs[1] = y_wall
        posObs_obs[2] = ht_wall + 0.1  # set z at ground level
        posObs[0] = posObs_obs
        colIdxs[0] = p.createCollisionShape(p.GEOM_BOX, halfExtents=[(xmax-xmin)/2,(ymax-ymin)/2,0.1])
        visIdxs[0] = p.createVisualShape(p.GEOM_BOX, 
                                         halfExtents=[(xmax-xmin)/2+0.1,(ymax-ymin)/2,wall_thickness/2], #[(xmax-xmin)/2,0.1,ht_wall] 
                                         rgbaColor=[0.7, 0, 0, 0.01])

        x_wall = xmin
        y_wall = (ymin+ymax)/2
        # ht_wall = 1
        posObs_obs = np.array([None] * 3)
        posObs_obs[0] = x_wall
        posObs_obs[1] = y_wall
        posObs_obs[2] = ht_wall/2  # set z at ground level
        posObs[1] = posObs_obs
        colIdxs[1] = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness/2,(ymax-ymin)/2,ht_wall/2])
        visIdxs[1] = p.createVisualShape(p.GEOM_BOX, 
                                         halfExtents=[wall_thickness/2,(ymax-ymin)/2,ht_wall/2],
                                         rgbaColor=[0.7, 0, 0, 1])

        x_wall = xmax
        y_wall = (ymin+ymax)/2
        # ht_wall = 1
        posObs_obs = np.array([None] * 3)
        posObs_obs[0] = x_wall
        posObs_obs[1] = y_wall
        posObs_obs[2] = ht_wall/2  # set z at ground level
        posObs[2] = posObs_obs
        colIdxs[2] = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness/2,(ymax-ymin)/2,ht_wall/2])
        visIdxs[2] = p.createVisualShape(p.GEOM_BOX, 
                                         halfExtents=[wall_thickness/2,(ymax-ymin)/2,ht_wall/2], 
                                         rgbaColor=[0.7, 0, 0, 1])

        for obs in range(3,numObs):  # Cylindrical obstacles
            posObs_obs = np.array([None] * 3)
            x_temp = 0
            y_temp = 1
            radiusObs = 0
            
            x_temp = xmin + (xmax - xmin) * np.random.random_sample(1)
            y_temp = ymin + (ymax - ymin) * np.random.random_sample(1)
            radiusObs = rmin + (rmax - rmin) * np.random.random_sample(1)
            
            posObs_obs[0] = x_temp
            posObs_obs[1] = y_temp
            posObs_obs[2] = 0  # set z at ground level
            posObs[obs] = posObs_obs
            # radiusObs = rmin + (rmax - rmin) * np.random.random_sample(1)
            colIdxs[obs] = p.createCollisionShape(p.GEOM_CYLINDER, 
                                                  radius=radiusObs, 
                                                  height=2*heightObs)

            cylinder_color = color_lib[np.random.randint(low=0, high=4)]
            visIdxs[obs] = p.createVisualShape(pybullet.GEOM_CYLINDER, 
                                               radius=radiusObs, 
                                               length=2*heightObs,
                                               rgbaColor=cylinder_color)

        return posObs, colIdxs, visIdxs
