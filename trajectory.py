#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
from numpy import linalg as LA
import modern_robotics as mr
from helper import Tsc, writeConfiguration

class TrajectoryGeneratorClass():
    '''Generate full reference trajectory of end-effector picking and dropping cube'''
    
    def __init__(self, Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff, k=1, 
                 filename='trajectory.csv', linear_speed=0.05, angular_speed=0.08):
        '''Tse_init: SE(3) - initial end-effector config in {s}-frame
           Tsc_init: SE(3) - initial cube config in {s}-frame
           Tsc_final: SE(3) - final cube config in {s}-frame
           Tce_grasp: SE(3) - end-effector config when grasping cube in cube-frame
           Tce_standoff: SE(3) - end-effector standoff config before grasping cube in cube-frame
           k: int - the number of trajectory reference configs per 0.01 seconds
           filename: str - file name for saving trajectory of end-effector configs
           linear_speed: float - average linear speed of configs
           angular_speed: float - average angular speed of configs'''
        
        self.method = 5  # Default: quintic time scaling
        self.decimals = 5  # Default: 4 decimal places
        
        self.configs = []  # Initialize configs storage
        self.k = k
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.filename = filename
        self.TrajectoryGenerator(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff)
        
        # Save configs
        writeConfiguration(self.configs, self.filename)
        print('Reference Trajectory Generated!')

    def readTrajectory(self):
        '''Read trajectories [r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper]
        from csv file and reformat to SE(3) transformation matrices'''

        trans_matrices = []
        grippers = []
        with open(self.filename, mode='r') as csvfile:
            csvreader = csv.reader(csvfile)
            for lines in csvreader:
                # Retrive rotation and position parts
                rotation = np.array([float(elem) for elem in lines[:9]])
                rotation = rotation.reshape((3, 3))
                position = np.array([float(elem) for elem in lines[9:-1]])
                # Reformat to transformation matrix
                trans_matrices.append(mr.RpToTrans(rotation, position))
                grippers.append(int(lines[-1]))
        return trans_matrices, grippers

    def TrajectoryTime(self, T_init, T_final):
        '''Calculate appropriate trajectory time given average speeds'''

        # Extract rotation and position from SE(3)
        R_init, p_init = mr.TransToRp(T_init)
        R_final, p_final = mr.TransToRp(T_final)
        # Calculate linear time
        linear_dist = LA.norm(p_init - p_final)
        linear_time = linear_dist / self.linear_speed
        # Calculate angular time [Note: chordal distance = | Ri - Rf |^2_F = 2 sqrt(2) * sin(theta/2)]
        chordal_dist = LA.norm(R_init - R_final, 'fro') ** 2
        angular_time = chordal_dist / self.angular_speed
        # Determine maximum time and round to nearest 0.01
        #print('Time:', linear_time, angular_time)
        time = np.maximum(linear_time, angular_time)
        return np.around(time, decimals=2)

    def addTrajectory(self, trajectories, gripper):
        '''Transform SE(3) end-effector trajectories into 
        [r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper] format'''

        for X in trajectories:
            rot_vec = np.around(np.ravel(X[:-1, :-1]), self.decimals)
            pos_vec = np.around(X[:-1, -1], self.decimals)
            config = rot_vec.tolist() + pos_vec.tolist() + [gripper]
            self.configs.append(config)

    def SubTrajectoryGenerator(self, T_init, T_final, gripper, traj_type, penalty=1.0):
        '''Generate sub-trajectory and add it to configs'''

        # Calculate total_time, time between configs, and total data points
        # Note: opening and closing the gripper takes up to 0.625 seconds
        total_time = 1.00 if np.array_equal(T_init, T_final) else self.TrajectoryTime(T_init, T_final)
        total_time *= penalty
        print('Sub-trajectory Time:', total_time)
        dt = 0.01 / self.k
        N = total_time // dt
        # Generate sub-trajectory
        if traj_type == 'Cartesian':
            traj = mr.CartesianTrajectory(T_init, T_final, dt, N, self.method)
        else:
            traj = mr.ScrewTrajectory(T_init, T_final, dt, N, self.method)
        # Add sub-trajectory to configs
        self.addTrajectory(traj, gripper)

    def TrajectoryGenerator(self, Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff):
        '''Main function for generating reference trajectory of end-effector'''
        
        print('\n*** Start Generate Trajectory ***')
        # Trajectory 1: Move the gripper from its initial config to a "standoff" config
        print('Sub-trajectory 1: initial to standoff')
        Tse_standoff_i = Tsc_init @ Tce_standoff
        self.SubTrajectoryGenerator(Tse_init, Tse_standoff_i, 0, 'Cartesian', 2.2)
        # Trajectory 2: Move the gripper down to the grasp position
        print('Sub-trajectory 2: standoff to grasp')
        Tse_grasp = Tsc_init @ Tce_grasp
        self.SubTrajectoryGenerator(Tse_standoff_i, Tse_grasp, 0, 'Cartesian', 2.0)
        # Trajectory 3: Closing of the gripper
        print('Sub-trajectory 3: close gripper')
        self.SubTrajectoryGenerator(Tse_grasp, Tse_grasp, 1, 'Cartesian')
        # Trajectory 4: Move the gripper back up to the "standoff" config
        print('Sub-trajectory 4: grasp to standoff')
        self.SubTrajectoryGenerator(Tse_grasp, Tse_standoff_i, 1, 'Cartesian')  
        # Trajectory 5: Move the gripper to a "standoff" config above the final config
        print('Sub-trajectory 5: standoff to above final')
        Tse_standoff_f = Tsc_final @ Tce_standoff
        self.SubTrajectoryGenerator(Tse_standoff_i, Tse_standoff_f, 1, 'Cartesian', 0.8)
        # Trajectory 6: Move the gripper to the final config of the object
        print('Sub-trajectory 6: above final to final')
        Tse_release = Tsc_final @ Tce_grasp
        self.SubTrajectoryGenerator(Tse_standoff_f, Tse_release, 1, 'Cartesian')
        # Trajectory 7: Opening of the gripper
        print('Sub-trajectory 7: open gripper')
        self.SubTrajectoryGenerator(Tse_release, Tse_release, 0, 'Cartesian')
        # Trajectory 8: Move the gripper back to the "standoff" config
        print('Sub-trajectory 8: final to standoff')
        self.SubTrajectoryGenerator(Tse_release, Tse_standoff_f, 0, 'Cartesian')
        
        
        
# # =============== FUNCTIONS TEST ===============
# # Configurations setting
# Tse_init = np.array([[ 0, 0, 1, 0.5], 
#                      [ 0, 1, 0,   0],
#                      [-1, 0, 0, 0.5],
#                      [ 0, 0, 0,   1]])
    
# Tsc_init = Tsc(1, 0, 0)

# Tsc_final = Tsc(0, -1, -np.pi/2)

# Tce_grasp = np.array([[-0.707, 0,  0.707, 0], 
#                       [     0, 1,      0, 0],
#                       [-0.707, 0, -0.707, 0],
#                       [     0, 0,      0, 1]])

# Tce_standoff = np.array([[-0.707, 0,  0.707,    0], 
#                          [     0, 1,      0,    0],
#                          [-0.707, 0, -0.707, 0.25],
#                          [     0, 0,      0,    1]])

# # Generate Trajectory 
# TrajectoryGeneratorClass(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff)


# In[ ]:




