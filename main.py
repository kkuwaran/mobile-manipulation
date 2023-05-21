#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from helper import Tsc
from trajectory import TrajectoryGeneratorClass
from manipulation import Manipulation

# Test case setting
# For customized case, set case = 0 and specify 
# 'init_cube_config', 'final_cube_config', 'kp', 'ki'
case = 2

# ==================== Default Test Cases ====================
# Parameters setting for each default test case
#   - init_cube_config: initial cube configuration (x, y, theta)
#   - final_cube_config: final cube configuration (x, y, theta)
#   - k: no. of trajectory reference configs per 0.01 seconds
#   - kp: scalar proportional gain
#   - ki: scalar integral gain
if case == 1:  # Case I: "best"
    init_cube_config = (1, 0, 0)
    final_cube_config = (0, -1, -np.pi/2)
    k = 1
    kp = 2.50
    ki = 0.00
elif case == 2:  # Case II: "overshoot"
    init_cube_config = (1, 0, 0)
    final_cube_config = (0, -1, -np.pi/2)
    k = 1
    kp = 1.50
    ki = 8.00
elif case == 3:  # Case III: "newTask"
    init_cube_config = (1, 0.5, -np.pi/4)
    final_cube_config = (0, -1, -np.pi/2)
    k = 2
    kp = 3.00
    ki = 0.50

# ==================== Default Robot's Initial Configuration ====================
chassis_config = [np.pi/6, -0.5, 0]
joint_config = [0, -0.2, -0.3, -1.6, 0]

# ==================== Default Planning ====================
# Planning configurations
linear_speed=0.12
angular_speed=0.20

Tse_init = np.array([[ 0, 0, 1,   0], 
                     [ 0, 1, 0,   0],
                     [-1, 0, 0, 0.5],
                     [ 0, 0, 0,   1]])

Tce_grasp = np.array([[-0.707, 0,  0.707,     0], 
                      [     0, 1,      0,     0],
                      [-0.707, 0, -0.707, -0.01],
                      [     0, 0,      0,     1]])

Tce_standoff = np.array([[-0.707, 0,  0.707,    0], 
                         [     0, 1,      0,    0],
                         [-0.707, 0, -0.707, 0.25],
                         [     0, 0,      0,    1]])

# ==================== Default Robot's Constraints ====================
max_joint_speeds = 5 * [2 * np.pi]   # joint speeds limit
max_wheel_speeds = 4 * [4 * np.pi]   # wheel speeds limit
joint_limit_flag = True   # enforce joint constraints
joint_limit = None   # use default values

# ==================== Default File Names ====================
d_traj_filename = 'trajectoryD.csv'
r_traj_filename = 'trajectoryR.csv'
twist_filename = 'errortwist.csv'

# ==================== Default Error Plotting ====================
ylimit = 0.5


# ==================== Execution ====================
case_name = ['I: best', 'II: overshoot', 'III: newTask']
print('runscript --- case {} ---'.format(case_name[case-1]))
# Cube configurations
Tsc_init = Tsc(init_cube_config[0], init_cube_config[1], init_cube_config[2])
Tsc_final = Tsc(final_cube_config[0], final_cube_config[1], final_cube_config[2])
# Initial robot's configuration
init_config = [chassis_config, joint_config]
# Constraints
controls_limit = [max_joint_speeds, max_wheel_speeds]

# Generate Trajectory 
traj = TrajectoryGeneratorClass(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff, 
                                k, d_traj_filename, linear_speed, angular_speed)
trans_matrices, grippers = traj.readTrajectory()
manipulation = Manipulation(init_config, trans_matrices, grippers, k, kp, ki, 
                            controls_limit, joint_limit_flag, joint_limit, 
                            r_traj_filename, twist_filename)
manipulation.ManipulationControl()
manipulation.plotErrorTwist(twist_filename, case, ylimit)


# In[ ]:




