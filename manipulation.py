#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import linalg as LA
import modern_robotics as mr
import matplotlib.pyplot as plt
from helper import F, writeConfiguration

class Manipulation():
    '''Generate real trajectory (using feedback controller) given desired trajectory and gripper states'''
    
    def __init__(self, init_config, trans_matrices, grippers, k=1, kp=0.0, ki=0.0, 
                 controls_limit=None, joint_limit_flag=False, joint_limit=None, 
                 traj_filename='trajectoryR.csv', twist_filename='errortwist.csv'):
        '''init_config: [init_chassis_config, init_joint_config] - initial configuration
                    where init_chassis_config: [phi, x, y] - initial chassis configuration
                          init_joint_config: [J1, J2, J3, J4, J5] - initial joint configuration
           trans_matrices: N * [SE(3)] - trajectory of end-effector in {s}-frame
           grippers: list of length N - gripper states
           k: int - the number of trajectory reference configs per 0.01 seconds
           kp: float - scalar proportional gain matrix
           ki: float - scalar integral gain matrix
           controls_limit: [max_joint_speed, max_wheel_speed] - limit of control inputs
           joint_limit_flag: bool - activate joint limit constraints
           joint_limit: [min_joint_config, max_joint_config] - min and max joint limit
           traj_filename: str - file name for saving generated real trajectory
           twist_filename: str - file name for saving error twists'''
        
        ### Robot's constants ###
        # Fixed offset from the chassis frame {b} to the base frame of the arm {0}
        self.Tb0 = np.array([[ 1, 0, 0, 0.1662],
                             [ 0, 1, 0,      0],
                             [ 0, 0, 1, 0.0026],
                             [ 0, 0, 0,      1]])
        # Home config: End-effector frame {e} relative to the arm base frame {0}
        self.M0e = np.array([[ 1, 0, 0,  0.033],
                             [ 0, 1, 0,      0],
                             [ 0, 0, 1, 0.6546],
                             [ 0, 0, 0,      1]])
        # Screw axes for the five joints are expressed in the end-effector frame {e}
        B1 = [ 0,  0, 1,       0, 0.033, 0]
        B2 = [ 0, -1, 0, -0.5076,     0, 0]
        B3 = [ 0, -1, 0, -0.3526,     0, 0]
        B4 = [ 0, -1, 0, -0.2176,     0, 0]
        B5 = [ 0,  0, 1,       0,     0, 0]
        self.Blist = np.array([B1, B2, B3, B4, B5]).T
        self.n = self.Blist.shape[1]  # number of joints
        # Construct spatial F-matrix for four mecanum wheel robot
        self.F_planar = F()
        self.m = self.F_planar.shape[1]  # number of wheels
        self.F6 = np.zeros((6, self.m))
        self.F6[2:-1] = self.F_planar
        
        ### Trajectory constants ###
        self.trans_matrices = trans_matrices
        self.grippers = grippers
        self.N = len(grippers)   # total number of data points
        self.k = k   # number of data points per 0.01 seconds
        self.dt = 0.01   # time-step (seconds)
        self.Kp = kp * np.eye(6)   # proportional gain matrix
        self.Ki = ki * np.eye(6)   # integral gain matrix
        
        ### Constraints (constant) ###
        # joint speed limit and wheel speed limit
        if controls_limit is not None:
            self.max_joint_speed = np.array(controls_limit[0])
            self.max_wheel_speed = np.array(controls_limit[1])
        else:
            self.max_joint_speed = None
            self.max_wheel_speed = None
        # Activate joint constraints
        self.joint_limit_flag = joint_limit_flag
        # Joint Constraints setting
        if joint_limit is None:
            min_joint_config = [-np.pi/2, np.NINF, np.NINF, np.NINF, np.NINF]
            max_joint_config = [np.pi/2, -0.1, -0.2, -0.2, np.inf]
            joint_limit = [min_joint_config, max_joint_config]
        self.min_joint_config = np.array(joint_limit[0])
        self.max_joint_config = np.array(joint_limit[1])
        assert (self.min_joint_config.shape[0] == self.n and 
                self.max_joint_config.shape[0] == self.n), 'incorrect joint dimension'
        
        ### Miscellaneous constants ###
        self.rcond = 1e-3
        self.decimals = 4
        self.print_period = 10  # print every 10 seconds of traj
        self.period = int(self.print_period // self.dt) * self.k
        
        ### Files name ###
        self.r_traj_filename = traj_filename
        self.twist_filename = twist_filename
        
        ### Storage initialization ###
        self.real_configs = []   # Real configurations
        self.Xerrs = []   # Error Twists
        
        ### Current states ###
        # Current robot configurations
        self.chassis_config = np.array(init_config[0])
        self.joint_config = np.array(init_config[1])
        self.wheel_config = np.array([0] * self.m)
        # Current controls
        self.joint_speed = None
        self.wheel_speed = None
        # Current main variables
        self.X = None  # real config
        self.V = None  # commanded twist
        # Current auxiliary variables
        self.Xerr = None  # twist error
        self.int_Xerr = 0  # integral of twist errors
        self.Teb = None  # body config in {e}-frame
        self.Jb = None   # body Jacobian
        # Current gripper state
        self.gripper = None
        
        
    def plotErrorTwist(self, filename=None, case=None, ylimit=0.1):
        legends = ['wx', 'wy', 'wz', 'vx', 'vy', 'vz']
        titles = ['Case I: best', 'Case II: overshoot', 'Case III: newTask']
        if filename is None: filename=self.twist_filename;
        twist_mat = np.genfromtxt(filename, delimiter=',')
        N, n_dim =  twist_mat.shape
        time = 0.01 * np.arange(N)
        for i in range(n_dim):
            plt.plot(time, twist_mat[:, i], label=legends[i])
        if case in [1, 2, 3]:
            plt.title(titles[case-1])
        plt.xlabel('Time (s)')
        plt.ylabel('Error Twist Xerr')
        plt.ylim([-ylimit, ylimit])
        plt.legend()
        plt.grid()
        plt.show()
    
    
    def testJointLimits(self, joint_config):
        '''Return a list of joint limits that are violated given the robot arm's configuration'''

        min_violations = joint_config < self.min_joint_config
        max_violations = joint_config > self.max_joint_config
        return np.logical_or(min_violations, max_violations)

    
    def addConfiguration(self):
        '''Transform [[phi, x, y], [J1, J2, J3, J4, J5], [W1, W2, W3, W4]] format into
        [phi, x, y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper] format'''

        # Convert chassis_config, joint_config and wheel_config to lists
        chassis = np.around(self.chassis_config, self.decimals).tolist()
        joint = np.around(self.joint_config, self.decimals).tolist()
        wheel = np.around(self.wheel_config, self.decimals).tolist()
        # Append configuration (+ gripper state)
        self.real_configs.append(chassis + joint + wheel + [self.gripper])

        
    def NextState(self):
        '''Compute the configuration [chassis_config, joint_config, wheel_config] of the robot at "dt" later'''

        # Clip joint_speed and wheel_speed
        if self.max_joint_speed is not None:
            self.joint_speed = np.clip(self.joint_speed, a_min=-self.max_joint_speed, a_max=self.max_joint_speed)
        if self.max_wheel_speed is not None:
            self.wheel_speed = np.clip(self.wheel_speed, a_min=-self.max_wheel_speed, a_max=self.max_wheel_speed)

        # First-order Euler step (joint angles, wheel angles)
        self.joint_config = self.joint_config + self.joint_speed * self.dt
        self.wheel_config = self.wheel_config + self.wheel_speed * self.dt

        # First-order Euler step (chassis config) using wheel_speeds
        Vb = self.F_planar @ self.wheel_speed
        Vb6 = np.zeros(6); Vb6[2:-1] = Vb;
        T = mr.MatrixExp6(mr.VecTose3(Vb6) * self.dt)
        self.chassis_config = self.chassis_config + mr.se3ToVec(T)[2:-1]
        

    def NextStatesConstantControls(self, controls, gripper=0, time=1.0, filename='NextStateTest.csv'):
        '''Supplement: Compute the robot's configurations (given constant controls)
        [chassis_config, joint_config, wheel_config, gripper] for 'time' seconds'''

        N = int(time // self.dt) + 1
        self.joint_speed = np.array(controls[0]) 
        self.wheel_speed = np.array(controls[1])
        self.gripper = gripper
        # Calculate subsequent configurations
        for i in range(N):
            self.NextState()
            self.addConfiguration()
        # Write a csv file from 'real_configs'
        writeConfiguration(self.real_configs, filename)
    

    def ConfigToEndEffector(self):
        ''' 1. Calculate end-effector configuration in SE(3) from chassis and arm configurations: X (or Tse)
            2. Calculate transformation matrix in SE(3) from frame {b} to frame {e}: Teb
            3. Calculate body Jacobian given joint angles: Jb'''

        # chassis (phi, x, y)
        phi, x, y = self.chassis_config
        # Calculate config of {b} of the mobile base relative to frame {s}
        Tsb = np.array([[np.cos(phi), -np.sin(phi), 0,      x],
                        [np.sin(phi),  np.cos(phi), 0,      y],
                        [          0,            0, 1, 0.0963],
                        [          0,            0, 0,      1]])
        # Calculate end-effector config relative to the arm base frame {0}
        T0e = mr.FKinBody(self.M0e, self.Blist, self.joint_config)
        # Calculate end-effector config relative to frame {s} (i.e., X), and Teb
        Tbe = self.Tb0 @ T0e
        self.X = Tsb @ Tbe
        self.Teb = LA.inv(Tbe)
        # Calculate Body Jarcobian Jb
        self.Jb = mr.JacobianBody(self.Blist, self.joint_config)


    def subTwistToCommand(self, Je):
        ''' 1. Calculate commanded speeds [wheel_speeds, joint_speeds] given twist V and Jacobian Je
            2. Determine joint indices subject to joint constraint violation (in testJointLimits)'''

        speeds = LA.pinv(Je, self.rcond) @ self.V
        ##print('Commanded Speed:', np.around(speeds, decimals=2))
        wheel_speeds, joint_speeds = speeds[:self.m], speeds[self.m:]
        joint_config_next = self.joint_config + joint_speeds * self.dt
        ##print('Next Joint Config:', np.around(joint_config_next, decimals=2))
        violations = self.testJointLimits(joint_config_next)
        ##print('violations:', violations)
        violation_indices = [self.m + i for i, violation in enumerate(violations) if violation]
        return [joint_speeds, wheel_speeds], violation_indices


    def TwistToCommand(self):
        '''Turn the commanded end-effector twist V expressed in the end-effector frame {e} 
        into commanded [joint_speeds, wheel_speeds]'''

        # Calculate Jacobian for wheel speeds J_base
        Jbase = mr.Adjoint(self.Teb) @ self.F6
        # Calculate Jacobian: Je = [Jbase, Jb]
        Je = np.concatenate((Jbase, self.Jb), axis=1)
        ##print('Jacobian:\n', np.around(Je, decimals=3))
        # Calculate commanded speeds given end-effector twist V
        # If flag = True, consider joint constraints
        speeds, violation_indices = self.subTwistToCommand(Je)
        acc_violation_indices = violation_indices
        while self.joint_limit_flag and violation_indices:
            Je_temp = Je.copy()
            Je_temp[:, acc_violation_indices] = np.zeros((Je.shape[0], len(acc_violation_indices)))
            speeds, violation_indices = self.subTwistToCommand(Je_temp)
            acc_violation_indices = [i for i in range(self.m + self.n) if i in acc_violation_indices or i in violation_indices]
        self.joint_speed, self.wheel_speed = speeds


    def FeedbackControl(self, i):
        '''Calculate the kinematic task-space feedforward plus feedback control law
        V(t) = [Ad_{Xinv Xd}] Vd(t) + Kp Xerr(t) + Ki \int_0^t Xerr(t) dt'''
        
        # Extract current and next reference state, and gripper state
        Xd, Xd_next = self.trans_matrices[i], self.trans_matrices[i+1]
        self.gripper = self.grippers[i]

        # Calculate feedforward reference twist Vd that takes Xd to Xd_next in time dt:
        # [Vd] = (1/dt) * log( inv(Xd) Xd_next )
        Vd = 1/self.dt * mr.se3ToVec(mr.MatrixLog6(LA.inv(Xd) @ Xd_next))
        ##print('Reference Twist:', np.around(Vd, self.decimals))
        # Calculate feedforward reference twist Vd expressed in end-effector frame {e}
        Vd_transfrom = mr.Adjoint(LA.inv(self.X) @  Xd) @ Vd
        ##print('Transformed Reference Twist:', np.around(Vd_transfrom, self.decimals))
        # Calculate error twist: [Xerr] = log(Xinv Xd)
        self.Xerr = mr.se3ToVec(mr.MatrixLog6(LA.inv(self.X) @ Xd))
        ##print('Error Twist:', np.around(self.Xerr, self.decimals))
        # Calculate running total of the integral: \int_0^t Xerr(t) dt
        self.int_Xerr += self.Xerr * self.dt
        # Calculate commanded twist V
        self.V = Vd_transfrom + self.Kp @ self.Xerr + self.Ki @ self.int_Xerr
        ##print('Commanded Twist:', np.around(self.V, self.decimals))


    def ManipulationControl(self):
        '''Generate real trajectory (in csv) using feedback control given desired trajectory'''
        
        print('\n*** Start Manipulation Control ***')
        print('Total number of configurations:', self.N)
        # Manipulation control start
        self.ConfigToEndEffector()   
        for i in range(self.N-1):
            # Calculate commanded twist given reference config
            self.FeedbackControl(i)
            # Calculate commanded speeds from the twist
            self.TwistToCommand()
            # Calculate next robot's configuration using odometry
            self.NextState()
            # Calculate next real end-effector configuration
            self.ConfigToEndEffector()
            
            # Store every k-th (i) real configuration for later animation, (ii) Xerr for plotting the evolution of error
            # Note: the reference trajectory has k reference configurations per 0.01 second step
            if i % self.k == 0:
                self.addConfiguration()
                self.Xerrs.append(self.Xerr.tolist())
            if i != 0 and i % self.period == 0:
                print('Real Trajectory: {} seconds'.format(str(self.print_period * (i // self.period))))
        print('Actual Trajectory Generated!')
        
        # Save 'real_configs' and 'Xerrs' to csv file
        writeConfiguration(self.real_configs, self.r_traj_filename)
        writeConfiguration(self.Xerrs, self.twist_filename)
        print('Writing error plot data. \nDone!')

    
# # =============== FUNCTIONS TEST (NextState) ===============
# # Default Settings
# chassis_config = [0, 0, 0]
# joint_config = [0, 0, 0, 0, 0]
# init_config = [chassis_config, joint_config]
# joint_speeds = [5, 5, 5, 5, 5]

# # Test Case I
# wheel_speeds = [10, 10, 10, 10]
# controls1 = [joint_speeds, wheel_speeds]
# max_speeds1 = None

# # Test Case II
# wheel_speeds = [-10, 10, -10, 10]
# controls2 = [joint_speeds, wheel_speeds]
# max_speeds2 = None

# # Test Case III
# wheel_speeds = [-10, 10, 10, -10]
# controls3 = [joint_speeds, wheel_speeds]
# max_speeds3 = None

# # Test Case IV
# wheel_speeds = [10, 10, 10, 10]
# controls4 = [joint_speeds, wheel_speeds]
# max_speeds4 = [[2.5] * 5, [5] * 4]

# # Test Execution (test_case 1-4)
# test_case = 4
# controls_case = [controls1, controls2, controls3, controls4]
# max_speeds_case = [max_speeds1, max_speeds2, max_speeds3, max_speeds4]
# manipulation = Manipulation(init_config, None, [], controls_limit=max_speeds_case[test_case-1])
# manipulation.NextStatesConstantControls(controls_case[test_case-1])


# # =============== FUNCTIONS TEST (testJointLimits, TwistToCommand) ===============
# # Parameters
# chassis_config = [0, 0, 0]
# joint_config = [0, 0, 0.2, -1.6, 0]
# init_config = [chassis_config, joint_config]

# Xd = np.array([[  0, 0, 1, 0.5],
#                [  0, 1, 0,   0],
#                [ -1, 0, 0, 0.5],
#                [  0, 0, 0,   1]])

# Xd_next = np.array([[  0, 0, 1, 0.6],
#                     [  0, 1, 0,   0],
#                     [ -1, 0, 0, 0.3],
#                     [  0, 0, 0,   1]])

# trans_matrices = [Xd, Xd_next]
# grippers = [0, 0]

# ### Test 'TwistToCommand' Function ###
# # Test Case I
# kp = 0.0
# manipulation = Manipulation(init_config, trans_matrices, grippers, kp=kp)
# manipulation.ConfigToEndEffector()
# manipulation.FeedbackControl(0)
# manipulation.TwistToCommand()
# print()
# # Test Case II
# kp = 1.0
# manipulation = Manipulation(init_config, trans_matrices, grippers, kp=kp)
# manipulation.ConfigToEndEffector()
# manipulation.FeedbackControl(0)
# manipulation.TwistToCommand()
# print()
# ### Test 'testJointLimits' Function ###
# joint_config = np.random.randn(5)
# vios = manipulation.testJointLimits(joint_config)
# print('Joint Config {}; Violation {}'.format(joint_config, vios))

