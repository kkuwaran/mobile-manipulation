# mobile-manipulation
Pick-n-place task of youBot

This is a software that plans a trajectory for the end-effector of the youBot mobile manipulator (a mobile base with four mecanum wheels and a 5R robot arm), performs odometry as the chassis moves, and performs feedback control to drive the youBot to pick up a block at a specified location, carry it to a desired location, and put it down.

Code Files:
- 'main': the main script for user's inputs
- 'trajectory': the class used to generate a reference end-effector trajectory 
- 'manipulation': the class used to generate a real trajectory (with a feedback controller) based on a given reference trajectory
- 'helper': helping functions

Inputs of 'main' file:
- three default cases are implemented (setting: case = 1, 2, or 3)
	- case = 1 'best': use a well-tuned controller (feedforward-plus-P) to solve a pick-and-place task where the initial and final configurations of the cube are at the default locations.
	- case = 2 'overshoot': use  less-well-tuned controller, one that exhibits overshoot and a bit of oscillation (feedforward-plus-PI) to solve a pick-and-place task where the initial and final configurations of the cube are at the default locations.
	- case = 3 'newTask': use a feedforward-plus-PI controller to solve a pick-and-place task with different initial and final block configurations.
- alternatively, users can specify parameters:
	- cube's initial configuration: 'init_cube_config', 'final_cube_config'
	- robot's inital configuration: 'chassis_config', 'joint_config'
	- control gains: 'kp', 'ki'
	- number of trajectory reference configs per 0.01 seconds: 'k'
	- average speeds for trajectory generation: 'linear_speed', 'angular_speed'
	- planning configurations: 'Tse_init', 'Tce_grasp', 'Tce_standoff'
	- robot's constraints: 'max_joint_speeds', 'max_wheel_speeds', 'joint_limit_flag', 'joint_limit'
	- files' names: 'd_traj_filename', 'r_traj_filename', 'twist_filename'
	- twist error plot: 'ylimit'
Note: the initial block configuration is at (x, y, theta) = (1 m, 0 m, 0 rad) and the final block configuration is at (x, y, theta) = (0 m, -1 m, -pi/2 rad).

Outputs of 'main' file:
	- The CoppeliaSim .csv file of the real trjectory with 0.01s time-step.
	- The error twist 'Xerr' .csv data file.
	- A plot of the six elements of the twist error 'Xerr' as a function of time.

*** Singularity Avoidance ***
- Approach: Ignore any requested twist components in directions that the near-singularity renders difficult to achieve.
- Detail: In 'subTwistToCommand' function in 'manipulation' file, treating small singular values (that are greater than the default tolerance) of the pseudoinverse of the Jacobian 'Je' as zero, this will avoid having pseudoinverse matrices with unreasonably large entries. 

*** Self-Collisions Avoidance ***
- Approach: Implement heuristic joint limits
- Detail: Using the arm joint angle sliders in 'Scene 3: Interactive youBot' to approximately find the joint-angle combinations that avoid self-collision. This joint limits are implemented in 'testJointLimits' function in 'manipulation' file. Then, each time the wheel and arm joint speeds are calculated using the pseudoinverse, use 'testJointLimits' function to check if the new configuration at a time 'dt' later will violate the joint limits. If so, the controls are recalculated by first changing the Jacobian 'Je' to indicate that the offending joint(s) should not be used---the robot must use other joints (if possible) to generate the desired end-effector twist 'V'. To recalculate the controls, each column of 'Je' corresponding to an offending joint is changed to all zeros. This indicates that moving these joints causes no motion at the end-effector, so the pseudoinverse solution will not request any motion from these joints.
