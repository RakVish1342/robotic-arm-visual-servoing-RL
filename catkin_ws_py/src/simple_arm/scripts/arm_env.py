import math
import rospy
import roslaunch
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose 
import subprocess
import start_ros
import numpy as np
from math import pi

class RoboticArm():
	"""docstring for RoboticArm"""
	def __init__(self, max_time_steps = None, dt = None):
		# super(RoboticArm, self).__init__()
		self.max_time_steps = max_time_steps
		self.joint_state = None
		self.box_loc = None
		self.reward = None
		self.dt = dt or 0.5

		rospy.init_node('init_position', anonymous=True)
		self.rate = rospy.Rate(1.0 / self.dt)
		uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
		roslaunch.configure_logging(uuid)
		# self.launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/ravipipaliya/project/robotic-arm/catkin_ws_py/src/simple_arm/launch/robot_spawn.launch"])
		self.launch = roslaunch.parent.ROSLaunchParent(uuid, ["../launch/robot_spawn.launch"])
		self.launch.start()
		self.pub_q1 = rospy.Publisher('/simple_arm/joint_1_position_controller/command',Float64, queue_size=10)
		self.pub_q2 = rospy.Publisher('/simple_arm/joint_2_position_controller/command',Float64, queue_size=10)
		self.sub_joint_state = rospy.Subscriber('/simple_arm/joint_states', JointState, self.cb_joint_state)
		self.sub_box_location = rospy.Subscriber("/simple_arm/box_location", Pose, self.cb_box_location)
		self.pub_reward = rospy.Subscriber('/simple_arm/reward', Float64, self.cb_reward)
		rospy.loginfo("started")
		rospy.sleep(5)

	def cb_joint_state(self, msg):
		self.joint_state = np.asarray(msg.position)

	def cb_box_location(self,msg):
		self.box_loc = np.array([msg.position.x,msg.position.y])

	def cb_reward(self, msg):
		self.reward = msg.data

	def step(self, action):
		"""
		Returns:
		    observation (object): agent's observation of the current environment.
		    reward (float) : amount of reward returned after previous action.
		    done (boolean): whether the episode has ended, in which case further step() calls will return undefined results.
		    info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
		"""
		js_prev = self.joint_state
		js_target = js_prev + action
		self.set_state(action)
		js = self.joint_state

		if abs(js_target[0]) < pi/2 and js_target[1] < pi and js_target[1] > 0: 
			while ((js_target - js)**2).max() > 0.01 and self.reward > -1:
				rospy.sleep(0.1)
				js = self.joint_state
				self.set_state(js_target - js)			
				# print("Error:" + str((js_target - js)[0]) +"\t" + str((js_target - js)[1]))
		else:
			print("Action out of range")
			self.reset(None)
			print(js_target)

		bloc = self.box_loc		
		reward = self.reward
		done = True if reward < -10 else False
		return js, bloc, reward, done

		# raise NotImplementedError

	def reset(self, state=None):
		if state is None:
			state = [0,math.pi/2]
		self.pub_q1.publish(state[0])
		self.pub_q2.publish(state[1])
		# self.rate.sleep()
		rospy.sleep(1)
		return self.box_loc

	def get_state(self):
		return self.joint_state

	def set_state(self, state):
		state = np.asarray(state)
		# if not abs(state.max()) < 0.001:
		q = self.joint_state + np.asarray(state) 
		self.pub_q1.publish(q[0])
		self.pub_q2.publish(q[1])
		# else:
		# 	print("Action skipped due to low change")
		# 	print(state)

	def render(self):
		pass

	def close(self):
		self.launch.shutdown()

	@property
	def action_space(self):
		raise NotImplementedError

	@property
	def observation_space(self):
		raise NotImplementedError



		

