#!/usr/bin/env python

#robot.py implementation goes here

import rospy
import numpy as np
from math import *
from read_config import read_config
from astar import *
from mdp import *
from tfmdp import *
from mfmdp import *
from cse_190_assi_3.msg import *
from std_msgs.msg import Bool

class Robot():
	def __init__(self):
		rospy.init_node("robot")
		self.config = read_config()
		self.move_list = self.config["move_list"]
		self.map_size = self.config["map_size"]
		self.start = self.config["start"]
		self.goal = self.config["goal"]
		self.walls = self.config["walls"]
		self.pits = self.config["pits"]
		self.max_iterations = self.config["max_iterations"]
		self.threshold_difference = self.config["threshold_difference"]
		self.reward_for_each_step = self.config["reward_for_each_step"]
		self.reward_for_hitting_wall = self.config["reward_for_hitting_wall"]
		self.reward_for_reaching_goal = self.config["reward_for_reaching_goal"]
		self.reward_for_falling_in_pit = self.config["reward_for_falling_in_pit"]
		self.discount_factor = self.config["discount_factor"]
		self.prob_move_forward = self.config["prob_move_forward"]
		self.prob_move_backward = self.config["prob_move_backward"]
		self.prob_move_left = self.config["prob_move_left"]
		self.prob_move_right = self.config["prob_move_right"]
		self.generate_video = self.config["generate_video"]
		self.astar_pub = rospy.Publisher("/results/path_list", AStarPath, queue_size = 100);
		self.mdp_pub = rospy.Publisher("/results/policy_list", PolicyList, queue_size = 100);
		self.sim_pub = rospy.Publisher("/map_node/sim_complete", Bool, queue_size = 10);
		

		self.qmdpPolicy = tdmdp();
		rospy.sleep(1)
		msg = Bool()
		msg.data = True
		
		self.sim_pub.publish(msg);
		rospy.sleep(1)
		rospy.signal_shutdown('robot')
		



if __name__ == '__main__':
	rb = Robot()
