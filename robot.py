#!/usr/bin/env python

import rospy
import random
from math import *
import numpy as np
from read_config import read_config
from helper_functions import get_pose, move_function
from map_utils import Map
from sklearn.neighbors import KDTree
from laser_model import laserModel

from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Quaternion, Point
from geometry_msgs.msg import Pose, PoseArray, PointStamped, Quaternion, Point, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String, Float32, Float32MultiArray

from PIL import Image

class Robot():
	def __init__(self):
		rospy.init_node("robot")
		self.config = read_config()
		self.move_list = self.config["move_list"]
		self.numParticles = self.config["num_particles"]

		random.seed(self.config['seed'])
		self.first_move_sigma_x = self.config["first_move_sigma_x"]
		self.first_move_sigma_y = self.config["first_move_sigma_y"]
		self.first_move_sigma_angle = self.config["first_move_sigma_angle"]
		self.resample_sigma_x = self.config["resample_sigma_x"]
		self.resample_sigma_y = self.config["resample_sigma_y"]
		self.resample_sigma_angle = self.config["resample_sigma_angle"]
		self.laser_z_hit = self.config["laser_z_hit"]
		self.laser_z_rand = self.config["laser_z_rand"]
		self.laser_sigma_hit = self.config["laser_sigma_hit"]

		self.weight = []
		for i in range(self.numParticles):
			self.weight.append(1.0/self.numParticles)
		self.oldX = []
		self.oldY = []
		self.oldTheta = []
		self.points_coord = []
		

		self.particlesPos = PoseArray()
		self.particlesPos.header.stamp = rospy.Time.now()
		self.particlesPos.header.frame_id = 'map'
		self.particlesPos.poses = []

		self.map_pub = rospy.Publisher("/particlecloud", PoseArray, queue_size = 10, latch=True);
		self.field_pub = rospy.Publisher("/likelihood_field", OccupancyGrid, queue_size = 10, latch=True);
		self.result_pub = rospy.Publisher("/result_update", Bool, queue_size = 10);
		self.sim_pub = rospy.Publisher("/sim_complete", Bool, queue_size = 10);
		self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_message_handle)
		self.laser_sub = rospy.Subscriber("/base_scan", LaserScan, self.laser_message_handle)
		self.robot_sub = rospy.Subscriber("/base_pose_ground_truth", Odometry, self.robot_handle)
		##rospy.wait_for_message("/map", OccupancyGrid);
		self.listIndex = 0;
		rospy.spin()

	def robot_handle(self, message):
		self.robotMsg = message


	def laser_message_handle(self, message):
		self.laserMsg = message;

	def construct_prob_field(self, sigma_hit):
		points = []
		prob_field = []
		obstacles = []
		for i in range(self.width):
			for j in range(self.height):
				x,y = self.map.cell_position(j,i)
				if(self.map.get_cell(x,y)==1):
					obstacles.append((x,y))
				points.append((x,y))

		KDT = KDTree(obstacles) 
		dist,indices = KDT.query(points,k=1)
		for i in range(self.width):
			for j in range(self.height):
				x ,y = points[i*self.height+j]
				self.map.set_cell(x,y,self.eval_gaussian(dist[i*self.height+j],sigma_hit))

	def map_message_handle(self, message):

		print "initial map_message_handle"
		self.height = message.info.height;
		self.width = message.info.width;
		self.origin_map = Map(message);
		self.map = Map(message);

		self.weight_map = np.ones([self.height,self.width])
		for i in range(self.numParticles):
			x = 0
			y = 0
			while(self.origin_map.get_cell(x, y) == 1):
				x = random.uniform(0,self.width)
				y = random.uniform(0,self.height)
			self.oldX.append(x)
			self.oldY.append(y)
			self.points_coord.append((self.oldX[i],self.oldY[i]))
			self.oldTheta.append(random.uniform(-pi,pi))
			self.particlesPos.poses.append(get_pose(self.oldX[i], self.oldY[i],self.oldTheta[i]))

		self.map_pub.publish(self.particlesPos)

		self.construct_prob_field(self.laser_sigma_hit);
		self.field_pub.publish(self.map.to_message())
		
		for l in range(len(self.move_list)):
			self.a = self.move_list[l][0]
			self.d = self.move_list[l][1]
			self.n = self.move_list[l][2]
			move_function(self.a, 0)
			self.firstMove = 0

			newX = []
			newY = []
			newTheta = []
			for j in range(0, self.n):
				move_function(0, self.d)
				print "robot pose is:\n", self.robotMsg.pose.pose.position
				self.moveParticle()
				rospy.wait_for_message("/base_scan", LaserScan)
				#print "laser scan: ", self.laserMsg.ranges
				self.weight = self.weight_update();
				# self.weight_map = self.weight_map_particles(self.map.grid,self.particlesPos.poses,self.weight)
				# self.weight_map = self.weight_map/np.sum(self.weight_map)
				new_points = self.reselect_particles(self.weight_map)
				old_particles = KDTree(self.points_coord) 
				dist,indices = old_particles.query(new_points,k=5)
				asd = 0
				for index in range(len(indices)):
					max_weight = 0
					max_x = -1
					for x in indices[index]:
						if(self.weight[x]>max_weight):
							max_weight = self.weight[x]
							max_x = x
					max_theta = self.oldTheta[max_x]
					max_theta = max_theta + random.gauss(max_theta,self.resample_sigma_angle)
					max_x_pos = new_points[index][0]
					max_y_pos = new_points[index][1]

					self.particlesPos.poses[asd] = get_pose(max_x_pos, \
						max_y_pos,\
						max_theta)
					asd += 1
					self.oldTheta[index] = max_theta
					self.oldX[index] = max_x_pos
					self.oldY[index] = max_y_pos
				#print "particle pose is: \n", self.particlesPos.poses.position
				#or t in range(len(self.particlesPos.poses)):
				print self.particlesPos.poses[0].position 
				self.map_pub.publish(self.particlesPos)
				self.result_pub.publish(True)
		self.sim_pub.publish(True);
		rospy.signal_shutdown('robot')
		print "end of map_message_handle"

	def weight_update(self):
		robotW = 0
		paticleW = 0
		total = np.ones(len(self.particlesPos.poses),dtype = 'float32')
		totalHitted = []
		for particleIndex in range(len(self.particlesPos.poses)):
			difference = 0
			
			ox = self.particlesPos.poses[particleIndex].position.x
			oy = self.particlesPos.poses[particleIndex].position.x
			prob1 = self.origin_map.get_cell(ox, oy);
			if prob1 == 1: 
				total[particleIndex] = 0
				continue
			else:
				for i in range(len(self.laserMsg.ranges)):
					j = self.laserMsg.angle_min + self.laserMsg.angle_increment*i;
					#if(j <= self.laserMsg.angle_max j >= self.laserMsg.angle_min):
					#estimate robot
					#robotW = self.laser_z_hit * random.gauss(self.laserMsg.ranges[i], \
					#	self.laser_sigma_hit) + self.laser_z_rand
					#estimate particle
					x = self.laserMsg.ranges[i] * cos(j+self.oldTheta[particleIndex]) 
					y = self.laserMsg.ranges[i] * sin(j+self.oldTheta[particleIndex]) 
					prob = self.map.get_cell(x, y);
					
					if(np.isnan(prob)):
						difference += 0#abs(robotW)  #or prob1 == 1

					else:
						paticleW = self.laser_z_hit * prob + self.laser_z_rand
						#difference += paticleW ** 3
						difference += paticleW ** 3
					#total[particleIndex] = -log(sqrt(difference))*self.weight[particleIndex]
				total[particleIndex] = (self.sigmoid(difference) + 1) *\
				self.weight[particleIndex]
			#total[particleIndex] = self.sigmoid(difference)*self.weight[particleIndex]
		return total / np.sum(total)

	def sigmoid(self, x):
		return 1.0/ (1 + exp(-x))


	# update paticles' position by chaning x, y, and theta, and then get_pose

	### update first move 
	def moveParticle(self):
		print "inside moveParticle"
		for i in range(0,self.numParticles):
			#print "old X is : ", self.oldX[i]
			if(self.firstMove == 0):
				self.oldX[i] = self.oldX[i] + self.d*cos(self.oldTheta[i] + radians(self.a)) + random.gauss(0, self.first_move_sigma_x)
				self.oldY[i] = self.oldY[i] + self.d*sin(self.oldTheta[i] + radians(self.a)) + random.gauss(0, self.first_move_sigma_y)
				self.oldTheta[i] = self.oldTheta[i] + radians(self.a) + random.gauss(0, self.first_move_sigma_angle)
			else:
				self.oldX[i] = self.oldX[i] + self.d*cos(self.oldTheta[i])
				self.oldY[i] = self.oldY[i] + self.d*sin(self.oldTheta[i])
				self.oldTheta[i] = self.oldTheta[i]
			#print "new X is :", self.oldX[i]
			self.particlesPos.poses[i] = get_pose(self.oldX[i], self.oldY[i],self.oldTheta[i])
			'''
			
			ox = self.particlesPos.poses[i].position.x
			oy = self.particlesPos.poses[i].position.x
			prob1 = self.origin_map.get_cell(ox, oy);
			if prob1 == 1: 
				self.particlesPos.poses[i] = get_pose(0, 0,self.oldTheta[i])
			'''
		#self.map_pub.publish(self.particlesPos)

		self.firstMove = 1	
		print "end of moveParticle"	

	def eval_gaussian(self, x, sigma):
		gauss = e**(-0.5*(float(x)/sigma)**2)
		return gauss
'''
	def gaussian_rendering(self,map_width,map_height,particle_x,particle_y,weight,sigma):
		width = map_width
		height = map_height
		x = particle_x
		y = particle_y
		prob_map = np.zeros([map_height,map_width])
		vertical = np.linspace(-width, width, 2 * width + 1)
		horizontal = np.linspace(-height, height, 2 * height + 1)
		vertical_v, horizontal_v = np.meshgrid(vertical, horizontal)
		dist    = np.sqrt(vertical_v**2 + horizontal_v**2);
		gaussian_window = (weight*(np.exp(-dist/sigma))).astype(np.float32)
		x, y= int(round(x + 1)), int(round(y + 1))
		delY, delX = height - y, width - x
		xSt, ySt   = max(0, delX), max(0, delY)
		yEn, xEn = delY + height, delX + width
		yEn      = min(2 * height + 1, yEn) 
		xEn      = min(2 * width + 1, xEn) 
		yImSt, xImSt = max(0, y - height), max(0, x - width)
		yImEn, xImEn = yImSt + (yEn - ySt), xImSt + (xEn - xSt)
		prob_map[yImSt:yImEn,xImSt:xImEn] = gaussian_window[ySt:yEn, xSt:xEn]
		return prob_map


	def weight_map_particles(self, map_arr,particles,weights):
		map_height = map_arr.shape[0]
		map_width = map_arr.shape[1]
		weight_map = np.zeros([map_arr.shape[0],map_arr.shape[1]])
		for index in range(len(particles)):
			weight_map = weight_map + self.gaussian_rendering(map_width,map_height,particles[index].position.x,\
			particles[index].position.y,weights[index],self.resample_sigma_y/self.resample_sigma_x)
		weight_map = weight_map/len(particles)

		return weight_map/np.sum(weight_map)
'''
	def reselect_particles(self,weight_map):

		map_arr = self.map.grid
		print 'map_shape:' , map_arr.shape
		candidates = {}

		good_particles = []
		good_particle_weight = []
		for i in range(self.numParticles):

			point_x = self.oldX[i]
			point_y = self.oldY[i]
			point = (point_x,point_y)
			weight = self.weight[i]
			new_weight = []

			if(self.origin_map.get_cell(point_x,point_y) == 1 or isnan(self.origin_map.get_cell(point_x,point_y))):
				candidates[point] = 0
			else:
				candidates[point] = self.weight[i]
		candidates_lst = sorted(candidates, key=lambda key: candidates[key])

		##choose good particles
		for p in candidates_lst:

			if(candidates[p] == 0):
				continue
			good_particles.append(p)
			good_particle_weight.append(candidates[p])

		'''
		##normalize particles
		good_particle_weight_arr = np.array(good_particle_weight,dtype = 'float32')/np.min(good_particle_weight)
		good_particle_weight_arr = good_particle_weight_arr/np.min(good_particle_weight_arr)

		for p in range(len(good_particles)):
			for c in range(int(round(good_particle_weight_arr[p]))):
				roll.append(good_particles[p])

		good_particle_weight = []
		good_particles = []
		for i in range(self.numParticles):
			point_x,point_y = roll[random.randint(0,len(roll)-1)]
			good_particle_weight.append[candidates[(point_x,point_y)]]
			point_x = point_x + random.gauss(0,self.resample_sigma_x)
			point_y = point_y + random.gauss(0,self.resample_sigma_y)
			good_particles.append((point_x,point_y))

		good_particle_weight = good_particle_weight / np.sum(good_particle_weight)
		#print "candidates_lst:" , candidates_lst
		self.weight = good_particle_weight
		'''

		choose_list = []
		tmp_sum = 0
		for i in range(len(good_particle_weight)):
			tmp_sum += good_particle_weight[i]
			choose_list[i] = tmp_sum

		good_particle_weight = []
		good_particles = []
		for i in range(self.numParticles):
			ranNum = random.uniform(0,1)
			point_x,point_y = roll[random.randint(0,len(roll)-1)]
			good_particle_weight.append[candidates[(point_x,point_y)]]
			point_x = point_x + random.gauss(0,self.resample_sigma_x)
			point_y = point_y + random.gauss(0,self.resample_sigma_y)
			good_particles.append((point_x,point_y))

		good_particle_weight = good_particle_weight / np.sum(good_particle_weight)
		#print "candidates_lst:" , candidates_lst
		self.weight = good_particle_weight

		return good_particles




if __name__ == '__main__':
	rb = Robot()
