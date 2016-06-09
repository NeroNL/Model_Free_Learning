
import rospy
import random as rand
import numpy as np
import sys
from math import *
from cse_190_assi_3.msg import *
from read_config import read_config

def tdmdp():
    config = read_config()
    move_list = config["move_list"]
    map_size = config["map_size"]
    start = config["start"]
    goal = config["goal"]
    walls = config["walls"]
    pits = config["pits"]
    max_iterations = config["max_iterations"]
    threshold_difference = config["threshold_difference"]
    reward_for_each_step = config["reward_for_each_step"]
    reward_for_hitting_wall = config["reward_for_hitting_wall"]
    reward_for_reaching_goal = config["reward_for_reaching_goal"]
    reward_for_falling_in_pit = config["reward_for_falling_in_pit"]
    discount_factor = config["discount_factor"]
    learning_rate = config["learning_rate"]
    mdp_pub = rospy.Publisher("/results/policy_list", PolicyList, queue_size = 100);

    row = map_size[0]
    col = map_size[1]
    sX = start[0]
    sY = start[1]
    gX = goal[0]
    gY = goal[1]

    mdpMap = []
    for i in range(row):
        mdpMap.append(np.zeros(col, dtype = 'float32'))
    mdpMap = np.array(mdpMap)

    for i in range(row):
        for j in range(col):
            mdpMap[i][j] = reward_for_each_step

    for wall in walls:
        mdpMap[wall[0]][wall[1]] = reward_for_hitting_wall

    for pit in pits:
        mdpMap[pit[0]][pit[1]] = reward_for_falling_in_pit

    mdpMap[gX][gY] = reward_for_reaching_goal

    countMap = np.zeros([row, col, 4], dtype = 'float32')
    oldCountMap = np.copy(countMap)

    
    probUp = 0.8#rand.random();
    probDown = 0#rand.random();
    probLeft = 0.1#rand.random();
    probRight = 0.1#rand.random();
    total = probUp + probDown + probLeft + probRight
    probUp /= total
    probDown /= total
    probLeft /= total
    probRight /= total
    probList = [probUp, probDown, probLeft, probRight]
    
    policyMap = np.zeros([row, col, 4], dtype = 'float32')
    oldPolicyMap = np.zeros([row, col, 4], dtype = 'float32')

    iteration_size = 100
    max_iteration_size = 100
    

    for w in range(max_iteration_size):
        for i in range(row):
            for j in range(col):
                if([i,j] in walls):
                    for d in range(4):
                        policyMap[i][j][d] = reward_for_hitting_wall
                elif([i,j] in pits):
                    for d in range(4):
                        policyMap[i][j][d] = reward_for_falling_in_pit
                elif([i,j] == goal):
                    for d in range(4):
                        policyMap[i][j][d] = 0
                else:
                    for s in range(iteration_size):
                        #for n in neighbors[i*col+j]:
                        upValue = moveUp(mdpMap, oldPolicyMap, i, j, row, col, discount_factor, reward_for_hitting_wall, walls, learning_rate)
                        #a = computeReward(probList, upValue)

                        downValue = moveDown(mdpMap, oldPolicyMap,i,j, row, col, discount_factor, reward_for_hitting_wall, walls, learning_rate)
                        #b = computeReward(probList, downValue)

                        leftValue = moveLeft(mdpMap, oldPolicyMap, i, j, row, col, discount_factor, reward_for_hitting_wall, walls, learning_rate)
                        #c = computeReward(probList, leftValue)

                        rightValue = moveRight(mdpMap, oldPolicyMap, i, j, row, col, discount_factor, reward_for_hitting_wall, walls, learning_rate)
                        #d = computeReward(probList, rightValue)

                        
                        for d in range(4):
                            num = rand.random();
                            reward = 0;
                            if(num < probUp):
                                reward = upValue[d]
                                policyMap[i][j][d] += reward
                            elif(num >= probUp and num < probUp + probDown):
                                reward = downValue[d]
                                policyMap[i][j][d] += reward
                            elif(num >= probUp + probDown and num < probUp + probDown + probLeft):
                                reward = leftValue[d]
                                policyMap[i][j][d] += reward
                            elif(num >= probUp + probDown + probLeft and num < probUp + probDown + probLeft + probRight):
                                reward = rightValue[d]
                                policyMap[i][j][d] += reward

                    policyMap[i][j][0] /= iteration_size
                    policyMap[i][j][1] /= iteration_size
                    policyMap[i][j][2] /= iteration_size
                    policyMap[i][j][3] /= iteration_size


        oldPolicyMap = np.copy(policyMap);
    
        
        maxScore = np.max(policyMap, axis=2)
        maxIndex = policyMap.argmax(axis=2)
        toBeReturn = []
        for k in range(maxScore.shape[0]):
            for l in range(maxScore.shape[1]):
                if([k,l] not in walls and [k,l] not in pits and [k,l] != goal):
                    if maxIndex[k,l] == 0:
                        toBeReturn.append('N');
                        
                    elif maxIndex[k,l] == 1:
                        toBeReturn.append('S');
                        
                    elif maxIndex[k,l] == 2:
                        toBeReturn.append('W');
                        
                    elif maxIndex[k,l] == 3:
                        toBeReturn.append('E');
                elif([k,l] in walls):
                    toBeReturn.append('WALL');
                elif([k,l] in pits):
                    toBeReturn.append('PIT');
                elif([k,l] == goal):
                    toBeReturn.append('GOAL');
        mdp_inst = PolicyList()
        mdp_inst.data = toBeReturn
        mdp_pub.publish(mdp_inst)
    
    return 1


def computeReward(cl, li):
    return cl[0]*li[0]+cl[1]*li[1]+cl[2]*li[2]+cl[3]*li[3]


def learning(learningRate, currentIndexValue, reward, df, nextIndexValue):
    return currentIndexValue + learningRate * (reward + df * nextIndexValue - currentIndexValue);

def moveUp(mdpMap, oldPolicyMap, x,y,row,col, df, rhw, walls, learningRate):
    forward = (x-1,y)
    backward = (x+1,y)
    below = (x, y-1)
    above = (x, y+1)

    toBeReturn = []
    if(forward[0] >= 0 and (list(forward) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[forward[0]][forward[1]], df, max(oldPolicyMap[forward[0]][forward[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(backward[0] < row and (list(backward) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[backward[0]][backward[1]], df, max(oldPolicyMap[backward[0]][backward[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(below[1] >= 0 and (list(below) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[below[0]][below[1]], df, max(oldPolicyMap[below[0]][below[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(above[1] < col and (list(above) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[above[0]][above[1]], df, max(oldPolicyMap[above[0]][above[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))

    return toBeReturn


def moveDown(mdpMap, oldPolicyMap, x,y,row,col, df, rhw,walls, learningRate):
    forward = (x+1,y)
    backward = (x-1,y)
    below = (x, y+1)
    above = (x, y-1)

    toBeReturn = []
    if(forward[0] < row and (list(forward) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[forward[0]][forward[1]], df, max(oldPolicyMap[forward[0]][forward[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(backward[0] >= 0 and (list(backward) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[backward[0]][backward[1]], df, max(oldPolicyMap[backward[0]][backward[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(below[1] < col and (list(below) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[below[0]][below[1]], df, max(oldPolicyMap[below[0]][below[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(above[1] >= 0 and (list(above) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[above[0]][above[1]], df, max(oldPolicyMap[above[0]][above[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))

    return toBeReturn


def moveLeft(mdpMap, oldPolicyMap, x,y,row,col, df, rhw, walls, learningRate):
    forward = (x,y-1)
    backward = (x,y+1)
    below = (x+1, y)
    above = (x-1, y)

    toBeReturn = []
    if(forward[1] >= 0 and (list(forward) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[forward[0]][forward[1]], df, max(oldPolicyMap[forward[0]][forward[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(backward[1] < col and (list(backward) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[backward[0]][backward[1]], df, max(oldPolicyMap[backward[0]][backward[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(below[0] < row and (list(below) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[below[0]][below[1]], df, max(oldPolicyMap[below[0]][below[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(above[0] >= 0 and (list(above) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[above[0]][above[1]], df, max(oldPolicyMap[above[0]][above[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))

    return toBeReturn


def moveRight(mdpMap, oldPolicyMap, x,y,row,col, df, rhw,walls, learningRate):
    forward = (x,y+1)
    backward = (x,y-1)
    below = (x-1, y)
    above = (x+1, y)

    toBeReturn = []
    if(forward[1] < col and (list(forward) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[forward[0]][forward[1]], df, max(oldPolicyMap[forward[0]][forward[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(backward[1] >= 0 and (list(backward) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[backward[0]][backward[1]], df, max(oldPolicyMap[backward[0]][backward[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(below[0] >= 0 and (list(below) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[below[0]][below[1]], df, max(oldPolicyMap[below[0]][below[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))
    if(above[0] < row and (list(above) not in walls)):
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], \
            mdpMap[above[0]][above[1]], df, max(oldPolicyMap[above[0]][above[1]])))
    else:
        toBeReturn.append(learning(learningRate, oldPolicyMap[x][y][0], rhw, df, max(oldPolicyMap[x][y])))

    return toBeReturn
