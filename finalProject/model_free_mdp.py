
import rospy
import numpy as np
import sys
from math import *
from read_config import read_config

def mfmdp():
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
    row = map_size[0]
    col = map_size[1]
    sX = start[0]
    sY = start[1]
    gX = goal[0]
    gY = goal[1]

    policyMap = np.zeros([row, col, 4], dtype = 'float32')
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

    oldMap = []
    for i in range(row):
        for j in range(col):
            oldMap.append((0,0))

    neighbors = []
    for i in range(row*col):
        b = []
        neighbors.append(b)

    for i in range(row):
        for j in range(col):
            if j-1 >=0:
                neighbors[i*col+j].append((i,j-1))
            if(j+1 < col):
                neighbors[i*col+j].append((i,j+1))
            if i-1 >= 0:
                neighbors[i*col+j].append((i-1,j))
            if i + 1 < row:
                neighbors[i*col+j].append((i+1,j))

    #print neighbors[gX*col+gY]

    #frontierList = neighbors[gX*col+gY]
    #for f in frontierList:
    for w in range(1000):
        for i in range(row):
            for j in range(col):
                #for n in neighbors[i*col+j]:
                upValue = moveUp(oldMap, mdpMap, i, j, row, col, discount_factor, reward_for_hitting_wall, walls, learning_rate)
                policyMap[i][j][0] = upValue

                downValue = moveDown(oldMap, mdpMap,i,j, row, col, discount_factor, reward_for_hitting_wall, walls, learning_rate)
                policyMap[i][j][1] = downValue

                leftValue = moveLeft(oldMap, mdpMap, i, j, row, col, discount_factor, reward_for_hitting_wall, walls, learning_rate)
                policyMap[i][j][2] = leftValue

                rightValue = moveRight(oldMap, mdpMap,i, j, row, col, discount_factor, reward_for_hitting_wall, walls, learning_rate)
                policyMap[i][j][3] = rightValue

                #print upValue, downValue, leftValue, rightValue

        maxScore = np.max(policyMap, axis=2)
        maxIndex = policyMap.argmax(axis=2)
                #print maxScore
                #print maxIndex
                #print '\n'
        totalDiff = 0.0
        for k in range(maxScore.shape[0]):
            for l in range(maxScore.shape[1]):
                if([k,l] in walls):
                    oldMap[k*col+l] = ('WALL',mdpMap[k][l])
                elif([k,l] in pits):
                     oldMap[k*col+l] = ('PIT',mdpMap[k][l])
                elif([k,l] == goal):
                     oldMap[k*col+l] = ('GOAL',0)
                else:
                    totalDiff += abs(maxScore[k][l] - oldMap[k*col+l][1])
                    if maxIndex[k,l] == 0:
                        oldMap[k*col+l] = ('N',maxScore[k,l])
                    elif maxIndex[k,l] == 1:
                        oldMap[k*col+l] = ('S',maxScore[k,l])
                    elif maxIndex[k,l] == 2:
                        oldMap[k*col+l] = ('W',maxScore[k,l])
                    elif maxIndex[k,l] == 3:
                        oldMap[k*col+l] = ('E',maxScore[k,l])

            #maxValue = max(nr,nr1,nr2,nr3)
        #print totalDiff
        if(totalDiff < threshold_difference):
            break
    #print neighbors
    print oldMap
    print mdpMap
    toBeReturn = []
    for i in range(len(oldMap)):
        toBeReturn.append(oldMap[i][0])

    print toBeReturn

    return toBeReturn

def learning(learningRate, currentIndexValue, reward, df, nextIndexValue):
    return (1-learningRate) * currentIndexValue + learningRate * (reward + df * nextIndexValue);




def moveUp(oldMap, mdpMap, x,y,row,col, df, rhw, walls, learningRate):
    forward = (x-1,y)

    if(forward[0] >= 0 and (list(forward) not in walls)):
        return learning(learningRate, oldMap[x*col+y][1], \
            mdpMap[forward[0]][forward[1]], df, oldMap[forward[0]*col+forward[1]][1])
    else:
        return learning(learningRate, oldMap[x*col+y][1], rhw, df, oldMap[x*col+y][1])



def moveDown(oldMap, mdpMap, x,y,row,col, df, rhw,walls, learningRate):
    backward = (x+1,y)

    if(backward[0] < row and (list(backward) not in walls)):
        return learning(learningRate, oldMap[x*col+y][1], \
            mdpMap[backward[0]][backward[1]], df, oldMap[backward[0]*col+backward[1]][1])
    else:
        return learning(learningRate, oldMap[x*col+y][1], rhw, df, oldMap[x*col+y][1])


def moveLeft(oldMap, mdpMap, x,y,row,col, df, rhw, walls, learningRate):
    below = (x, y-1)

    if(below[1] >= 0 and (list(below) not in walls)):
        return learning(learningRate, oldMap[x*col+y][1], \
            mdpMap[below[0]][below[1]], df, oldMap[below[0]*col+below[1]][1])
    else:
        return learning(learningRate, oldMap[x*col+y][1], rhw, df, oldMap[x*col+y][1])


def moveRight(oldMap, mdpMap, x,y,row,col, df, rhw,walls, learningRate):
    below = (x, y+1)

    if(below[1] < col and (list(below) not in walls)):
        return learning(learningRate, oldMap[x*col+y][1], \
            mdpMap[below[0]][below[1]], df, oldMap[below[0]*col+below[1]][1])
    else:
        return learning(learningRate, oldMap[x*col+y][1], rhw, df, oldMap[x*col+y][1])



