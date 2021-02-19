
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    alpha,beta = arm.getArmAngle()
    alpha_range = arm.getArmLimit()[0]
    beta_range = arm.getArmLimit()[1]
    rows = int(((alpha_range[1] - alpha_range[0])/(granularity)) + 1)
    cols = int(((beta_range[1] - beta_range[0])/(granularity)) + 1)
    map = [[SPACE_CHAR for c in range(cols)] for r in range(rows)]
    offset = [alpha_range[0], beta_range[0]]
    
    start_ind = angleToIdx([alpha,beta], offset, granularity)
    map[start_ind[0]][start_ind[1]] = START_CHAR
    
    for r in range(rows):
        for c in range(cols):
            if(start_ind[0] == r and start_ind[1] == c):
                continue
            ang = idxToAngle([r,c], offset, granularity)
            arm.setArmAngle(ang)
            pos = arm.getArmPosDist()
            tip = arm.getEnd()
            
            object = doesArmTouchObjects(pos, obstacles, isGoal=False)
            goal = doesArmTouchObjects(pos,goals, isGoal=True)
            goaltip = doesArmTipTouchGoals(tip, goals)
            wind = isArmWithinWindow(pos, window)
            
            if(not wind):
                map[r][c] = WALL_CHAR
            elif(goaltip):
                map[r][c] = OBJECTIVE_CHAR
            elif(object):
                map[r][c] = WALL_CHAR
            elif(goal):
                map[r][c] = WALL_CHAR
    maze = Maze(map, offset, granularity)
    return maze
