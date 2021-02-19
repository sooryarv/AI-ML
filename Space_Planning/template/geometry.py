# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
#import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position (int,int):of the arm link, (x-coordinate, y-coordinate)
    """
    end_x = start[0] + int(length * math.cos(math.radians(angle)))
    end_y = start[1] - int(length * math.sin(math.radians(angle)))
    return (end_x, end_y)
    


import numpy as np
def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """
    #print(armPosDist)
    if isGoal:
        for pos in armPosDist:
            for object in objects:
                radius = object[2]
                l2 = math.pow(pos[0][0] - pos[1][0], 2) + math.pow(pos[0][1] - pos[1][1], 2)
                if l2 == 0:
                    dis = math.sqrt(math.pow(object[0] - pos[0][0], 2) + math.pow(object[1] - pos[0][1], 2))
                else:
                    t = max(0, min(1, ((object[0] - pos[0][0]) * (pos[1][0] - pos[0][0]) +  (object[1] - pos[0][1]) * (pos[1][1] - pos[0][1]))/l2))
                    
                    dis = math.sqrt(math.pow(object[0] - (pos[0][0] + t * (pos[1][0] - pos[0][0])), 2) + math.pow(object[1] - (pos[0][1] + t * (pos[1][1] - pos[0][1])), 2))
               
                if dis <= radius:
                    return True
    else:
        for pos in armPosDist:
            for object in objects:
                radius = object[2]
                l2 = math.pow(pos[0][0] - pos[1][0], 2) + math.pow(pos[0][1] - pos[1][1], 2)
                if l2 == 0:
                    dis = math.sqrt(math.pow(object[0] - pos[0][0], 2) + math.pow(object[1] - pos[0][1], 2))
                else:
                    t = max(0, min(1, ((object[0] - pos[0][0]) * (pos[1][0] - pos[0][0]) +  (object[1] - pos[0][1]) * (pos[1][1] - pos[0][1]))/l2))
                   
                    dis = math.sqrt(math.pow(object[0] - (pos[0][0] + t * (pos[1][0] - pos[0][0])), 2) + math.pow(object[1] - (pos[0][1] + t * (pos[1][1] - pos[0][1])), 2))
                
                if dis <= radius + pos[2]:
                    return True
    return False

def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tip touch goals

        Args:
            armEnd (tuple): the arm tip position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tip touches any goal. False if not.
    """
    for goal in goals:
        if math.sqrt(math.pow(armEnd[0] - goal[0], 2) + math.pow(armEnd[1] - goal[1], 2)) < goal[2]:
            return True
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """
    for pos in armPos:
        x_start = pos[0][0]
        y_start = pos[0][1]
        x_end = pos[1][0]
        y_end = pos[1][1]
        if x_start > window[0] or y_start > window[1] or x_end > window[0] or y_end > window[1]  or x_start < 0  or x_end < 0 or y_start < 0 or y_end < 0:
            return False
    return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
