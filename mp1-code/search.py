# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

from collections import deque
def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    ends = maze.getObjectives()
    if(start in ends):
        return [start]
    frontier = deque([start])
    explored = {}
    path = []
    while frontier:
        current = frontier.popleft()
        if (current in ends):
            break
        neighbors = maze.getNeighbors(current[0], current[1])
        for i in neighbors:
            if (i not in explored and i not in frontier):
                frontier.append(i)
                explored[i] = current
    path.append(current)
    path.append(explored[current])
    cur = explored[current]
    while cur != start:
        upd = explored[cur]
        cur = upd
        path.append(cur)
    path.reverse()
    #print(maze.isValidPath(path))
    return path

from queue import PriorityQueue
import heapq
def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    ends = maze.getObjectives()
    if(start in ends):
        return [start]
    frontier = []
    g = 0
    f = abs(start[0] - ends[0][0]) + abs(start[1] - ends[0][1])
    frontier.append((f, start))
    heapq.heapify(frontier)
    explored = {}
    path = []
    while frontier:
        current = heapq.heappop(frontier)
        if (current[1] in ends):
            break
        neighbors = maze.getNeighbors(current[1][0], current[1][1])
        for i in neighbors:
            if (i not in explored and i not in frontier):
                g = 1 + current[0]
                h = abs(i[0] - ends[0][0]) + abs(i[1] - ends[0][1])
                fn = g + h
                if (h < current[0]):
                    frontier.append((fn, i))
                    heapq.heapify(frontier)
                    explored[i] = current
    path.append(current[1])
    path.append(explored[current[1]][1])
    cur = explored[current[1]][1]
    while cur != start:
        upd = explored[cur][1]
        cur = upd
        path.append(cur)
    path.reverse()
    print(maze.isValidPath(path))
    return path
    
def astar_end(maze, startpoint, endpoint):
    start = startpoint
    ends = [endpoint]
    if(start in ends):
        #print('got here immediately')
        return [start]
    frontier = []
    g = 0
    f = abs(start[0] - ends[0][0]) + abs(start[1] - ends[0][1])
    frontier.append((f, start))
    heapq.heapify(frontier)
    explored = {}
    path = []
    #print(frontier)
    while frontier:
        #print(frontier)
        current = heapq.heappop(frontier)
        if (current[1] in ends):
            #print('got here?')
            break
        neighbors = maze.getNeighbors(current[1][0], current[1][1])
        for i in neighbors:
            #print(i)
            if (i not in explored and i not in frontier):
                g = 1 + current[0]
                h = abs(i[0] - ends[0][0]) + abs(i[1] - ends[0][1])
                fn = g + h
                #print(current[0], h)
                #if (h < current[0]):
                frontier.append((fn, i))
                heapq.heapify(frontier)
                explored[i] = current
    #print(frontier)
    path.append(current[1])
    #print(path, current[1])
    path.append(explored[current[1]][1])
    cur = explored[current[1]][1]
    while cur != start:
        upd = explored[cur][1]
        cur = upd
        path.append(cur)
    path.reverse()
    #print(maze.isValidPath(path))
    return path
    


from itertools import groupby
def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    start = maze.getStart()
    ends = maze.getObjectives()
    #print(start, ends)
    frontiers = []
    path = []
    nstart = start
    while ends:
        #frontiers = []
        for i in ends:
            #print(i)
            f = len(path) + abs(nstart[0] - i[0]) + abs(nstart[1] - i[1])
            if (f < 34):
                frontiers.append((f, i))
                heapq.heapify(frontiers)
                #print(frontiers)
                end = heapq.heappop(frontiers)[1]
                #print(nstart, ends, end)
                ends.remove(end)
                path.append(astar_end(maze, nstart, end))
                explored = []
                nstart = end
    npath = [item for sublist in path for item in sublist]
    #print(npath)
    opath = [i[0] for i in groupby(npath)]
    print(opath)
    return opath

def prims_mst(maze, objectives):
   mst_pq = []
   mst_path = []
   heapq.heappush(mst_pq, (0, objectives[0], [objectives[0]]))  # (f, state, path to state)
   closed = set()  # init closed to close anything already in path

   while len(mst_pq):
       (cur_path_len, vertex, cur_path_) = heapq.heappop(mst_pq)
       if vertex in closed:
           continue
       closed.add(vertex)
       if len(mst_path) > 0 and mst_path[-1] == cur_path_[0]:
           mst_path += cur_path_[1:]
       else:
           mst_path += cur_path_
       for object in objectives:
           if vertex != object:
               if object not in closed:
                   path_ = astar_end(maze, vertex, object)
                   heapq.heappush(mst_pq, (len(path_), object, path_))
   return mst_path

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    ends = maze.getObjectives()
    frontier = []
    f = prims_mst(maze, ends)
    frontier.append((len(f), start, ends))
    heapq.heapify(frontier)
    #print(frontier)
    explored = {}
    explored[start] = f
    heapq.heappop(frontier)
    #print(explored[start])
    '''for end in ends:
        fn = abs(start[0] - end[0]) + abs(start[1] - end[1])
        heapq.heappush(frontier, (fn, 0, start, end))
    first = heapq.heappop(frontier)[3]'''
    l = astar_end(maze, start, explored[start][0])
    #print(l)
    l.extend(explored[start])
    res = [i[0] for i in groupby(l)]
    print(res)
    return res


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []


