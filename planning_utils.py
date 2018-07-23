from enum import Enum
from queue import PriorityQueue
import numpy as np
from bresenham import bresenham


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)
    
    NORTHWEST = (-1, -1, np.sqrt(2))
    NORTHEAST = (-1, 1, np.sqrt(2))
    SOUTHWEST = (1, -1, np.sqrt(2))
    SOUTHEAST = (1, 1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
        
        if Action.NORTHEAST in valid_actions:
            valid_actions.remove(Action.NORTHWEST)
        if Action.NORTHWEST in valid_actions:
            valid_actions.remove(Action.NORTHWEST)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
        
        if Action.SOUTHEAST in valid_actions:
            valid_actions.remove(Action.SOUTHEAST)
        if Action.SOUTHWEST in valid_actions:
            valid_actions.remove(Action.SOUTHWEST)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
        
        if Action.NORTHWEST in valid_actions:
            valid_actions.remove(Action.NORTHWEST)
        if Action.SOUTHWEST in valid_actions:
            valid_actions.remove(Action.SOUTHWEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)
        
        if Action.SOUTHEAST in valid_actions:
            valid_actions.remove(Action.SOUTHEAST)
        if Action.NORTHEAST in valid_actions:
            valid_actions.remove(Action.NORTHEAST)

    return valid_actions


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost



def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def attraction(position, goal, alpha):
    # TODO: implement attraction
    ax_force = -alpha * (position[0] - goal[0])
    ay_force = -alpha * (position[1] - goal[1])
    return [ax_force, ay_force]

def repulsion(position, obstacle, beta, q_max):
    # TODO: implement replusion force
    rx_force = beta * ((1/q_max) - (1/(position[0] - obstacle[0]))) * (1/pow(position[0] - obstacle[0],2))
    ry_force = beta * ((1/q_max) - (1/(position[1] - obstacle[1]))) * (1/pow(position[1] - obstacle[1],2))
    return [rx_force, ry_force]


def potential_field(grid, goal, alpha, beta, q_max):
    x = []
    y = []
    fx = []
    fy = []

    obs_i, obs_j = np.where(grid == 1)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 0:

                # add attraction force
                force = attraction([i, j], goal, alpha)
                print("attractive", force)
                
                for (oi, oj) in zip(obs_i, obs_j):
                    if np.linalg.norm(np.array([i, j]) - np.array([oi, oj])) < q_max:
                        # add replusion force
                        if i != oi and j != oj:
                            r_force = repulsion([i, j], [oi, oj], beta, q_max)
                            force[0] = force[0] - r_force[0]
                            force[1] = force[1] - r_force[1]
                            print("i:", i, " j:", j)
                            print("oi:", oi, " oj:", oj)
                            print("repulsive", r_force)
                        #pass

                print("force", force)
                x.append(i)
                y.append(j)
                fx.append(force[0])
                fy.append(force[1])

    return x, y, fx, fy

def parse_text(str):
    alph = ['1','2','3','4','5','6','7','8','9','0','.','-']
    new_s = ''
    float_list = []
    for c in range(len(str)):
        if str[c] in alph:
            new_s += str[c]
        else:
            if len(new_s) > 1:
                float_list.append(float(new_s))
                new_s = ''

    return (float_list[0], float_list[1])

def is_colinear(p1, p2, p3, epsilon = 1e-6):
    m = np.concatenate((p1, p2, p3), 0)
    det = np.linalg.det(m)
    return abs(det) < epsilon 

def point(p):
    return np.array([p[0], p[1], 1.]).reshape(1, -1)

def prune_path(path, grid):
    new_path = [p for p in path]

    i = 0
    while i < len(new_path) - 2:

        #prune path for extraneous diagonals
        p1 = new_path[i]
        p2 = new_path[i+1]
        p3 = new_path[i+2]

        cells = list(bresenham(p1[0], p1[1], p3[0], p3[1]))

        hits = False
        #print("*******************")
        #print(cells)
        #print() 
        for c in cells:
            #print(c)
            #print(grid[c[0]][c[1]])
            if grid[c[0]][c[1]] == 1:
                hits = True
            

        p1 = point(p1)
        p2 = point(p2)
        p3 = point(p3)

        #prune path for colinear points
        if is_colinear(p1, p2, p3):
            new_path.remove(new_path[i+1])

        elif not hits:
            new_path.remove(new_path[i+1])

        else:
            i += 1
    return new_path


