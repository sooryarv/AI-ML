import numpy as np
import utils
import random
import math


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        
        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        self.reset()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def discretize(self, state):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state
        snake_head_x = math.floor(snake_head_x/40)
        snake_head_y = math.floor(snake_head_y/40)
        food_dir_x = math.floor(food_x/40) - snake_head_x
        food_dir_y = math.floor(food_y/40) - snake_head_y
    
        adjoining_wall_x = 0
        adjoining_wall_y = 0
        
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        body = []
        for i, j in snake_body:
            body.append((math.floor(i/40), math.floor(j/40)))
        
        if(food_dir_x == 0):
            food_dir_x = 0
        elif(food_dir_x < 0):
            food_dir_x = 1
        else:
            food_dir_x = 2
                
        if(food_dir_y == 0):
            food_dir_y = 0
        elif(food_dir_y < 0):
            food_dir_y = 1
        else:
            food_dir_y = 2
        
        
        if (snake_head_x == 1):
            adjoining_wall_x = 1
        elif (snake_head_x == 12):
            adjoining_wall_x = 2
        
        if (snake_head_y == 0):
            adjoining_wall_y = 1
        elif(snake_head_y == 12):
            adjoining_wall_y = 2
        

        if((snake_head_x, snake_head_y+1) in body):
            adjoining_body_top = 1
        if((snake_head_x, snake_head_y-1) in body):
            adjoining_body_bottom = 1
        if((snake_head_x - 1, snake_head_y) in body):
            adjoining_body_left = 1
        if((snake_head_x + 1, snake_head_y) in body):
            adjoining_body_right = 1
                
        mdp = (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        
        return mdp
    
    
    def updateQ(self, s, a, state, points, dead):
        s = self.discretize(s)
        if points > self.points:
            reward = 1
        elif dead:
            reward = -1
        else:
            reward = -0.1
        
        state = self.discretize(state)
        
        optimal = max(self.Q[state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], 0], self.Q[state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], 1], self.Q[state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], 2], self.Q[state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], 3])
        
        alpha = self.C / (self.C + self.N[s[0],s[1], s[2], s[3], s[4], s[5], s[6], s[7], a])
        
        q = self.Q[s[0],s[1], s[2], s[3], s[4], s[5], s[6], s[7], a]
        return q + alpha*(reward + (self.gamma * optimal)  - q)
        
        
    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        
        mdp = self.discretize(state)
        
        if dead:
            s = self.discretize(self.s)
            q = self.updateQ(self.s, self.a, state, points, dead)
            self.Q[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], self.a] = q
            self.reset()
            return
            
            
        if self._train:
            if (self.s != None):
                s = self.discretize(self.s)
                q = self.updateQ(self.s, self.a, state, points, dead)
                self.Q[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], self.a] = q #Update Q
            
        action = []
        for i in range(4):
            if self.N[mdp[0], mdp[1], mdp[2], mdp[3], mdp[4], mdp[5], mdp[6], mdp[7], i] < self.Ne: #Compute next action based on exploration
                action.append(1)
            else:
                action.append(self.Q[mdp[0], mdp[1], mdp[2], mdp[3], mdp[4], mdp[5], mdp[6], mdp[7], i])
        
        acts = np.argmax(action)
        
        a_set = set(action)
        contains_duplicates = len(action) != len(a_set)
        if contains_duplicates:
            for i in range(len(action)-1, -1, -1):
                if action[i] == max(action):
                    acts = i
                    break
        self.N[mdp[0], mdp[1], mdp[2], mdp[3], mdp[4], mdp[5], mdp[6], mdp[7], acts] += 1#Update N if not dead
        self.s = state
        self.a = acts
        self.points = points #Cache
        return acts
