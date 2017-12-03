"""
Deterministic, fully observable Grid World environment. Taken from Sutton & Barto 1996
source: http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf
"""

import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import pyglet
import math
from random import *

logger = logging.getLogger(__name__)


class field_state:
    def __init__(self, red_robot_data,blue_robot_data, cones_in_tiles, mobile_goal_data):
        self.red_data = red_robot_data #location and holding goal flag
        self.blue_data = blue_robot_data
        self.tile_data = cones_in_tiles
        self.goal_data = mobile_goal_data
        
    def get_Key_Red(self,action):
        bigList =  self.red_data +\
                [self.blue_data[0]] + \
                self.tile_data + \
                sum(self.goal_data[:4],[]) + \
                [self.goal_data[4][0],self.goal_data[5][0],self.goal_data[6][0],self.goal_data[7][0],action]    
        string = ','.join([str(i) for i in bigList])               
        return string
#    def __repr__(self):
#        return "({}, {})".format(self.red_data[0], self.blue_data[0])
def getxy(index):
    x = index % 6
    y = int(index / 6)
    return (x,y)
def getDistance(index1,index2):
    start = getxy(index1)
    stop = getxy(index2)
    xDist = start[0] - stop[0]
    yDist = start[1] - stop[1]
    distance = round(math.sqrt(xDist**2 + yDist **2))
    if(distance == 0):
        return 1
    else:
        return distance

def calculate_score(field):    
    twentyPoint = 0
    tenPoint = 0
    cones = 0 
    for i in range(4):
        cones += field.goal_data[i][1]
        if field.goal_data[i][0] == 30:
            twentyPoint = 1
        if field.goal_data[i][0] == 24 or\
           field.goal_data[i][0] == 25 or\
           field.goal_data[i][0] == 31 :
               tenPoint +=1
    redScore = twentyPoint *20 + tenPoint*10 + cones * 2
    
    twentyPoint = 0
    tenPoint = 0
    cones = 0 
    for i in range(4):
        cones += field.goal_data[4 + i][1]
        if field.goal_data[4 + i][0] == 5:
            twentyPoint = 1
        if field.goal_data[4 + i][0] == 4 or\
           field.goal_data[4 + i][0] == 10 or\
           field.goal_data[4 + i][0] == 11 :
               tenPoint +=1
    blueScore = twentyPoint *20 + tenPoint*10 + cones * 2

    return(redScore,blueScore)
    

class GridWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # row, column        
        self.viewer = None
        self.robots = 2
        self.field_locations = 36
        self.max_cones = 16
        self.red_goals = 4
        self.blue_goals = 4
        self.max_time = 120
        self.state = 0
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def _step(self, action):
        failChance = 0.5
        reward = 0
        if(action <36):#move
            self.state.red_data[2] += getDistance(action,self.state.red_data[0])
            if getDistance(self.state.red_data[0],self.state.blue_data[0]) >=2 or random() > failChance:#still take time but fail action
                self.state.red_data[0] = action
                if self.state.red_data[1] >0: #if holding goal, move that goal with me
                    self.state.goal_data[self.state.red_data[1] - 1][0] = action
    #                print('got goal number ' + str(self.state.red_data[1]))
    #                print(self.state.goal_data)
        elif(action ==36):#goal
           # print('goal action')
            if self.state.red_data[1] > 0 :#holding goal
                self.state.red_data[1]=0
                #print('dropping goal')
                self.state.red_data[2] += 1

            else: 
                #print('picking up')
                for i in range(self.red_goals):
                   # print('me ' +str(self.state.red_data[0]) + '    ' + str(self.state.goal_data[i]))
                    if self.state.red_data[0] == self.state.goal_data[i][0]:#if same location
                       # print('picking up')
                        self.state.red_data[2] += 1
                        if getDistance(self.state.red_data[0],self.state.blue_data[0]) >=2 or random() > failChance:#still take time but fail action
                            self.state.red_data[1] = i + 1 #say im holding goal
                        break
        elif(action ==37):#cone
           # print('cone action')
            if self.state.red_data[1] > 0 :#holding goal
                if self.state.tile_data[self.state.red_data[0]] > 0:
                    #print('cone pickup success')
                    self.state.red_data[2] += 1
                    if getDistance(self.state.red_data[0],self.state.blue_data[0]) >=2 or random() > failChance:#still take time but fail action
                        self.state.tile_data[self.state.red_data[0]] -=1
                        self.state.goal_data[self.state.red_data[1]-1][1] +=1
                   # print('scoring on goal ' + str(self.state.red_data[1]))
                elif self.state.red_data[0] == 12 and self.state.red_data[3] >0:
                    self.state.red_data[2] += 1
                    if getDistance(self.state.red_data[0],self.state.blue_data[0]) >=2 or random() > failChance:#still take time but fail action
                        self.state.red_data[3] -=1
                        self.state.goal_data[self.state.red_data[1]-1][1] +=1
        
        elif(action-100 <36):#move
            action = action - 100 #moving
            self.state.blue_data[2] += getDistance(action,self.state.blue_data[0])
            if getDistance(self.state.red_data[0],self.state.blue_data[0]) >=2 or random() > failChance:#still take time but fail action
                self.state.blue_data[0] = action
                if self.state.blue_data[1] >0: #if holding goal, move that goal with me
                    self.state.goal_data[self.red_goals+ self.state.blue_data[1] - 1][0] = action
        elif(action-100 ==36):#cone
           # print('goal action')
            if self.state.blue_data[1] > 0 :#holding goal
                self.state.blue_data[1]=0
#                print('dropping goal')
                self.state.blue_data[2] += 1

            else: 
                #print('picking up')
                for i in range(self.blue_goals):
                    #print('me ' +str(self.state.red_data[0]) + '    ' + str(self.state.goal_data[i]))
                    if self.state.blue_data[0] == self.state.goal_data[self.red_goals + i][0]:#if same location
                        self.state.blue_data[2] += 1                    
                        if getDistance(self.state.red_data[0],self.state.blue_data[0]) >=2 or random() > failChance:#still take time but fail action
                            self.state.blue_data[1] = i + 1 #say im holding goal
                        break
        elif(action-100 ==37):#cone
           # print('cone action')
            if self.state.blue_data[1] > 0 :#holding goal
                if self.state.tile_data[self.state.blue_data[0]] > 0:
                    #print('cone pickup success')
                    self.state.blue_data[2] += 1
                    if getDistance(self.state.red_data[0],self.state.blue_data[0]) >=2 or random() > failChance:#still take time but fail action
                        self.state.tile_data[self.state.blue_data[0]] -=1
                        self.state.goal_data[self.red_goals + self.state.blue_data[1] - 1][1] +=1
                    
                elif self.state.blue_data[0] == 2 and self.state.blue_data[3] >0:
                    self.state.blue_data[2] += 1
                    if getDistance(self.state.red_data[0],self.state.blue_data[0]) >=2 or random() > failChance:#still take time but fail action
                        self.state.blue_data[3] -=1
                        self.state.goal_data[self.red_goals + self.state.blue_data[1] - 1][1] +=1
        reward = 0
        done = False
        if self.state.red_data[2]>=120 and self.state.blue_data[2]>=120:
            scores = calculate_score(self.state)
            #print('times ' + str(self.state.red_data[2]) + ' ' + str(self.state.blue_data[2]) )
            #print('scores ' + str(scores[0]) + ' ' +str(scores[1]))
            reward = (scores[0],scores[1])
            done = True
        return self.state, reward, done, {}

    def _reset(self):#totes wrong
        red_robot_data = [25,0,0,13]#location,goal holding, time, cones left in loads
        blue_robot_data = [10,0,0,13]
        cones_in_tiles = [9,4,0,0,0,0,
                          4,0,0,0,0,0,
                          0,0,5,0,0,0,
                          0,0,0,5,0,4,
                          0,0,0,0,0,4,
                          0,0,0,4,4,9]
        mobile_goal_data = []
        goal_locations = [14,6,34,15,1,20,21,29]#red first
        for i in range(self.red_goals + self.blue_goals):
            mobile_goal_data.append([goal_locations[i],0])
        
        self._seed()
        self.state = field_state(red_robot_data,blue_robot_data, cones_in_tiles, mobile_goal_data)
        return self.state

    def _render(self, mode='human', close=False, values=None):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        size = 6
        px_per_cell = 120
        size_px = size * px_per_cell

        from gym.envs.classic_control import rendering

        self.viewer = rendering.Viewer(size_px, size_px)
        for x in range(size + 1):
            for y in range(size + 1):
                x_line = rendering.Line((px_per_cell * x, 0), (px_per_cell * x, size_px))
                y_line = rendering.Line((0, px_per_cell * y), (size_px, px_per_cell * y))
                self.viewer.add_geom(x_line)
                self.viewer.add_geom(y_line)
        robot_width = px_per_cell*3/4
        poly = [(-robot_width/2,-robot_width/2),
                (-robot_width/2,robot_width/2),
                (robot_width/2,robot_width/2),
                (robot_width/2,-robot_width/2)]
        
        
        self.robot = self.viewer.draw_polygon(poly,filled=False)
        self.viewer.add_geom(self.robot)
        self.robot_transform = rendering.Transform()
        self.robot.add_attr(self.robot_transform)
        self.robot.attrs[0].vec4 = (0,0,1,1)
        self.robot.attrs[1] = rendering.LineWidth(5)
        
        self.opponent = self.viewer.draw_polygon(poly,filled=False)
        self.viewer.add_geom(self.opponent)
        self.opponent_transform = rendering.Transform()
        self.opponent.add_attr(self.opponent_transform)
        self.opponent.attrs[0].vec4 = (1,0,0,1)
        self.opponent.attrs[1] = rendering.LineWidth(5)
        

        location = getxy(self.state.blue_data[0])
        self.robot_transform.set_translation(location[0] * px_per_cell + px_per_cell / 2,
                                             size_px - location[1] * px_per_cell - px_per_cell / 2 - px_per_cell/8)
        location = getxy(self.state.red_data[0])
        self.opponent_transform.set_translation(location[0] * px_per_cell + px_per_cell / 2,
                                             size_px - location[1] * px_per_cell - px_per_cell / 2 - px_per_cell/8)
        for i in range(self.red_goals):
            goal = self.viewer.draw_circle(radius=px_per_cell / 5, filled=True)
            self.viewer.add_geom(goal)
            goal_transform = rendering.Transform()
            goal.add_attr(goal_transform)
            goal.attrs[0].vec4 = (1,0,0,0.5)
            
            location = getxy(self.state.goal_data[i][0])
            goal_transform.set_translation(location[0] * px_per_cell + px_per_cell / 2,
                                             size_px - location[1] * px_per_cell - px_per_cell / 2)
            label = pyglet.text.Label(str(self.state.goal_data[i][1]),
                          font_name='Times New Roman',
                          font_size=15,
                          color=(0,0,0,255),
                          x=location[0] * px_per_cell + px_per_cell / 2, y=size_px - location[1] * px_per_cell - px_per_cell / 2,
                          anchor_x='center', anchor_y='center')
            self.viewer.add_geom(label)
            
        for i in range(self.blue_goals):
            goal = self.viewer.draw_circle(radius=px_per_cell / 5, filled=True)
            self.viewer.add_geom(goal)
            goal_transform = rendering.Transform()
            goal.add_attr(goal_transform)
            goal.attrs[0].vec4 = (0,0,1,0.5)
            
            location = getxy(self.state.goal_data[i+self.red_goals][0])
            goal_transform.set_translation(location[0] * px_per_cell + px_per_cell / 2,
                                             size_px - location[1] * px_per_cell - px_per_cell / 2)
            
            label = pyglet.text.Label(str(self.state.goal_data[i+self.red_goals][1]),
                          font_name='Times New Roman',
                          font_size=15,
                          color=(0,0,0,255),
                          x=location[0] * px_per_cell + px_per_cell / 2, y=size_px - location[1] * px_per_cell - px_per_cell / 2,
                          anchor_x='center', anchor_y='center')
            self.viewer.add_geom(label)
        for i in range(self.field_locations):
            if self.state.tile_data[i] >0:
                location = getxy(i)
                label = pyglet.text.Label(str(self.state.tile_data[i]),
                          font_name='Times New Roman',
                          font_size=15,
                          color=(0,0,0,255),
                          x=location[0] * px_per_cell + px_per_cell / 2, y=size_px - location[1] * px_per_cell - px_per_cell / 8,
                          anchor_x='center', anchor_y='center')
                self.viewer.add_geom(label)
        location = getxy(12)       
        redLoads = pyglet.text.Label(str(self.state.red_data[3]),
              font_name='Times New Roman',
              font_size=15,
              color=(255,0,0,255),
              x=location[0] * px_per_cell + px_per_cell / 2, y=size_px - location[1] * px_per_cell - px_per_cell / 8,
              anchor_x='center', anchor_y='center')
        self.viewer.add_geom(redLoads)  
        location = getxy(2)       
        blueLoads = pyglet.text.Label(str(self.state.blue_data[3]),
              font_name='Times New Roman',
              font_size=15,
              color=(0,0,255,255),
              x=location[0] * px_per_cell + px_per_cell / 2, y=size_px - location[1] * px_per_cell - px_per_cell / 8,
              anchor_x='center', anchor_y='center')
        self.viewer.add_geom(blueLoads)
                        
                
                
                
        
        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
