# a_star.py

import sys
import time
import random
import numpy as np
import cv2
from matplotlib.patches import Rectangle

import point
import random_map

class AStar:
    def __init__(self, map,start,end,name):
        self.map=map
        self.open_set = []
        self.close_set = []
        self.start_x = start[0]
        self.start_y = start[1]
        self.end_y = end[1]
        self.end_x = end[0]
        self.r = random.randint(1,100)
        self.name =name
    def BaseCost(self, p):
        x_dis = p.x - self.start_x
        y_dis = p.y - self.start_y
        # Distance to start point
        return x_dis + y_dis + (np.sqrt(2) - 2) * min(x_dis, y_dis)

    def HeuristicCost(self, p):
        x_dis = self.end_x - p.x
        y_dis = self.end_y - p.y
        # Distance to end point
        return x_dis + y_dis + (np.sqrt(2) - 2) * min(x_dis, y_dis)
    def ConnerCost(self,p):
        if p.parent == None:
            return 0
        if (p.parent.x == self.start_x) and (p.parent.y == 0):
            return 0
        else:
            dif_x = (p.x != p.parent.parent.x)
            dif_y = (p.y != p.parent.parent.y)
            return dif_x * dif_y* 10
        #如果与grandpa点的x,y都不同，则增加cost
    def GuideCost(self,p):
        
        if self.r>50:#优先走横的
            if p.x>self.start_x:
                if p.y ==self.end_y:
                    return 0
                return self.BaseCost(p)+3
            return 0
        if self.r<= 50:
            if p.y >0:
                if p.x == self.end_x :
                    return 0
                return self.BaseCost(p)+3
            return 0
    def TotalCost(self, p):
        return self.BaseCost(p) + self.HeuristicCost(p) +self.GuideCost(p)

    def IsValidPoint(self, x, y):
        if x < 0 or y < 0:
            return False
        if x >= self.map.size or y >= self.map.size:
            return False
        return not self.map.IsObstacle(x, y)

    def IsInPointList(self, p, point_list):
        for point in point_list:
            if point.x == p.x and point.y == p.y:
                return True
        return False

    def IsInOpenList(self, p):
        return self.IsInPointList(p, self.open_set)

    def IsInCloseList(self, p):
        return self.IsInPointList(p, self.close_set)

    def IsStartPoint(self, p):
        return p.x == self.start_x and p.y == self.start_y

    def IsEndPoint(self, p):
        return p.y == self.end_y and p.x == self.end_x 
    def SaveImage(self, plt):
        millis = int(round(time.time() * 1000))
        filename = './' + str(millis) + '.png'
        plt.savefig(filename)

    def ProcessPoint(self, x, y, parent):
        if not self.IsValidPoint(x, y):
            return # Do nothing for invalid point
        p = point.Point(x, y)
        if self.IsInCloseList(p):
            return # Do nothing for visited point
        print('Process Point [', p.x, ',', p.y, ']', ', cost: ', p.cost)
        if not self.IsInOpenList(p):
            p.parent = parent
            p.cost = self.TotalCost(p)
            self.open_set.append(p)

    def SelectPointInOpenList(self):
        index = 0
        selected_index = -1
        min_cost = sys.maxsize
        for p in self.open_set:
            cost = self.TotalCost(p)
            if cost < min_cost:
                min_cost = cost
                selected_index = index
            index += 1
        return selected_index

    def BuildPath(self, p, ax, plt, start_time):
        path = []
        while True:
            path.insert(0, p) # Insert first
            if self.IsStartPoint(p):
                break
            else:
                p = p.parent
        name = self.name
        filename = './' + str(name) + 'pins_input.png'
        pict = self.map.map
        '''cv2.imwrite(filename,pict*255)'''
        for p in path:
            x = p.x
            y = p.y
            self.map.map[x,y,0] = 1
            '''rec = Rectangle((p.x, p.y), 1, 1, color='g')
            ax.add_patch(rec)
            plt.draw()
            self.SaveImage(plt)'''
        endname = './Multi_pin_routing_solver/amaster/firstdata/' + str(name) + 'route_input.png'
        cv2.imwrite(endname, pict*255)
        end_time = time.time()
        print('r=',self.r)
        print('===== Algorithm finish in', int(end_time-start_time), ' seconds')

    def RunAndSaveImage(self, ax, plt,):
        start_time = time.time()

        start_point = point.Point(self.start_x, self.start_y)
        start_point.cost = 0
        self.open_set.append(start_point)
        
        while True:
            index = self.SelectPointInOpenList()
            
            if index < 0:
                print('No path found, algorithm failed!!!')
                return
            p = self.open_set[index]
            rec = Rectangle((p.x, p.y), 1, 1, color='c')
            ax.add_patch(rec)
            '''self.SaveImage(plt)'''

            if (self.IsEndPoint(p) == True):
                return self.BuildPath(p, ax, plt, start_time)
                
            del self.open_set[index]
            self.close_set.append(p)

            # Process all neighbors
            x = p.x
            y = p.y
            '''self.ProcessPoint(x-1, y+1, p)'''
            self.ProcessPoint(x-1, y, p)
            '''self.ProcessPoint(x-1, y-1, p)'''
            self.ProcessPoint(x, y-1, p)
            '''self.ProcessPoint(x+1, y-1, p)'''
            self.ProcessPoint(x+1, y, p)
            '''self.ProcessPoint(x+1, y+1, p)'''
            self.ProcessPoint(x, y+1, p)
            




