import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle
import cv2
from generater import genMap
import random_map
from a_star import AStar
import point


class bmgen():

    def __init__(self,num,h,w,pinpair,name):
        self.bm =genMap(num,h,w,pinpair,name) 
        
    def bmgen(self):
        output = self.bm.gen(self.bm.num,self.bm.h,self.bm.w,self.bm.pinnum)
        return output


    def IsObstacle(self,map,input_x,input_y):
        if map is None:
            print('错了')
        if map[input_x][input_y][1] == 1:
            return True
        else:
            return False

  
    def ax_init(self,map,name):
        size = self.bm.h
        ax = plt.gca()
        ax.set_xlim([0, size])
        ax.set_ylim([0, size])
        zat = map

        for i in range(size):
            for j in range(size):
                if zat.IsObstacle(i,j):
                    rec = Rectangle((i, j), width=1, height=1, color='gray')
                    ax.add_patch(rec)
                else:
                    rec = Rectangle((i, j), width=1, height=1, edgecolor='gray', facecolor='w')
                    ax.add_patch(rec)

        Is_pin = np.argwhere(zat.map[:,:,0] == 1)
        rec = Rectangle(Is_pin[0], width = 1, height = 1, facecolor='b')
        ax.add_patch(rec)

        rec = Rectangle(Is_pin[1], width = 1, height = 1, facecolor='r')
        ax.add_patch(rec)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        #plt.show()

        a_star = AStar(map,Is_pin[0],Is_pin[1],name) 
        a_star.RunAndSaveImage(ax, plt)
class m2c():
    def __init__(self,map):
        self.map = map
        self.obstacle_point = []
        self.GenerateObstacle()
        self.size = len(map[:,])
    def GenerateObstacle(self):
        indices = np.argwhere(self.map[:,:,1] == 1)
        for dian in indices:
            self.obstacle_point.append(point.Point_arr(dian))
    def IsObstacle(self, i ,j):
        for p in self.obstacle_point:
            if i==p.x and j==p.y:
                return True
        return False    


num,h,w,pinpairs = 3,32,32,1
for i in range(1323,3000):

    name = i
    new = bmgen(num,h,w,pinpairs,1)
    map0 = new.bmgen()
    map = m2c(map0)
    print('now is iteration:',i)
    new.ax_init(map,name)
