import cv2
import numpy as np


'''name = './Multi_pin_routing_solver/amaster/firstdata/6201route_input.png'
arr = cv2.imread(name)
if arr is not None:
    arr2= arr.transpose(1,0,2)#行列转置
    arr_new = arr2[::-1]
    cv2.imshow('Saved PNG Image', arr_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Failed to load the saved image.')'''


class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y

class edge():
    def __init__(self,list,dire,valid):
        self.max_dir = 0
        self.min_dir = 0
        self.point = 0
        self.dire = dire
        self.valid = valid
        self.list = list
        self.assign()
        
    def assign(self):
        
        if self.valid:
            lenth = len(self.list)
            if  lenth==1:
                self.point = 1
            self.max_dir = self.list[lenth-1]
            self.min_dir = self.list[0]


        
class map():
    def __init__(self,map):
        self.map = map
        self.routed = self.map[:,:,0]
        self.obstacle = self.map[:,:,1]
        self.pin = self.map[:,:,2]
        self.size = len(self.map[0])
        self.obstacle_point = []
        self.edge = []
        self.top = 1
        self.left = 2
        self.bottom = 3
        self.right = 4
    def row_is_zero(self,arr, rowIndex):
        node = 0
        index = rowIndex
        for i in range(self.size):
            if(arr[index][i] != 0):
                node = node + 1
        return node == 0
    def column_is_zero(self,arr, columnIndex):
        node = 0
        index = columnIndex
        for i in range(self.size):
            if(arr[i][index] != 0):
                node = node + 1
        return node == 0
    def findend(self,arr,index,is_row):
        point = []
        if is_row == 1:
            for i in range(self.size):
                if arr[index][i] != 0:
                    point.append(i)
            return point
        else:
            for i in range(self.size):
                if arr[i][index] != 0:
                    point.append(i)
            return point
    #assign a list to record pixel at egde of the arr
    def app_edge(self):
        #top:1,left:2,bottom:3,,right:4
        if not self.row_is_zero(self.routed,0):
            point = self.findend(self.routed,0,1)
            #list,dire,valid
            E_edge = edge(point,1,1)
            self.edge.append(E_edge)
        else :
            E_edge = edge(0,1,0)
            self.edge.append(E_edge) 
        #top
        if not self.column_is_zero(self.routed,0):
            point = self.findend(self.routed,0,0)
            E_edge = edge(point,2,1)
            self.edge.append(E_edge)
        else :
            E_edge = edge(0,2,0)
            self.edge.append(E_edge) 
        #left
        if not self.row_is_zero(self.routed,self.size-1):
            point = self.findend(self.routed,self.size-1,1)
            E_edge = edge(point,3,1)
            self.edge.append(E_edge)
        else :
            E_edge = edge(0,3,0)
            self.edge.append(E_edge) 
        #bottom
        
        if not self.column_is_zero(self.routed,self.size-1):
            point = self.findend(self.routed,self.size-1,0)
            E_edge = edge(point,4,1)
            self.edge.append(E_edge)
        else :
            E_edge = edge(0,4,0)
            self.edge.append(E_edge) 
        #right
    def revolve(self):
        for yam in self.edge:
            if yam.dire ==4:
                yam.dire = 1
            else:
                yam.dire = yam.dire +1
        A = self.edge[3]
        ###
        min_2 = self.edge[2].min_dir
        self.edge[2].min_dir = 31 - self.edge[2].max_dir
        self.edge[2].max_dir = 31 - min_2
        self.edge[3] = self.edge[2]
        ######
        self.edge[2] = self.edge[1]
        ######
        min_1 = self.edge[1].min_dir
        self.edge[0].min_dir = 31 - self.edge[1].max_dir
        self.edge[0].max_dir = 31 - min_1
        self.edge[1] = self.edge[0]
        ###
        self.edge[0] = A
        
             
class merge():
    def __init__(self,img_path):
        self.mode = 'first'
        self.maplist = []
        '''self.map = map()'''
        self.img_path = img_path
    def rotate(self,map):
        arr1 =map
        arr1.map= arr1.map.transpose(1,0,2)#行列转置
        arr1.map = arr1.map[::-1]
        arr1.revolve()
        return arr1


    def can_mer(self,status,map1,map2):
        t1,t2 = 0,0
        b1,b2 = 0,0
        l1,l2 = 0,0
        r1,r2 = 0,0
        pe_match = 0
        ep_match = 0
        ee_match = 0
        for yam in map1.edge:
            if yam.dire == 1 and yam.valid ==1:
                t1 = 1
            if yam.dire ==2 and yam.valid ==1:
                l1 = 1
            if yam.dire == 3 and yam.valid ==1:
                b1 = 1
            if yam.dire ==4 and yam.valid ==1:
                r1 = 1
        '''if status == 0:
            if b1 ==0:
                return False'''
        for yam in map2.edge:
            if yam.dire == 1 and yam.valid ==1:
                t2 = 1
            if yam.dire ==2 and yam.valid ==1:
                l2 = 1
            if yam.dire == 3 and yam.valid ==1:
                b2 = 1
            if yam.dire ==4 and yam.valid ==1:
                r2 = 1
        #edge[t,l,b,r]
        if status == 1:
            if b1 == 1 and t2 ==1:#1b,2t
                if map1.edge[2].point == 1:
                    if map2.edge[0].min_dir <= map1.edge[2].max_dir and map1.edge[2].max_dir <= map2.edge[0].max_dir:
                        
                        return True
                elif map2.edge[0].point == 1:
                    if map1.edge[2].min_dir <= map2.edge[0].max_dir and map2.edge[0].max_dir <= map2.edge[2].max_dir:
                        
                        return True
                else:
                    '''elif (map1.edge[2].min_dir< map2.edge[0].max_dir) or (map1.edge[2].max_dir > map2.edge[0].min_dir) :
                    ee_match = 1
                    return ee_match'''
                    return False
            return False
        if status == 2:
            if r1 == 1 and l2 ==1:
                if map1.edge[3].point == 1:
                    if map2.edge[1].min_dir <= map1.edge[3].max_dir and map1.edge[3].max_dir <= map2.edge[1].max_dir:
                        
                        return True
                elif map2.edge[2].point == 1:
                    if map1.edge[3].min_dir <= map2.edge[1].max_dir and map2.edge[1].max_dir <= map2.edge[3].max_dir:
                        
                        return True
                else:
                    '''elif (map1.edge[3].min_dir< map2.edge[1].max_dir) or (map1.edge[3].max_dir > map2.edge[1].min_dir) :
                    ee_match = 1
                    return ee_match'''
                    return False
            return False
        if status == 3:
            if t1 == 1 and b2 ==1:
                if map1.edge[0].point == 1:
                    if map2.edge[2].min_dir <= map1.edge[0].max_dir and map1.edge[0].max_dir <= map2.edge[2].max_dir:
                        
                        return True
                elif map2.edge[2].point == 1:
                    if map1.edge[0].min_dir <= map2.edge[2].max_dir and map2.edge[2].max_dir <= map2.edge[0].max_dir:
                        
                        return True
                else:
                    '''elif (map1.edge[0].min_dir< map2.edge[2].max_dir) or (map1.edge[0].max_dir > map2.edge[2].min_dir) :
                    ee_match = 1
                    return ee_match'''
                    return False
            return False
        if status == 4:#check map1 and map4
            if l1 == 1 and r2 ==1:
                if map1.edge[1].point == 1:
                    if map2.edge[3].min_dir <= map1.edge[1].max_dir and map1.edge[1].max_dir <= map2.edge[3].max_dir:
                        
                        return False
                elif map2.edge[3].point == 1:
                    if map1.edge[1].min_dir <= map2.edge[3].max_dir and map2.edge[3].max_dir <= map2.edge[1].max_dir:
                        
                        return False
                elif (map1.edge[1].min_dir< map2.edge[3].max_dir) or (map1.edge[1].max_dir > map2.edge[3].min_dir) :
                    
                    return False
                else:
                    return True
            return True

    def availible(self,map1,edge):
        t1 = 0
        b1 = 0
        l1 = 0
        r1 = 0
        for yam in map1.edge:
            if yam.dire == 1 and yam.valid ==1:
                t1 = 1
            if yam.dire ==2 and yam.valid ==1:
                l1 = 1
            if yam.dire == 3 and yam.valid ==1:
                b1 = 1
            if yam.dire ==4 and yam.valid ==1:
                r1 = 1
        match edge:
            case 0 :return t1
            case 1 :return l1
            case 2 :return b1
            case 3 :return r1

    def first_c(self):
        while True:#first map， check bottom is availible
            ran = np.random.randint(0,999)
            name = self.img_path + str(ran) + 'route_input.png'
            img = cv2.imread(name)
            map_c = map(img)
            map_c.app_edge()
            for i in range(0,3):
                if self.availible(map_c,2) :
                    self.maplist.append(map_c)
                    return 
                map_c = self.rotate(map_c)
    def second_b(self):
        while True:#second map， check right is availible and match top2_bottom1
            ran = np.random.randint(0,999)
            name = self.img_path + str(ran) + 'route_input.png'
            img = cv2.imread(name)
            map_d = map(img)
            map_d.app_edge()
            for i in range(0,3):
                if self.availible(map_d,3):
                    if self.can_mer(1,self.maplist[0],map_d):
                        self.maplist.append(map_d)
                        return
                map_d = self.rotate(map_d)
    def third_r(self):
        while True:#third map， check top is availible and match left3_right2
            ran = np.random.randint(0,999)
            name = self.img_path + str(ran) + 'route_input.png'
            img = cv2.imread(name)
            map_r = map(img)
            map_r.app_edge()
            for i in range(0,3):
                if self.availible(map_r,0):
                    if self.can_mer(2,self.maplist[1],map_r):
                        self.maplist.append(map_r)
                        return
                map_r = self.rotate(map_r)
    def fourth_t(self):
        while True:#fourth map， match bottom4_top3 and we want the other edge isn't left so not connect map1
            ran = np.random.randint(0,999)
            name = self.img_path + str(ran) + 'route_input.png'
            img = cv2.imread(name)
            map_t = map(img)
            map_t.app_edge()
            for i in range(0,3):
                if not self.availible(map_t,1):
                    if self.can_mer(3,self.maplist[2],map_t):
                        if self.can_mer(4,map_t,self.maplist[0]):
                            self.maplist.append(map_t)
                            return
                map_t = self.rotate(map_t)
    def poolmerge(self):
        result = np.concatenate(
                        (np.concatenate((self.maplist[0].map,self.maplist[1].map),axis=0),
                        np.concatenate((self.maplist[3].map, self.maplist[2].map), axis=0)),axis=1)
        name = './Multi_pin_routing_solver/amaster/secdata/route_input.png'
        name0 = './Multi_pin_routing_solver/amaster/secdata/0route_input.png'
        name1 = './Multi_pin_routing_solver/amaster/secdata/1route_input.png'
        name2 = './Multi_pin_routing_solver/amaster/secdata/2route_input.png'
        name3 = './Multi_pin_routing_solver/amaster/secdata/3route_input.png'
        cv2.imwrite(name,result) 
        cv2.imwrite(name0,self.maplist[0].map)
        cv2.imwrite(name1,self.maplist[1].map)
        cv2.imwrite(name2,self.maplist[2].map)
        cv2.imwrite(name3,self.maplist[3].map)
        print('merge completed !!')
name = './Multi_pin_routing_solver/amaster/firstdata/'
gen = merge(name)
gen.first_c()
gen.second_b()
gen.third_r()
gen.fourth_t()
gen.poolmerge() 


