import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import cv2
import torch 
from torchvision.transforms import ToPILImage
import torchvision




class genMap():
     def __init__(self,num,h,w,pinnum,name):
          Osize = h/4
          self.size = h
          self.num = num
          self.h = h
          self.w = w
          self.pinnum = pinnum
          self.name = name

     def gmap(self,num,h,w):
          map_range = h*w*num
          map = np.zeros((w,h,num))
          return map
     
     def gpins(self,num,h,w,map):
          """在分辨度图像内选择点生成引脚,一次生成的数量为预设num"""  

          for i in range(num):
               select_pinx =  random.randint(0,w-1)
               select_piny =  random.randint(0,h-1)
               p = random.randint(0,7)
               if p <2:#B,R
                    map[select_pinx,0,0] = 1  
                    map[select_pinx,0,2] = 1  
                    map[h-1,select_piny,2] = 1
                    map[h-1,select_piny,0] = 1
               elif p <4:#B,T
                    map[select_pinx,0,0] = 1  
                    map[select_pinx,0,2] = 1  
                    map[select_piny,h-1,2] = 1
                    map[select_piny,h-1,0] = 1
               elif p<6:#B,L
                    map[select_pinx,0,0] = 1  
                    map[select_pinx,0,2] = 1  
                    map[0,select_piny,2] = 1
                    map[0,select_piny,0] = 1
               else:#L,R
                    map[0,select_pinx,0] = 1  
                    map[0,select_pinx,2] = 1  
                    map[h-1,select_piny,2] = 1
                    map[h-1,select_piny,0] = 1
          return map
          
     def gobstacle(self,Osize,map,h,w): 
          x_len = random.randint(0,Osize)
          y_len = random.randint(0,Osize)
          things = np.ones((x_len,y_len))
          x = random.randint(0,w-Osize-1)
          y = random.randint(0,h-Osize-1)
          zero_map = np.zeros((x_len,y_len))
          if (map[x:x+x_len,y:y+y_len,1].all() == 0):
               map[x:x+x_len,y:y+y_len,2] = 1
               return map
          else :
               print('no obstacle')
               return map
     def a2i(self,map):
          name = self.name
          filename = './' + str(name) + 'output.png'
          cv2.imwrite(filename, map*255)
     def gen(self,num,h,w,pins):
          pinnums = pins
          map = self.gmap(num,h,w)
          pinmap = self.gpins(pinnums,h,w,map)
          '''plt.imshow(pinmap)
          plt.show()'''
          self.a2i(pinmap)
          return pinmap
"""sixteen generater"""
'''
h = w = 32
num = 3
pinnums = 2
Osize = int(4)
name = 1

genmap = genMap(num,h,w,pinnums,name)
map = genmap.gmap(num,h,w)
pinmap = genmap.gpins(pinnums,h,w,map)'''
'''allmap = genmap.gobstacle(Osize,pinmap,h,w)
plt.imshow(pinmap)
plt.show()
print(pinmap.shape)
genmap.a2i(pinmap)'''

'''tensor = torchvision.transforms.ToTensor()(pinmap)
print(tensor.shape)
tensor = torchvision.transforms.ToPILImage()(tensor)
tensor.save('E:\EDA\code\Multi_pin_routing_solver\mydata/test.jpg')
"""genmap.a2i(allmap)"""'''
     
          



