import os
import glob
import csv

class_to_num = {}
path = 'Multi_pin_routing_solver/mydata'
class_name_list = os.listdir(path)

for class_name in class_name_list:
    class_to_num[class_name] = len(class_to_num.keys())

image_dir = []
for class_name in class_name_list:
    image_dir += glob.glob(os.path.join(path,class_name,'*jpg'))

with open('data_csv.py',mode='w',newline='') as d:
    writer = csv.writer(d)
    for image in image_dir:
        class_name = image.splitg(os.sep)[-2]
        label = class_to_num(class_name)
        writer.writerow([image,label])