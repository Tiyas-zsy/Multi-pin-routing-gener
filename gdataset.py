import torch
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
'''class makeDataSet(Dataset):
    def __init__(self,csv_filename,resize,mode):
        super(makeDataSet,self).__init__()

        self.cvs_filename = csv_filename
        self.resize = resize
        self.image,self.label = self.load_csv()

        if mode == 'train':
            self.image = self.image[:int(0.6*len(self.image))]
            self.label = self.label[:int(0.6*len(self.label))]
        
        if mode == 'val':#20%
            self.image = self.image[int(0.6*len(self.image))int(0.8*len(self.image))]
            self.label = self.label[int(0.6*len(self.label))int(0.8*len(self.label))]
        else:
            self.image = self.image[int(0.8*len(self.image))int(len(self.image))]
            self.label = self.label[int(0.8*len(self.label))int(len(self.label))]

    def load_csv(self):
        image,label = [],[]
        with open (self.csv_filename) as d:
            reader = csv.reader(d)
            for row in reader:
                i,l=row
                image.append(int(i))
                label.append(int(i))
        return image,label'''
    
'''    def __len__(self):
        return len(self.image)

    def __getItem__(self,idx):
        tf = transforms.Compose([lambda x:Image.open(x).convert('RGB',
                                transform.Resize(self.resize),
                                transforms.ToTensor())])
        image_tensor = tf(self.image[idx])
        label_tensor = torch.tensor(self.label[idx])
        return image_tensor,label_tensor'''



# 创建一个随机数组，所有数值都在(0, 1)之间
map = np.random.rand(3, 3)
print(map)

# 将大于0.5的元素变为1，小于等于0.5的元素变为0
map[map > 0.5] = 1
map[map <= 0.5] = 0

print(map)


