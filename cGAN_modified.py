import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import cv2

from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from VAEgenerater_modified import VAEgen
from  myDataset import get_dataloader




os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class function():
    def __init__(self,gen,label):
        
        self.gen = gen
        self.label = label
        
    def tile_cnt(self,map) :
        y_ = torch.sum(map >= 0.5).item()#统计布线瓦片数量
        return y_
    def lr_distance(self):
        num_gen = self.tile_cnt(self.gen)
        num_label = self.tile_cnt(self.label)
        distance = abs(num_label - num_gen)
        return distance
    def net_save(self,model,model_name):
        
        torch.save(model.state_dict(), model_name)
    def to_image(self,input,num):
        
        #看一眼输入是什么
        '''cgen_routed_map = input.detach().to('cpu')
        plt.imshow(cgen_routed_map[0,0,:, :], cmap='viridis')
        plt.show()'''


        map = torch.zeros((8,3,128,128))

        agen_routed_map = input.to('cpu',non_blocking=True)
        print(agen_routed_map.shape)
        '''agen_routed_map = torch.from_numpy(agen_routed_map)'''
        map[:,2,:,:] = agen_routed_map[:,0,:,:]*255
        name = 'E:/EDA/code/Multi_pin_routing_solver/amaster/gendata/aaa' + str(num) + '.png'
        save_image(map,name,nrow = 4)
        ''' N, C, H, W = map.shape
        assert N == 16
        img = torch.permute(map, (1, 0, 2, 3))
        img = torch.reshape(img, (C, 4, 4 * H, W))
        img = torch.permute(img, (0, 2, 1, 3))
        img = torch.reshape(img, (C, 4 * H, 4 * W))
        img = transforms.ToPILImage()(img)

        name = 'E:/EDA/code/Multi_pin_routing_solver/amaster/gendata/' + str(num) + '.png'
        img.save(name)'''
        '''cv2.imwrite(name,map)'''

    
#预先准备

memory_summary = torch.cuda.memory_summary()
print(memory_summary)

generator_losses = []
gen_losses = []
org_losses = []
discriminator_losses = []

class Generator(nn.Module):
    def __init__(self,in_dim):
        super(Generator,self).__init__()
        self.embedding = nn.Embedding(10,label_emb_dim)
        self.model = nn.Sequential(
            nn.Linear(in_dim,64),
            torch.nn.GELU(inplace = True),
            nn.Linear(64,128),
            torch.nn.GELU(inplace = True),
            nn.Linear(128,256),
            torch.nn.GELU(inplace = True),
            nn.Linear(256,512),
            torch.nn.GELU(inplace = True),
            nn.Linear(512,1024),
            torch.nn.GELU(inplace = True),
            nn.Linear(1024,torch.prod(image_size,dtype=torch.int32)),
            nn.Tanh(),
            )

    def forward(self,z,labels):
         
         label_embedding = self.embedding(labels)
         z = torch.cat([z,label_embedding],axis = -1)
         output = self.model(z)
         image = output.reshape(z.shape[0],*image_size)

         return image

class Discriminator(nn.Module):
    def __init__(self) :
        super(Discriminator,self).__init__()
        image_size = [1,128,128]
        label_emb_dim = 32
        self.embedding = nn.Embedding(10,label_emb_dim)

        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size,dtype = np.int32),512),
            torch.nn.GELU(),

            torch.nn.utils.spectral_norm(nn.Linear(512,256)),
            torch.nn.GELU(),

            torch.nn.utils.spectral_norm(nn.Linear(256,128)),
            torch.nn.GELU(),

            torch.nn.utils.spectral_norm(nn.Linear(128,64)),
            torch.nn.GELU(),

            torch.nn.utils.spectral_norm(nn.Linear(64,32)),
            torch.nn.GELU(),

            torch.nn.utils.spectral_norm(nn.Linear(32,1)),
            nn.Sigmoid(),
            )
    def forward(self,image):
         '''label_embedding = self.embedding(labels)'''
          
         prob= self.model(image.reshape(image.shape[0],-1))
          
         return prob
    
#training
batch_size = 8

dataloader = get_dataloader()

generator = VAEgen()
discriminator = Discriminator()


g_optimizer = torch.optim.Adam(generator.parameters(),lr = 0.003)
d_optimizer = torch.optim.Adam(discriminator.parameters(),lr = 0.003)

loss_fn = nn.BCELoss()

labels_one = torch.ones(batch_size,1)
labels_zero = torch.zeros(batch_size,1)
g_name = 'Gen.pth'
d_name = 'disc.pth'
use_gpu =1
if use_gpu:
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
    labels_one = labels_one.to("cuda")
    labels_zero = labels_zero.to("cuda")

num_epoch = 100
latent_dim=128
start = time.time()
k = 101
for epoch in range(num_epoch):
    for i,mini_batch in enumerate(dataloader):
        #unrouted pins
        #数据集是BGR格式，输入是RGB格式，所以0层是引脚，1层是障碍，2层是布线
        '''#将数组数值为255的图片格式转化为数组数值为1的模式'''
        '''mini_batch = mini_batch[mini_batch > 0] = 1'''
        '''print(mini_batch.shape)'''
        vae_input = np.zeros((8,1,128,128),dtype=np.float32)
        vae_input[:,0,:,:] = mini_batch[:,0,:,:]
        t_vae_input = torch.from_numpy(vae_input).to("cuda")

        #routed labels
        a_labels= np.zeros((8,1,128,128),dtype=np.float32)
        a_labels[:,0,:,:] = mini_batch[:,2,:,:]
        t_labels = torch.from_numpy(a_labels).to("cuda")
        t_labels = t_labels.float()
        #生成输出
        pred_images = generator(t_vae_input)
        g_optimizer.zero_grad()
        #将输出概率转化为输出路径
        gen_routed_map = (pred_images > 0.5).float()
        
        
        '''print(gen_routed_map.type())'''
        '''cpred_images = pred_images.detach().to('cpu')
        plt.imshow(cpred_images[0,0,:128, :128], cmap='viridis')
        plt.show()'''
        '''gen_routed_map[gen_routed_map >= 0.5] = 1
        gen_routed_map[gen_routed_map < 0.5] = 0'''
        '''cgen_routed_map = gen_routed_map.detach().to('cpu')
        plt.imshow(cgen_routed_map[0,0,:128, :128], cmap='viridis')
        plt.show()'''

        #物理损失重构函数加上原始损失函数
        recons_loss = torch.abs(gen_routed_map - t_labels).mean()

        lr = function(pred_images,t_labels)
        lr_dist = lr.lr_distance()

        '''torch.autograd.set_detect_anomaly(True)'''
        gen_loss = generator.loss_fnn(lr_dist,pred_images,t_labels)#数量级：1k
        org_loss = loss_fn(discriminator(pred_images),labels_one)#数量级：100
        
        g_loss = generator.loss_fnn(lr_dist,pred_images,t_labels) 
        #+loss_fn(discriminator(pred_images),labels_one)

        g_loss.backward()
        '''torch.autograd.set_detect_anomaly(False)'''
        g_optimizer.step()
        memory_summary = torch.cuda.memory_summary()
        d_optimizer.zero_grad()
        real_loss =loss_fn(discriminator(t_labels),labels_one)
        fake_loss = loss_fn(discriminator(pred_images.detach()),labels_zero)
        d_loss = real_loss + fake_loss

        d_loss.backward()
        d_optimizer.step()
        
        if i%50 == 0:
            now = time.time()
            print('now is dataloader :',i)
            print('now is iteration :',epoch)
            print('now is ',int(now - start),'seconds')

            lr.net_save(generator,g_name)
            lr.net_save(discriminator,d_name)
        torch.cuda.empty_cache()
    k = k+1
    lr.to_image(gen_routed_map,k)
plt.plot(generator_losses, label='g_loss')
plt.plot(gen_losses, label='g_loss_vae')
plt.plot(org_losses, label='g_loss_d2g')
plt.plot(discriminator_losses, label='d_loss')

# 设置图表标题和坐标轴标签
plt.title('Multiple losses Plot')
plt.xlabel('iteration ')
plt.ylabel('loss')

# 添加图例
plt.legend()

# 显示图表
plt.show()

