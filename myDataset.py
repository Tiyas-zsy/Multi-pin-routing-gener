import os
import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class my_Dataset(Dataset):

    def __init__(self, root, img_shape=(128, 128)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return 2400

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        '''img = cv2.imread(path)'''
        img = Image.open(path).convert('RGB')
        pipeline = transforms.Compose([
            transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor()
        ])
        return pipeline(img)
    #img 0:pin,1:obst,2:routed


def get_dataloader(root=r'E:\EDA\code\Multi_pin_routing_solver\amaster\128data', **kwargs):
    dataset = my_Dataset(root, **kwargs)
    return DataLoader(dataset, 16, shuffle=True,num_workers=0)


if __name__ == '__main__':
    dataloader = get_dataloader()
    img = next(iter(dataloader))
    print(img.type)
    print(img.shape)
    '''img[:,2,:,:,] = 0'''
    # Concat 4x4 images
    N, C, H, W = img.shape
    assert N == 16
    img = torch.permute(img, (1, 0, 2, 3))
    img = torch.reshape(img, (C, 4, 4 * H, W))
    img = torch.permute(img, (0, 2, 1, 3))
    img = torch.reshape(img, (C, 4 * H, 4 * W))
    img = transforms.ToPILImage()(img)
    img.save('E:\EDA\code\Multi_pin_routing_solver/tmp.png')
