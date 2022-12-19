import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from skimage import io


class Mydataset(Dataset):
    def __init__(self, image_dir):
        # 数据集地址
        self.image_dir = image_dir
        # 获取文件地址（列表）
        self.image_path = os.listdir(self.image_dir)

        # 相关预处理的初始化，将PIL或numpy数据转换并归一化为torch.FloatTensor类型
        self.toTensor = transforms.ToTensor()

        # 标准化预处理，给定均值和标准差，用公式channel = （channel - mean）/std进行规范化
        self.normalize = transforms.Normalize(
            mean=0.5,
            std=0.5
        )

    def __getitem__(self, idx):
        img_name = self.image_path[idx]  # 获取各个图片索引
        img_item_path = os.path.join(self.image_dir, img_name)  # 地址相加获取文件中各个图片的地址
        # 读取图像数据
        img_PIL = io.imread(img_item_path)

        # 避免totensor时HWC变为CHW对图像像素分布产生的影响，但不知道是不是要加这个操作
        # 从zxy到xyz
        img_PIL = img_PIL.transpose((1, 2, 0))

        # 调用预处理
        # totensor又会变成zxy
        img_tensor = self.data_preproccess(img_PIL)

        # 维度扩展，需要得到通道单通道1
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        return img_tensor


    def __len__(self):
        return len(self.image_path)  # 返回数据集长度


    def data_preproccess(self, data):
        # 数据预处理
        data = self.toTensor(data)
        data = self.normalize(data)
        return data


train_dir = r"D:\fsrcnn_demo2\label_A_crop"
test_dir = r"D:\fsrcnn_demo2\test_A_crop"

train_data = Mydataset(train_dir)
test_data = Mydataset(test_dir)



train_loader = DataLoader(dataset=train_data, batch_size=6, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=6, num_workers=0)

