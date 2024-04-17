import os
import sys
import glob
import torch
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

IMG_SIZE = (224, 224)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# train_transform = transforms.Compose(
#     [
#         # transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
#         # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
#         # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
#         # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
#         # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
#         # # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相 全部是随机变化
#         # transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
#     ]
# )

# test_transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]
# )


# 类别名
label_name = [
    "battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass",
    "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]
 
# 将类别保存到字典中
label_dict = {}

# 将字符串类别转化成数字
for idx, name in enumerate(label_name):
    label_dict[name] = idx


# 返回通过PIL读取的图片数据并转换成RGB格式
def default_loader(path):
    return Image.open(path).convert("RGB").resize(IMG_SIZE)


# MyDataset需要完成三个函数的定义
class MyDataset(Dataset):
    # 初始化函数
    def __init__(self, img_list, transform=None, loader = default_loader):
        super(MyDataset, self).__init__()
        # 定义数据列表
        imgs = []
        for img_item in img_list:
            # 获取每一张图片的路径，格式如下
            #"datasets/garbage/train/airplane/aeroplane_s_000021.png"
            # 获取类别名
            if sys.platform == "linux":
                img_label_name = img_item.split("/")[-2]
            else:
                img_label_name = img_item.split("\\")[-2]
            # 保存图片路径和类别名对应的id
            imgs.append([img_item, label_dict[img_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    # 定义数据的读取和增强，返回数据和类别
    def __getitem__(self, index):
        # index：训练时传入的索引值
        img_path, img_label = self.imgs[index]
        # 读取图片
        img_data = self.loader(img_path)
        # 数据增强
        if self.transform is not None:
            img_data = self.transform(img_data)
 
        return img_data, img_label
 
    # 返回样本的数量
    def __len__(self):
        return len(self.imgs)


# 训练集图片路径
train_img_path = glob.glob(os.path.join(path, r"datasets/garbage/train/*/*.jpg"))
# 测试集图片路径
test_img_path = glob.glob(os.path.join(path, r"datasets/garbage/val/*/*.jpg"))

# 训练集数据增强
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
    ]
)

# 测试集数据增强
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
    ]
)

train_dataset = MyDataset(train_img_path, transform=train_transform)
test_dataset = MyDataset(test_img_path, transform=test_transform)

# 加载数据
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=0)

dataloaders = {
    "train": train_loader,
    "test": test_loader
}


# 在原有models.resnet基础上修改
model = torchvision.models.resnet50()
# 修改全连接层的输出
num = model.fc.in_features
model.fc = nn.Linear(num, 12)
model.to(device)
