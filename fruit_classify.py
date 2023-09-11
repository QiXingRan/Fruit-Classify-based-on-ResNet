# 数据预处理
import os
from functools import partial
from random import shuffle
import torch
from mmengine import to_2tuple
# from mmseg.models.utils import DropPath
# from mmseg.utils import get_root_logger
from mmseg import *
# from mmcv.runner import load_checkpoint
from torch.nn.init import trunc_normal_



# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
device = torch.device("cuda:0")
#%%
# 定义公共变量"水果"
name_dict = {"apple":0,"banana":1,"grape":2,"orange":3,"pear":4}
# 定义数据集路径
data_root_path = "data/fruits/"
# 测试集与训练集文件
test_file_path = data_root_path + "test.txt"
trainer_file_path = data_root_path + "trainer.txt"
# 记录每类水果有多少训练与测试图片
name_data_list = {}
trainer_list = []
test_list = []

# 将图片的路径、序号存入字典
def save_train_test_file(path,name):
    if name not in name_data_list:
        img_list = []
        img_list.append(path) # 将路径用列表存储
        name_data_list[name] = img_list # 使用字典name：path
    else:
        name_data_list[name].append(path) # 另一种情况直接存入

# 遍历数据集目录，提取出图片路径，分训练集、测试集
dirs = os.listdir(data_root_path) # 列出fruits目录下所有内容
for d in dirs:
    full_path = data_root_path + d # 拼出某水果一个完整路径
    if os.path.isdir(full_path): # 某水果的目录
        imgs = os.listdir(full_path) # 列出该水果下的img文件昵称
        for img in imgs:
            # 凑完整水果图片路径后，用自定义函数存入字典
            save_train_test_file(full_path + "/" + img,d)
    else:
        pass
#%%
# 将字典中的内容写入测试集/训练集文件
# 清空文件
with open(test_file_path, "w") as f:       #清空测试集文件
    pass
with open(trainer_file_path, "w") as f:     #清空训练集文件
    pass
#%%
# 遍历字典，区分测试、训练集
for name,img_list in name_data_list.items():
    i = 0
    num = len(img_list)
    print(f"{name}: {num}张") #打印xxx水果有xx张
    for img in img_list:
        if i%10 == 0:
            # 给添加的数据标记序号，每10个为测试集数据
             test_list.append(f"{img}\t{name_dict[name]}\n")
        else:
            trainer_list.append(f"{img}\t{name_dict[name]}\n")
        i += 1
#%%
#保存所需数据
with open(trainer_file_path, "w") as f:
    shuffle(trainer_list)                 #打乱数据顺序
    f.writelines(trainer_list)            #写入训练用数据

with open(test_file_path, "w") as f:
    f.writelines(test_list)               #写入测试用数据
#%%
# 读取数据
# train_mapper: 对训练样本数据进行整理，返回img数组和标签
def train_mapper(sample):                #sample是一个元组（样本文件中的一行）
    img, label = sample
    if not os.path.exists(img):
        print(f"{img}文件不存在")
    # 读取图片，并对图片进行大小尺寸变换
    img = torch.load(img) #读取
    img = torch.transpose(
            im=img,                     #要转换的图像
            resize_size=100,            #设置大小
            crop_size=100,
            is_color=True,              #彩色图像
            is_train=True)              #返回(3, 100, 100)数组
    # 将img中所有数据值压缩到0~1之间（数据归一化）
    img = img.flatten().astype("float32") / 255.0
    return img, label
#%%
# 读取训练集
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
class TrainerDataset(Dataset):
    def __init__(self, train_list):
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f]
        self.img_paths = [line.split("\t")[0] for line in lines]
        self.labels = [int(line.split("\t")[1]) for line in lines]


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        img = np.array(Image.open(img_path).convert('RGB'))

        transform = transforms.Compose([

            transforms.ToTensor(),
            transforms.Resize((224, 224)),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ])

        img = transform(img)
        return img, label

    def __len__(self):
        return len(self.img_paths)

def train_r(train_list, buffered_size=1024):
    train_dataset = TrainerDataset(train_list)
    return DataLoader(train_dataset, batch_size=buffered_size, shuffle=True)

trainer_reader = trainer_file_path
test_reader = test_file_path
print("训练集初始化完成")
#%%
from torch.utils.data import dataset

# 搭建模型
BATCH_SIZE = 8 # 一次读取数据的大小



# 定义输入数据: 图片、标签
image = torch.randn(BATCH_SIZE, 3, 100, 100)
label = torch.randn(BATCH_SIZE, 1).long()


train_dataset = TrainerDataset(trainer_reader)
train_loader1 = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TrainerDataset(test_reader)
test_loader1 = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
#%%
# 搭建CNN神经网络
from torch import nn
import torch.nn.functional as F
# 输入层 --> 卷积/池化/dropout --> 卷积/池化/dropout
#       --> 卷积/池化/dropout --> fc --> dropout --> 输出层

class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self, type_size):
        super(ConvolutionNeuralNetwork, self).__init__()
        # 第一组
        self.conv_pool_1 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32, # 卷积核数量
                      kernel_size=3), # 卷积核大小
            # relu激活函数
            nn.ReLU(),
            # maxpool层的核大小、步幅
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5) # 使用dropout层来减轻过拟合
        )
        # 第二组
        self.conv_pool_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5)
        )
        # 第三组
        self.conv_pool_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5)
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(in_features=64*3*3, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        # 输出层
        self.predict = nn.Linear(in_features=512, out_features=type_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #########################
        x = self.conv_pool_1(x)
        x = self.conv_pool_2(x)
        x = self.conv_pool_3(x)
        #########################
        x = x.view(x.size(0), -1)
        #########################
        x = self.fc(x)
        #########################
        x = self.predict(x)
        # x = self.softmax(x)
        return x
#%%
# 搭建ResNet神经网络
from torch import nn
import torch.nn.functional as F
# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        # 如果输入和输出通道数不同，需要使用1*1的卷积核进行降维
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, num_classes=5, in_channel=64):
        super(ResNet,self).__init__()
        #输入通道
        self.in_channels = in_channel
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        # 对输入的4D张量进行批量归一化操作,输入张量通道数64
        self.bn1 = nn.BatchNorm2d(64)
        # inplace=True表示该操作是原址操作，即在原输入张量上进行操作，节省内存空间。
        self.relu = nn.ReLU(inplace=True) # 激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.layer1 = self.make_layer(64,3) # 3x3x64
        self.layer2 = self.make_layer(128,1,stride=2) #3x3x128
        # self.layer2 = self.make_layer(128,1,stride=2)
        self.layer3 = self.make_layer(256,1,stride=2) #3x3x256
        self.layer4 = self.make_layer(512,1,stride=2) #3x3x512
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # 平均池化,空间维度压缩为1
        self.fc = nn.Linear(512,num_classes) # 线性层输入512维向量，输出num_classes维向量（特征\分类）

    # 定义残差块层
    def make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    # 前向传播
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape((-1,512))
        x = self.fc(x)
        return x

#%%
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 输入维度 隐藏层特征维度 输出维度 激活函数GELU dropout比例
        out_features = out_features or in_features # 无指定情况下 输出维度 = 输入维度
        hidden_features = hidden_features or in_features # ... 隐藏层维度 = 输入维度
        self.fc1 = nn.Linear(in_features, hidden_features) # 类似卷积网络的结构
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        # 确保维度可以被注意力头数量整除
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 # 缩放大小，默认为头维度的倒数平方根

        self.q = nn.Linear(dim, dim, bias=qkv_bias) # 对q矩阵的线性变换（是否偏置none）
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias) # kv矩阵线性变换，变成两个部分
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio # 缩减率R
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape # 输入参数为论文的 H W，x为输入特征
        # 查询矩阵的线性变换得到查询张量，并对维度进行重组和排列，以适应多头注意力的计算
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1) # 执行下采样降维操作
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # 拆分键值张量，得到键和值

        attn = (q @ k.transpose(-2, -1)) * self.scale # 点积计算并乘以缩放因子
        attn = attn.softmax(dim=-1) # 归一化，得到权重并使用dropout防止过拟合
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # 计算注意力加权后的值，并进行维度转置和重组，以适应线性变换层的输入（transformer的计算）
        x = self.proj(x) #  self.proj = nn.Linear(dim, dim) 线性变换操作
        x = self.proj_drop(x)

        return x

# 定义一个block
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim) # 归一化第一层
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio) # 注意力机制模块参数
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # 随机路径丢弃，否则无操作
        self.norm2 = norm_layer(dim) # 第二个层的归一化
        mlp_hidden_dim = int(dim * mlp_ratio) # mlp隐藏层维度：dim*倍率（4）
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) # MLP函数部分

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W)) # 将x进行归一化->注意力机制x H W->dropout
        x = x + self.drop_path(self.mlp(self.norm2(x))) # 上一步结果归一化->MLP模块->dropout

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # img patch 转换为二元组
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # 确保img patch之间可以整除
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        # 计算每个patch的H W
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W # patch数量(为什么没有除？)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 卷积层，将patch变为embedding， 输入3维（RGB） -> 经典768维向量
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape # 输入图像批量大小，通道数，高，宽
        # 通过卷积将patch转换为嵌入向量，展平并转置
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        # 经过以上操作后的HW，并且输出
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=5, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, F4=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths # 每个阶段的block数量
        self.F4 = F4
        self.num_stages = num_stages # 阶段数

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]
            # 第i轮的各个参数
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            trunc_normal_(pos_embed, std=.02)

        # init weights
        self.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        return
        # if isinstance(pretrained, str):
        #     logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
    # 初始化权重参数
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 位置编码 嵌入特征向量 H W
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            # 这段代码判断当前阶段的特征数量是否与图像块的数量相同。如果相同，说明当前阶段的特征已经对齐了位置编码，直接返回原始的位置编码
            return pos_embed
        else:
            # 不同采用插值操作，最终将位置编码调整为当前阶段特征尺寸（H，W）相匹配
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []
        # 图像batch大小
        B = x.shape[0]
        # 遍历阶段
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}") # 获取对应模块参数
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x) # 提取特征
            if i == self.num_stages - 1: # 调整位置编码
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block: # 传入block做转换
                x = blk(x, H, W)
            # reshape为(B, H, W, C)，再调整维度顺序为（B，C，H，W）
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x) # out列表储存结果

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # 仅返回第四阶段特征 或者 返回所有阶段特征
        if self.F4:
            x = x[3:4]

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            # 变为（B, 3, patch_size, patch_size）
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v # 结果保存字典
    return out_dict


class pvt_tiny(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_tiny, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

#%%
from torch import optim
import torch
#初始化模型
# predict = ConvolutionNeuralNetwork(type_size=5)
model = ResNet().to(device)
model2 = pvt_tiny().to(device)
#定义损失问题(交叉熵作为损失函数)
criterion = nn.CrossEntropyLoss()
#定义优化器(自适应梯度下降优化器)
optimizer = optim.Adam(model.parameters(), lr=0.001)
#优化器 学习率 defualt：Adam  0.001

#执行器
use_gpu = True

model2.train()
print(torch.cuda.is_available ())
print("ResNet初始化完成")
#%%
#from torchcrepe.load import model

# 训练模型
epoch_num = 11
times = 0
# 训练过程的可视化数据
iter_count = []
cost_list = []
accs_list = []
print("开始训练...")

times = 0

def train(model, train_loader1, criterion,optimizer, epoch):
    # model.train()
    acc = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader1,0):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        predicted = (output.argmax(dim=1) == target).sum().item()
        acc += predicted
        total += BATCH_SIZE


        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc: {}%'.format(
                    epoch, batch_idx * len(data), len(train_loader1.dataset),
                        100. * batch_idx / len(train_loader1), loss.item(), 100*acc/total))



