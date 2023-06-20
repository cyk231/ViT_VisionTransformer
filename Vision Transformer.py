import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# 残差模块的定义
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x  # 将输入与函数的输出相加作为最终的输出结果

# 层归一化模块的定义
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) # 做层归一化处理

# MLP块的定义
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):# 保证输入输出的dim相同，因此可以堆叠多个Transformer Block
        super().__init__()
        # 定义第一个全连接层
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)# 使用Xavier均匀初始化权重
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)# 使用正态分布初始化偏置项
        self.af1 = nn.GELU() # 参考论文中使用GELU激活函数
        self.do1 = nn.Dropout(dropout) # 添加Dropout层, 防止过拟合

        # 定义第二个全连接层(同理)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std=1e-6)
        self.do2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.nn1(x) # 前向传播：第一个全连接层
        x = self.af1(x) # 前向传播：GELU激活函数
        x = self.do1(x) # 前向传播：第一个Dropout层
        x = self.nn2(x) # 前向传播：第二个全连接层
        x = self.do2(x) # 前向传播：第二个dropout层

        return x

# 注意力模块的定义
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads # 注意力头数
        self.scale = dim ** -0.5  # 缩放因子，scale = 1/sqrt(dim)

        # 定义全连接层
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True) # 每个向量的Wq,Wk,Wv ,因此需要乘3倍(对照Transformer Encoder)

        # 使用Xavier均匀初始化权重和偏置项
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)

        # 定义全连接层和Dropout层
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)  # 拆分为多个注意力头

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # 计算点积注意力的分数

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # 进行Softmax归一化，计算注意力权重

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # v与Softmax内部的矩阵相乘
        out = rearrange(out, 'b h n d -> b n (h d)')  # 将多注意力头连接成一个矩阵，准备进行下一个编码器块
        out = self.nn1(out)
        out = self.do1(out)
        return out

# Transformer模块的定义，包含多个层
class Transformer(nn.Module):

    # Block层前应用层归一化模块LayerNormalize，Block层后应用残差连接模块Residual
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        # 按照标准的Transformer架构，每个层先是多头自注意力层Multi-Attention，再是多层感知器MLP，通过层归一化和残差模块进行连接
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),  # Attention
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))  # MLP
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # Attention
            x = mlp(x)  # MLP
        return x

# 图像Transformer模型
class ImageTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        # 要求image_size能够被patch_size整除
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

        num_patches = (image_size // patch_size) ** 2  # patch数
        patch_dim = channels * patch_size ** 2  # patch的维度

        self.patch_size = patch_size

        # 位置编码，用于表示每个patch的位置的信息
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_patches + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)  # 根据论文中的内容做初始化

        # patch投影层，将图像分割成patch并进行线性投影
        self.patch_conv = nn.Conv2d(3, dim, patch_size, stride=patch_size)  # 等效于 x matmul E, E= Embedd Matrix，这是线性patch投影

        # self.E = nn.Parameter(nn.init.normal_(torch.empty(BATCH_SIZE_TRAIN,patch_dim,dim)),requires_grad = True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # 类别标记的表示，同样根据论文中的内容做初始化
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout) # Transformer模块

        self.to_cls_token = nn.Identity() # 图像的分类结果由cls决定

        self.nn1 = nn.Linear(dim, num_classes)  # 根据论文：如果进行微调，只使用一个线性层，没有更多的隐藏层
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

        # 只有在训练大型数据集时才使用额外的隐藏层

        # self.af1 = nn.GELU()
        # self.do1 = nn.Dropout(dropout)
        # self.nn2 = nn.Linear(mlp_dim, num_classes)
        # torch.nn.init.xavier_uniform_(self.nn2.weight)
        # torch.nn.init.normal_(self.nn2.bias)
        # self.do2 = nn.Dropout(dropout)

    def forward(self, img, mask=None):
        p = self.patch_size

        x = self.patch_conv(img)  # 将图像划分为patch，并进行线性投影
        # x = torch.matmul(x, self.E)
        x = rearrange(x, 'b c h w -> b (h w) c')  # 将patch重排成序列的形式

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1) # 扩展类别标记的表示
        x = torch.cat((cls_tokens, x), dim=1) # 将类别标记的表示与patch表示拼接起来
        x += self.pos_embedding # 加上位置编码（这里用的是加法，在论文中还有提到拼接concat的方法）
        x = self.dropout(x)

        x = self.transformer(x, mask)  # Transformer模块，主要计算过程

        x = self.to_cls_token(x[:, 0]) # 根据cls的信息作为预测图像分类的结果

        x = self.nn1(x)
        # x = self.af1(x)
        # x = self.do1(x)
        # x = self.nn2(x)
        # x = self.do2(x)

        return x




# 定义超参数batch_size
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST  = 100


# 图像数据预处理，包括随机水平翻转、随机旋转、随机仿射变换、转为张量、归一化
transform = torchvision.transforms.Compose(
    [torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
     torchvision.transforms.RandomAffine(8, translate=(.15, .15)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


'''
结论：当Vision Transformer(ViT)模型在超大规模的数据集上预训练，再在中小规模的图像分类数据集上做微调后，能够得到很好的效果，甚至优于ResNet
     若在中小规模数据集上做预训练，由于缺少归纳偏置，得到的模型效果可能不如普通的CNN
     
     
论文中，作者在JFT-300M数据集上进行预训练，使用TPUv3-core，模型训练天数为2.5k，最终测试得到在CIFAR-10数据集上的Accuracy能到达99.5%


考虑到GPU算力有限等原因，这里仅根据论文搭建了Vision Transformer的模型结构，并直接在CIFAR-10上做训练


取迭代轮次为150，搭建的ViT模型在CIFAR-10数据集上的Accuracy最终达到了78%左右，与实验6中LeNet的效果相仿(仅作参考)
'''


# 加载CIFAR-10数据集和DataLoader
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)


# 训练函数，包括模型训练、损失计算和反向传播更新参数
def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    # 在训练集上进行模型训练
    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


# 评估函数，用于计算测试集上的损失和准确率
def evaluate(model, data_loader, loss_history):
    model.eval()# 评估模型性能

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    # 在测试集上计算loss和accuracy
    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)# Softmax
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)

    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')



# 程序开始处
if __name__ == '__main__':

    N_EPOCHS = 150# 迭代次数

    model = ImageTransformer(image_size=32, patch_size=4, num_classes=10, channels=3, dim=64, depth=6, heads=8, mlp_dim=128)# ViT模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)# 参考论文中，使用Adam优化算法，取learning rate=0.003

    train_loss_history, test_loss_history = [], []

    for epoch in range(1, N_EPOCHS + 1):# 开始迭代训练模型
        print('Epoch:', epoch)
        start_time = time.time()
        train(model, optimizer, train_loader, train_loss_history)# 训练函数，包括模型训练、损失计算和反向传播更新参数
        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')# 训练时间
        evaluate(model, test_loader, test_loss_history)# 评估函数，用于计算测试集上的损失和准确率

    print('Execution time')

    PATH = "./model"
    torch.save(model.state_dict(), PATH)# 保存model到本地



    # =============================================================================
    # model = ViT()
    # model.load_state_dict(torch.load(PATH))
    # model.eval()# 评估模型性能
    # =============================================================================