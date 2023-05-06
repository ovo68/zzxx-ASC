import torch
# from model import *
# 准备数据集
from torch import nn
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter  # tensorboard显示

from model.SSKGNN import SSKModel
from utils import ZXinDataset
from utils import load_data

# 定义训练的设备   gpu 适用对象   数据（非dataset）model  loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
instances = load_data()
# for i in instances:
#     print(i)
#
# print()

train_dataset = ZXinDataset(instances)

# length 长度
train_data_size = len(train_dataset)

# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))

train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=3,
    shuffle=True
)

# 创建网络模型

net = SSKModel()
# net = net.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# loss_fn = loss_fn.to(device)
# 优化器

learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 3

# 添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤开始
    net.train()
    for data in train_dataloader:
        item, label, dt, kt, at = data
        # item = item.to(device)
        # targets = label.to(device)
        # adj_tensor = torch.randn(len(item), 80, 80)
        outputs = net(item, dt, kt, at)
        loss = loss_fn(outputs, label)

        # 优化器优化模型
        optimizer.zero_grad()  # 清理前一次梯度
        loss.backward()  # 反向求导
        optimizer.step()  # 更新网络数据

        total_train_step = total_train_step + 1
        print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), total_train_step)
        # if total_train_step % 100 == 0:
        #     print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

    # 测试步骤开始
    # net.eval()
    # total_test_loss = 0
    # total_accuracy = 0
    # with torch.no_grad():
    #     for data in test_dataloader:
    #         imgs, targets = data
    #         imgs = imgs.to(device)
    #         targets = targets.to(device)
    #         outputs = net(imgs)
    #         loss = loss_fn(outputs, targets)
    #         total_test_loss = total_test_loss + loss.item()
    #         accuracy = (outputs.argmax(1) == targets).sum()  # outputs.argmax(1) 横向比较 找出最大值的索引
    #         total_accuracy = total_accuracy + accuracy
    #
    # print("整体测试集上的Loss: {}".format(total_test_loss))
    # print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    # total_test_step = total_test_step + 1

    torch.save(net, "savemodel/myModel_{}.pth".format(i))
    print("模型已保存")

writer.close()  # 关闭 tensorborad
