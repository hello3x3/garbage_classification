import os
import torch
import copy, time
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from build import device, model, dataloaders, path

epochs = 50
lr = 0.1

torch.backends.cudnn.benchmark = True

# 创建损失函数
loss_func = nn.CrossEntropyLoss()
loss_func.to(device)

counter = 0
steps = {'train': 0, 'test': 0}

valid_loss_min = np.Inf
save_num = 0

best_acc = 0
train_log_file = os.path.join(path, "train/train.log")

if os.path.exists(train_log_file):
    print("log was built")
else:
    open(train_log_file, "w", encoding="utf-8")

for epoch in range(epochs):
    # 动态调整学习率
    if counter == 10:
        counter = 0
        lr = lr * 0.5

    # 在每个epoch里重新创建优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print('Epoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    # 训练和验证 每一轮都是先训练train 再验证valid
    for phase in ['train', 'test']:
        # 调整模型状态
        if phase == 'train':
            model.train()  # 训练
        else:
            model.eval()  # 验证

        # 记录损失值
        running_loss = 0.0
        # 记录正确个数
        running_corrects = 0

        # 一次读取一个batch里面的全部数据
        for inputs, labels in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 梯度清零
            optimizer.zero_grad()
            # 只有训练的时候计算和更新梯度
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = loss_func(outputs, labels)

                # torch.max() 返回的是一个元组 第一个参数是返回的最大值的数值 第二个参数是最大值的序号
                _, predictions = torch.max(outputs, 1)  # 前向传播 这里可以测试 在valid时梯度是否变化

                # 训练阶段更新权重
                if phase == 'train':
                    loss.backward()  # 反向传播
                    optimizer.step()  # 优化权重

            # 计算损失值
            running_loss += loss.item() * inputs.size(0)  # loss计算的是平均值，所以要乘上batch-size，计算损失的总和
            running_corrects += (predictions == labels).sum()  # 计算预测正确总个数
            # 每个batch加1次
            steps[phase] += 1

        # 一轮训练完后计算损失率和正确率
        epoch_loss = running_loss / len(dataloaders[phase].sampler)  # 当前轮的总体平均损失值
        epoch_acc = float(running_corrects) / len(dataloaders[phase].sampler)  # 当前轮的总正确率

        print('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format(phase, epoch_loss, counter, epoch_acc))

        if phase == 'test':
            # 得到最好那次的模型
            if epoch_acc > best_acc:  # epoch_acc > best_acc:
                best_acc = epoch_acc
                valid_loss_min = epoch_loss

                # 保存当前模型
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                # 只保存最近2次的训练结果
                save_num = 0 if save_num > 1 else save_num
                save_name_t = os.path.join(path, f"ckpt/best_model_{save_num}.pth")
                torch.save(state, save_name_t)  # \033[1;31m 字体颜色：红色\033[0m
                print("已保存最优模型，准确率:\033[1;31m {:.2f}%\033[0m，文件名：{}".format(best_acc * 100, save_name_t))
                if best_acc > 0.90:
                    torch.save(state, os.path.join(path, f"ckpt/resnet_{best_acc * 100:.2f}%.pth"))
                with open(train_log_file, "a", encoding="utf-8") as f:
                    t = time.localtime()
                    f.write(f"准确率: {best_acc*100:.2f}%, epoch: {epoch+1}, 时间: {t.tm_hour}时{t.tm_min}分, "
                            f"loss: {epoch_loss:.4f}\n")
                save_num += 1
                counter = 0
            else:
                counter += 1
    print('当前学习率 : {:.7f}\n'.format(optimizer.param_groups[0]['lr']))


# 训练结束
print('*'*50)
print('任务完成！')
print('最高验证集准确率: {:4f}'.format(best_acc))
save_num = save_num - 1
save_num = save_num if save_num < 0 else 1
save_name_t = 'best_model_{}.pth'.format(save_num)
print('最优模型保存在：{}'.format(save_name_t))
