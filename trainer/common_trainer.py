import torch
from torch.optim import Adam
from torch import nn
import os
import time
from data import data_process
from tqdm import tqdm

# 设置需要训练的参数
def setParam(model, layers_need_to_train):
    param_need_to_train = []
    print("Params need to learn:")
    if layers_need_to_train == 1:
        for name, param in model.named_parameters():
            if name == "model.fc.weight" or name == "model.fc.bias":
                param.requires_grad = True
                param_need_to_train.append(param)
                print(name)
                continue
            param.requires_grad = False
    elif layers_need_to_train == 2:
        pass
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True
            param_need_to_train.append(param)
            print(name)
        print('\n')
    return param_need_to_train

def train(model, Dataloader, writer, **kwargs):
    print("Parametres setting...")
    param_need_to_train = setParam(model, kwargs['layers_need_to_train'])
    # 制作损失函数和优化器
    loss_func = nn.CrossEntropyLoss()
    optim = Adam(param_need_to_train, kwargs['lr'])
    train_count = 0  # 用于计算runtime_loss和输出图像
    max_precision = 0
    val_epoch = 0

    epoch_loop = tqdm(range(kwargs['epoches']), total=kwargs['epoches'])
    print("Training...")
    for epoch in epoch_loop:
        print("Epoch:{:}".format(epoch))
        runtime_loss = 0
        precision = 0
        with torch.enable_grad():
            model.train()
            print("Learning rate = {}".format(optim.state_dict()['param_groups'][0]['lr']))
        data_num = 0
        correct_num = 0
        data_loop = tqdm(Dataloader, total=len(Dataloader))
        for data in data_loop:
            video, target = data
            data_num += video.size(dim=0)
            # 展示数据
            data_process.showData(video,[writer])
            # 梯度清零，如果不清零梯度就会叠加
            optim.zero_grad()
            with torch.set_grad_enabled(True):
                # 计算输出
                output = model(video)
                # 计算损失
                loss = loss_func(output, target)
                # 迭代和更新
                loss.backward()
                optim.step()
                train_count += 1
                runtime_loss += loss
            if (train_count % 5 == 0):  # 每5次迭代输出一次平均loss
                writer.add_scalar("Train/Loss", runtime_loss, train_count )
                print("runtime_loss={}".format(runtime_loss / 5))
                runtime_loss = 0
            _, preds = torch.max(output, dim=1)
            # 展示正确识别的图片
            data_process.showData(video, [writer], preds, target)
            correct_num += torch.sum(preds == target)
        # 计算精度
        precision = correct_num / data_num * 100        # 计算一个epoch下来的精度
        print("Precision = {}%".format(precision))
        # 更新最优参数
        state = {
            "state_dict": model.state_dict(),
            **kwargs
        }

        torch.save(state, os.path.join(model.save_model_path, "best_model.pt"))
        writer.add_scalar("{:s}/Precision".format("Train"), precision, epoch)
    writer.close()