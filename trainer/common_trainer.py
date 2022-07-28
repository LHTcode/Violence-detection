import torch
from torch.optim import Adam
from torch import nn
import os
from tools.valid import valid
from data import data_process
from tqdm import tqdm

# 设置需要训练的参数
def setParam(model, layers_need_to_train):
    param_need_to_train = []
    print("\nParams need to learn:")
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

def train(model: dict, Dataloader: dict, writer, **kwargs):
    #===================== init =====================
    train_model = model["train"]
    valid_model = model["valid"]
    train_dataloader = Dataloader["train"]
    valid_dataloader = Dataloader["valid"]
    print("\nParametres setting...")
    param_need_to_train = setParam(train_model, kwargs['layers_need_to_train'])
    # 制作损失函数和优化器
    loss_func = nn.CrossEntropyLoss()
    optim = Adam(param_need_to_train, kwargs['lr'])
    scheduler = optim.lr_scheduler.StepLR(optim, kwargs[step_size], kwargs[gamma])  # 学习率每7个epoch衰减成原来的1/10
    train_count = 0  # 用于计算runtime_loss和输出图像
    max_precision = 0
    val_count = 1

    epoch_loop = tqdm(range(kwargs['epoches']), total=kwargs['epoches'])
    #===================== train =====================
    print("\nTraining...")
    for epoch in epoch_loop:
        print("\nEpoch:{:}".format(epoch))
        runtime_loss = 0
        precision = 0
        with torch.enable_grad():
            train_model.train()
            print("Learning rate = {}".format(optim.state_dict()['param_groups'][0]['lr']))
        data_num = 0
        correct_num = 0
        data_loop = tqdm(train_dataloader, total=len(train_dataloader))
        for data in data_loop:
            video, target = data
            data_num += video.size(dim=0)
            # 展示数据
            data_process.showData(video,[writer])
            # 梯度清零，如果不清零梯度就会叠加
            optim.zero_grad()
            with torch.set_grad_enabled(True):
                # 计算输出
                output = train_model(video)
                # 计算损失
                loss = loss_func(output, target)
                # 迭代和更新
                loss.backward()
                optim.step()
                train_count += 1
                runtime_loss += loss
            if (train_count % 5 == 0):  # 每5次迭代输出一次平均loss
                writer.add_scalar("Train/Loss", runtime_loss, train_count )
                print("\nruntime_loss={}".format(runtime_loss / 5))
                runtime_loss = 0
            _, preds = torch.max(output, dim=1)
            # 展示正确识别的图片
            data_process.showData(video, [writer], preds, target)
            correct_num += torch.sum(preds == target)
        scheduler.step() #更新学习率
        # 计算精度
        precision = correct_num / data_num * 100        # 计算一个epoch下来的精度
        # 更新最优参数
        if precision >= max_precision:
            print("\nSaving new model...")
            max_precision = precision
            state = {
                "state_dict": train_model.state_dict(),
                **kwargs
            }
            torch.save(state,train_model.save_model_path)
            print("\nSuccess!")
        print("\nTrain Precision = {}%".format(precision))
        writer.add_scalar("Train/Precision", precision, epoch)
        # ======================== 验证部分 ========================
        if val_count % kwargs["eval_interval"] == 0:
            val_count = 0
            valid(valid_model, valid_dataloader, epoch, writer)
            print("\nNow backout to train...")
        val_count += 1
    writer.close()
