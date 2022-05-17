import random
import time
import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from LoadData import VideoDataset
from Model import R3D_18
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as T

# train+valid模块
def train_model(model, Dataloader, loss_func, optim, epochs):
    max_precision = 0
    writer = SummaryWriter()
    train_count = 0  # 用于计算runtime_loss和输出图像
    device = model.device
    is_new_model = 0
    val_epoch = 0
    model_path = ''
    local_time_dir = ''
    for epoch in range(epochs):
        print("Epoch:{:}".format(epoch))
        for phase in ["train", "valid"]:
            runtime_loss = 0
            precision = 0
            if phase == "train":
                model.train()
                print("Training...")
                print("Learning rate = {}".format(optim.state_dict()['param_groups'][0]['lr']))
            elif phase == "valid" and is_new_model == 1:    # 只有当模型参数被更新的时候才计算valid
                # model_path = "models/{:s}/model{:s}.pt".format(local_time_dir,str(val_epoch))
                model_path = "models/best_model.pt"
                if os.path.exists(model_path):
                    model_dict = torch.load(model_path,map_location=torch.device(device))
                    model.load_state_dict(model_dict["state_dict"])
                    is_new_model = 0    # 重新将模型置为待更新状态
                else:
                    raise RuntimeError("Can't find \'./model.pt\'")
                model.eval()    # 评估模式
                print("Validating...")
            else:
                break       # 如果模型参数是旧版则就不需要计算测试集了
            data_num = 0
            correct_num = 0
            for data in Dataloader[phase]:
                video, target = data
                data_num += video.size(dim=0)
                target = torch.tensor(target).to(torch.long).to(device)
                # 展示数据
                # VideoDataset.showData(video,[writer])
                # 梯度清零，如果不清零梯度就会叠加
                optim.zero_grad()
                with torch.set_grad_enabled(True if phase == "train" else False):
                    # 计算输出
                    output = model(video)
                    if phase == "train":
                        # 计算损失
                        loss = loss_func(output, target)
                        # 迭代和更新
                        loss.backward()
                        optim.step()
                        train_count += 1
                        runtime_loss += loss
                    if (phase == "train" and train_count % 5 == 0):  # 每5次迭代输出一次平均loss
                        writer.add_scalar("Train/Loss", runtime_loss, train_count )
                        print("runtime_loss={}".format(runtime_loss / 5))
                        runtime_loss = 0
                # 计算精度
                _, preds = torch.max(output, dim=1)
                print(output)
                print(preds)
                # 展示正确识别的图片
                VideoDataset.showData(video,[writer],preds,target)

                correct_num += torch.sum(preds == target)
            precision = correct_num / data_num * 100        # 计算一个epoch下来的精度
            print("Precision = {}%".format(precision))
            if precision >= max_precision + 0.03 and phase == "train":  # 找到训练时所有epoch中精度最高的那个模型参数,+0.03是为了使得参数更新有意义
                val_epoch += 1      # 叠加模型更新次数
                max_precision = precision
                state = {
                    "state_dict": model.state_dict(),
                    "optim": optim.state_dict()
                }
                # 每天新建一个文件夹储存model
                local_time = time.localtime()
                local_time_dir = str(local_time[0]) + '.' + str(local_time[1]) \
                                 + '.' + str(local_time[2])
                model_path = os.path.join(os.getcwd(),"models",local_time_dir)
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(state, "models/{:s}/model{:s}.pt".format(local_time_dir,str(val_epoch)))
                is_new_model = 1    # 将模型标志为新版
            writer.add_scalar("{:s}/Precision".format("Train" if phase == "train" else "Valid"), precision, epoch if phase == "train" else val_epoch)  # 制作两张图，一个Train一个Valid
    writer.close()

# 简单的推理端
@torch.no_grad()        # 禁用此函数的梯度计算
def inference_model(long_vedio_path, model, video_size):
    import cv2
    model.eval()    # 这个设置一定要加上，会禁用一些跟训练相关的操作如Dropout，BatchNorm

    long_vedio_path_list = os.listdir(long_vedio_path)
    long_vedio_path_list.sort(key=lambda x: int(x.split("input")[1].split(".avi")[0]))
    print(long_vedio_path_list)

    root_path = os.path.abspath("RWF-2000")
    data_path = os.path.join(root_path, "long_video")
    frameSize = (640, 360)
    video_writer = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

    for long_vedio in long_vedio_path_list:
        print("Now pred: {:s}".format(long_vedio.split("input")[1].split(".avi")[0]))
        # 数据准备
        input_abspath = os.path.join(data_path, long_vedio)
        assert os.path.exists(input_abspath), "Vedio path {:s} may be wrong!".format(input_abspath)
        # 按照时间间隔来取帧
        origin_frames, _, info = torchvision.io.read_video(input_abspath, pts_unit="sec")

        fps = info["video_fps"]
        sampling_rate = 2   # 采样率是每采样8帧送入推理网络所需的时间, 单位：s
        sampling_scale = int(((sampling_rate * fps) // video_size) * video_size)

        origin_frames = origin_frames[0: ((origin_frames.shape[0] // sampling_scale) * sampling_scale), :, :, :] # 如果帧数不能整除则去掉末尾的一些帧
        # reshape to [time_batch, sampling_scale, H, W, C]
        origin_frames = origin_frames.reshape((int(origin_frames.shape[0] / sampling_scale), sampling_scale, origin_frames.shape[1], origin_frames.shape[2], origin_frames.shape[3]))

        import itertools
        # 数据预处理
        basic_transform = T.Compose(
            [
                T.Resize(size=[128, 171]),
                T.CenterCrop(112),
                T.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
            ]
        )
        temp_frames = torch.clone(origin_frames.permute(0, 1, 4, 2, 3)).to(torch.float64)    # switch to [time_batch, sampling_scale, C, H, W]

        input_frames = torch.zeros((1, 3, int(sampling_scale / video_size) + 1, 112, 112))
        for slice in temp_frames:
            temp_list = []
            for frame in itertools.islice(slice, 0, sampling_scale, int(sampling_scale / video_size)):
                temp_list.append(basic_transform(frame))
            temp = torch.stack(temp_list, 0).permute(1, 0, 2, 3).unsqueeze(0)
            input_frames = torch.cat((input_frames, temp), dim=0)

        # 标签提取
        import json
        labels = {}
        with open(os.path.join(root_path, "violent_classification.json"), 'r') as f:
            classes = f.read()
            labels = json.loads(classes)

        # 开始推理
        input_frames = input_frames[1: ,:, :, :, :]
        for outer_idx, input_frame in enumerate(input_frames):
            input_frame = input_frame.unsqueeze(0)  # BCTHW
            model_path = "models/best_model.pt"
            device = model.device
            model_dict = torch.load(model_path, map_location=torch.device(device))
            model.load_state_dict(model_dict["state_dict"])
            output = model(input_frame)
            _, pred = torch.max(output, 1)
            # 绘制文字
            output_frame = origin_frames[outer_idx].numpy()
            org = (50, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1.5
            thickness = 3
            for inner_idx, image in enumerate(output_frame):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                color = (0, 0, 255) if pred.item() == 1 else (0, 255, 0)
                output_frame[inner_idx] = cv2.putText(image, labels[str(pred.item())], org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
                cv2.imshow("Frame", output_frame[inner_idx])
                video_writer.write(output_frame[inner_idx])
                cv2.waitKey(25)
    video_writer.release()



if __name__ == "__main__":

    state = "inference"
    # 制作数据集
    root_path = os.path.abspath("RWF-2000")
    batch_size = 8
    dataset = {x: VideoDataset(root_path, video_size=8, phase=x, transform=None) for x in ["train", "valid"]}
    Dataloader = {x: DataLoader(dataset[x], batch_size, shuffle=True) for x in ["train", "valid"]}

    # 制作模型
    model = R3D_18(pretrained=True)     # 使用训练好的参数作为初始化参数

    if state == "train":
        # 设置需要训练的参数
        '''
        layers_need_to_train    ==  0   --> 训练所有层
                                ==  1   --> 仅训练全连接层
                                ==  2   --> 待更新...
        '''
        layers_need_to_train = 0
        param_need_to_update = []
        print("Params need to learn:")
        if layers_need_to_train == 1:
            for name, param in model.named_parameters():
                if name == "model.fc.weight" or name == "model.fc.bias":
                    param.requires_grad = True
                    param_need_to_update.append(param)
                    print(name)
                    continue
                param.requires_grad = False
        elif layers_need_to_train == 2:
            pass
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True
                param_need_to_update.append(param)
                print(name)
            print('\n')
        # 制作损失函数和优化器
        loss_func = nn.CrossEntropyLoss()
        lr = 1e-4
        optim = optim.Adam(param_need_to_update, lr)
        # 开始训练
        train_model(model,Dataloader,loss_func,optim,epochs=30)

    elif state == "inference":
        vedio_size = 8  # 模型接受输入的T大小
        inference_model("/home/lihanting/PycharmProjects/DeepLearning/RWF-2000/long_video", model, vedio_size)
