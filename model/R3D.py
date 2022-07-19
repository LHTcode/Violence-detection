import os

import torch
from torch import nn
from torchvision import models
import time

class R3D_18(nn.Module):
    def __init__(self, root_path, is_train_phase: bool=True, pretrained=True):
        super(R3D_18, self).__init__()
        self.pretrained = pretrained
        self.root_path = root_path
        self.is_train_phase = is_train_phase
        self.save_model_path = self.getSaveDir()

        if torch.cuda.is_available():
            self.device = "cuda:0"
            print("Using GPU...")
        else:
            self.device = "cpu"
            raise RuntimeError("Can't use cuda")    # 我的电脑用不了cpu训练
            # print("Using CPU to train...")
        self.model = models.video.r3d_18(pretrained=self.pretrained,progress=True).to(torch.float64).to(self.device)
        self.model.fc = nn.Linear(self.model.fc.in_features,2,True).to(torch.float64).to(self.device)

    def forward(self,input):
        output = self.model(input.to(self.device))
        return output

    def getSaveDir(self):
        save_model_path = ""
        models_parameters_path = os.path.join(self.root_path, "models_parameters")
        if self.is_train_phase:
            # 每天新建一个文件夹储存best_model
            local_time = time.localtime()
            local_time_dir = str(local_time[0]) + '.' + str(local_time[1]) \
                             + '.' + str(local_time[2])
            save_model_path = os.path.join(models_parameters_path, local_time_dir)
            if not os.path.exists(save_model_path):  # 如果没有这个目录就创建
                os.mkdir(save_model_path)
        elif not self.is_train_phase:
            # 暂时是读取一个时间最新的bestmodel，后续将会读取一个精度最高的modle
            dir_list = os.listdir(models_parameters_path)
            for idx, dir in enumerate(dir_list):
                dir_list[idx] = [x for x in dir.split('.')]
            dir_list.sort(key=lambda x: (x[0], x[1], x[0]))     # 按照时间顺序降序排列
            lastest_date = '.'.join(dir_list[0])
            save_model_path = os.path.join(models_parameters_path, lastest_date, "best_model.pt")
            
        return save_model_path
