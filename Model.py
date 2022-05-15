import torch
from torch import nn
from torchvision import models

class R3D_18(nn.Module):
    def __init__(self,pretrained=True):
        super(R3D_18, self).__init__()
        self.pretrained = pretrained
        if torch.cuda.is_available():
            self.device = "cuda:0"
            print("Using GPU to train...")
        else:
            self.device = "cpu"
            raise RuntimeError("Can't use cuda")    # 我的电脑用不了cpu训练
            # print("Using CPU to train...")
        self.model = models.video.r3d_18(pretrained=self.pretrained,progress=True).to(torch.float64).to(self.device)
        self.model.fc = nn.Linear(self.model.fc.in_features,2,True).to(self.device).to(torch.float64)

    def forward(self,input):
        output = self.model(input.to(self.device))
        return output