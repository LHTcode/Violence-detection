import cv2
import os
import torch
from torchvision.transforms import transforms as T
import torchvision
import itertools
from model.R3D import R3D_18
from data import data_process
import sys

# base = os.path.abspath("../")
# sys.path.append(base)

# 简单的推理端
@torch.no_grad()        # 禁用此函数的梯度计算
def inference_and_draw(root_path, input_video_path, inference_model, video_size, **kwargs):
    import cv2
    inference_model.eval()    # 这个设置一定要加上，会禁用一些跟训练相关的操作如Dropout，BatchNorm

    frameSize = (640, 360)
    output_video_path = os.path.join(root_path, kwargs["output_video_name"])
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)
    # 按照时间间隔来取帧
    origin_frames, _, info = torchvision.io.read_video(input_video_path, pts_unit="sec")

    fps = info["video_fps"]
    sampling_rate = 2   # 采样率是每采样8帧送入推理网络所需的时间, 单位：s
    sampling_scale = int(((sampling_rate * fps) // video_size) * video_size)

    origin_frames = origin_frames[0: ((origin_frames.shape[0] // sampling_scale) * sampling_scale), :, :, :] # 如果帧数不能整除则去掉末尾的一些帧
    # reshape to [time_batch, sampling_scale, H, W, C]
    origin_frames = origin_frames.reshape((
        int(origin_frames.shape[0] / sampling_scale),
         sampling_scale,
         origin_frames.shape[1],
         origin_frames.shape[2],
         origin_frames.shape[3]
    ))
    # 数据预处理
    basic_transform = T.Compose(
        [
            T.Resize(size=[128, 171]),
            T.CenterCrop(112),
            T.Normalize(data_process.MEAN, data_process.STD)
        ]
    )
    temp_frames = torch.clone(origin_frames.permute(0, 1, 4, 2, 3))    # switch to [time_batch, sampling_scale, C, H, W]

    input_frames = torch.zeros((1, 3, int(sampling_scale / video_size) + 1, 112, 112))
    for slice in temp_frames:
        temp_list = []
        for frame in itertools.islice(slice, 0, sampling_scale, int(sampling_scale / video_size)):
            temp_list.append(basic_transform(frame.to(torch.float64)))
        temp = torch.stack(temp_list, 0).permute(1, 0, 2, 3).unsqueeze(0)
        input_frames = torch.cat((input_frames, temp), dim=0)

    # 标签提取
    labels = data_process.getLabel(os.path.join(root_path, "dataset"), kwargs['label_file_name'])
    # 开始推理
    print("Inferencing...")
    input_frames = input_frames[1: ,:, :, :, :]
    for outer_idx, input_frame in enumerate(input_frames):
        input_frame = input_frame.unsqueeze(0)  # BCTHW
        model_path = inference_model.save_model_path
        device = inference_model.device
        model_dict = torch.load(model_path, map_location=torch.device(device))
        inference_model.load_state_dict(model_dict["state_dict"])
        output = inference_model(input_frame)
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
            video_writer.write(output_frame[inner_idx])
    print("Successfully save output_video.avi!")
    video_writer.release()

if __name__ == '__main__':
    # 制作数据集
    root_path = os.path.abspath("../")

    # 制作模型
    inference_model = R3D_18(root_path, is_train_phase=False, pretrained=False)

    vedio_size = 8  # 模型接受输入的T大小
    video_name = "input.avi"
    input_video_path = os.path.join(root_path, video_name)
    assert os.path.exists(input_video_path), "Vedio path {:s} may be wrong!".format(input_video_path)

    dataset_setting = {
        "label_file_name": "violent_classification.json",
        "input_video_name": "input.avi",
        "output_video_name": "output.avi"
    }
    inference_and_draw(root_path, input_video_path, inference_model, vedio_size, **dataset_setting)