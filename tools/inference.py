
vedio_size = 8  # 模型接受输入的T大小
        video_name = "input.avi"
        input_video_path = os.path.join(os.getcwd(), video_name)

        inference_model(input_video_path, model, vedio_size)
        # inference_and_draw(input_video_path, model, vedio_size)


# 简单的推理端
@torch.no_grad()        # 禁用此函数的梯度计算
def inference_and_draw(input_video_path, model, video_size):
    import cv2
    model.eval()    # 这个设置一定要加上，会禁用一些跟训练相关的操作如Dropout，BatchNorm

    frameSize = (640, 360)
    video_writer = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

    # 数据准备
    assert os.path.exists(input_video_path), "Vedio path {:s} may be wrong!".format(input_video_path)
    # 按照时间间隔来取帧
    origin_frames, _, info = torchvision.io.read_video(input_video_path, pts_unit="sec")

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
    with open(os.path.join(root_path, "../dataset/violent_classification.json"), 'r') as f:
        classes = f.read()
        labels = json.loads(classes)

    # 开始推理
    input_frames = input_frames[1: ,:, :, :, :]
    for outer_idx, input_frame in enumerate(input_frames):
        input_frame = input_frame.unsqueeze(0)  # BCTHW
        model_path = "../models_parameters/best_model.pt"
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


@torch.no_grad()        # 禁用此函数的梯度计算
def inference_model(input_video_path, model, video_size):
    model.eval()  # 这个设置一定要加上，会禁用一些跟训练相关的操作如Dropout，BatchNorm
    # 数据准备
    assert os.path.exists(input_video_path), "Vedio path {:s} may be wrong!".format(input_video_path)
    # 按照时间间隔来取帧
    origin_frames, _, info = torchvision.io.read_video(input_video_path, pts_unit="sec")

    fps = info["video_fps"]
    sampling_rate = 2   # 采样率是指多少秒采样一次, 单位：s
    sampling_scale = int(((sampling_rate * fps) // video_size) * video_size)

    origin_frames = origin_frames[0: ((origin_frames.shape[0] // sampling_scale) * sampling_scale), :, :, :] # 如果帧数不能整除则去掉末尾的一些帧
    # reshape to [time_batch, sampling_scale, H, W, C]
    origin_frames = origin_frames.reshape((int(origin_frames.shape[0] / sampling_scale), sampling_scale, origin_frames.shape[1], origin_frames.shape[2], origin_frames.shape[3]))
    basic_transform = T.Compose(
        [
            T.Resize(size=[128, 171]),
            T.CenterCrop(112),
            T.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
        ]
    )
    # 标签提取
    import json
    labels = {}
    with open(os.path.join(root_path, "violent_classification.json"), 'r') as f:
        classes = f.read()
        labels = json.loads(classes)
    origin_frames = origin_frames.permute(0, 1, 4, 2, 3)
    import itertools
    input_frames = torch.zeros((1, 3, int(sampling_scale / video_size) + 1, 112, 112))
    for slice in origin_frames:
        temp_list = []
        for frame in itertools.islice(slice, 0, sampling_scale, int(sampling_scale / video_size)):
            temp_list.append(basic_transform(frame.to(torch.float64)))
        temp = torch.stack(temp_list, 0).permute(1, 0, 2, 3).unsqueeze(0)
        input_frames = torch.cat((input_frames, temp), dim=0)

    # 开始推理
    input_frames = input_frames[1: ,:, :, :, :]
    for outer_idx, input_frame in enumerate(input_frames):
        input_frame = input_frame.unsqueeze(0)  # BCTHW
        model_path = "../models_parameters/best_model.pt"
        device = model.device
        model_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(model_dict["state_dict"])
        output = model(input_frame)
        _, pred = torch.max(output, 1)
        print(labels[str(pred.item())])

