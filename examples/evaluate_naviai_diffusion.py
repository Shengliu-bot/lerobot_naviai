"""
此脚本演示如何从数据集中抽取样本评估预训练策略
"""

from pathlib import Path
import h5py
import imageio
import numpy
import torch
import random
import sys
import time

sys.path.append("/home/diffusion/src/lerobot-main/lerobot")
sys.path.append("/home/diffusion/src/lerobot-main")
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# 创建目录存储评估视频
output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# 加载预训练策略
pretrained_policy_path = Path("/home/diffusion/src/lerobot-main/outputs/train/050000/pretrained_model")
policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.eval()

# 检查GPU可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU可用。使用设备:", device)
else:
    device = torch.device("cpu")
    print(f"GPU不可用。使用设备: {device}。推理速度将较慢。")
    policy.diffusion.num_inference_steps = 10

policy.to(device)

# 加载数据集
dataset_path = "/home/diffusion/src/lerobot-main/data/record_2.hdf5"
for i in range(200):
    sample_idx = i
    with h5py.File(dataset_path, 'r') as f:
        # 随机选择一个样本
        total_samples = len(f['observation/end_effector/right_tcp'])
        # sample_idx = random.randint(0, total_samples - 1)
        
        # 获取样本数据
        observation = {
            'end_effector': f['observation/end_effector/right_tcp'][sample_idx][()],
            'realsense_rgb': f['observation/images/realsense_rgb'][sample_idx][()]
        }
        true_action = f['actions/end_effector/right_tcp'][sample_idx][()] -  f['observation/end_effector/right_tcp'][sample_idx][()]

    # 准备模型输入
    state = torch.from_numpy(observation['end_effector'])
    image1 = torch.from_numpy(observation['realsense_rgb'])

    # 确保state维度正确
    current_dim = state.shape[0]
    if current_dim < 7:
        random_padding = torch.rand(7 - current_dim)
        state = torch.cat([state, random_padding])
        print("原始state维度:", current_dim)
        print("填充后的state维度:", state.shape)

    # 处理图像数据
    image1 = image1.to(torch.float32) / 255
    image1 = image1.permute(2, 0, 1)

    # 移动数据到设备
    state = state.to(device, non_blocking=True)
    image1 = image1.to(device, non_blocking=True)

    # 添加batch维度
    state = state.unsqueeze(0)
    image1 = image1.unsqueeze(0)

    # 创建策略输入字典
    model_input = {
        "observation.state": state,
        "observation.images.realsense_rgb": image1,
    }

    # 预测动作
    with torch.inference_mode():
        start_time = time.time()
        

        
        predicted_action = policy.select_action(model_input)
        end_time = time.time()
        print(f"当前样本推理时间: ",end_time-start_time)
    # 转换为numpy数组用于比较
    predicted_action = predicted_action.squeeze(0).cpu().numpy()

    # 计算预测动作与真实动作的误差
    error = numpy.mean(numpy.abs(predicted_action - true_action))
    print("预测动作:", predicted_action)
    print("真实动作:", true_action)
    print("平均绝对误差:", error/true_action * 100)

    # 可视化结果
    fig_path = output_directory / f"action_comparison_{i}.png"
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(observation['realsense_rgb'])
    plt.title("RGB1")

    plt.subplot(1, 3, 2)
    plt.imshow(observation['realsense_rgb'])
    plt.title("RGB2")

    plt.subplot(1, 3, 3)
    # 绘制预测动作和真实动作的对比
    x = numpy.arange(len(predicted_action))
    width = 0.35
    plt.bar(x - width/2, predicted_action, width, label='predicted')
    plt.bar(x + width/2, true_action, width, label='real')
    plt.title("action comparison")
    plt.legend()
    plt.savefig(fig_path)
    plt.close()

    print(f"对比结果已保存至 '{fig_path}'")
