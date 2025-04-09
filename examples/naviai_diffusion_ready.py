"""
此脚本演示如何从数据集中抽取样本评估预训练策略
"""
import sys
import time

sys.path.append("/home/diffusion/src/lerobot-main/lerobot")
sys.path.append("/home/diffusion/src/lerobot-main")
sys.path.append("/home/diffusion/src")
sys.path.append("/home/diffusion")
from pathlib import Path
import numpy
import torch
# 添加ROS相关导入
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from naviai_manip_msgs.msg import *
from naviai_manip_srvs.srv import *
from cv_bridge import CvBridge
# 创建目录存储评估视频
output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

def right_arm_movej_client(target_pose, not_wait):
    rospy.wait_for_service('right_arm_movej_service')  # 等待服务提供者启动
    try:
        # 创建服务代理
        right_arm_movej = rospy.ServiceProxy('right_arm_movej_service', MoveJ)
        # 调用服务并传递请求
        response = right_arm_movej(target_pose, not_wait)
        return response.finish  # 返回服务响应中的finish字段
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def right_arm_movel_client(target_pose, not_wait):
    rospy.wait_for_service('right_arm_movel_service')  # 等待服务提供者启动
    try:
        # 创建服务代理
        right_arm_movel = rospy.ServiceProxy('right_arm_movel_service', MoveL)
        # 调用服务并传递请求
        response = right_arm_movel(target_pose, not_wait)
        return response.finish  # 返回服务响应中的finish字段
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


if __name__ == "__main__":
    # # 加载预训练策略
    # pretrained_policy_path = Path("/home/diffusion/src/lerobot-main/outputs/train/090000/pretrained_model")
    # policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    # policy.eval()

    # # 检查GPU可用性
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("GPU可用。使用设备:", device)
    # else:
    #     device = torch.device("cpu")
    #     print(f"GPU不可用。使用设备: {device}。推理速度将较慢。")
    #     policy.diffusion.num_inference_steps = 10

    # policy.to(device)



    rospy.init_node('evaluate_policy', anonymous=True)

    # 初始化CvBridge
    bridge = CvBridge()
    # 关节角度数据
    jnt_angle_1 = [-0.07720238, -0.27499003,  0.02398043,  0.022364,    0.04290353,  0.0715358, 0.13130902]
    jnt_angle_2 = [-0.37509427, -1.09652066,  0.02402837,  0.02206631,  0.0429994,   0.09588217, 0.11671971]
    jnt_angle_3 = [-1.01242735, -1.13959196,  1.16175079, -1.19537781, -1.11349032, -0.10592222, -0.09411509]
    jnt_angle_4 = [-0.63267122, -0.90872785,  0.60841515, -1.64891009, -1.05761986,  0.14467416, -0.56197497]

    # 目标关节角度列表
    joint_angles = [jnt_angle_1, jnt_angle_2, jnt_angle_3, jnt_angle_4]

    # 设置是否等待的标志
    not_wait = False  # 设置为True表示不等待

    # 遍历四组关节角度并依次调用服务
    for i, angles in enumerate(joint_angles):
        print(f"Requesting movement to joint angles {i+1}: {angles}")

        # 调用服务
        result = right_arm_movej_client(angles, not_wait)
        print(f"Movement to joint angles {i+1} finish status: {result}")



