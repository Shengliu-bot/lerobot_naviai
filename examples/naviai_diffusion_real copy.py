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
import numpy as np 
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



if __name__ == "__main__":
    rospy.init_node('evaluate_policy', anonymous=True)
    bridge = CvBridge()

    for i in range(1000):
        try:
            
            pose_msg = rospy.wait_for_message("/right_arm_tcp_pose", Pose, timeout=3.0)
            image_msg = rospy.wait_for_message("/image_streaming", Image, timeout=3.0)
        except rospy.ROSException as e:
            rospy.logwarn("接收传感器数据超时: %s" % e)
            continue

        observation_state = np.array([
            pose_msg.position.x,
            pose_msg.position.y,
            pose_msg.position.z,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w
        ])
        state_list = observation_state.tolist()

        # abtain action from GPU
        try:
            predicted_action_msg = rospy.wait_for_message("/predicted_action", Float64MultiArray, timeout=3.0)
            predicted_action = predicted_action_msg.data
        except rospy.ROSException as e:
            rospy.logwarn("接收预测动作超时: %s" % e)
            continue

        print("predicted_action: ", predicted_action)

        target_pose = Pose()
        target_pose.position.x = observation_state[0] + predicted_action[0]
        target_pose.position.y = observation_state[1] + predicted_action[1]
        target_pose.position.z = observation_state[2] + predicted_action[2]
        target_pose.orientation.x = observation_state[3] + predicted_action[3]
        target_pose.orientation.y = observation_state[4] + predicted_action[4]
        target_pose.orientation.z = observation_state[5] + predicted_action[5]
        target_pose.orientation.w = observation_state[6] + predicted_action[6]

        # 发送运动指令
        not_wait = True
        result = right_arm_movel_client(target_pose, not_wait)
        print(f"Movement to position {i+1} finish status: {result}")

        rospy.sleep(0.1)

"""
if __name__ == "__main__":
    # 加载预训练策略
    pretrained_policy_path = Path("/home/diffusion/src/lerobot-main/outputs/train/090000/pretrained_model")
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


    rospy.init_node('evaluate_policy', anonymous=True)

    # 初始化CvBridge
    bridge = CvBridge()
    # # 关节角度数据
    # jnt_angle_1 = [-0.07720238, -0.27499003,  0.02398043,  0.022364,    0.04290353,  0.0715358, 0.13130902]
    # jnt_angle_2 = [-0.37509427, -1.09652066,  0.02402837,  0.02206631,  0.0429994,   0.09588217, 0.11671971]
    # jnt_angle_3 = [-1.01242735, -1.13959196,  1.16175079, -1.19537781, -1.11349032, -0.10592222, -0.09411509]
    # jnt_angle_4 = [-0.63267122, -0.90872785,  0.60841515, -1.64891009, -1.05761986,  0.14467416, -0.56197497]

    # # 目标关节角度列表
    # joint_angles = [jnt_angle_1, jnt_angle_2, jnt_angle_3, jnt_angle_4]

    # # 设置是否等待的标志
    # not_wait = False  # 设置为True表示不等待

    # # 遍历四组关节角度并依次调用服务
    # for i, angles in enumerate(joint_angles):
    #     print(f"Requesting movement to joint angles {i+1}: {angles}")

    #     # 调用服务
    #     result = right_arm_movej_client(angles, not_wait)
    #     print(f"Movement to joint angles {i+1} finish status: {result}")

    # return




    for i in range(1000):
        sample_idx = i


        # # 获取样本数据
        # observation = {
        #     'end_effector': rospy.wait_for_message("/right_arm_tcp_pose", Pose) # 从ROS话题获取末端位姿
        #     'realsense_rgb': rospy.wait_for_message("/image_streaming", Image)  # 从ROS话题获取相机图像
        # }
        # 获取样本数据
        observation = {
            'end_effector': numpy.array([
                rospy.wait_for_message("/right_arm_tcp_pose", Pose).position.x,
                rospy.wait_for_message("/right_arm_tcp_pose", Pose).position.y, 
                rospy.wait_for_message("/right_arm_tcp_pose", Pose).position.z,
                rospy.wait_for_message("/right_arm_tcp_pose", Pose).orientation.x,
                rospy.wait_for_message("/right_arm_tcp_pose", Pose).orientation.y,
                rospy.wait_for_message("/right_arm_tcp_pose", Pose).orientation.z,
                rospy.wait_for_message("/right_arm_tcp_pose", Pose).orientation.w
            ]),  # 从ROS话题获取末端位姿(位置+四元数)
            'realsense_rgb': rospy.wait_for_message("/image_streaming", Image)  # 从ROS话题获取相机图像
        }

        receive_and_republish()


        # 准备模型输入
        state = torch.from_numpy(observation['end_effector'])
        #print(state)


        # 将 ROS 图像消息转换为 OpenCV 格式的图像
        cv_image = bridge.imgmsg_to_cv2(observation['realsense_rgb'])

        # 提取 RGB 通道（16位无符号整数类型的每个像素包含 4 个通道：RGBA）
        # 假设图像是 16 位的深度图像，RGB 通道在前三个位置，Alpha 通道在第四个位置
        rgb_image = cv_image[:, :, :3]  # 提取前三个通道 (RGB)

        # 转换为 8 位，便于后续处理
        # rgb_image_8bit = cv2.convertScaleAbs(rgb_image)  # 转换为 8 位图像

        # # 转换为 BGR 格式（OpenCV 默认使用 BGR 格式）
        # bgr_image = cv2.cvtColor(rgb_image_8bit, cv2.COLOR_RGB2BGR)



        image1 = torch.from_numpy(rgb_image).float()

        # 确保state维度正确
        current_dim = state.shape[0]
        if current_dim < 7:
            print("维度错误:", current_dim)



        # 处理图像数据
        image1 = image1.to(torch.float32) / 255
        image1 = image1.permute(2, 0, 1)

        # 移动数据到设备
        state = state.to(torch.float32) 
        state = state.to(device, non_blocking=True)
        image1 = image1.to(device, non_blocking=True)

        # 添加batch维度
        state = state.unsqueeze(0)
        image1 = image1.unsqueeze(0)

        # 创建策略输入字典
        model_input = {
            "observation.state": state,
            "observation.images.rgb1": image1,
        }

        # 预测动作
        with torch.inference_mode():
            # start_time = torch.cuda.Event(enable_timing=True)
            # end_time = torch.cuda.Event(enable_timing=True)

            # start_time.record()
            start_time = time.time()
            predicted_action = policy.select_action(model_input)
            end_time = time.time()
            print(f"推理时间: {(end_time - start_time) * 1000:.2f} 毫秒")
            # end_time.record()

            # 等待GPU操作完成
            # torch.cuda.synchronize()

            # # 计算耗时(毫秒)
            # elapsed_time = start_time.elapsed_time(end_time)
            # total_inference_time += elapsed_time
            # print(f"当前样本推理时间: {elapsed_time:.2f} ms")

        # 转换为numpy数组用于比较
        # 将预测动作转换为numpy数组
        predicted_action = predicted_action.squeeze(0).cpu().numpy()
        print("predicted_action: ",predicted_action)

        # 创建目标位姿消息

        target_pose = Pose()
        target_pose.position.x = observation['end_effector'][0] + predicted_action[0]
        target_pose.position.y = observation['end_effector'][1] + predicted_action[1] 
        target_pose.position.z = observation['end_effector'][2] + predicted_action[2]
        target_pose.orientation.x = observation['end_effector'][3] + predicted_action[3]
        target_pose.orientation.y = observation['end_effector'][4] + predicted_action[4]
        target_pose.orientation.z = observation['end_effector'][5] + predicted_action[5]
        target_pose.orientation.w = observation['end_effector'][6] + predicted_action[6]
        not_wait = True

        # 调用服务
        result = right_arm_movel_client(target_pose, not_wait)
        print(f"Movement to position {i+1} finish status: {result}")

"""