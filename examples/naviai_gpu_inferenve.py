import rospy
import sys
import torch
import time
sys.path.append("/home/diffusion/src/lerobot-main/lerobot")
sys.path.append("/home/diffusion/src/lerobot-main")
sys.path.append("/home/diffusion/src")
sys.path.append("/home/diffusion")
import numpy as np
from pathlib import Path


from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
import torchvision.transforms as transforms
from naviai_manip_msgs.msg import *
from naviai_manip_srvs.srv import *
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray


output_directory = Path("outputs/eval/gpu_inference")
output_directory.mkdir(parents=True, exist_ok=True)

def receive_data():
    rospy.loginfo("正在接收位姿数据...")
    pose_msg = rospy.wait_for_message('/right_arm_tcp_pose', Pose, timeout=3.0)
    rospy.loginfo("接收到位姿数据.")

    rospy.loginfo("正在接收图像数据...")
    image_msg = rospy.wait_for_message('/image_streaming', Image, timeout=3.0)
    rospy.loginfo("接收到图像数据.")
    return pose_msg, image_msg


if __name__ == "__main__":
    pretrained_policy_path = Path("/home/diffusion/src/lerobot-main/outputs/train/090000/pretrained_model")
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU可用。使用设备:", device)
    else:
        device = torch.device("cpu")
        print(f"GPU不可用。使用设备: {device}。推理速度将较慢。")
        policy.diffusion.num_inference_steps = 10
    policy.to(device)

    # 初始化ROS节点和CvBridge
    rospy.init_node('gpu_inference_node', anonymous=True)
    bridge = CvBridge()

    # 创建一个发布器，将预测动作以Float64MultiArray格式发布到'/predicted_action'话题上
    action_pub = rospy.Publisher('/predicted_action', Float64MultiArray, queue_size=10)

    for i in range(1000):
        rospy.loginfo("样本 %d: 开始接收数据..." % (i+1))
        try:
            pose_msg, image_msg = receive_data()
        except rospy.ROSException as e:
            rospy.logwarn("接收数据超时: %s" % e)
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

        cv_image = bridge.imgmsg_to_cv2(image_msg)
        rgb_image = cv_image[:, :, :3]
        state_tensor = torch.from_numpy(observation_state).to(torch.float32).to(device, non_blocking=True)
        state_tensor = state_tensor.unsqueeze(0)

        image_tensor = torch.from_numpy(rgb_image).to(torch.float32) / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).to(device, non_blocking=True)
        image_tensor = image_tensor.unsqueeze(0)

        model_input = {
            "observation.state": state_tensor,
            "observation.images.rgb1": image_tensor,
        }


        with torch.inference_mode():
            start_time = time.time()
            predicted_action = policy.select_action(model_input)
            end_time = time.time()
            rospy.loginfo("样本 %d 推理时间: %.2f 毫秒" % (i+1, (end_time - start_time) * 1000.0))

        predicted_action = predicted_action.squeeze(0).cpu().numpy()
        rospy.loginfo("predicted_action: %s" % predicted_action)

        action_msg = Float64MultiArray()
        action_msg.data = predicted_action.tolist()
        action_pub.publish(action_msg)
        rospy.loginfo("样本 %d 的预测动作已发布。" % (i+1))

        rospy.sleep(0.1)