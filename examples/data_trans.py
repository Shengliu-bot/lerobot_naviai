#!/usr/bin/env python

import os
import sys

# 确保使用正确的Python环境
if __name__ == "__main__":
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    """This scripts demonstrates how to train Diffusion Policy on the PushT environment.

    Once you have trained a model with this script, you can try to evaluate it on
    examples/2_evaluate_pretrained_policy.py
    """


    import argparse
    import json
    import shutil
    import warnings
    from pathlib import Path
    from typing import Any
    import torch

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.common.datasets.push_dataset_to_hub.naviai_hdf5_format import from_raw_to_lerobot_format
    from safetensors.torch import save_file

    from lerobot.common.datasets.compute_stats import compute_stats
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id
    from lerobot.common.datasets.utils import create_branch, create_lerobot_dataset_card, flatten_dict
    import warnings
    from lerobot.common.datasets.factory import make_dataset, resolve_delta_timestamps

    def save_meta_data(
        info: dict[str, Any], stats: dict, episode_data_index: dict[str, list], meta_data_dir: Path
    ):
        meta_data_dir.mkdir(parents=True, exist_ok=True)

        # save info
        info_path = meta_data_dir / "info.json"
        with open(str(info_path), "w") as f:
            json.dump(info, f, indent=4)

        # save stats
        stats_path = meta_data_dir / "stats.safetensors"
        save_file(flatten_dict(stats), stats_path)

        # save episode_data_index
        # episode_data_index = {key: episode_data_index[key].clone().detach() for key in episode_data_index}
        episode_data_index = {key: episode_data_index[key].clone().detach() for key in episode_data_index}
        ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
        save_file(episode_data_index, ep_data_idx_path)


    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/example_pusht_diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 5000
    device = torch.device("cuda")
    log_freq = 250

    # Set up the dataset.
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    # dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
    repo_id = "lerobot/pusht"

    fps = 10  # 每秒 30 帧
    video = True  # 将图像保存为视频文件
    raw_dir = Path("/home/rxjqr/Docker/lerobot-main/lerobot/common/datasets/push_dataset_to_hub/data/raw_data")  # 存储 HDF5 文件的原始数据目录
    videos_dir = Path("/home/rxjqr/Docker/lerobot-main/lerobot/common/datasets/push_dataset_to_hub/data/videos")  # 存储生成的视频文件的目录

    # 自动获取raw_dir目录下的HDF5文件数量来确定回合数
    episodes = list(range(len([f for f in raw_dir.glob("*.hdf5")])))  # 自动获取所有回合

    encoding = {"crf": 23}  # 示例编码设置（例如使用 ffmpeg 编码质量）

    user_id, dataset_id = repo_id.split("/")

    # Robustify when `raw_dir` is str instead of Path
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise NotADirectoryError(
            f"{raw_dir} does not exists. Check your paths or run this command to download an existing raw dataset on the hub: "
            f"`python lerobot/common/datasets/push_dataset_to_hub/_download_raw.py --raw-dir your/raw/dir --repo-id your/repo/id_raw`"
        )
    local_dir = Path("/home/rxjqr/Docker/lerobot-main/lerobot/common/datasets/push_dataset_to_hub/data/local")  
    cache_dir = Path("/home/rxjqr/Docker/lerobot-main/lerobot/common/datasets/push_dataset_to_hub/data/cache")  

    force_override = True
    resume = True
    batch_size = 32
    num_workers = 2
    if local_dir:
        # Robustify when `local_dir` is str instead of Path
        local_dir = Path(local_dir)

        # Send warning if local_dir isn't well formated
        if local_dir.parts[-2] != user_id or local_dir.parts[-1] != dataset_id:
            warnings.warn(
                f"`local_dir` ({local_dir}) doesn't contain a community or user id `/` the name of the dataset that match the `repo_id` (e.g. 'data/lerobot/pusht'). Following this naming convention is advised, but not mandatory.",
                stacklevel=1,
            )

        # Check we don't override an existing `local_dir` by mistake
        if local_dir.exists():
            if force_override:
                shutil.rmtree(local_dir)
            elif not resume:
                raise ValueError(f"`local_dir` already exists ({local_dir}). Use `--force-override 1`.")

        meta_data_dir = local_dir / "meta_data"
        videos_dir = local_dir / "videos"
    else:
        # Temporary directory used to store images, videos, meta_data
        meta_data_dir = Path(cache_dir) / "meta_data"
        videos_dir = Path(cache_dir) / "videos"



    fmt_kwgs = {
        "raw_dir": raw_dir,
        "videos_dir": videos_dir,
        "fps": fps,
        "video": video,
        "episodes": episodes,
        "encoding": encoding,
    }

    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(**fmt_kwgs)

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    stats = compute_stats(lerobot_dataset, batch_size, num_workers)

    if local_dir:
        hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
        hf_dataset.save_to_disk(str(local_dir / "train"))

    if local_dir:
        # mandatory for upload
        save_meta_data(info, stats, episode_data_index, meta_data_dir)
