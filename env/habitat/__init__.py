# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api

import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from .exploration_env import Exploration_Env
from .habitat_api.habitat.core.vector_env import VectorEnv
from .habitat_api.habitat_baselines.config.default import get_config as cfg_baseline

from . import vistarget_nav_task
from .vistarget_nav_task import CustomVisTargetSensor, PanoramicDepthSensor, PanoramicRGBSensor
from .vistarget_nav_dataset import VisTargetNavDatasetV1
def make_env_fn(args, config_env, config_baseline, rank):
    #dataset = PointNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = config_env.DATASET.CONTENT_SCENES
    print("Loading {}".format(config_env.SIMULATOR.SCENE))
    config_env.freeze()

    env = Exploration_Env(args=args, rank=rank,
                          config_env=config_env, config_baseline=config_baseline
                          )

    env.seed(rank)
    return env

import habitat
def add_panoramic_camera(task_config):
    num_of_camera = int(360 // task_config.SIMULATOR.RGB_SENSOR.HFOV)
    #assert isinstance(num_of_camera, int)
    angles = [2 * np.pi * idx / num_of_camera for idx in range(num_of_camera - 1, -1, -1)]
    half = num_of_camera // 2
    angles = angles[half:] + angles[:half]
    sensors = []
    use_semantic = False#'PANORAMIC_SEMANTIC_SENSOR' in task_config.TASK.SENSORS
    use_depth = True#'PANORAMIC_DEPTH_SENSOR' in task_config.TASK.SENSORS
    for camera_idx in range(num_of_camera):
        curr_angle = angles[camera_idx]
        if curr_angle > 3.14:
            curr_angle -= 2 * np.pi
        new_camera_config = task_config.SIMULATOR.RGB_SENSOR.clone()
        new_camera_config.TYPE = "PanoramicPartRGBSensor"

        new_camera_config.ORIENTATION = [0, curr_angle, 0]
        new_camera_config.ANGLE = "{}".format(camera_idx)
        task_config.SIMULATOR.update({'RGB_SENSOR_{}'.format(camera_idx): new_camera_config})
        sensors.append('RGB_SENSOR_{}'.format(camera_idx))

        if use_depth:
            new_depth_camera_config = task_config.SIMULATOR.DEPTH_SENSOR.clone()
            new_depth_camera_config.TYPE = "PanoramicPartDepthSensor"
            new_depth_camera_config.ORIENTATION = [0, curr_angle, 0]
            new_depth_camera_config.ANGLE = "{}".format(camera_idx)
            task_config.SIMULATOR.update({'DEPTH_SENSOR_{}'.format(camera_idx): new_depth_camera_config})
            sensors.append('DEPTH_SENSOR_{}'.format(camera_idx))
        if use_semantic:
            new_semantic_camera_config = task_config.SIMULATOR.SEMANTIC_SENSOR.clone()
            new_semantic_camera_config.TYPE = "PanoramicPartSemanticSensor"
            new_semantic_camera_config.ORIENTATION = [0, curr_angle, 0]
            new_semantic_camera_config.ANGLE = "{}".format(camera_idx)
            task_config.SIMULATOR.update({'SEMANTIC_SENSOR_{}'.format(camera_idx): new_semantic_camera_config})
            sensors.append('SEMANTIC_SENSOR_{}'.format(camera_idx))

    task_config.SIMULATOR.AGENT_0.SENSORS = sensors
    task_config.TASK.SENSORS = ['CUSTOM_VISTARGET_SENSOR',"PANORAMIC_SENSOR", "PANORAMIC_DEPTH_SENSOR"]
    task_config.TASK.PANORAMIC_SENSOR = habitat.Config()
    task_config.TASK.PANORAMIC_SENSOR.TYPE = 'PanoramicRGBSensor'
    task_config.TASK.PANORAMIC_SENSOR.WIDTH = task_config.SIMULATOR.RGB_SENSOR.HEIGHT * 4
    task_config.TASK.PANORAMIC_SENSOR.HEIGHT = task_config.SIMULATOR.RGB_SENSOR.HEIGHT
    task_config.TASK.PANORAMIC_SENSOR.NUM_CAMERA = num_of_camera
    if use_depth:
        task_config.TASK.PANORAMIC_DEPTH_SENSOR = task_config.SIMULATOR.DEPTH_SENSOR.clone()
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.TYPE = 'PanoramicDepthSensor'
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.WIDTH = task_config.SIMULATOR.DEPTH_SENSOR.HEIGHT * 4
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.HEIGHT = task_config.SIMULATOR.DEPTH_SENSOR.HEIGHT
        task_config.TASK.PANORAMIC_DEPTH_SENSOR.NUM_CAMERA = num_of_camera
    if use_semantic:
        task_config.TASK.PANORAMIC_SEMANTIC_SENSOR = habitat.Config()
        task_config.TASK.PANORAMIC_SEMANTIC_SENSOR.TYPE = 'PanoramicSemanticSensor'
        task_config.TASK.PANORAMIC_SEMANTIC_SENSOR.WIDTH = task_config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT * 4
        task_config.TASK.PANORAMIC_SEMANTIC_SENSOR.HEIGHT = task_config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT
        task_config.TASK.PANORAMIC_SEMANTIC_SENSOR.NUM_CAMERA = num_of_camera

    task_config.TASK.CUSTOM_VISTARGET_SENSOR = habitat.Config()
    task_config.TASK.CUSTOM_VISTARGET_SENSOR.TYPE = 'CustomVisTargetSensor'
    task_config.TASK.CUSTOM_VISTARGET_SENSOR.NUM_CAMERA = num_of_camera
    task_config.TASK.CUSTOM_VISTARGET_SENSOR.WIDTH = task_config.SIMULATOR.RGB_SENSOR.HEIGHT * 4
    task_config.TASK.CUSTOM_VISTARGET_SENSOR.HEIGHT = task_config.SIMULATOR.RGB_SENSOR.HEIGHT

    if "STOP" not in task_config.TASK.POSSIBLE_ACTIONS:
        task_config.TASK.SUCCESS.TYPE = "Success_woSTOP"
    task_config.TASK.SUCCESS.SUCCESS_DISTANCE = task_config.TASK.SUCCESS_DISTANCE

    return task_config


def construct_envs(args):
    env_configs = []
    baseline_configs = []
    args_list = []

    basic_config = cfg_env(config_paths=
                           ["env/habitat/habitat_api/configs/" + args.task_config])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    for i in range(args.num_processes):
        config_env = cfg_env(config_paths=
                             ["env/habitat/habitat_api/configs/" + args.task_config])
        config_env.defrost()

        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[
                                                i * scene_split_size: (i + 1) * scene_split_size
                                                ]

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        # agent_sensors = []
        # agent_sensors.append("RGB_SENSOR")
        # agent_sensors.append("DEPTH_SENSOR")
        #
        # config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.TURN_ANGLE = 10
        config_env.DATASET.SPLIT = args.split

        config_env = add_panoramic_camera(config_env)

        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, baseline_configs,
                    range(args.num_processes))
            )
        ),
    )

    return envs
