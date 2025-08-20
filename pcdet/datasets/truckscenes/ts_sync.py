from truckscenes import TruckScenes
trucksc = TruckScenes('v1.0-mini', '/home/cx/Downloads/man-truckscenes/', True)
trucksc.list_scenes()
terminal_scene_list = []
count = 0
for scene in trucksc.scene:
    if 'highway' in scene['description']:
        # print(scene['description'])
        terminal_scene_list.append(scene)
        count += 1

for index, terminal_scene in enumerate(terminal_scene_list):
    print(index, terminal_scene['description'])


from truckscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from truckscenes.utils.geometry_utils import transform_matrix
import numpy as np
import os.path as osp
import json
import math
from functools import reduce
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import cProfile
import pstats
from concurrent.futures import ThreadPoolExecutor
import time
import bisect

LIDAR_OUTPUT_PATH = '/home/cx/Downloads/TruckScenes/v1.0-trainval/sample'
LABEL_OUTPUT_PATH = '/home/cx/Downloads/TruckScenes/v1.0-trainval/label'
LIDAR_REF = "LIDAR_LEFT"
USE_REF_COORD = True


def yaw_to_quaternion(yaw_rad):
    """将yaw角转换为四元数
    Args:
        yaw_rad: 弧度制偏航角
    Returns:
        [x, y, z, w] 格式四元数
    """
    cy = math.cos(yaw_rad * 0.5)
    sz = math.sin(yaw_rad * 0.5)
    return [0.0, 0.0, sz, cy]   

def get_window(ordered_dict, query_timestamp):
    """
    从OrderedDict中获取最接近给定时间戳的值
    
    参数:
        ordered_dict: OrderedDict，键为时间戳
        query_timestamp: 要查询的时间戳
        
    返回:
        最接近query_timestamp的值(当两个时间戳距离相等时，返回较小的)
    """
    # 获取所有时间戳列表
    timestamps = list(ordered_dict.keys())
    
    # 使用bisect找到查询时间戳的插入位置
    pos = bisect.bisect_left(timestamps, query_timestamp)
    
    # 计算前后100个时间戳的范围
    start = max(0, pos - 100)
    end = min(len(timestamps), pos + 100)
    
    # 提取局部范围的时间戳
    window_timestamps = timestamps[start:end]
    
    # 如果没有数据，返回None
    if not window_timestamps:
        return None
    return window_timestamps
   

def get_closest_in_window(window_timestamps, query_timestamp):
     # 在局部范围内进行二分查找
    closest_pos = bisect.bisect_left(window_timestamps, query_timestamp)
    
    # 处理边界情况
    if closest_pos == 0:
        return window_timestamps[0]
    if closest_pos == len(window_timestamps):
        return window_timestamps[-1]
    
    # 获取前后相邻的时间戳
    before = window_timestamps[closest_pos - 1]
    after = window_timestamps[closest_pos]
    
    # 比较距离，选择更接近的(距离相等时选较小的)
    if (after - query_timestamp) < (query_timestamp - before):
        return after
    else:
        return before


def process(current_sample_token):
    sample = trucksc.get('sample', current_sample_token)
    print(sample['data'].keys())
    timestamp = sample['timestamp']
    ego_pose = trucksc.getclosest('ego_pose', timestamp)
    pose_time_stamp_window = get_window(trucksc._timestamp2token['ego_pose'], timestamp)
    if (pose_time_stamp_window is None):
        print('No pose time stamp window')
        return
    # print(ego_pose)
    
    # sample_data = trucksc.get('sample_data', sample['sample_token'])

    ref_lidar_sd_token = sample['data'][LIDAR_REF]
    ref_lidar_sample_data = trucksc.get('sample_data', ref_lidar_sd_token)
    ref_cs = trucksc.get('calibrated_sensor', ref_lidar_sample_data['calibrated_sensor_token'])
    ref_pose_rec = trucksc.get('ego_pose', ref_lidar_sample_data['ego_pose_token'])
    ref_from_car = transform_matrix(ref_cs['translation'],
                                    Quaternion(ref_cs['rotation']),
                                    inverse=True)

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'],
                                        Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    point_data_list = []
    pose_cache = {}
    
    for key, val in sample['data'].items():
        if 'LIDAR' in key :
            # print(key, val)
            sensor_token = val
            lidar_sample_data = trucksc.get('sample_data', sensor_token)
            # print(lidar_sample_data)
            cs_record = trucksc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
            # get vehicle frame lidar
            pcl_path = osp.join(trucksc.dataroot, lidar_sample_data['filename'])
            pc = LidarPointCloud.from_file(pcl_path)
            # print(pc.timestamps)
            last = None
            count_diff = 0
            current_pose_rec = trucksc.get('ego_pose', lidar_sample_data['ego_pose_token'])

            global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']),
                                            inverse=False)

            car_from_current = transform_matrix(cs_record['translation'],
                                                Quaternion(cs_record['rotation']),
                                                inverse=False)
            
            
            pc.transform(car_from_current)
            begin = time.time()
            unique_timestamps = np.unique(pc.timestamps[0, :])
            pose_cache_little = {ts: trucksc.get('ego_pose', trucksc._timestamp2token['ego_pose'][get_closest_in_window(pose_time_stamp_window, ts)]) for ts in unique_timestamps if ts not in pose_cache}
            pose_cache.update(pose_cache_little)
            end = time.time()
            print('update pose cache', end - begin)
            for index in tqdm(range(pc.points.shape[1])): 
                point_new = pc.points[:, index]
                # point_pose = trucksc.getclosest('ego_pose', pc.timestamps[:, index])
                ts = pc.timestamps[:, index][0]
                if ts not in pose_cache:
                    pose_cache[ts] = trucksc.getclosest('ego_pose', ts)
                point_pose = pose_cache[ts]
                point_transform_matrix = transform_matrix(point_pose['translation'],
                                            Quaternion(point_pose['rotation']),
                                            inverse=False)
                new_point = point_transform_matrix.dot(np.array([point_new[0], point_new[1], point_new[2], 1]))
                point_new[:3] = new_point[:3]

            pc.transform(car_from_global)
            point_data_list.append(pc.points)

    if (len(point_data_list) > 0):
        points = np.concatenate(point_data_list, axis=1).T
        # print(points)
        out_file = osp.join(LIDAR_OUTPUT_PATH, '{}.bin'.format(str(timestamp)))
        print(out_file)
        points.tofile(out_file)

    # print(sample_data)
    annos = sample['anns']
    out_file = osp.join(LABEL_OUTPUT_PATH, '{}.json'.format(str(timestamp)))
    label_data = []
    with open(out_file, 'w', encoding='utf-8') as f:
        for anno in annos:
            anno_data = trucksc.get('sample_annotation', anno)
            cls_name = anno_data['category_name']
            # print(anno_data)
            box = trucksc.get_box(anno)
            box.translate(-np.array(ref_pose_rec['translation']))
            box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)
            # if USE_REF_COORD:
            #     box.translate(-np.array(ref_cs['translation']))
            #     box.rotate(Quaternion(ref_cs['rotation']).inverse)

            box_data = {
                "track_id": 1, 
                "label": cls_name, 
                "subtype": cls_name, 
                "xyz": [box.center[0], box.center[1], box.center[2]], 
                "lwh": [box.wlh[1], box.wlh[0], box.wlh[2]], 
                "rotation": box.orientation.yaw_pitch_roll[0], 
                "num_lidar_pts": 10
            }
            label_data.append(box_data)
        json.dump(label_data, f)

for index, terminal_scene in enumerate(terminal_scene_list):
    first_sample_token = terminal_scene['first_sample_token']
    last_sample_token = terminal_scene['first_sample_token']
    current_sample_token = first_sample_token
    # pose_cache = {}
    sample_list = []
    while True:
        if current_sample_token == '':
            break
        sample = trucksc.get('sample', current_sample_token)
        sample_list.append(current_sample_token)
        
        if (current_sample_token == last_sample_token and current_sample_token != first_sample_token):
            break
        
        current_sample_token = sample['next']
    for sample_token in sample_list:
        process(sample_token)
    
    # with ThreadPoolExecutor() as executor:
    #     list(tqdm(executor.map(process, sample_list), total=len(sample_list)))