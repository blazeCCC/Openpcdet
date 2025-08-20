#pip install truckscenes-devkit
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from truckscenes.utils.geometry_utils import transform_matrix
import numpy as np
import os.path as osp
import json
import math
from functools import reduce
from tqdm import tqdm
from fire import Fire
from pathlib import Path


def conver2kitti(root_path='/home/cx/server3/man-truckscenes/', version='v1.0-trainval',
                 output_path='/home/cx/server3/TruckScenes/', use_ref_coord=False, lidar_ref='LIDAR_LEFT'):
    
    LIDAR_OUTPUT_PATH = Path(output_path) / version / 'sample'
    LABEL_OUTPUT_PATH = Path(output_path) / version / 'label'
    LIDAR_REF = lidar_ref
    USE_REF_COORD = use_ref_coord
    trucksc = TruckScenes(version, root_path, True)
    root_path = Path(root_path) / version

    trucksc.list_scenes()
    terminal_scene_list = []
    for scene in trucksc.scene:
        # if 'terminal' in scene['description']:
            # print(scene['description'])
        terminal_scene_list.append(scene)

    for index, terminal_scene in enumerate(terminal_scene_list):
        print(index, terminal_scene['description'])

    for index, terminal_scene in tqdm(enumerate(terminal_scene_list)):
        first_sample_token = terminal_scene['first_sample_token']
        last_sample_token = terminal_scene['first_sample_token']
        current_sample_token = first_sample_token
        while True:
            if current_sample_token == '':
                break
            # if (current_sample_token == last_sample_token and current_sample_token != first_sample_token):
            #     break
            sample = trucksc.get('sample', current_sample_token)
            # print(sample.keys())
            timestamp = sample['timestamp']
            ego_pose = trucksc.getclosest('ego_pose', timestamp)
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
            for key, val in sample['data'].items():
                if 'LIDAR' in key:
                    # print(key, val)
                    sensor_token = val
                    lidar_sample_data = trucksc.get('sample_data', sensor_token)
                    # print(lidar_sample_data)
                    cs_record = trucksc.get('calibrated_sensor', lidar_sample_data['calibrated_sensor_token'])
                    # get vehicle frame lidar
                    pcl_path = osp.join(trucksc.dataroot, lidar_sample_data['filename'])
                    pc = LidarPointCloud.from_file(pcl_path)
                    
                    current_pose_rec = trucksc.get('ego_pose', lidar_sample_data['ego_pose_token'])

                    global_from_car = transform_matrix(current_pose_rec['translation'],
                                                Quaternion(current_pose_rec['rotation']),
                                                inverse=False)

                    car_from_current = transform_matrix(cs_record['translation'],
                                                        Quaternion(cs_record['rotation']),
                                                        inverse=False)

                    # Fuse four transformation matrices into one and perform transform.
                    if USE_REF_COORD:
                        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global,
                                                    global_from_car, car_from_current])
                    else:
                        trans_matrix = reduce(np.dot, [car_from_global,
                                                    global_from_car, car_from_current])
                    pc.transform(trans_matrix)

                    # pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
                    # pc.translate(np.array(cs_record['translation']))


                    # poserecord = trucksc.get('ego_pose', lidar_sample_data['ego_pose_token'])
                    # pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
                    # pc.translate(np.array(poserecord['translation']))

                    # pc.translate(-np.array(ego_pose['translation']))
                    # pc.rotate(Quaternion(ego_pose['rotation']).rotation_matrix.T)
                    point_data_list.append(pc.points)


                    # print(pc.points.shape)
            if (len(point_data_list) > 0):
                points = np.concatenate(point_data_list, axis=1).T
                # print(points)
                out_file = osp.join(str(LIDAR_OUTPUT_PATH), '{}.bin'.format(str(timestamp)))
                print(out_file)
                points.tofile(out_file)

            # print(sample_data)
            annos = sample['anns']
            out_file = osp.join(str(LABEL_OUTPUT_PATH), '{}.json'.format(str(timestamp)))
            label_data = []
            with open(out_file, 'w', encoding='utf-8') as f:
                for anno in annos:
                    anno_data = trucksc.get('sample_annotation', anno)
                    cls_name = anno_data['category_name']
                    # print(anno_data)
                    box = trucksc.get_box(anno)
                    box.translate(-np.array(ref_pose_rec['translation']))
                    box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)
                    if USE_REF_COORD:
                        box.translate(-np.array(ref_cs['translation']))
                        box.rotate(Quaternion(ref_cs['rotation']).inverse)

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
            current_sample_token = sample['next']


if __name__=='__main__':
    Fire(conver2kitti)
