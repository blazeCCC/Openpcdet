import copy

import numpy as np
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
from typing import List, Tuple
import shutil
from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from scipy.spatial.transform import Rotation as R
from pcdet.utils import common_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_truckscenes
import random



class TruckScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.infos = []
        self.include_ts_data(self.mode)
    
    def include_ts_data(self, mode):
        self.logger.info('Loading TS dataset')
        ts_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                self.logger.warning(f'{info_path} does not exist, this is in generate process?', )
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                ts_infos.extend(infos)

        self.infos.extend(ts_infos)
        self.logger.info('Total samples for KL dataset: %d' % (len(ts_infos)))
        self.train_sample_lst = []
        self.val_sample_lst = []

    def generateSplit(self):
        anno_path = self.root_path / 'label'
        all_file_list = []
        #遍历目录下所有的文件
        for anno_file in anno_path.iterdir():
            if anno_file.is_file():
                all_file_list.append(anno_file.stem)
        random.shuffle(all_file_list)
        self.train_sample_lst = all_file_list[:int(len(all_file_list)*0.8)]
        self.val_sample_lst = all_file_list[int(len(all_file_list)*0.2):]
    
    def get_label(self, idx):
        label_file = self.root_path / 'label' / ('%s.json' % idx)
        assert label_file.exists()
        return object3d_truckscenes.get_objects_from_label(label_file)
    
    def get_lidar(self, idx):
        lidar_file = self.root_path / 'sample' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float64).reshape(-1, 4).astype(np.float32)

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('sample_idx: %s' % (sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                # annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                # annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                # annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                # annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
                
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                # loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc, l, w, h, rots.reshape(-1, 1)], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                
                info['annos'] = annotations
                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt
            return info
        # sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
        from pcdet.datasets.kitti import kitti_utils

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict
    
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('ts_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            # bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):

                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]
                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': 0}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
                                
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)
    
    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)

        input_dict = {
            'frame_id': sample_idx,
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def create_ts_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = TruckScenesDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False,
                                logger=common_utils.create_logger())
    train_split, val_split = 'train', 'val'
    
    train_filename = save_path / ('ts_infos_%s.pkl' % train_split)
    val_filename = save_path / ('ts_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'ts_infos_trainval.pkl'
    test_filename = save_path / 'ts_infos_test.pkl'

    print('---------------Start to generate data infos---------------')
    dataset.generateSplit()
    # dataset.set_split(train_split)
    ts_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True, sample_id_list=dataset.train_sample_lst)
    with open(train_filename, 'wb') as f:
        pickle.dump(ts_infos_train, f)
    print('ts info train file is saved to %s' % train_filename)

    # dataset.set_split(val_split)
    ts_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True, sample_id_list=dataset.val_sample_lst)
    with open(val_filename, 'wb') as f:
        pickle.dump(ts_infos_val, f)
    print('ts info val file is saved to %s' % val_filename)

    # with open(trainval_filename, 'wb') as f:
    #     pickle.dump(ts_infos_train + ts_infos_val, f)
    # print('ts info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # ts_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(ts_infos_test, f)
    # print('ts info test file is saved to %s' % test_filename)

    # print('---------------Start create groundtruth database for data augmentation---------------')
    # dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, used_classes=class_names, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':

    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="cfgs/dataset_configs/truck_scenes_dataset.yaml", help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_kl_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
    args = parser.parse_args()
    
    


    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset_cfg.VERSION = args.version
    create_ts_infos(

        dataset_cfg = dataset_cfg, 
        class_names = dataset_cfg.CLASS_NAMES,
        data_path=ROOT_DIR / 'data' / 'ts',
        save_path=ROOT_DIR / 'data' / 'ts'/ args.version ,

    )

    # ts_dataset = TruckScenesDataset(
    #     dataset_cfg=dataset_cfg, class_names=None,
    #     root_path=ROOT_DIR / 'data' / 'ts',
    #     logger=common_utils.create_logger(), training=True
    # )

    # ts_dataset.create_groundtruth_database("/home/cx/server3/TruckScenes/v1.0-trainval/ts_infos_train.pkl", used_classes=dataset_cfg.CLASS_NAMES, split = 'train')