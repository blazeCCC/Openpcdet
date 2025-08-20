from pcdet.visualize.ptqt_visual import PointCloudViewer
from pcdet.datasets.truckscenes.truckscenes_dataset import TruckScenesDataset
from pcdet.utils import common_utils
from pathlib import Path
import numpy as np
import json
import yaml
from easydict import EasyDict


def load_data():

    file_dir = Path('/home/cx/Downloads/TruckScenes/v1.0-trainval/sample/')
    pcd_files = list(file_dir.glob('*.bin'))
    for pcd_file in pcd_files:
        # json_label = pcd_file.parent.parent / 'label' / (pcd_file.stem + '.json')
        pcd_file = "/home/cx/Downloads/TruckScenes/v1.0-trainval/sample/1696700124497510.bin"
        json_label = "/home/cx/Downloads/TruckScenes/v1.0-trainval/label/1696700124497510.json"
        pcd_data = np.fromfile(pcd_file, dtype=np.float64).reshape(-1, 4)
        print(pcd_data.shape, pcd_data[0])

        with open(json_label, "r") as f:
            json_data = json.load(f)
        gt_boxes = []
        gt_label = []
        for item in json_data:
            gt_boxes.append([item['xyz'][0], item['xyz'][1], item['xyz'][2], item['lwh'][0], item['lwh'][1], item['lwh'][2], item['rotation']])
            gt_label.append(item['label'])
        yield pcd_data[:, :3], gt_boxes, gt_label

# def load_dataset():
#     dataset_cfg = EasyDict(yaml.safe_load(open("/home/cx/code/OpenPCDet/tools/cfgs/dataset_configs/truck_scenes_dataset.yaml")))
#     ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
#     dataset_cfg.VERSION = "v1.0-trainval"
#     ts_dataset = TruckScenesDataset(
#         dataset_cfg=dataset_cfg, class_names=None,
#         root_path=ROOT_DIR / 'data' / 'ts',
#         logger=common_utils.create_logger(), training=True
#     )
#     for index  in range(10):
#         print(ts_dataset.infos[index])
#         info = ts_dataset.infos[index]['annos']
#         gt_box = info['gt_boxes_lidar'].tolist()
#         gt_label = info['name'].tolist()
#         pcd_data = ts_dataset.get_lidar(ts_dataset.infos[index]['point_cloud']['lidar_idx'])
#         yield pcd_data[:, :3], gt_box, gt_label

        
    


pointCloudViewer = PointCloudViewer(load_data)
pointCloudViewer.run()
# load_dataset()


# import argparse

# from typing import Any, Dict

# import matplotlib
# import numpy as np
# import open3d as o3d

# from truckscenes import TruckScenes
# from truckscenes.utils.data_classes import LidarPointCloud


# def get_fused_pointcloud(trucksc: TruckScenes, sample: Dict[str, Any]) -> LidarPointCloud:
#     """ Returns a fused lidar point cloud for the given sample.

#     Fuses the point clouds of the given sample and returns them in the reference
#     sensor frame at the timestamp of the reference sample data record. Uses the
#     timestamp of the sample data record of the individual sensors to transform
#     them to a uniform frame.

#     Does not consider the timestamps of the individual points during the
#     fusion. Therefore, motion distortion is not considered and deskewing
#     is not performed.

#     Arguments:
#         trucksc: TruckScenes dataset instance.
#         sample: Reference sample to fuse the point clouds of.

#     Returns:
#         fused_point_cloud: Fused lidar point cloud in the ego vehicle frame at the
#             timestamp of the sample.
#     """
#     # Initialize
#     points = np.zeros((LidarPointCloud.nbr_dims(), 0), dtype=np.float64)
#     timestamps = np.zeros((1, 0), dtype=np.uint64)
#     fused_point_cloud = LidarPointCloud(points, timestamps)

#     # Define reference sensor
#     ref_chan = "LIDAR_LEFT"
#     bboxes = []
#     # Iterate over all lidar sensors and fuse their point clouds
#     for sensor in sample['data'].keys():
#         if 'lidar' not in sensor.lower():
#             continue

#         # Load pointcloud
#         point_cloud, _ = LidarPointCloud.from_file_multisweep(trucksc, sample, chan=sensor, ref_chan=ref_chan, nsweeps=1)
        
#         # Merge with reference point cloud.
#         fused_point_cloud.points = np.hstack((fused_point_cloud.points, point_cloud.points))
#         if point_cloud.timestamps is not None:
#             fused_point_cloud.timestamps = np.hstack((fused_point_cloud.timestamps, point_cloud.timestamps))
#         data_path, boxes_, cam_intrinsic = trucksc.get_sample_data(sample['data'][sensor])
#         # print(boxes_)
#         if (sensor == ref_chan):
#             for box in boxes_:
#                 bboxes.append([box.center[0], box.center[1], box.center[2], box.wlh[1], box.wlh[0], box.wlh[2],box.orientation.yaw_pitch_roll[0]])


#     return fused_point_cloud, bboxes


# def visualize_pointcloud(point_cloud: LidarPointCloud, boxes: list = None) -> None:
#     """Visualizes the given point cloud.

#     Arguments:
#         point_cloud: LidarPointCloud object to visualize.
#     """
#     # Extract points and intensities
#     points = point_cloud.points[:3, :].T
#     intensities = point_cloud.points[3, :].T

#     # Convert intensities to colors
#     rgb = matplotlib.colormaps['viridis'](intensities)[..., :3]

#     # Initialize vizualization objets
#     vis_obj = []

#     # Define point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(rgb)
#     vis_obj.append(pcd)

#     if boxes is not None:
#         for box in boxes:
#             center = box[:3]
#             length, width, height = box[3:6]
#             yaw = box[6]
            
#             # Create oriented bounding box
#             obb = o3d.geometry.OrientedBoundingBox()
#             obb.center = center
#             obb.extent = np.array([length, width, height])
            
#             # Convert yaw to rotation matrix (around Z-axis)
#             rot_mat = np.array([
#                 [np.cos(yaw), -np.sin(yaw), 0],
#                 [np.sin(yaw), np.cos(yaw), 0],
#                 [0, 0, 1]
#             ])
#             obb.R = rot_mat
            
#             # Set box color (red by default)
#             obb.color = [1, 0, 0]
#             vis_obj.append(obb)



#     # Set visualization options
#     rend = o3d.visualization.RenderOption()
#     rend.line_width = 8.0
#     vis = o3d.visualization.Visualizer()
#     vis.update_renderer()
#     vis.create_window()

#     # Visualize all objects (point cloud and boxes)
#     for obj in vis_obj:
#         vis.add_geometry(obj)
#         vis.poll_events()
#         vis.update_geometry(obj)

#     # Visualize
#     vis.run()


# def main(src: str, version: str = 'v1.0-mini') -> None:
#     """Main function to fused and visualize lidar point clouds
#     of the MAN TruckScenes dataset.

#     Arguments:
#         src: Dataset root path.
#         version: Dataset version.
#     """
#     # Load TruckScenes dataset
#     trucksc = TruckScenes(version=version, dataroot=src)
#     count = 0
#     for scene in trucksc.scene:
#         print(count)
#         # Get first sample of the scene
#         sample = trucksc.get('sample', scene['first_sample_token'])

#         # Get fused point cloud
#         point_cloud, boxes = get_fused_pointcloud(trucksc, sample)
            
#         # Visualize fused point cloud
#         if (count > 100):
#             trucksc.render_sample(scene['first_sample_token'], nsweeps = 1)
#             visualize_pointcloud(point_cloud, boxes)
#         count += 1


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser('DPRT data preprocessing')
#     parser.add_argument('--src', type=str, default='/home/cx/server3/man-truckscenes/',
#                         help="Path to the dataset folder.")
#     parser.add_argument('--version', type=str, default='v1.0-trainval',
#                         help="Dataset version.")
#     args = parser.parse_args()
#     main(args.src, args.version)