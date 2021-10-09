# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import io
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage
from abc import abstractmethod

from .. import data as g_data
from GraphicsDL.graphicsutils import g_cfg, g_io


def get_cfg(cfg_type, cfg_path, shared_scope=''):
    cfg = cfg_type()
    cfg.load_from_yaml(cfg_path, shared_scope=shared_scope)
    return cfg


def get_cfg_from_pipeline(cfg_type, cfg_path, assemble_name=None):
    all_cfg = [p_p for p_p in get_cfg(cfg_type, cfg_path).dataset_list[0].process_pipelines]
    eval_cfg = [p_p for p_p in all_cfg if assemble_name is None or p_p.assemble_name == assemble_name]
    assert len(eval_cfg) == 1, f'cfg_path: {cfg_path}, assemble_name: {assemble_name}, {len(eval_cfg)} cfg'
    return eval_cfg[0]


def get_map_dict_from_csv(map_file, sep_str, id_key, map_key):
    map_dict = dict()
    map_csv = pd.read_csv(map_file, sep=sep_str)
    for k, v in map_csv[[id_key, map_key]].values:
        map_dict[k] = str(v)
    return map_dict


class AxisAlignBoundingBox(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.max = list([float(0.), float(0.), float(0.)])
        self.min = list([float(0.), float(0.), float(0.)])

    def assign_box_size(self, maximum, minimum):
        self.max = np.asarray(maximum, dtype=np.float32)
        self.min = np.asarray(minimum, dtype=np.float32)

    def scale(self, scale_num):
        self.max = np.asarray(self.max) * scale_num
        self.min = np.asarray(self.min) * scale_num

    def translation(self, translation_offset):
        self.max = np.asarray(self.max) + translation_offset
        self.min = np.asarray(self.min) + translation_offset

    def rotation(self, rotation_m):
        min_rot = np.matmul(rotation_m, np.asarray(self.min))
        max_rot = np.matmul(rotation_m, np.asarray(self.max))
        self.max = np.maximum(min_rot, max_rot)
        self.min = np.minimum(min_rot, max_rot)

    def center(self):
        return (np.array(self.min) + self.max) / 2

    def center_floor(self):
        c_floor = np.asarray(self.center())
        c_floor[1] = self.min[1]
        return c_floor

    def box_size(self):
        return np.array(self.max) - self.min

    def box_area(self):
        return np.prod(self.box_size())


class UniformObject(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.label = str()
        self.label_id = int(0)
        self.model_id = str()
        self.scale = int(1)
        self.bbox = AxisAlignBoundingBox()
        self.world_pose = list([[float(0.)] * 4] * 4)
        self.interpolation = int(1)
        self.rotation = list([float(0.)] * 3)

    def mesh_center_floor(self, model_dir):
        mesh_dir = os.path.join(model_dir, '../../object')
        mesh_path = os.path.join(mesh_dir, self.model_id, f'{self.model_id}_0.obj')
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(mesh_dir, self.model_id, f'{self.model_id}.obj')
            if not os.path.exists(mesh_path):
                raise NotImplementedError
        with g_io.OpenText(mesh_path, 'r') as obj_io:
            obj_local = np.array([list(map(float, obj_line.split()[1:4])) for obj_line in obj_io.read_lines()
                                  if obj_line.strip() != '' and obj_line.split()[0] == 'v'])

        obj_world = np.dot(np.insert(obj_local, 3, values=1, axis=1), self.world_pose)[:, :3]
        obj_max, obj_min = np.max(obj_world, axis=0), np.min(obj_world, axis=0)
        obj_center_floor = (obj_max + obj_min) / 2
        obj_center_floor[1] = obj_min[1]
        return obj_center_floor

    def read_points(self, objs_zip_meta):
        npz_meta = io.BytesIO(objs_zip_meta.read(f'{self.model_id}.npz'))
        obj_points = np.load(npz_meta)['arr_0'].astype(np.float32)
        if self.interpolation > 1:
            obj_min = np.min(obj_points, axis=0)
            obj_max = np.max(obj_points, axis=0)
            obj_center = (obj_max + obj_min) / 2
            obj_vox_scale = 126 / np.max(obj_max - obj_min)
            obj_vox_indices = ((obj_points - obj_center) * obj_vox_scale).astype(np.int32) + 64

            obj_voxel = np.zeros((128, 128, 128), dtype=np.uint8)
            obj_voxel[tuple(np.split(obj_vox_indices, 3, axis=-1))] = 1

            obj_vox_scaled = ndimage.zoom(obj_voxel, self.interpolation, np.uint8, mode='nearest')
            obj_points_scaled = np.argwhere(obj_vox_scaled > 0)
            obj_points = (obj_points_scaled / self.interpolation - 64) / obj_vox_scale + obj_center

        obj_points_world = np.concatenate([obj_points / self.scale, np.ones([obj_points.shape[0], 1])], axis=-1)
        obj_points_world = np.matmul(obj_points_world, self.world_pose)[:, :-1]
        return obj_points_world


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)


class UniformScene(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.scene_id = str()
        self.room_type = str()
        self.label_type = str()
        self.bbox = AxisAlignBoundingBox()
        self.objects = list([UniformObject()])

    def dump_to_json(self):
        return json.dumps(self.save(), cls=JsonEncoder, indent=4)

    @staticmethod
    def bbox2points(obj_max, obj_min, obj_label):
        bbox_edge_x_range = np.linspace(obj_min[0], obj_max[0], max((obj_max[0] - obj_min[0]) // 0.025, 2))
        bbox_edge_x_zeros = np.zeros_like(bbox_edge_x_range)
        bbox_edge_x_min = np.stack([bbox_edge_x_range, bbox_edge_x_zeros, bbox_edge_x_zeros + obj_min[2]], 1)
        bbox_edge_x_max = np.stack([bbox_edge_x_range, bbox_edge_x_zeros, bbox_edge_x_zeros + obj_max[2]], 1)

        bbox_edge_z_range = np.linspace(obj_min[2], obj_max[2], max((obj_max[2] - obj_min[2]) // 0.025, 2))
        bbox_edge_z_zeros = np.zeros_like(bbox_edge_z_range)
        bbox_edge_z_min = np.stack([bbox_edge_z_zeros + obj_min[0], bbox_edge_z_zeros, bbox_edge_z_range], 1)
        bbox_edge_z_max = np.stack([bbox_edge_z_zeros + obj_max[0], bbox_edge_z_zeros, bbox_edge_z_range], 1)

        bbox_edge_points = np.concatenate([bbox_edge_x_min, bbox_edge_x_max, bbox_edge_z_min, bbox_edge_z_max])
        bbox_edge_points_label = np.zeros([bbox_edge_points.shape[0]]) + obj_label
        return bbox_edge_points, bbox_edge_points_label

    def parse_scene(self, objs_zip_meta, bbox=False, vox_stride=0.2):
        assert objs_zip_meta is not None or bbox
        objs_points = list()
        objs_points_label = list()
        for obj in self.objects:
            if objs_zip_meta is not None:
                obj_points = obj.read_points(objs_zip_meta)
                obj_points_label = np.array([obj.label_id] * obj_points.shape[0])
                objs_points.append(obj_points)
                objs_points_label.append(obj_points_label)
            if bbox:
                bbox_max = np.array(obj.bbox.max) - vox_stride
                bbox_edge_points, bbox_edge_points_label = self.bbox2points(bbox_max, obj.bbox.min, obj.label_id)
                objs_points.append(bbox_edge_points)
                objs_points_label.append(bbox_edge_points_label)
        return objs_points, objs_points_label

    def get_scene_voxel(self, objs_zip_meta, room_size, room_stride):
        if len(self.objects) == 0:
            return None

        objs_points, points_label = self.parse_scene(objs_zip_meta)

        objs_points = np.concatenate(objs_points, axis=0)
        points_label = np.concatenate(points_label, axis=0)
        obj_vox_indices = (objs_points / room_stride + 0.1).astype(np.int32)
        point_xyz_max = (np.array(room_size) / room_stride).astype(np.int32)
        valid_indices = np.logical_and(np.all(obj_vox_indices >= 0, axis=1),
                                       np.all(obj_vox_indices < point_xyz_max, axis=1))
        obj_vox_indices, points_label = obj_vox_indices[valid_indices], points_label[valid_indices]

        voxel_size = (np.array(room_size) / room_stride).astype(np.int32)
        voxel_room = np.zeros(voxel_size, dtype=np.uint8)
        voxel_room[tuple(np.split(obj_vox_indices, 3, axis=-1))] = np.expand_dims(points_label, axis=-1)

        return voxel_room

    def vis_scene_voxel(self, objs_zip_meta, room_size, room_stride, vis_path):
        voxel_room = self.get_scene_voxel(objs_zip_meta, room_size, room_stride)
        if voxel_room is None:
            return None

        colors_map = getattr(g_data, self.label_type.upper())().color_map_arr()
        g_io.PlyIO().dump_vox(vis_path, voxel_room, vox_scale=room_stride, colors_map=colors_map)


class ProcessPipeline(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.label_type = str()
        self.assemble_name = str()
        self.room_types = list([str()])
        self.room_size = list([float(0), float(0)])
        self.room_height = float(0)
        self.room_stride = float(0)
        self.render_cfg = str()


class DataGenConfig(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.data_dir = str()
        self.out_dir = str()
        self.data_type = str()
        self.process_pipelines = list([ProcessPipeline()])


class GroupDataGenConfig(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.dataset_list = list([DataGenConfig()])


class BaseDataGen(object):
    def __init__(self, data_dir, out_dir, process_pipelines, **kargs):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.process_pipelines = process_pipelines
        self.out_assemble_dir = os.path.join(self.out_dir, 'AssembleData')

    @abstractmethod
    def mediate_process(self): pass

    @staticmethod
    def get_assemble_writer(assemble_path):
        train_samples_path = assemble_path + '_samples'
        train_samples_zip = g_io.ZipIO(assemble_path + '.zip', 'w')
        option = tf.io.TFRecordOptions()
        option.compression_type = tf.compat.v1.python_io.TFRecordCompressionType.ZLIB
        train_samples_record = tf.io.TFRecordWriter(assemble_path + '.records', option)
        return train_samples_path, train_samples_zip, train_samples_record

    @staticmethod
    def filter_far_view(cam_t, cam_r): pass

    def filter_far_views(self, cam_t_list, cam_r_list, camera_id_list=None):
        filtered_cam_t_list, filtered_cam_r_list, filtered_cam_id_list = list(), list(), list()
        assert len(cam_t_list) == len(cam_r_list)
        for cam_i in range(len(cam_t_list)):
            cam_t, cam_r = cam_t_list[cam_i], cam_r_list[cam_i]
            if self.filter_far_view(cam_t, cam_r):
                filtered_cam_t_list.append(cam_t)
                filtered_cam_r_list.append(cam_r)
                if camera_id_list is not None:
                    filtered_cam_id_list.append(camera_id_list[cam_i])
        return np.asarray(filtered_cam_t_list), np.asarray(filtered_cam_r_list), np.array(filtered_cam_id_list)
