# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import io
import copy
import json
import trimesh
import logging
import numpy as np
import pandas as pd
from typing import List

from .. import data as g_data
from ..data.base import get_cfg_from_pipeline
from ..analyzer.uniform_scene import VisualizationUtilize
from ..data.base import GroupDataGenConfig, AxisAlignBoundingBox, UniformScene
from GraphicsDL.graphicsutils import g_str, g_io, g_perf, g_cfg


trimesh.util.attach_to_log(level=logging.ERROR)


class RetrievalUtilize(object):
    @staticmethod
    def get_pose(floor_center, rotation):
        cos_rotation, sin_rotation = np.cos(rotation), np.sin(rotation)
        obj_pose = np.array([[cos_rotation, 0, -sin_rotation, floor_center[0]],
                             [0, 1, 0, floor_center[1]],
                             [sin_rotation, 0, cos_rotation, floor_center[2]],
                             [0, 0, 0, 1]],
                            dtype=np.float32).T
        return obj_pose

    @staticmethod
    def vec_bbox(vox, size_ext=0):
        vox_min, vox_max = np.min(vox, axis=0), np.max(vox, axis=0)
        vox_size = vox_max - vox_min + size_ext
        vox_floor_center = (vox_max + vox_min) / 2
        vox_floor_center[1] = vox_min[1]
        return vox_min, vox_max, vox_size, vox_floor_center

    @staticmethod
    def xz_rotation_self(phi, vec):
        _, _, _, vec_floor_center = RetrievalUtilize.vec_bbox(vec)
        center_vec = vec - vec_floor_center
        rotate_matrix = np.array([[np.cos(phi), 0, -np.sin(phi)],
                                  [0, 1, 0],
                                  [np.sin(phi), 0, np.cos(phi)]], dtype=np.float32)
        rotated_center_vec = np.matmul(rotate_matrix, center_vec.T).T
        rotated_vec = rotated_center_vec + vec_floor_center
        return rotated_vec

    @staticmethod
    def get_vox_points(obj_points, voxel_size, vox_stride, floor_center):
        target_vox = np.zeros(voxel_size)
        t_points_vox = (obj_points + floor_center) / vox_stride
        target_vox[tuple(np.split(t_points_vox.astype(np.int32), 3, axis=-1))] = 1
        target_points = np.argwhere(target_vox > 0) * vox_stride
        target_points = target_points - RetrievalUtilize.vec_bbox(target_points)[-1]
        return target_points

    @staticmethod
    def chamfer_distance(source_obj_points, target_obj_points):
        s_num, t_num, f_num = source_obj_points.shape[0], target_obj_points.shape[0], source_obj_points.shape[1]
        source_expanded_points = np.tile(source_obj_points, (t_num, 1))
        target_expanded_points = np.reshape(np.tile(np.expand_dims(target_obj_points, 1), (1, s_num, 1)), (-1, f_num))
        distances = np.linalg.norm(source_expanded_points - target_expanded_points, axis=1)
        distances = np.reshape(distances, (t_num, s_num))
        chamfer_distance = np.mean(np.min(distances, axis=0)) + np.mean(np.min(distances, axis=1))
        return chamfer_distance

    @staticmethod
    def collision_vox_ratio(scene_occupy, obj_pts, obj_center_floor, vox_stride):
        obj_pts_vox = (obj_pts + obj_center_floor) / vox_stride
        obj_pts_vox = np.clip(obj_pts_vox, 0, np.array(scene_occupy.shape) - 1).astype(np.int32)
        obj_vox = np.zeros_like(scene_occupy)
        obj_vox[tuple(np.split(obj_pts_vox, 3, axis=-1))] = 1
        collision_ratio = np.sum(np.logical_and(scene_occupy == 1, obj_vox == 1)) / np.sum(obj_vox)
        return collision_ratio

    @staticmethod
    def iou_distance(source_obj_points, target_obj_points, voxel_size, vox_stride, floor_center):
        source_vox = np.zeros(voxel_size)
        target_vox = np.zeros(voxel_size) - 1
        s_points_vox = (source_obj_points + floor_center) / vox_stride
        t_points_vox = (target_obj_points + floor_center) / vox_stride
        source_vox[tuple(np.split(s_points_vox.astype(np.int32), 3, axis=-1))] = 1
        target_vox[tuple(np.split(t_points_vox.astype(np.int32), 3, axis=-1))] = 1

        inter_area = np.sum(source_vox == target_vox)
        union_area = np.sum(source_vox == 1) + np.sum(target_vox == 1) - inter_area
        iou = inter_area / union_area
        return 1 - iou

    @staticmethod
    def get_support_objects(scene_info, object_info, support_label, distance=0.3):
        def floor_intersection(a_bbox, b_bbox):
            bbox_intersection = np.maximum(np.minimum(a_bbox.max, b_bbox.max) - np.maximum(a_bbox.min, b_bbox.min), 0)
            return np.prod(bbox_intersection[[0, 2]])
        stand_list = [obj for obj in scene_info.objects if obj.label == support_label or obj.label in support_label]
        below_stand_list = [obj for obj in stand_list if floor_intersection(obj.bbox, object_info.bbox) > 0]
        support_stand_list = [obj for obj in below_stand_list if object_info.bbox.min[1] - obj.bbox.max[1] < distance]
        return support_stand_list

    @staticmethod
    def read_points_from_binvox(shape_zip_reader, model_id, label_name):
        try:
            if label_name == 'curtain':
                vox_path = os.path.join(g_str.abs_dir_path(__file__), 'resource', 'curtain', model_id[:-4] + '.binvox')
                model_points = BinvoxIO(vox_path).points_xyz
            else:
                binvox = BinvoxIO(io.BytesIO(shape_zip_reader.read(model_id[:-4] + '.solid.binvox')))
                model_points = np.matmul(binvox.points_xyz, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).T)
                model_norm = ModelNormalizedConfig()
                model_norm.load(json.loads(shape_zip_reader.read(model_id[:-4] + '.json')))
                model_vox_size = np.max(model_points, 0) - np.min(model_points, 0)
                model_points = model_points / model_vox_size * model_norm.norm_size()
            return model_points
        except ValueError:
            return None

    @staticmethod
    def read_objects_points(objects_list, shape_zip_reader):
        objs_points, objs_label = list(), list()
        for obj_i, obj_info in enumerate(objects_list):
            model_points = RetrievalUtilize.read_points_from_binvox(shape_zip_reader, obj_info.model_id, obj_info.label)
            _, _, _, model_points_floor_center = RetrievalUtilize.vec_bbox(model_points)
            center_model_points = model_points - model_points_floor_center
            model_points_ones = np.concatenate([center_model_points, np.ones([center_model_points.shape[0], 1])], -1)
            rotated_model_points = np.matmul(model_points_ones, obj_info.world_pose)[:, :-1]
            objs_points.append(rotated_model_points)
            objs_label.append(np.ones([rotated_model_points.shape[0], 1], dtype=np.int32) * obj_info.label_id)
        return objs_points, objs_label


class BinvoxIO(object):
    def __init__(self, bin_path):
        self.bin_path = bin_path
        self.scale = None
        self.dimension = None
        self.translation = None
        self.volumetric = None
        self.points_xyz = None
        self.read_bin_vox()

    def read_bin_vox(self):
        fp = self.bin_path
        if not isinstance(self.bin_path, io.BytesIO):
            fp = open(self.bin_path, 'rb')
        _ = fp.readline().rstrip().decode('utf-8')
        dim_str = g_str.reorganize_by_symbol(fp.readline().rstrip().decode('utf-8'), ' ', slice(1, 4))
        self.dimension = np.fromstring(dim_str, sep=' ', dtype=np.int32)
        translate_str = g_str.reorganize_by_symbol(fp.readline().rstrip().decode('utf-8'), ' ', slice(1, 4))
        self.translation = np.fromstring(translate_str, sep=' ', dtype=np.float32)
        self.scale = float(g_str.reorganize_by_symbol(fp.readline().rstrip().decode('utf-8'), ' ', slice(1, 2)))
        scale_norm = self.scale / self.dimension[0]
        _ = fp.readline().rstrip().decode('utf-8')
        data_pair = np.transpose(np.reshape(np.frombuffer(fp.read(), dtype=np.uint8), [-1, 2]))
        data_stop = np.cumsum(data_pair[1])
        data_start = np.concatenate([np.array([0], dtype=np.uint32), data_stop[:-1]])
        data_stop = data_stop[data_pair[0] != 0]
        data_start = data_start[data_pair[0] != 0]

        points_indices = np.concatenate([np.arange(stt, stp) for stt, stp in zip(data_start, data_stop)])
        points_xy = (points_indices / self.dimension[2]).astype(np.uint32)
        points_x = (points_xy / self.dimension[1]).astype(np.uint32)
        points_y = points_xy % self.dimension[1]
        points_z = points_indices % self.dimension[2]
        points_xyz = np.stack([points_x, points_y, points_z], axis=-1).astype(np.int32)
        self.points_xyz = points_xyz.astype(np.float32) * scale_norm + self.translation
        self.volumetric = np.zeros(self.dimension, np.uint8)
        self.volumetric[tuple(np.split(points_xyz, 3, axis=-1))] = 1


class ZipResolver(trimesh.resolvers.Resolver):
    def __init__(self, zip_reader, model_id):
        self.zip_reader = zip_reader
        self.model_dir = '/'.join(model_id.split('/')[:-1])

    def get(self, name):
        file_path = os.path.normpath(os.path.join(self.model_dir, name)).replace('\\', '/')
        return io.BytesIO(self.zip_reader.read(file_path)).read()


class TaxonomyItem(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.synsetId: str = str()
        self.name: str = str()
        self.children: List[int] = list([int(0)])
        self.numInstances: int = int(0)


class ShapeTaxonomy(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.taxonomy: List[TaxonomyItem] = list([TaxonomyItem()])


class ModelNormalizedConfig(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.id: str = str()
        self.numVertices: int = int(0)
        self.max: List[float] = list([float(0)])
        self.min: List[float] = list([float(0)])
        self.centroid: List[float] = list([float(0)])

    def norm_size(self):
        return np.array(self.max) - self.min


class ObjectsRetrieval(object):
    def __init__(self, eval_dir, output_dir, shapenet_path, cfg_path, assemble_name=None):
        self.eval_dir = eval_dir
        self.output_dir = g_str.mkdir_automated(os.path.join(output_dir, 'retrieval'))

        self.cfg = get_cfg_from_pipeline(GroupDataGenConfig, cfg_path, assemble_name)
        self.room_size, self.vox_stride = np.insert(self.cfg.room_size, 1, self.cfg.room_height), self.cfg.room_stride
        self.vox_size = (self.room_size / self.vox_stride).astype(np.int32)
        self.room_center = self.room_size * [0.5, 0, 0.5]

        self.label_type = self.cfg.label_type
        self.data_label = getattr(g_data, self.label_type.replace('-', '').upper())()
        self.label_list, self.color_map = self.data_label.label_id_map_arr(), self.data_label.color_map_arr()

        self.rotation_split = 8
        self.align_model = [
            'bed', 'cabinet', 'night_stand', 'desk', 'shelves', 'table', 'tv_stand', 'picture', 'curtain',
            'refridgerator', 'sofa', 'television',
        ]
        self.no_collision_model_list = ['television']

        self.shapenet_path = shapenet_path
        self.shape_zip_reader = g_io.ZipIO(self.shapenet_path)
        self.floor_mesh_path = os.path.join(g_str.abs_dir_path(__file__), 'resource', 'floor', 'floor.obj')
        self.category_model_list = self.get_model_id_list()

        self.wall_flag_list = [9, 10, 6, 5]
        self.camera_fov = [8, 6]
        self.render_resolution = [640, 480]
        room_half_size = self.cfg.room_size[0] / 2
        cam_height = 1.3 / 3.2 * room_half_size
        self.camera_matrix_list = VisualizationUtilize.get_render_camera_parameter(room_half_size, cam_height)

    def get_model_id_list(self):
        shape_list_dic = dict()
        shape_path = os.path.join(g_str.abs_dir_path(__file__), 'resource', 'shape', f'shape_list.csv')
        shape_list_map = pd.read_csv(shape_path, sep=',')
        for label_name, shape_path, source in shape_list_map[['label', 'shape', 'shapenet']].values:
            if label_name not in shape_list_dic:
                shape_list_dic[label_name] = list()
            shape_path = shape_path if source == 0 else f'ShapeNetCore.v2/{shape_path}/models/model_normalized.obj'
            shape_list_dic[label_name].append(shape_path)
        if 'kitchen' in self.label_type.lower():
            shape_list_dic['cabinet'] = shape_list_dic['kitchen_cabinet']
        shape_list_dic = {key: np.asarray(value) for key, value in shape_list_dic.items()}
        return shape_list_dic

    def gen_rotate_list(self, scene_info, object_info):
        rotation_split = 4 if self.label_list[object_info.label_id] in self.align_model else self.rotation_split
        assert self.rotation_split in [4, 8]
        two_pi = 2 * np.pi
        half_pi = np.pi / 2
        unit_degree = two_pi / rotation_split
        room_max, room_min = np.array(scene_info.bbox.max), np.array(scene_info.bbox.min)
        obj_max, obj_min = np.array(object_info.bbox.max), np.array(object_info.bbox.min)
        wall_distances = np.append(obj_min - room_min, room_max - obj_max)[[5, 1, 2, 3]]
        distances_sort = np.argsort(wall_distances)
        rotate_list = [half_pi * r_i for r_i in distances_sort]
        for r_i in distances_sort:
            for m_r in range(1, 1 + rotation_split//8):
                rotate = half_pi * r_i + m_r * unit_degree
                if rotate not in rotate_list:
                    rotate_list.append(rotate)
                rotate = (half_pi * r_i - m_r * unit_degree) % two_pi
                if rotate not in rotate_list:
                    rotate_list.append(rotate)
        return rotate_list

    def get_fine_grained_class(self, scene_info, object_info):
        lamp_type = self.label_list[object_info.label_id]
        if lamp_type != 'lamp':
            return lamp_type
        if object_info.bbox.min[1] < 1.5:
            support_label = ['table', 'desk', 'night_stand', 'stand']
            if len(RetrievalUtilize.get_support_objects(scene_info, object_info, support_label)) > 0:
                lamp_type = 'table_lamp'
            else:
                lamp_type = 'floor_lamp'
        else:
            lamp_type = 'ceiling_lamp'
        return lamp_type

    def retrieval_object(self, scene_info, object_info, objects_zip, scene_occupy, shape_zip_reader):
        scene_occupy_ = scene_occupy.copy()
        label_name = self.get_fine_grained_class(scene_info, object_info)
        model_list = self.category_model_list[label_name]

        obj_points = object_info.read_points(objects_zip)
        if object_info.label == 'bed' and 'pillow' in self.label_list:
            pillow_list = RetrievalUtilize.get_support_objects(scene_info, object_info, 'pillow')
            pillow_pts = [obj.read_points(objects_zip) for obj in pillow_list]
            obj_points = np.concatenate([obj_points, *pillow_pts], axis=0)

        obj_vox_indices = (obj_points / self.vox_stride + 0.1).astype(np.int32)
        scene_occupy_[tuple(np.split(obj_vox_indices, 3, axis=-1))] = 0
        obj_center_floor = RetrievalUtilize.vec_bbox(obj_points)[-1]
        obj_points = obj_points - obj_center_floor

        rotate_list = self.gen_rotate_list(scene_info, object_info)
        matched_model_id, matched_model_rotation, match_distance, matched_center = 0, 0, 1e10, obj_center_floor
        for m_i, t_model_id in enumerate(model_list):
            t_model_points = RetrievalUtilize.read_points_from_binvox(shape_zip_reader, t_model_id, label_name)
            if t_model_points is None or np.any(RetrievalUtilize.vec_bbox(t_model_points)[2] >= self.room_size):
                continue
            for r_i, rotation in enumerate(rotate_list):
                t_center = obj_center_floor.copy()
                t_rotated_points = RetrievalUtilize.xz_rotation_self(rotation, t_model_points)
                _, _, t_points_size, t_points_floor_center = RetrievalUtilize.vec_bbox(t_rotated_points)
                max_size = np.maximum(object_info.bbox.box_size(), t_points_size)
                voxel_size = np.ceil(max_size / self.vox_stride).astype(np.int32) + 1

                floor_center = max_size / 2
                floor_center[1] = 0
                t_obj_points = t_rotated_points - t_points_floor_center
                t_obj_vox_pts = RetrievalUtilize.get_vox_points(t_obj_points, voxel_size, self.vox_stride, floor_center)
                obj_distance = RetrievalUtilize.chamfer_distance(obj_points, t_obj_vox_pts)
                if label_name not in self.no_collision_model_list:
                    collision_ratio = RetrievalUtilize.collision_vox_ratio(scene_occupy_, t_obj_points, t_center,
                                                                           self.vox_stride)
                    obj_distance += collision_ratio * 5
                if match_distance > obj_distance:
                    matched_model_id = t_model_id
                    matched_model_rotation = rotation
                    match_distance = obj_distance
                    matched_center = t_center
        return matched_model_id, matched_model_rotation, match_distance, matched_center

    def visualize_mesh(self, scene_path, vis_path, shape_zip_reader, floor_mesh_path=None, force_refresh=False):
        if not force_refresh and os.path.exists(f'{vis_path}_view0.png') and os.path.exists(f'{vis_path}_view1.png') \
                and os.path.exists(f'{vis_path}_view2.png') and os.path.exists(f'{vis_path}_view3.png'):
            return
        logging.info(f'Visualize {scene_path} mesh')
        scene_info = UniformScene()
        scene_info.load_from_json(scene_path)
        scene_mesh = None
        for obj_i, obj_info in enumerate(scene_info.objects):
            if obj_info.label == 'curtain':
                curtain_path = os.path.join(g_str.abs_dir_path(__file__), 'resource', 'curtain', obj_info.model_id)
                obj_mesh = trimesh.load(curtain_path)
            else:
                obj_resolver = ZipResolver(shape_zip_reader, obj_info.model_id)
                obj_file = io.BytesIO(shape_zip_reader.read(obj_info.model_id))
                obj_mesh = trimesh.load(obj_file, file_type='obj', resolver=obj_resolver)
                model_norm = ModelNormalizedConfig()
                model_norm.load(json.loads(shape_zip_reader.read(obj_info.model_id[:-4] + '.json')))
                norm_scale = model_norm.norm_size() / obj_mesh.bounding_box.extents
                obj_mesh.apply_transform(np.eye(4) * np.insert(norm_scale, 3, 1))
            if isinstance(obj_mesh, trimesh.Trimesh):
                obj_mesh = trimesh.Scene(geometry=obj_mesh)
            obj_mesh_center_floor = np.mean(obj_mesh.bounding_box.bounds, axis=0)
            obj_mesh_center_floor[1] = obj_mesh.bounding_box.bounds[0, 1]
            obj_mesh.apply_translation(-obj_mesh_center_floor)
            obj_mesh.apply_transform(np.asarray(obj_info.world_pose).T)
            if scene_mesh is not None:
                scene_mesh.add_geometry(obj_mesh.geometry)
            else:
                scene_mesh = obj_mesh

        if scene_mesh is None:
            return
        scene_bbox = AxisAlignBoundingBox()
        scene_max = np.array(scene_mesh.bounding_box.bounds[1])
        scene_max[1] = max(scene_max[1] + 0.2, 2.7)
        scene_min = scene_mesh.bounding_box.bounds[0]
        scene_max, scene_min = np.maximum(scene_max, scene_info.bbox.max), np.minimum(scene_min, scene_info.bbox.min)
        scene_bbox.assign_box_size(scene_max, scene_min)
        VisualizationUtilize.mesh_scene_visualization(vis_path, scene_mesh, scene_bbox, self.camera_matrix_list,
                                                      floor_mesh_path, self.wall_flag_list,
                                                      self.camera_fov, self.render_resolution)

    def visualize_vox(self, scene_path, vis_path, shape_zip_reader, cfg, force_refresh=False):
        vox_vis_path = f'{vis_path}_vox_view'
        if not force_refresh and os.path.exists(vox_vis_path + '0.png') and os.path.exists(vox_vis_path + '1.png') \
                and os.path.exists(vox_vis_path + '2.png') and os.path.exists(vox_vis_path + '3.png'):
            return
        logging.info(f'Visualize {scene_path} voxel')
        scene_info = UniformScene()
        scene_info.load_from_json(scene_path)
        voxel_size = (np.array(self.room_size) / cfg.room_stride).astype(np.int32)
        voxel_room = np.zeros(voxel_size, dtype=np.uint8)
        objs_points, objs_label = RetrievalUtilize.read_objects_points(scene_info.objects, shape_zip_reader)
        objs_points = np.concatenate(objs_points, axis=0)
        objs_label = np.concatenate(objs_label, axis=0)
        points_obj = (objs_points / cfg.room_stride).astype(np.int32)
        if np.any(np.any(points_obj >= voxel_size, axis=-1)) or np.any(points_obj < 0):
            points_ind = np.logical_and(np.all(points_obj < voxel_size, -1), np.all(points_obj >= 0, -1))
            objs_label = objs_label[points_ind]
            points_obj = points_obj[points_ind]
        voxel_room[tuple(np.split(points_obj, 3, axis=-1))] = objs_label
        VisualizationUtilize.vol_top_view_visualization(vis_path + '_vox', voxel_room, colors_map=self.color_map)
        g_io.PlyIO().dump_vox(vis_path + '_vox.ply', voxel_room, vox_scale=cfg.room_stride, colors_map=self.color_map)
        VisualizationUtilize.scene_vox_visualization(f'{vis_path}_vox.ply', f'{vis_path}_vox', self.camera_matrix_list,
                                                     self.camera_fov, self.render_resolution)

    @staticmethod
    def refine_support_relation(scene_info, obj_info, shape_zip_reader):
        if 'lamp' in obj_info.label:
            support_label = ['table', 'desk', 'night_stand', 'stand']
        elif obj_info.label == 'pillow':
            support_label = ['chair', 'bed', 'sofa']
        elif obj_info.label == 'television':
            support_label = ['cabinet', 'table', 'desk', 'tv_stand']
        elif obj_info.label == 'computer':
            support_label = ['stand', 'table', 'desk', 'tv_stand']
        elif obj_info.label == 'laptop':
            support_label = ['stand', 'table', 'desk', 'ottoman', 'sofa', 'bed']
        else:
            raise NotImplementedError

        support_object_list = RetrievalUtilize.get_support_objects(scene_info, obj_info, support_label)
        if len(support_object_list) == 0:
            return obj_info
        support_object_pts, _ = RetrievalUtilize.read_objects_points(support_object_list, shape_zip_reader)
        support_object_pts = np.concatenate(support_object_pts, axis=0)
        support_object_min, support_object_max = np.min(support_object_pts, axis=0), np.max(support_object_pts, axis=0)
        if obj_info.bbox.max[1] - support_object_max[1] > 0.2:
            obj_info.world_pose[3][1] = support_object_max[1]
        return obj_info

    def refine_objects_position(self, scene_info: UniformScene, shape_zip_reader):
        support_list = ['television', 'table_lamp', 'computer', 'laptop', 'pillow']
        floor_list = ['bed', 'wardrobe_cabinet', 'floor_lamp', 'night_stand', 'stand', 'chair', 'rug',
                      'desk', 'dressing_table', 'table', 'tv_stand', 'ottoman', 'sofa', 'refridgerator', ]
        floor_list = [*floor_list, 'cabinet'] if 'kitchen' not in self.label_type.lower() else floor_list
        ceil_list = ['ceiling_lamp']
        for obj_i, obj_info in enumerate(scene_info.objects):
            obj_label = self.get_fine_grained_class(scene_info, obj_info)
            if obj_label not in ['lamp'] + support_list + floor_list + ceil_list:
                continue
            if obj_label in ceil_list:
                scene_info.objects[obj_i].world_pose[3][1] = max(scene_info.bbox.max[1] - obj_info.bbox.box_size()[1],
                                                                 obj_info.world_pose[3][1])
            elif obj_label in support_list:
                scene_info.objects[obj_i] = self.refine_support_relation(scene_info, obj_info, shape_zip_reader)
            elif obj_label in floor_list:
                scene_info.objects[obj_i].world_pose[3][1] = 0
            else:
                raise NotImplementedError
        return scene_info

    def retrieval_objects(self, samples, vis_dir, scenes_path, objs_path, start_index=0, worker_id=0):
        del start_index
        scenes_zip, objs_zip = g_io.ZipIO(scenes_path), g_io.ZipIO(objs_path)
        shape_zip_reader = g_io.ZipIO(self.shapenet_path) if self.shape_zip_reader is None else self.shape_zip_reader
        for s_i, sample in enumerate(samples):
            out_path = os.path.join(vis_dir, sample[:-5] + '_retrieval')
            if not os.path.exists(out_path + '.json'):
                scene_info = UniformScene()
                scene_info.load(json.loads(scenes_zip.read(sample)))
                logging.info(f'Retrieval{worker_id}: {s_i}/{len(samples)} {sample} {len(scene_info.objects)} objs')
                scene_info_retrieval = copy.deepcopy(scene_info)

                objects_list = list()
                scene_vox = scene_info.get_scene_voxel(objs_zip, self.room_size, self.vox_stride)
                scene_occupy = np.where(scene_vox > 0, np.ones_like(scene_vox), np.zeros_like(scene_vox))
                for obj_i, obj_info in enumerate(scene_info_retrieval.objects):
                    retrieve_out = self.retrieval_object(scene_info, obj_info, objs_zip, scene_occupy, shape_zip_reader)
                    if retrieve_out is None:
                        continue
                    model_id, rotation, _, floor_center = retrieve_out
                    obj_info.model_id = model_id
                    obj_info.world_pose = RetrievalUtilize.get_pose(floor_center, rotation).tolist()
                    objects_list.append(obj_info)
                scene_info_retrieval.objects = objects_list
                scene_info_retrieval = self.refine_objects_position(scene_info_retrieval, shape_zip_reader)
                scene_info_retrieval.save_to_json(out_path + '.json')
            self.visualize_mesh(out_path + '.json', out_path, shape_zip_reader, self.floor_mesh_path)
            self.visualize_vox(out_path + '.json', out_path, shape_zip_reader, self.cfg)
        scenes_zip.close()
        objs_zip.close()

    def eval_retrieval(self, num_workers=4, stride=1, sample_num=100):
        scenes_path = os.path.join(self.eval_dir, f'eval_meta_uni_rep.zip')
        objs_path = os.path.join(self.eval_dir, f'eval_meta_obj_rep.zip')
        if not os.path.exists(scenes_path) or not os.path.exists(objs_path):
            logging.warning(f'There is no post processed uniform scene.')
            return

        scenes_zip = g_io.ZipIO(scenes_path)
        samples_list = [s_n for s_i, s_n in enumerate(scenes_zip.namelist()) if s_i % stride == 0][:sample_num]
        scenes_zip.close()

        self.shape_zip_reader = None
        g_perf.multiple_processor(self.retrieval_objects, samples_list, num_workers,
                                  (self.output_dir, scenes_path, objs_path))
