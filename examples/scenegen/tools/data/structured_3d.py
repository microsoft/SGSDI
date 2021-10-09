# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import io
import cv2
import json
import logging
import numpy as np
import tensorflow as tf
from typing import List
from scipy.spatial.transform import Rotation as R

from .. import data as g_data
from .base import BaseDataGen, ProcessPipeline, AxisAlignBoundingBox
from GraphicsDL.graphicsutils import g_io, g_cfg, g_math, g_str, g_perf


class SemanticsPlane(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.ID: int = int(0)
        self.planeID: List[int] = list([int(0)])
        self.type: str = str()


class Junction(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.ID: int = int(0)
        self.coordinate: List[float] = list([float(0)])


class Line(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.ID: int = int(0)
        self.point: List[float] = list([float(0)])
        self.direction: List[float] = list([float(0)])


class Plane(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.offset: float = float(0)
        self.type: str = str()
        self.ID: int = int(0)
        self.normal: List[float] = list([float(0)])


class Annotation3D(g_cfg.DictRecursive):
    def __init__(self):
        super().__init__()
        self.junctions: List[Junction] = list([Junction()])
        self.lines: List[Line] = list([Line()])
        self.planes: List[Plane] = list([Plane()])
        self.planeLineMatrix: List[List[int]] = list([list([0])])
        self.lineJunctionMatrix: List[List[int]] = list([list([0])])
        self.semantics: List[SemanticsPlane] = list([SemanticsPlane()])

    def get_semantics_by_room_id(self, room_id):
        for k_id, k_ in enumerate(self.semantics):
            if k_.ID == room_id:
                return k_
        return None

    def get_rooms_by_type(self, r_type) -> List[int]:
        room_list = list()
        for k_id, k_ in enumerate(self.semantics):
            if k_.type != r_type:
                continue
            room_list.append(k_.ID)
        return room_list

    def get_semantic_bounding_box(self, room_id) -> AxisAlignBoundingBox:
        planes_id = self.get_semantics_by_room_id(int(room_id)).planeID
        planes = [p_ for p_ in self.planes if p_.ID in planes_id]
        plane_lines_matrix = np.asarray(self.planeLineMatrix)
        lines_id = [np.argwhere(plane_lines_matrix[p_.ID])[..., 0] for p_ in planes]
        lines_id = np.unique(np.concatenate(lines_id))
        line_junctions_matrix = np.asarray(self.lineJunctionMatrix)
        junctions_id = [np.argwhere(line_junctions_matrix[l_])[..., 0] for l_ in lines_id]
        junctions_id = np.unique(np.concatenate(junctions_id))
        junctions = [j_ for j_ in self.junctions if j_.ID in junctions_id]
        points = [p_.coordinate for p_ in junctions]
        semantic_box = AxisAlignBoundingBox()
        semantic_box.assign_box_size(np.max(points, axis=0).tolist(), np.min(points, axis=0).tolist())
        return semantic_box


class S3DUtilize(object):
    @staticmethod
    def get_fov_normal(image_size, cam_focal):
        u2x, v2y = [(np.arange(1, image_size[a_i] + 1) - image_size[a_i] / 2) / cam_focal[a_i] for a_i in [0, 1]]
        cam_m_u2x = np.tile([u2x], (image_size[1], 1))
        cam_m_v2y = np.tile(v2y[:, np.newaxis], (1, image_size[0]))
        cam_m_depth = np.ones(image_size).T
        fov_normal = np.stack((cam_m_depth, -1 * cam_m_v2y, cam_m_u2x), axis=-1)
        fov_normal = fov_normal / np.sqrt(np.sum(np.square(fov_normal), axis=-1, keepdims=True))
        return fov_normal

    @staticmethod
    def cast_perspective_to_local_coord(depth_img: np.ndarray, fov_normal):
        return np.expand_dims(depth_img, axis=-1) * fov_normal

    @staticmethod
    def cast_points_to_voxel(points, labels, room_size=(6.4, 3.2, 6.4), room_stride=0.2):
        vol_resolution = (np.asarray(room_size) / room_stride).astype(np.int32)
        vol_index = np.floor(points / room_stride).astype(np.int32)
        in_vol = np.logical_and(np.all(vol_index < vol_resolution, axis=1), np.all(vol_index >= 0, axis=1))
        x, y, z = [d_[..., 0] for d_ in np.split(vol_index[in_vol], 3, axis=-1)]
        vol_label = labels[in_vol]
        vol_data = np.zeros(vol_resolution, dtype=np.uint8)
        vol_data[x, y, z] = vol_label
        return vol_data

    @staticmethod
    def get_rotation_matrix_from_tu(cam_front, cam_up):
        cam_n = np.cross(cam_front, cam_up)
        cam_m = np.stack((cam_front, cam_up, cam_n), axis=1).astype(np.float32)
        return cam_m


class Structured3DDataGen(BaseDataGen):
    def __init__(self, data_dir, out_dir, process_pipelines, cfg=None, **kargs):
        super().__init__(data_dir, out_dir, process_pipelines, **kargs)
        cfg = cfg if cfg is not None else process_pipelines[0]
        room_size = np.insert(cfg.room_size, 1, cfg.room_height)
        self.room_size, self.room_stride = np.array(room_size), cfg.room_stride
        self.room_center = self.room_size * [0.5, 0, 0.5]
        self.vox_size = (self.room_size / self.room_stride).astype(np.int32)

        self.label_type = cfg.label_type
        self.data_label = getattr(g_data, self.label_type.upper())()
        self.label_list = self.data_label.label_id_map_arr()
        self.color_map = np.concatenate([self.data_label.color_map_arr(), [[0, 0, 0]]], axis=0)

        self.fov_n = None
        self.select_nyu_label_id, self.label_mapping, self.category_mapping = None, None, None
        self.init_config()

    def init_config(self):
        nyu40_label = g_data.NYU40().label_id_map_arr()
        select_nyu_label = self.label_list + ['desk'] if 'living' in self.label_type else self.label_list
        self.select_nyu_label_id = [nyu40_label.index(s_l) for s_l in select_nyu_label]
        self.category_mapping = np.zeros(len(g_data.NYU40().label_id_map_arr()), dtype=np.uint8)
        for s_i, s_l in enumerate(self.select_nyu_label_id):
            self.category_mapping[s_l] = s_i
        if 'living' in self.label_type:
            self.category_mapping[nyu40_label.index('desk')] = self.label_list.index('table')

        image_size = np.array([1280, 720], np.int32)
        cam_half_fov = np.array([0.698132, 0.440992])
        self.fov_n = S3DUtilize.get_fov_normal(image_size, image_size / 2 / np.tan(cam_half_fov))

    def load_zips(self, filter_regex='Structured3D') -> g_io.GroupZipIO:
        ctx_files = [f for f in os.listdir(self.data_dir) if filter_regex in f]
        zip_reader = g_io.GroupZipIO([os.path.join(self.data_dir, f) for f in ctx_files])
        return zip_reader

    @staticmethod
    def read_file_from_zip(zip_reader, scene_id, file_, filter_regex='Structured3D'):
        ctx = zip_reader.read('/'.join((filter_regex, scene_id, file_)))
        return io.BytesIO(ctx)

    def load_scene_anno_from_zip(self, zip_reader, scene_id: str):
        anno_3d = Annotation3D()
        anno_3d.load(json.load(self.read_file_from_zip(zip_reader, scene_id, 'annotation_3d.json')))
        return anno_3d

    def get_room_box_from_zip(self, zip_reader, scene_id: str, room_id: str):
        scene_anno = self.load_scene_anno_from_zip(zip_reader, scene_id)
        room_box = scene_anno.get_semantic_bounding_box(room_id)
        room_box.scale(1 / 1000)
        room_box.rotation(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
        return room_box

    def assemble_semantic_points_from_img(self, depth_img, semantic_pano, cos_threshold=0.15):
        points = S3DUtilize.cast_perspective_to_local_coord(depth_img, self.fov_n)
        points_normal = g_math.normal_from_cross_product(points)
        view_dist = np.maximum(np.linalg.norm(points, axis=-1, keepdims=True), float(10e-5))
        cosine_dist = np.sum((points * points_normal / view_dist), axis=-1)
        cosine_dist = np.abs(cosine_dist)
        point_valid = np.logical_and(cosine_dist > cos_threshold, depth_img < 65535)
        label_valid = semantic_pano > 0
        all_valid = np.logical_and(point_valid, label_valid)
        return points[all_valid], semantic_pano[all_valid]

    def get_all_rooms_by_type(self, room_type):
        room_list_path = os.path.join(g_str.mkdir_automated(self.out_assemble_dir), f'{room_type}_list')
        if os.path.exists(room_list_path):
            with open(room_list_path, 'r') as fp:
                room_list = [f.rstrip() for f in fp.readlines()]
        else:
            room_list = list()
            data_zip_meta = self.load_zips()
            scene_list = [c.split('/')[1] for c in data_zip_meta.namelist() if 'annotation_3d.json' in c]
            for scene_id in scene_list:
                scene_anno = self.load_scene_anno_from_zip(data_zip_meta, scene_id)
                room_ids = scene_anno.get_rooms_by_type(room_type)
                room_list.extend([f'{scene_id}/2D_rendering/{r_i}' for r_i in room_ids])
            with open(room_list_path, 'w') as fp:
                fp.writelines('\n'.join(room_list))
            data_zip_meta.close()
        return room_list

    def load_camera_and_image(self, zip_meta, cam_path):
        # load camera_pose
        camera_para = io.BytesIO(zip_meta.read(cam_path)).read().decode('utf-8').split(' ')
        assert np.all(np.array(camera_para[-3:-1]) == np.array(['0.698132', '0.440992']))
        camera_para = np.asarray([float(i_) for i_ in camera_para], np.float32)
        cam_r = S3DUtilize.get_rotation_matrix_from_tu(camera_para[3:6], camera_para[6:9])
        cam_r = np.matmul(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), cam_r)
        cam_t = np.matmul(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), camera_para[:3] / 1000)

        # load depth image
        depth_path = cam_path.replace('camera_pose.txt', 'depth.png')
        depth_img_data = np.frombuffer(io.BytesIO(zip_meta.read(depth_path)).read(), np.uint8)
        depth_img = cv2.imdecode(depth_img_data, cv2.IMREAD_UNCHANGED)
        depth_img[depth_img == 0] = 65535

        # load semantic image
        semantic_path = cam_path.replace('camera_pose.txt', 'semantic.png')
        semantic_img_data = np.frombuffer(io.BytesIO(zip_meta.read(semantic_path)).read(), np.uint8)
        semantic_img = cv2.imdecode(semantic_img_data, cv2.IMREAD_UNCHANGED)[..., ::-1]
        semantic_id_img = np.zeros(semantic_img.shape[:2], dtype=np.uint8)
        for l_id in self.select_nyu_label_id:
            color = np.asarray(g_data.NYU40().color_map(l_id), dtype=np.uint8)
            semantic_id_img[np.all(semantic_img == color, axis=-1)] = l_id
        label_img = np.take(self.category_mapping, semantic_id_img)
        return cam_r, cam_t, depth_img, label_img

    @staticmethod
    def filter_far_view(cam_t, cam_r):
        cam_r_rotvec = R.from_dcm(cam_r).as_rotvec()
        if cam_t[0] <= 0.5 and cam_t[2] <= 0.5:
            return -np.pi / 2 < cam_r_rotvec[1] < 0
        elif cam_t[0] > 0.5 and cam_t[2] <= 0.5:
            return -np.pi < cam_r_rotvec[1] < -np.pi / 2
        elif cam_t[0] > 0.5 and cam_t[2] > 0.5:
            return np.pi / 2 < cam_r_rotvec[1] < np.pi
        elif cam_t[0] <= 0.5 and cam_t[2] > 0.5:
            return 0 < cam_r_rotvec[1] < np.pi / 2
        else:
            raise NotImplementedError

    def visualize_image(self, depth_image, label_image, vis_path):
        depth_img_vis = (np.where(depth_image < 65535, depth_image, 0) / 1000 / 6.4 * 255).astype(np.uint8)
        cv2.imwrite(vis_path + '_depth.png', depth_img_vis)
        label_image_vis = self.color_map[label_image][..., ::-1]
        cv2.imwrite(vis_path + '_category.jpg', label_image_vis)

    def visualize_surface_voxel(self, surface_voxel, cam_t_list, vis_path):
        voxel_room_cam = surface_voxel.copy()
        cam_t_vox_list = cam_t_list / self.room_stride
        voxel_room_cam[tuple(np.split(cam_t_vox_list.astype(np.uint8), 3, axis=-1))] = len(self.color_map)
        color_map = np.concatenate([self.color_map, [[0, 0, 0]]])
        g_io.PlyIO().dump_vox(vis_path, voxel_room_cam, vox_scale=self.room_stride, colors_map=color_map)

    def single_thread_perspective_vol(self, room_list, room_type, _, worker_id):
        vis_dir = g_str.mkdir_automated(os.path.join(self.out_dir, f'vis_{room_type}'))
        tmp_dir = g_str.mkdir_automated(os.path.join(self.out_dir, f'tmp_{room_type}'))
        assemble_zip_path = os.path.join(tmp_dir, f'assemble_worker{worker_id}.zip')
        if os.path.exists(assemble_zip_path):
            logging.info(f'Skip {assemble_zip_path} generation')
            return

        train_samples = list()
        vol_zip = g_io.ZipIO(assemble_zip_path, 'w')
        zip_meta = self.load_zips()
        for r_i, room_path in enumerate(room_list):
            if r_i % 100 == 0:
                logging.info(f'{worker_id} {r_i}th/{len(room_list)}')
            cam_path_list = [c for c in zip_meta.namelist() if room_path in c and 'camera_pose.txt' in c]
            if len(cam_path_list) == 0:
                continue

            scene_id, _, room_id = room_path.split('/')
            room_box = self.get_room_box_from_zip(zip_meta, scene_id, room_id)
            if np.any(room_box.box_size() > np.asarray(self.room_size)):
                continue

            room_samples, point_list, label_list, cam_t_list, cam_r_list = list(), list(), list(), list(), list()
            for cam_path in cam_path_list:
                _, scene_id, _, room_id, _, _, view_id, _ = cam_path.split('/')
                room_view_id = '%s-room_%s-view_%03d' % (scene_id, room_id, int(view_id))

                cam_r, cam_t, depth_img, label_img = self.load_camera_and_image(zip_meta, cam_path)
                r_points, r_labels = self.assemble_semantic_points_from_img(depth_img, label_img)
                if len(r_points) == 0:
                    continue
                r_points = np.matmul(r_points / 1000, cam_r.T).astype(np.float32) + cam_t

                # remove wrong label
                if 'sofa' in self.label_list and self.label_list.index('sofa') in r_labels:
                    p_valid = np.logical_or(r_points[..., 1] < 2.0, r_labels != self.label_list.index('sofa'))
                    if not np.all(p_valid):
                        r_points, r_labels = r_points[p_valid], r_labels[p_valid]
                    if len(r_points) == 0:
                        continue

                [arr.append(d) for arr, d in zip([room_samples, point_list, label_list, cam_t_list, cam_r_list],
                                                 [room_view_id, r_points, r_labels, cam_t, cam_r])]
                if r_i < 20 and worker_id == 0:
                    self.visualize_image(depth_img, label_img, os.path.join(vis_dir, f'{room_view_id}'))
            if len(point_list) == 0:
                continue

            # remove outside points
            point_list, label_list = np.concatenate(point_list), np.concatenate(label_list)
            p_valid = np.logical_and(np.all(point_list < room_box.max, -1), np.all(point_list > room_box.min, -1))
            point_list, label_list = point_list[p_valid], label_list[p_valid]
            if len(point_list) == 0:
                continue
            point_floor_center = room_box.center_floor()
            point_list = point_list - point_floor_center + np.asarray(self.room_center)

            surface_voxel = S3DUtilize.cast_points_to_voxel(point_list, label_list, self.room_size, self.room_stride)
            if len(np.unique(surface_voxel)) - 1 == 0:
                continue

            # camera
            cam_t_list, cam_r_list = np.asarray(cam_t_list), np.asarray(cam_r_list)
            cam_t_list = cam_t_list - point_floor_center + np.asarray(self.room_center)
            valid_c = np.logical_and(np.all(cam_t_list < self.room_size, axis=1), np.all(cam_t_list >= 0, axis=1))
            cam_t_list, cam_r_list = cam_t_list[valid_c], cam_r_list[valid_c]
            if len(cam_t_list) == 0:
                continue
            room_samples = np.asarray(room_samples)[valid_c].tolist()
            cam_t_norm_list = cam_t_list / self.room_size[0]

            npz_meta = io.BytesIO()
            np.savez_compressed(npz_meta, camera_id=room_samples, label=surface_voxel, room_size=room_box.box_size(),
                                cam_t=cam_t_norm_list, cam_r=cam_r_list)
            npz_meta.seek(0)
            room_out_name = '%s-room_%s' % (scene_id, room_id)
            vol_zip.writestr(f'{room_out_name}.npz', npz_meta.read())
            train_samples.extend(room_samples)

            if r_i < 20 and worker_id == 0:
                vis_path = os.path.join(vis_dir, room_out_name + '.ply')
                self.visualize_surface_voxel(surface_voxel, cam_t_list, vis_path)
        vol_zip.close()
        zip_meta.close()

    def assemble_training_sample(self, process_pipeline: ProcessPipeline, far_view=True, assemble_name='train_view',
                                 num_workers=8):
        for room_type in process_pipeline.room_types:
            room_list = self.get_all_rooms_by_type(room_type)
            logging.info(f'Assemble {len(room_list)} {room_type} from {self.data_dir} to {self.out_dir}')
            g_perf.multiple_processor(self.single_thread_perspective_vol, room_list, num_workers, (room_type,))

            tmp_zip_meta = g_io.GroupZipIO(os.path.join(self.out_dir, f'tmp_{room_type}'))
            target_files = [f for f in tmp_zip_meta.namelist() if f.endswith('npz')]

            out_path = os.path.join(g_str.mkdir_automated(self.out_assemble_dir), f'{assemble_name}_{room_type}')
            _, assemble_zip, assemble_record = self.get_assemble_writer(out_path)

            train_samples = list()
            for t_i, t_f in enumerate(target_files):
                sample_data = np.load(io.BytesIO(tmp_zip_meta.read(t_f)))
                vol_data, room_size = sample_data['label'], sample_data['room_size']
                cam_t, cam_r, camera_id = sample_data['cam_t'], sample_data['cam_r'], sample_data['camera_id']
                if far_view:
                    cam_t, cam_r, camera_id = self.filter_far_views(cam_t, cam_r, camera_id)
                if len(cam_t) == 0:
                    continue

                feature = {
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[vol_data.tobytes()])),
                    'cam_t': tf.train.Feature(bytes_list=tf.train.BytesList(value=[cam_t.tobytes()])),
                    'cam_r': tf.train.Feature(bytes_list=tf.train.BytesList(value=[cam_r.tobytes()]))
                }
                assemble_record.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

                npz_meta = io.BytesIO()
                np.savez_compressed(npz_meta, camera_id=camera_id, label=vol_data, room_size=room_size,
                                    cam_t=cam_t, cam_r=cam_r)
                npz_meta.seek(0)
                assemble_zip.writestr(t_f, npz_meta.read())
                train_samples.append(t_f[:-4])
            with open(out_path + '_samples', 'w') as fp:
                fp.writelines([f'{t_s}\n' for t_s in train_samples])
            assemble_zip.close()
            assemble_record.close()

    def mediate_process(self):
        process_pipelines = self.process_pipelines
        for p_p in process_pipelines:
            self.__init__(self.data_dir, os.path.join(self.out_dir, p_p.label_type), self.process_pipelines, p_p)
            self.assemble_training_sample(p_p, far_view=True, assemble_name=p_p.assemble_name, num_workers=8)
