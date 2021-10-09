# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import io
import wget
import logging
import numpy as np
import pandas as pd
from plyfile import PlyData

from .structured_3d import Structured3DDataGen
from GraphicsDL.graphicsutils import g_str, g_io, g_seg


class Matterport3DDataGen(Structured3DDataGen):
    def __init__(self, data_dir, out_dir, process_pipelines, cfg=None, **kargs):
        super().__init__(data_dir, out_dir, process_pipelines, cfg=cfg, **kargs)
        self.scans_dir = os.path.join(self.data_dir, 'v1', 'scans')

        self.cam_num, self.yaw_num = 3, 6
        self.image_list = [f'{c_i}_{y_i}' for c_i in range(self.cam_num) for y_i in range(self.yaw_num)]

        self.region_type_map = {'a': 'bathroom', 'b': 'bedroom', 'c': 'closet', 'd': 'dining room',
                                'e': 'entryway/foyer/lobby', 'f': 'familyroom', 'g': 'garage', 'h': 'hallway',
                                'i': 'library', 'j': 'laundryroom/mudroom', 'k': 'kitchen', 'l': 'living room',
                                'm': 'meetingroom/conferenceroom', 'n': 'lounge', 'o': 'office',
                                'p': 'porch/terrace/deck/driveway', 'r': 'rec/game', 's': 'stairs', 't': 'toilet',
                                'u': 'utilityroom/toolroom ', 'v': 'tv', 'w': 'workout/gym/exercise',
                                'x': 'outdoor areas containing grass, plants, bushes, trees, etc.', 'y': 'balcony',
                                'z': 'other room', 'B': 'bar', 'C': 'classroom', 'D': 'dining booth', 'S': 'spa/sauna',
                                'Z': 'junk', '-': 'no label '}

        self.init_config()

    def init_config(self):
        category_mapping_path = os.path.join(g_str.abs_dir_path(__file__), 'resource', 'category_mapping.tsv')
        if not os.path.exists(category_mapping_path):
            mapping_url = 'https://raw.githubusercontent.com/niessner/Matterport/master/metadata/category_mapping.tsv'
            wget.download(mapping_url, category_mapping_path)

        category_mapping_dict = dict()
        category_mapping_csv = pd.read_csv(category_mapping_path, sep='\t')
        for k, v in category_mapping_csv[['index', 'nyu40id']].values:
            category_mapping_dict[int(k)] = int(v) if str(v) != 'nan' else 0

        self.category_mapping = np.zeros(max(map(int, set(category_mapping_dict.keys())))+1, np.uint8)
        for k, v in category_mapping_dict.items():
            self.category_mapping[int(k)] = int(v)

        self.label_mapping = np.zeros(np.max(self.category_mapping) + 1, np.uint8)
        nyu40_mapping = [
            ['void', 'void'], ['wall', 'void'], ['floor', 'void'], ['cabinet', 'void'], ['bed', 'bed'],
            ['chair', 'chair'], ['sofa', 'void'], ['table', 'table'], ['door', 'void'], ['window', 'void'],
            ['bookshelf', 'void'], ['picture', 'picture'], ['counter', 'void'], ['blinds', 'curtain'], ['desk', 'void'],
            ['shelves', 'void'], ['curtain', 'curtain'], ['dresser', 'void'], ['pillow', 'pillow'], ['mirror', 'void'],
            ['floor_mat', 'void'], ['clothes', 'void'], ['ceiling', 'void'], ['books', 'void'],
            ['refridgerator', 'void'], ['television', 'void'], ['paper', 'void'], ['towel', 'void'],
            ['shower_curtain', 'void'], ['box', 'void'], ['whiteboard', 'void'], ['person', 'void'],
            ['night_stand', 'night_stand'], ['toilet', 'void'], ['sink', 'void'], ['lamp', 'lamp'], ['bathtub', 'void'],
            ['bag', 'void'], ['otherstructure', 'void'], ['otherfurniture', 'void'], ['otherprop', 'void']
        ]
        for nyu_i, nyu_map in enumerate(nyu40_mapping):
            self.label_mapping[nyu_i] = self.label_list.index(nyu_map[1])

    def get_region(self, sample, room_type=None):
        house_seg_zip = g_io.ZipIO(os.path.join(self.scans_dir, sample, 'house_segmentations.zip'), 'r')
        seg_path = f'{sample}/house_segmentations'
        house_seg_lines = house_seg_zip.read(f'{seg_path}/{sample}.house').decode().split('\n')
        panorama2region_lines = house_seg_zip.read(f'{seg_path}/panorama_to_region.txt').decode().split('\n')
        house_seg_zip.close()

        region_list = [line_str for line_str in house_seg_lines if len(line_str) > 0 and line_str[0] == 'R']
        region_list = [r_s.split()[1] for r_s in region_list
                       if room_type is None or self.region_type_map[r_s.split()[5]] == room_type]

        region_panorama_list = {region_id: list() for region_id in region_list}
        panorama2region_list = [line_str.split() for line_str in panorama2region_lines if len(line_str) > 0]
        [region_panorama_list[p_l[2]].append(p_l[1]) for p_l in panorama2region_list if p_l[2] in region_panorama_list]
        return region_list, region_panorama_list

    def read_camera_parameters(self, sample):
        camera_zip = g_io.ZipIO(os.path.join(self.scans_dir, sample, 'undistorted_camera_parameters.zip'), 'r')
        camera_lines = camera_zip.read(f'{sample}/undistorted_camera_parameters/{sample}.conf').decode().split('\n')
        camera_split_list = [cam_str.split() for cam_str in camera_lines if len(cam_str.split()) > 0]
        camera_zip.close()

        cam_poses_list = [str_split[3:19] for str_split in camera_split_list if str_split[0] == 'scan']
        cam_poses_list = np.array(cam_poses_list, np.float32).reshape([-1, 4, 4])  # * [1, -1, -1, 1]
        nz2x_m = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        cam_poses_list = np.matmul(cam_poses_list, nz2x_m)
        image_id_list = [str_split[1][:-4] for str_split in camera_split_list if str_split[0] == 'scan']
        cam_pose_dict = {image_id: cam_pose for image_id, cam_pose in zip(image_id_list, cam_poses_list)}

        cam_intrinsics_list = [str_split[1:] for str_split in camera_split_list if str_split[0] == 'intrinsics_matrix']
        cam_intrinsics_list = np.array(cam_intrinsics_list, np.float32).reshape([-1, 3, 3])
        cam_id_list = [image_id[:-2] for s_i, image_id in enumerate(image_id_list) if s_i % self.yaw_num == 0]
        cam_intrinsics_dict = {cam_id: cam_int for cam_id, cam_int in zip(cam_id_list, cam_intrinsics_list)}

        return cam_intrinsics_dict, cam_pose_dict

    def read_region_ply(self, region_zip, scene_id, region_id):
        ply_raw = region_zip.read(f'{scene_id}/region_segmentations/region{region_id}.ply')
        region_ply_data = PlyData.read(io.BytesIO(ply_raw))
        vertices = np.stack((region_ply_data['vertex']['x'], region_ply_data['vertex']['y'],
                             region_ply_data['vertex']['z']), axis=1)
        faces = np.stack(region_ply_data['face']['vertex_indices'], axis=0)
        faces_raw_label = region_ply_data['face']['category_id']
        faces_nyu_label = np.take(self.category_mapping, np.where(faces_raw_label > 0, faces_raw_label, 0))
        vertices_nyu_label = np.zeros((vertices.shape[0]), np.uint8)
        vertices_nyu_label[faces.reshape(-1)] = np.tile(faces_nyu_label[:, np.newaxis], [1, 3]).reshape((-1))

        faces_label = np.take(self.label_mapping, faces_nyu_label)
        vertices_label = np.zeros((vertices.shape[0]), np.uint8)
        vertices_label[faces.reshape(-1)] = np.tile(faces_label[:, np.newaxis], [1, 3]).reshape((-1))

        if self.label_list.index('bed') not in np.unique(faces_label):
            vertices_label = None

        return vertices, vertices_label

    def repair_voxel(self, vox):
        if 'mirror' not in self.label_list:
            return vox
        repair_label_id = [self.label_list.index(c_n) for c_n in ['mirror']]  # 'window', 'door'
        instances, smnt_list = g_seg.instance_segmentation_from_semantic(vox)
        for ins_num, label_id in enumerate(smnt_list[1:], start=1):
            if label_id not in repair_label_id:
                continue
            instance_indices = np.argwhere(instances == ins_num)
            instance_min, instance_max = np.min(instance_indices, axis=0), np.max(instance_indices, axis=0)
            instance_range = [np.arange(instance_min[a_i], instance_max[a_i] + 1) for a_i in range(3)]

            instance_count_z = np.sum(instances == ins_num, axis=(0, 1))
            instance_count_x = np.sum(instances == ins_num, axis=(1, 2))
            if np.max(instance_count_x) > np.max(instance_count_z):
                instance_x_max = np.argmax(instance_count_x)
                instance_range[0] = np.arange(instance_x_max, instance_x_max + 1)
            else:
                instance_z_max = np.argmax(instance_count_z)
                instance_range[2] = np.arange(instance_z_max, instance_z_max + 1)
            filling_indices = np.stack(np.meshgrid(*instance_range, indexing='ij'), axis=-1)
            vox[tuple(np.split(filling_indices, 3, axis=-1))] = label_id
        return vox

    def single_thread_perspective_vol(self, samples_list, room_type, _, worker_id):
        tmp_dir = g_str.mkdir_automated(os.path.join(self.out_dir, f'tmp_{room_type}'))
        assemble_zip_path = os.path.join(tmp_dir, f'assemble_worker{worker_id}.zip')
        if os.path.exists(assemble_zip_path):
            logging.info(f'Skip {assemble_zip_path} generation')
            return
        zip_writer = g_io.ZipIO(assemble_zip_path, 'w')
        for s_i, sample in enumerate(samples_list):
            if s_i % 10 == 0:
                logging.info(f'{worker_id}: {s_i}th/{len(samples_list)}')

            cam_intrinsics_dict, cam_pose_dict = self.read_camera_parameters(sample)
            region_list, region_panorama_list = self.get_region(sample, room_type)
            region_seg_zip = g_io.ZipIO(os.path.join(self.scans_dir, sample, 'region_segmentations.zip'), 'r')
            for r_i, region_id in enumerate(region_list):
                pano_list = np.array(region_panorama_list[region_id])
                if len(pano_list) == 0:
                    continue
                vertices, vertices_vox_label = self.read_region_ply(region_seg_zip, sample, region_id)
                if vertices_vox_label is None:
                    continue

                vertices_vox = np.matmul(vertices, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T)
                vertices_vox = vertices_vox[vertices_vox_label != 0]
                vertices_vox_label = vertices_vox_label[vertices_vox_label != 0]

                room_size = np.max(vertices_vox, axis=0) - np.min(vertices_vox, axis=0)
                if np.any(room_size > np.array(self.room_size)):
                    continue
                center_floor = (np.max(vertices_vox, axis=0) + np.min(vertices_vox, axis=0)) / 2
                center_floor[1] = np.min(vertices_vox, axis=0)[1]
                vertices_vox = ((vertices_vox - center_floor + self.room_center) / self.room_stride).astype(np.int32)
                if np.sum(np.logical_or(np.any(vertices_vox >= self.vox_size, 1), np.any(vertices_vox < 0, 1))) > 0:
                    indices_valid = np.logical_and(np.all(vertices_vox < self.vox_size, axis=1),
                                                   np.all(vertices_vox >= 0, axis=1))
                    vertices_vox, vertices_vox_label = vertices_vox[indices_valid], vertices_vox_label[indices_valid]

                voxel_room = np.zeros(self.vox_size, dtype=np.uint8)
                voxel_room[tuple(np.split(vertices_vox, 3, axis=-1))] = vertices_vox_label[:, np.newaxis]
                voxel_room = self.repair_voxel(voxel_room)

                pano_cam_list = [[cam_pose_dict[f'{p_i}_d{img_i}'] for img_i in self.image_list] for p_i in pano_list]
                pano_cam_list = np.matmul(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
                                          np.stack(pano_cam_list, axis=0))
                cam_t_translation = pano_cam_list[:, :, :3, 3] - center_floor + self.room_center
                cam_t = cam_t_translation / self.room_stride
                valid_c = np.logical_and(np.all(cam_t < self.vox_size, axis=(1, 2)), np.all(cam_t >= 0, axis=(1, 2)))
                pano_list, cam_t, cam_r = pano_list[valid_c], cam_t[valid_c], pano_cam_list[:, :, :3, :3][valid_c]
                pano_id_list = [f'{sample}_{region_id}_{p_i}_{img}' for p_i in pano_list for img in self.image_list]

                npz_meta = io.BytesIO()
                np.savez_compressed(npz_meta, camera_id=pano_id_list, label=voxel_room, room_size=room_size,
                                    cam_t=cam_t / self.vox_size[0], cam_r=cam_r)
                npz_meta.seek(0)
                zip_writer.writestr(f'{sample}_{region_id}.npz', npz_meta.read())

            region_seg_zip.close()
        zip_writer.close()

    def get_all_rooms_by_type(self, room_type):
        scenes_list = [scene for scene in os.listdir(self.scans_dir)]
        return scenes_list

    @staticmethod
    def filter_far_view(cam_t, cam_r):
        cam_direction = np.matmul(cam_r, np.array([1, 0, 0]).reshape([3, 1]))[..., 0]
        if cam_t[0] <= 0.5 and cam_t[2] <= 0.5:
            return cam_direction[0] > 0 and cam_direction[2] > 0
        elif cam_t[0] > 0.5 and cam_t[2] <= 0.5:
            return cam_direction[0] < 0 and cam_direction[2] > 0
        elif cam_t[0] > 0.5 and cam_t[2] > 0.5:
            return cam_direction[0] < 0 and cam_direction[2] < 0
        elif cam_t[0] <= 0.5 and cam_t[2] > 0.5:
            return cam_direction[0] > 0 and cam_direction[2] < 0
        else:
            raise NotImplementedError

    def filter_far_views(self, cam_t_all, cam_r_all, camera_id_all=None):
        cam_t_m, cam_r_m = cam_t_all[:, 6:12].reshape([-1, 3]), cam_r_all[:, 6:12].reshape([-1, 3, 3])
        cam_t_local = cam_t_m[:, 1] * self.room_size[0]
        valid_c = np.logical_and(1.3 < cam_t_local, cam_t_local < 1.7)
        cam_t, cam_r = cam_t_m[valid_c], cam_r_m[valid_c]
        camera_id = np.reshape(camera_id_all, [-1, 18])[:, 6:12].reshape(-1)[valid_c]
        return super().filter_far_views(cam_t, cam_r, camera_id)
