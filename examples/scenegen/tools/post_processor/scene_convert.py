# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import re
import io
import os
import json
import shutil
import logging
import numpy as np
import pandas as pd

from .. import data as g_data
from ..data.base import get_cfg_from_pipeline, GroupDataGenConfig, UniformScene, UniformObject, AxisAlignBoundingBox
from GraphicsDL.graphicsutils import g_io, g_str, g_perf, g_seg


class Vol2UniSceneConverter(object):
    def __init__(self, vol_path, zip_out_dir, cfg_path, assemble_name=None) -> None:
        super().__init__()
        self.vol_path = vol_path
        self.zip_out_dir = zip_out_dir
        self.zip_out_tmp_dir = g_str.mkdir_automated(os.path.join(self.zip_out_dir, 'uniform_scene_generation_tmp'))

        self.sample_list = None
        self.vol_data, self.vol_bbox = None, None

        self.cfg = get_cfg_from_pipeline(GroupDataGenConfig, cfg_path, assemble_name)
        self.label_type = getattr(g_data, self.cfg.label_type.replace('-', '').upper())()
        self.label_list = self.label_type.label_id_map_arr()

    def load_vol_data(self):
        if self.vol_path.endswith('npz'):
            eval_data, eval_bbox_data = self.load_eval_data(self.vol_path)
            sample_list = list(range(len(eval_data)))
            return eval_data, eval_bbox_data * self.cfg.room_stride, sample_list
        else:
            raise NotImplementedError

    @staticmethod
    def rename_file_in_zip(src_zip_path, target_zip_path):
        src_zip = g_io.ZipIO(src_zip_path)
        target_zip = g_io.ZipIO(target_zip_path, 'w')
        json_file_list = [f[:-5] for f in src_zip.namelist() if f.endswith('json')]
        for j_i, json_file in enumerate(json_file_list):
            src_data = src_zip.read(json_file+'.json')
            target_zip.writestr(json_file+'.npz', src_data)
        target_zip.close()
        src_zip.close()

    @staticmethod
    def load_eval_data(vol_path, auto_squeeze=True):
        eval_meta: np.lib.npyio.NpzFile = np.load(vol_path)
        eval_z, eval_data, eval_bbox_data = list(), list(), list()
        for e_k in eval_meta.keys():
            if 'Z' in e_k:
                e_z = np.squeeze(eval_meta[e_k]) if auto_squeeze else eval_meta[e_k]
                eval_z.append(e_z)
            elif 'Bbox' in e_k:
                scene_bbox = np.squeeze(eval_meta[e_k]) if auto_squeeze else eval_meta[e_k]
                eval_bbox_data.append(scene_bbox)
            else:
                e_d = np.squeeze(eval_meta[e_k]) if auto_squeeze else eval_meta[e_k]
                eval_data.append(e_d)

        assert len(eval_data) == len(eval_bbox_data) == 1, f'eval_meta: {[[k, v.shape] for k, v in eval_meta.items()]}'
        assert len(eval_data[0]) == len(eval_bbox_data[0]), f'data{eval_data[0].shape}, bbox{eval_bbox_data[0].shape}'
        for vox, bbox in zip(eval_data[0], eval_bbox_data[0]):
            bbox_max, bbox_min = np.ceil(bbox[:3]).astype(np.int32), np.floor(bbox[3:6]).astype(np.int32)
            vox[bbox_max[0]:] = 0
            vox[:, bbox_max[1]:] = 0
            vox[:, :, bbox_max[2]:] = 0
            vox[:bbox_min[0]] = 0
            vox[:, :bbox_min[1]] = 0
            vox[:, :, :bbox_min[2]] = 0
        return eval_data[0], eval_bbox_data[0]

    @staticmethod
    def denoise_volume_data(data, denoise_threshold, category_list, vox_stride):
        vol_denoised = np.zeros_like(data, dtype=np.uint8)
        vol_noise = np.zeros_like(data, dtype=np.uint8)
        if np.all(data == 0):
            return vol_denoised, vol_noise
        if data.ndim < 3:
            return data, vol_noise

        s_i, s_l = g_seg.instance_segmentation_from_semantic(data)
        for ins_num, label_id in enumerate(s_l[1:], start=1):
            ins_indices = np.argwhere(s_i == ins_num)
            ins_max = np.max(ins_indices, axis=0)
            ins_min = np.min(ins_indices, axis=0)
            ins_box_size = (ins_max - ins_min + 1) * vox_stride
            ins_size = np.around(np.prod(ins_box_size), decimals=10)

            label_denoise_threshold = denoise_threshold.get(category_list[label_id])
            if label_denoise_threshold is None:
                raise NotImplementedError

            if ins_size > label_denoise_threshold['V']['min']:
                vol_denoised[tuple(ins_indices.T)] = label_id
            else:
                vol_noise[tuple(ins_indices.T)] = label_id
        return vol_denoised, vol_noise

    def get_objects_from_vol(self, label_map, cell_stride, scene_name, obj_zip_writer=None):
        obj_info_list = list()
        instances, smnt_list = g_seg.instance_segmentation_from_semantic(label_map)
        for ins_num, label_id in enumerate(smnt_list[1:], start=1):
            ins_points = np.argwhere(instances == ins_num) * cell_stride
            if ins_points.shape[-1] < 3:
                ins_points = np.insert(ins_points, 1, 0, axis=-1)

            obj = UniformObject()
            obj.scale = 1
            obj.interpolation = 1
            obj.bbox = AxisAlignBoundingBox()
            obj.bbox.max = (np.max(ins_points, axis=0) + cell_stride).tolist()
            obj.bbox.min = np.min(ins_points, axis=0).tolist()
            obj.model_id = 'scene%s_object%06d' % (scene_name, ins_num)
            obj.label_id = int(label_id)
            obj.label = self.label_list[label_id]
            obj.world_pose = np.eye(4, dtype=np.float32)
            obj.world_pose[3, :3] = obj.bbox.center()
            obj.world_pose = obj.world_pose.tolist()
            obj_info_list.append(obj)

            if obj_zip_writer is not None:
                npz_file = f'{obj.model_id}.npz'
                obj_points = ins_points - obj.bbox.center()
                npz_io = io.BytesIO()
                np.savez_compressed(npz_io, obj_points)
                npz_io.seek(0)
                obj_zip_writer.writestr(npz_file, npz_io.read())

        return obj_info_list

    def dump_uniform_scene_from_vol(self, scenes_writer, objs_writer, scene_data, scene_bbox, scene_name):
        uniform_scene = UniformScene()
        uniform_scene.scene_id = scene_name
        uniform_scene.room_type = '-'.join(self.cfg.room_types)
        uniform_scene.label_type = self.cfg.label_type
        uniform_scene.bbox = AxisAlignBoundingBox()
        uniform_scene.bbox.assign_box_size(scene_bbox[:3], scene_bbox[3:6])
        uniform_scene.objects = self.get_objects_from_vol(scene_data, self.cfg.room_stride, scene_name, objs_writer)
        uniform_scene_dump = uniform_scene.dump_to_json()
        scenes_writer.writestr('%s.json' % scene_name, uniform_scene_dump)

    @staticmethod
    def remove_dir(rm_dir):
        try:
            shutil.rmtree(rm_dir)
        except PermissionError as E:
            logging.warning(f'Fail to delete {rm_dir}: {str(E)}')

    @staticmethod
    def merger_zip_file(target_path, source_dir, filter_regex='', source_rm=False):
        zip_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if re.match(filter_regex, f)]
        if len(zip_files) == 1:
            shutil.move(zip_files[0], target_path)
        elif len(zip_files) > 1:
            zip_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if re.match(filter_regex, f)]
            zip_group = g_io.GroupZipIO(zip_files)
            out_zip_meta = g_io.ZipIO(target_path, 'w')
            for f in zip_group.namelist():
                out_zip_meta.writestr(f, zip_group.read(f))
            out_zip_meta.close()
            zip_group.close()
        else:
            logging.warning(f'Fail to find {source_dir}')
            raise NotImplementedError
        if source_rm:
            Vol2UniSceneConverter.remove_dir(source_dir)

    def post_process_single_thread(self, sample_list, denoise, _, worker_id):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource', 'denoise.json'), 'r') as f:
            threshold = json.load(f)

        scene_writer_zip = g_io.ZipIO(os.path.join(self.zip_out_tmp_dir, f'uni_rep_work{worker_id}.zip'), mode='w')
        obj_writer_zip = g_io.ZipIO(os.path.join(self.zip_out_tmp_dir, f'obj_rep_work{worker_id}.zip'), mode='w')
        for s_i, sample_index in enumerate(sample_list):
            scene_name = '%06d' % sample_index
            scene_vol, scene_bbox = self.vol_data[sample_index], self.vol_bbox[sample_index]
            vol_dump = scene_vol.copy()
            if denoise:
                vol_dump, _ = self.denoise_volume_data(scene_vol, threshold, self.label_list, self.cfg.room_stride)
            self.dump_uniform_scene_from_vol(scene_writer_zip, obj_writer_zip, vol_dump, scene_bbox, scene_name)
        scene_writer_zip.close()
        obj_writer_zip.close()

    def mediate_process(self, denoise=True, num_workers=1):
        out_file_name = os.path.splitext(os.path.basename(self.vol_path))[0]
        scene_out_path = os.path.join(self.zip_out_dir, f'{out_file_name}_uni_rep.zip')
        obj_out_path = os.path.join(self.zip_out_dir, f'{out_file_name}_obj_rep.zip')
        if os.path.exists(scene_out_path) and os.path.exists(obj_out_path):
            return

        logging.info(f'\tStart processing generated data {self.vol_path}')
        self.vol_data, self.vol_bbox, self.sample_list = self.load_vol_data()
        g_perf.multiple_processor(self.post_process_single_thread, self.sample_list, num_workers, (denoise,))

        self.merger_zip_file(scene_out_path, self.zip_out_tmp_dir, filter_regex=r'uni_rep_work[0-9]*\.zip')
        self.merger_zip_file(obj_out_path, self.zip_out_tmp_dir, filter_regex=r'obj_rep_work[0-9]*\.zip')

        self.remove_dir(self.zip_out_tmp_dir)
        logging.info(f'\tGenerated data has been transformed into uniform scene representation')
