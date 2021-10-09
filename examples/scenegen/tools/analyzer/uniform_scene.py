# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import logging
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .visualization_utilize import VisualizationUtilize
from ..data.base import UniformScene
from .. import data as g_data
from GraphicsDL.graphicsutils import g_io, g_str


class AnalysisUtilize(object):
    @staticmethod
    def path_specifier(tar_path_list, force_refresh):
        tar_path = os.path.join(*tar_path_list)
        csv_path = '.'.join((tar_path, 'csv'))
        png_path = '.'.join((tar_path, 'png'))
        csv_path = None if not force_refresh and os.path.exists(csv_path) else csv_path
        png_path = None if not force_refresh and os.path.exists(png_path) else png_path
        return csv_path, png_path


class UniSceneAnalyzer(object):
    def __init__(self, data_source, scenes_path, objects_path, output_dir, data_cfg, samples_list=None):
        self.data_source = data_source
        self.scenes_path = scenes_path
        self.objects_path = objects_path
        self.output_dir = output_dir
        self.eval_meta_path = '_'.join(scenes_path.split('_')[:-3]) + '.npz'
        self.eval_bbox = None

        self.cfg = data_cfg
        room_size = np.insert(self.cfg.room_size, 1, self.cfg.room_height)
        self.room_size, self.room_stride = np.array(room_size), self.cfg.room_stride
        self.room_center = self.room_size * [0.5, 0, 0.5]
        self.vox_size = (self.room_size / self.room_stride).astype(np.int32)

        self.samples_list = samples_list
        self.sample_num = 0

        self.label_type = self.cfg.label_type
        label_type = getattr(g_data, self.label_type.replace('-', '').upper())()
        self.label_list, self.short_label_list = label_type.label_id_map_arr(), label_type.label_id_map_arr(True)
        self.num_categories = len(self.label_list)
        self.color_map = np.concatenate([label_type.color_map_arr(), [[0, 0, 0]]], axis=0)
        self.ignored_label = ['void']

        self.scenes_list = []

        self.wall_list = [9, 10, 6, 5]
        self.camera_fov = [8, 6]
        self.render_resolution = [640, 480]
        room_half_size = self.room_size[0] / 2
        cam_height = 1.2 / 3.2 * room_half_size
        self.camera_matrix_list = VisualizationUtilize.get_render_camera_parameter(room_size=room_half_size,
                                                                                   cam_height=cam_height)

    def load_data(self):
        if len(self.scenes_list) != 0:
            return

        logging.info(f'\t\tData loading')
        scenes_zip = g_io.ZipIO(self.scenes_path)
        self.samples_list = scenes_zip.namelist()[:10240]
        self.sample_num = len(self.samples_list)
        for s_name in self.samples_list:
            scene_info = UniformScene()
            scene_info.load(json.loads(scenes_zip.read(s_name)))
            self.scenes_list.append(scene_info)
        logging.info(f'\t\t{len(self.scenes_list)} Data loaded')
        if self.label_type != self.scenes_list[0].label_type:
            logging.warning(f'\t\tLabel type: Uniform scene - {self.scenes_list[0].label_type}, '
                            f'analyzer - {self.label_type}')
        scenes_zip.close()

    def co_occupied(self, num_cls, cor_length=None) -> np.ndarray:
        cor_length = cor_length if cor_length else num_cls
        acc_list = np.zeros([cor_length], dtype=np.int32)
        cor_map = np.zeros([cor_length, cor_length], dtype=np.int32)
        for scene in self.scenes_list:
            scene_label_ids = np.unique([obj.label_id for obj in scene.objects if obj.label not in self.ignored_label])
            if len(scene_label_ids) == 0:
                continue
            cart_indices = np.transpose(np.asarray([x for x in itertools.product(scene_label_ids, scene_label_ids)]))
            cor_map[tuple(cart_indices)] = cor_map[tuple(cart_indices)] + 1
            acc_list[scene_label_ids] = acc_list[scene_label_ids] + 1
        cor_map = cor_map / np.maximum(np.expand_dims(acc_list, axis=-1), 1)
        return cor_map

    def obj_number_bincount(self, num_cls: int) -> np.ndarray:
        bincount_list = list()
        for scene in self.scenes_list:
            bincount = np.bincount([obj.label_id for obj in scene.objects if obj.label not in self.ignored_label],
                                   minlength=num_cls)
            bincount_list.append(bincount)
        return np.asarray(bincount_list, dtype=np.int64)

    def vis_3d_analysis(self, force_refresh=False, vis_num=100, stride=1):
        self.load_data()
        scene_index_list = np.arange(vis_num) * stride
        vis_dir = g_str.mkdir_automated(os.path.join(self.output_dir, f'vis_3d'))
        objects_zip = g_io.ZipIO(self.objects_path)
        for s_i, scene_index in enumerate(scene_index_list):
            if s_i >= vis_num:
                break
            scene = self.scenes_list[scene_index]
            vis_path = os.path.join(vis_dir, scene.scene_id)
            vox_vis_path = f'{vis_path}_view'
            if not force_refresh and os.path.exists(vox_vis_path + '0.png') and os.path.exists(vox_vis_path + '1.png') \
                    and os.path.exists(vox_vis_path + '2.png') and os.path.exists(vox_vis_path + '3.png'):
                continue

            objs_points, points_label = scene.parse_scene(objects_zip)
            voxel_size = (np.array(self.room_size) / self.room_stride).astype(np.int32)
            voxel_room = np.zeros(voxel_size, dtype=np.uint8)
            if len(objs_points) > 0:
                objs_points = np.concatenate(objs_points, axis=0)
                points_label = np.concatenate(points_label, axis=0)
                obj_vox_indices = (objs_points / self.room_stride + 0.1).astype(np.int32)

                point_xyz_max = (np.array(self.room_size) / self.room_stride).astype(np.int32)
                valid_indices = np.logical_and(np.all(obj_vox_indices >= 0, axis=1),
                                               np.all(obj_vox_indices < point_xyz_max, axis=1))
                obj_vox_indices, points_label = obj_vox_indices[valid_indices], points_label[valid_indices]
                voxel_room[tuple(np.split(obj_vox_indices, 3, axis=-1))] = np.expand_dims(points_label, axis=-1)

            VisualizationUtilize.vol_top_view_visualization(vis_path, voxel_room, colors_map=self.color_map)
            g_io.PlyIO().dump_vox(vis_path + '.ply', voxel_room, vox_scale=self.room_stride, colors_map=self.color_map)
            VisualizationUtilize.scene_vox_visualization(vis_path + '.ply', vis_path, self.camera_matrix_list,
                                                         self.camera_fov, self.render_resolution)

    def correlation_analysis(self, force_refresh=False):
        csv_path, png_path = AnalysisUtilize.path_specifier((self.output_dir, 'correlation'), force_refresh)
        if csv_path is None:
            return

        self.load_data()
        cor_map = self.co_occupied(self.num_categories)
        cor_df = pd.DataFrame(cor_map, self.short_label_list, self.short_label_list)
        cor_df = cor_df.drop(columns=self.ignored_label, index=self.ignored_label)
        cor_df.to_csv(csv_path)

        heat_map = sns.heatmap(cor_df, vmin=0., vmax=1., cmap='jet')
        heat_map.figure.savefig(png_path, bbox_inches='tight')
        plt.close('all')

    def object_distribution_analysis(self, force_refresh=False, obj_num_max=20):
        csv_path, png_path = AnalysisUtilize.path_specifier((self.output_dir, 'object_distribution'), force_refresh)
        if csv_path is None:
            return

        self.load_data()
        scene_obj_num = np.sum(self.obj_number_bincount(self.num_categories), axis=-1, keepdims=False)
        scene_obj_num_bc = np.bincount(scene_obj_num, minlength=obj_num_max + 1)
        scene_obj_num_bc[obj_num_max] = np.sum(scene_obj_num_bc[obj_num_max:])
        scene_obj_num_bc = scene_obj_num_bc[:obj_num_max + 1]

        scene_obj_num_df = pd.DataFrame(scene_obj_num_bc).reset_index()
        scene_obj_num_df['source'] = self.data_source
        scene_obj_num_df.columns = ['object_number', 'counts', 'source']
        scene_obj_num_df['object_number'] = list(np.arange(obj_num_max).astype(str)) + [f'>={obj_num_max}']
        scene_obj_num_df.to_csv(csv_path)

        dist_plot = sns.barplot(scene_obj_num_df['object_number'], scene_obj_num_df['counts'], color=(0.2, 0.4, 0.6))
        dist_plot.figure.savefig(png_path)
        plt.close('all')

    def category_distribution_analysis(self, force_refresh=False):
        csv_path, png_path = AnalysisUtilize.path_specifier((self.output_dir, 'category_distribution'), force_refresh)
        if csv_path is None:
            return

        self.load_data()
        category_num = np.sum(self.obj_number_bincount(self.num_categories), axis=0, keepdims=False)
        category_num_df = pd.DataFrame(category_num, index=self.short_label_list)
        category_num_df = category_num_df.drop(index=self.ignored_label).reset_index()
        category_num_df['source'] = self.data_source
        category_num_df.columns = ['categories', 'counts', 'source']
        category_num_df.to_csv(csv_path)

        bar_plot = sns.barplot(category_num_df['categories'], category_num_df['counts'], color=(0.2, 0.4, 0.6))
        bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45)
        bar_plot.figure.savefig(png_path)
        plt.close('all')
