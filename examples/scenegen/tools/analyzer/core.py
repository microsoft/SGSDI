# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging

from .uniform_scene import UniSceneAnalyzer
from ..post_processor import scene_convert
from ..data.base import get_cfg_from_pipeline, GroupDataGenConfig
from GraphicsDL.graphicsutils import g_str


class CoreAnalyzer(object):
    def __init__(self, eval_dir, output_dir, cfg_path, assemble_name=None):
        self.eval_dir = eval_dir
        self.output_dir = g_str.mkdir_automated(output_dir)

        self.cfg_path = cfg_path
        self.cfg = get_cfg_from_pipeline(GroupDataGenConfig, cfg_path, assemble_name)

        self.analysis_methods = ['vis_3d', 'correlation', 'object_distribution', 'category_distribution']

    def analysis_generation(self):
        generated_data_path = os.path.join(self.eval_dir, 'eval_meta.npz')
        if not os.path.exists(generated_data_path):
            logging.warning(f'{generated_data_path} not found')
        logging.info(f'analysis {self.eval_dir}')
        converter = scene_convert.Vol2UniSceneConverter(generated_data_path, self.eval_dir, self.cfg_path)
        converter.mediate_process()

        scenes_path = os.path.join(self.eval_dir, f'eval_meta_uni_rep.zip')
        objs_path = os.path.join(self.eval_dir, f'eval_meta_obj_rep.zip')
        if not os.path.exists(scenes_path) or not os.path.exists(objs_path):
            return
        analyzer = UniSceneAnalyzer('generation', scenes_path, objs_path, self.output_dir, self.cfg)
        for a in self.analysis_methods:
            if not a:
                continue
            try:
                logging.info(f'\t{a}_analysis')
                getattr(analyzer, f'{a}_analysis')()
            except AttributeError:
                logging.warning(f'\tFail to find target {a}_analysis, ignored')
