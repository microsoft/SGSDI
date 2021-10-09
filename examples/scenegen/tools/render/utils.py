# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import io
import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional, List

from examples.scenegen import net_def as net_module
from examples.scenegen import custom_reader as reader_module
from examples.scenegen.cfg_def import CustomRunnerConfigurator, CustomNetworkConfigurator
from GraphicsDL.graphicsutils import g_io, g_str
from GraphicsDL.modules_v2.solver import ExecutionTree
from GraphicsDL.modules_v2.config import FlowConfigurator


logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG)


def load_scene_data():
    scene_matrix_path = os.path.join(g_str.abs_dir_path(__file__), '../unittests_resource', 'scene_matrix.npz')
    scene_matrix = np.load(scene_matrix_path)['matrix']
    return scene_matrix


class ValidatorCallbackArgs(object):
    def __init__(self, model_dir):
        self.model_dir = model_dir


class NetworkFlower(object):
    def __init__(self, flow_cfg: FlowConfigurator, nets: Dict):
        self.flow_cfg = flow_cfg
        self.execution_tree = ExecutionTree(self.flow_cfg.sequence.flow)
        self.execution_tree.build([nets[k_] for k_ in self.flow_cfg.sequence.nets])


def instance_case(cls_dict, cls_module, cls_list, name_list, extra_kargs):
    cls_cfg_list = [n_ for n_ in cls_list if n_.name in name_list and n_.name not in cls_dict]
    for cls_cfg in cls_cfg_list:
        cls_net = getattr(cls_module, cls_cfg.type)
        cls_kargs = cls_cfg.match_function_args({}, cls_net.__init__)
        for k, v in extra_kargs.items():
            if k in cls_kargs:
                cls_kargs[k] = v
        cls_ins = cls_net(**cls_kargs)
        cls_dict[cls_cfg.name] = cls_ins
    return cls_dict


def initialize_network_flow(runner_cfg: CustomRunnerConfigurator, flow_cfg_list: List[FlowConfigurator],
                            extra_kargs: Dict = dict()):
    readers_dict = dict()
    networks_dict = dict()
    flow_list = list()
    for f_i, flow_cfg in enumerate(flow_cfg_list):
        readers_dict = instance_case(readers_dict, reader_module, runner_cfg.readers, flow_cfg.sequence.readers, extra_kargs)
        networks_dict = instance_case(networks_dict, net_module, runner_cfg.nets, flow_cfg.sequence.nets, extra_kargs)

        flow_list.append(NetworkFlower(flow_cfg, networks_dict))

    return flow_list, networks_dict, readers_dict


def restore_ckp(net_dict, ckp_path):
    ckp = tf.train.Checkpoint(**net_dict)
    epoch = 0
    if ckp_path is not None:
        if os.path.isdir(ckp_path):
            ckp_path = tf.train.latest_checkpoint(ckp_path)
        epoch = int(ckp_path.split('-')[-1])
        ckp.restore(ckp_path).expect_partial()
    return ckp, epoch


def initialize_network(net_cfg: CustomNetworkConfigurator, model_dir: Optional[str], extra_kargs: Dict = dict()) \
        -> Tuple[tf.keras.Model, tf.train.Checkpoint]:
    cls_net = getattr(net_module, net_cfg.type)
    cls_kargs = net_cfg.match_function_args(extra_kargs, cls_net.__init__)
    cls_ins = cls_net(**cls_kargs)
    net_dict = {net_cfg.name: cls_ins}
    ckpt, _ = restore_ckp(net_dict, model_dir)
    return cls_ins, ckpt


def initialize_data_set(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(data_path)
    return data['semantic'], data['depth']


def load_eval_data(vol_path, auto_squeeze=True):
    eval_meta: np.lib.npyio.NpzFile = np.load(vol_path)
    eval_data = list()
    for e_k in eval_meta.keys():
        e_d = np.squeeze(eval_meta[e_k]) if auto_squeeze else eval_meta[e_k]
        eval_data.append(e_d)
    assert len(eval_data) == 1
    return eval_data[0]


def load_zip_data(vol_zip_path):
    train_vol_list = list()
    vol_zip_meta = g_io.ZipIO(vol_zip_path)
    vol_samples_list = [f for f in vol_zip_meta.namelist() if f.endswith('npz')]
    for t_i, t_f in enumerate(vol_samples_list):
        raw_data = vol_zip_meta.read(t_f)
        vol_data_np = np.asarray(np.load(io.BytesIO(raw_data))['label']).astype(np.uint8)
        train_vol_list.append(vol_data_np)
    train_vol_list = np.asarray(train_vol_list)
    return train_vol_list


def load_vol_and_camera(vol_zip_path):
    vol_zip_meta = g_io.ZipIO(vol_zip_path)
    vol_samples_list = [f for f in vol_zip_meta.namelist() if f.endswith('npz')]
    vol_data_dict = dict(view_id=list(), label=list(), cam_r=list(), cam_t=list())
    for t_i, t_f in enumerate(vol_samples_list):
        vol_meta_data = np.load(io.BytesIO(vol_zip_meta.read(t_f)))
        cam_t = vol_meta_data['cam_t'].reshape([-1, 3])
        cam_r = vol_meta_data['cam_r'].reshape([-1, 3, 3])
        vol_data_dict['cam_r'].append(cam_r)
        vol_data_dict['cam_t'].append(cam_t)
        view_shape = cam_t.shape
        vox_tile_shape = [*view_shape[:-1], 1, 1, 1]
        vol_label = np.reshape(vol_meta_data['label'], [*[1]*(len(vox_tile_shape)-3), *vol_meta_data['label'].shape])
        vol_label_tile = np.tile(vol_label, vox_tile_shape)
        vol_data_dict['label'].append(vol_label_tile.reshape([-1, *vol_label_tile.shape[-3:]]))
        camera_id = vol_meta_data['camera_id'] if 'camera_id' in vol_meta_data and 'cam_r' in vol_meta_data \
            else ['%02d' % v_i for v_i in range(view_shape[0])]
        view_id = np.asarray([f'{t_f[:-4]}_{c_i}' for c_i in camera_id]) if t_f[:-4] not in camera_id[0] else camera_id
        if len(view_shape) == 3:
            view_id = np.array([f'{c_i}_{v_i}' for c_i in view_id for v_i in range(view_shape[1])])
        vol_data_dict['view_id'].append(view_id)
    for key, value in vol_data_dict.items():
        vol_data_dict[key] = np.concatenate(value, axis=0)
        print(f'{key} data shape: {vol_data_dict[key].shape}')
    return vol_data_dict


def load_data(data_path):
    if data_path.endswith('npz'):
        return load_eval_data(data_path)
    elif data_path.endswith('zip'):
        return load_zip_data(data_path)
    else:
        raise NotImplementedError
