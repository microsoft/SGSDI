# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import cv2
import logging
import numpy as np
import tensorflow as tf

from .utils import load_vol_and_camera, initialize_network
from GraphicsDL.graphicsutils import g_str
from examples.scenegen.cfg_def import CustomRunnerConfigurator
from examples.scenegen.custom_layers import get_color_palettes
from examples.scenegen.tools.data.base import get_cfg, GroupDataGenConfig


def view_rendering(vol_path: str, out_path: str, cfg_path: str, num_cls: int, batch_size: int = 32):
    logging.info(f'Render {vol_path} to {out_path}: {cfg_path}, {num_cls} classes')
    # load network config
    tf.keras.backend.set_learning_phase(0)
    runner_cfg = CustomRunnerConfigurator()
    runner_cfg.load_from_yaml(cfg_path)
    nets_cfg = [n_ for n_ in runner_cfg.nets if n_.type.find('ViewRenderer') != -1]
    assert len(nets_cfg) == 1
    net_cfg, = tuple(nets_cfg)
    render_ops, _ = initialize_network(net_cfg=net_cfg, model_dir=None, extra_kargs=dict())

    # load data
    vol_data_dict = load_vol_and_camera(vol_path)
    vol_data, cam_r, cam_t = vol_data_dict['label'], vol_data_dict['cam_r'], vol_data_dict['cam_t']

    # render training image
    render_iters = vol_data.shape[0] // batch_size
    cam_r_list, cam_t_list = list(), list()
    label_views_list, depth_views_list = list(), list()
    for i in range(0, render_iters * batch_size, batch_size):
        iter_vol = vol_data[i: i + batch_size, ...]
        iter_cam_r = cam_r[i: i + batch_size, ...]
        iter_cam_t = cam_t[i: i + batch_size, ...]
        iter_vol_label = tf.one_hot(iter_vol, num_cls, dtype=tf.float32)
        iter_out = render_ops([iter_vol_label, iter_cam_r, iter_cam_t])['out']
        iter_label_views, iter_depth_views = iter_out[:2]
        label_views_list.extend(tf.argmax(iter_label_views, axis=-1).numpy().astype(np.uint8))
        depth_views_list.extend(tf.squeeze(iter_depth_views, axis=-1).numpy().astype(np.float32))
        cam_r_list.extend(iter_cam_r)
        cam_t_list.extend(iter_cam_t)

    np.savez_compressed(out_path, category=np.asarray(label_views_list), depth=np.asarray(depth_views_list),
                        cam_r=np.asarray(cam_r_list), cam_t=np.asarray(cam_t_list))

    view_id = vol_data_dict['view_id']
    vis_dir = g_str.mkdir_automated(out_path[:-4] + '_vis')
    color_map = get_color_palettes(num_cls, color_norm=False)[..., ::-1]
    for v_i in range(min(len(label_views_list), 20)):
        view_shape = label_views_list[v_i].shape
        resize_scale = 256 // view_shape[1]
        image_size = (view_shape[1] * resize_scale, view_shape[0] * resize_scale)
        label_view = cv2.resize(label_views_list[v_i], image_size, interpolation=cv2.INTER_NEAREST)
        depth_view = cv2.resize(depth_views_list[v_i], image_size, interpolation=cv2.INTER_NEAREST)
        label_view_color = color_map[label_view]

        cv2.imwrite(os.path.join(vis_dir, f'{view_id[v_i]}_label.png'), label_view_color)
        cv2.imwrite(os.path.join(vis_dir, f'{view_id[v_i]}_depth.png'), depth_view * 255)


def assemble_final_view_data(in_path, out_dir, assemble_name='train_image', view_num=1):
    logging.info(f'{assemble_name} assemble {in_path} to {out_dir}: {view_num} views')
    if not os.path.exists(in_path):
        logging.info(f'Not found {in_path}')
        return
    data_sample = np.load(in_path)

    tf_option = tf.io.TFRecordOptions()
    tf_option.compression_type = tf.compat.v1.python_io.TFRecordCompressionType.ZLIB
    assemble_record = tf.io.TFRecordWriter(os.path.join(out_dir, f'{assemble_name}.records'), tf_option)

    data_type = ['category', 'depth']
    assemble_data = {d_t: data_sample[d_t] for d_t in data_type}
    logging.info(f'Data: {[[d_t, assemble_data[d_t].shape, assemble_data[d_t].dtype] for d_t in data_type]}')
    assemble_data = {d_k: d_s.reshape([-1, view_num, *d_s.shape[1:]]) for d_k, d_s in assemble_data.items()}
    logging.info(f'Assemble data: {[[d_t, assemble_data[d_t].shape, assemble_data[d_t].dtype] for d_t in data_type]}')
    for d_i in range(len(assemble_data[data_type[0]])):
        if len(np.unique(assemble_data['category'][d_i])) == 1:
            continue
        feature = dict()
        for d_t in data_type:
            data = np.squeeze(assemble_data[d_t][d_i])
            feature[d_t] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()]))
        assemble_record.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
    assemble_record.close()


def assemble_training_data(cfg_path):
    cfg = get_cfg(GroupDataGenConfig, cfg_path)
    for data_cfg in cfg.dataset_list:
        for p_p in data_cfg.process_pipelines:
            if not os.path.exists(data_cfg.out_dir):
                logging.warning(f'{data_cfg.out_dir} not found')
                continue
            assemble_dir = os.path.join(data_cfg.out_dir, p_p.label_type, 'AssembleData')
            surface_vol_path = os.path.join(assemble_dir, f'{p_p.assemble_name}_{p_p.room_types[0]}.zip')
            render_out_path = os.path.join(assemble_dir, 'rendered_image.npz')
            assemble_data_dir = g_str.mkdir_automated(os.path.join(data_cfg.out_dir, p_p.label_type, 'TrainViewData'))
            num_cls = int(p_p.label_type.split('_')[-1]) + 1
            render_cfg = os.path.join(os.path.split(cfg_path)[0], p_p.render_cfg)
            view_rendering(surface_vol_path, render_out_path, render_cfg, num_cls)
            assemble_final_view_data(render_out_path, assemble_data_dir, view_num=1)
