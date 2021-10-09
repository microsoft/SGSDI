# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
import platform
import argparse
import importlib
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_arguments():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--backbone', required=False, type=str, default='v2')
    parser.add_argument('--cfg_path', required=True, type=str, help='config files')
    parser.add_argument('--measure', required=True, type=str, default='Train or Test')
    parser.add_argument('--example', required=True, type=str, help='the example to run')
    parser.add_argument('--log_dir', required=True, type=str, help='path to save training log')
    parser.add_argument('--model_dir', required=True, type=str, help='path to save trained model')
    parser.add_argument('--data_dir', required=True, type=str, help='path of train/validation data')
    parser.add_argument('--pretrain_dir', required=False, type=str, help='pretrain directory', default='pretrain')
    return parser.parse_known_args()


def active_logging():
    config_dict = dict(handlers=[logging.FileHandler('runtime.log'), logging.StreamHandler()],
                       format='[%(asctime)s] %(levelname)s: %(message)s',
                       level=logging.DEBUG)
    logging.basicConfig(**config_dict)
    # logging setting
    logging.info('===============================')
    logging.info('os=%s', platform.system())
    logging.info('host=%s', platform.node())
    try:
        logging.info('visible_device=%s', os.environ['CUDA_VISIBLE_DEVICES'])
    except KeyError:
        logging.info('visible_device=not specify')


def set_config():
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    for device in physical_devices:
        logging.info(f'set_memory_growth on device{device}')
        tf.config.experimental.set_memory_growth(device, True)


if __name__ == '__main__':
    active_logging()
    set_config()
    (args, unknown) = parse_arguments()
    if args.backbone == 'v1':
        config = importlib.import_module('GraphicsDL.modules.config')
        runner = importlib.import_module('GraphicsDL.modules.runner')
    elif args.backbone == 'v2':
        custom_config_file = f'{args.example}/cfg_def.py'
        if os.path.exists(custom_config_file):
            cfg_import_module = custom_config_file.split('.')[0].replace('/', '.').replace('\\', '.')
        else:
            cfg_import_module = 'GraphicsDL.modules_v2.config'
        config = importlib.import_module(cfg_import_module)
        runner = importlib.import_module('GraphicsDL.modules_v2.runner')
    else:
        raise NotImplementedError

    try:
        exp_args = config.CustomRunnerConfigurator()
    except AttributeError:
        exp_args = config.RunnerConfigurator()
    exp_args.load_from_yaml(args.cfg_path, 'shared')
    for r in exp_args.readers:
        r.data_dir = args.data_dir
    exp_args.model_dir = args.model_dir
    exp_args.log_dir = args.log_dir
    exp_args.eval_dir = args.eval_dir
    exp_args.pretrain_dir = args.pretrain_dir
    exp_args.example = args.example

    exp_runner = getattr(runner, exp_args.type)(exp_args)
    if args.measure == 'train':
        print("Begin to train ...")
        exp_runner.train()
    elif args.measure == 'test':
        exp_runner.args.validate_stride = 1
        exp_runner.test()
    elif args.measure == 'perf':
        exp_runner.perf()
    else:
        raise NotImplementedError
