# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
import argparse

from examples.scenegen.tools.data.group import GroupDataGen
from examples.scenegen.tools.analyzer.core import CoreAnalyzer
from examples.scenegen.tools.post_processor.retrieval import ObjectsRetrieval
from examples.scenegen.tools.render.drc_renderer import assemble_training_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG)


def parse_arguments():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--task', required=True, type=str, help='the task to run')
    parser.add_argument('--output_dir', required=False, type=str, help='dir to output')
    parser.add_argument('--cfg_path', required=True, type=str, help='path of config file')
    parser.add_argument('--eval_dir', required=False, type=str, help='dir of eval_meta.npz')
    parser.add_argument('--shapenet_path', required=False, type=str, help='path of ShapeNetCore.v2.zip')
    return parser.parse_known_args()


if __name__ == '__main__':
    (args, unknown) = parse_arguments()
    if args.task == 'data_gen':
        dataset = GroupDataGen(args.cfg_path)
        dataset.execute_pipeline()
        assemble_training_data(args.cfg_path)
    elif args.task == 'evaluation':
        analyzer_case = CoreAnalyzer(args.eval_dir, args.output_dir, args.cfg_path)
        analyzer_case.analysis_generation()
    elif args.task == 'retrieval':
        retrieval = ObjectsRetrieval(args.eval_dir, args.output_dir, args.shapenet_path, args.cfg_path)
        retrieval.eval_retrieval()
    else:
        raise NotImplementedError
