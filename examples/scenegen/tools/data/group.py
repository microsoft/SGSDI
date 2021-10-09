# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import List

from .base import GroupDataGenConfig, BaseDataGen
from .structured_3d import Structured3DDataGen
from .matter_port_3d import Matterport3DDataGen


class GroupDataGen(object):
    def __init__(self, cfg_file):
        self.cfg = GroupDataGenConfig()
        self.cfg.load_from_yaml(cfg_file)
        dataset_type = [globals()[b.data_type] for b in self.cfg.dataset_list]
        dataset_zip = zip(dataset_type, self.cfg.dataset_list)
        dataset_kargs = [c.match_function_args(dict(), b) for b, c in dataset_zip]
        self.dataset_list: List[BaseDataGen] = [b(**k) for b, k in zip(dataset_type, dataset_kargs)]

    def mediate_process(self):
        for b in self.dataset_list:
            b.mediate_process()

    def execute_pipeline(self):
        # Assemble mediate results
        self.mediate_process()
