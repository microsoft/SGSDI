# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from GraphicsDL.graphicsutils import g_str
from .cfg_def import CustomRunnerConfigurator
from .custom_discriminator import *
from .custom_conditional import *
from .custom_generator import *
from .custom_sampler import *
from .custom_layers import *
from .custom_losses import *


class ValidatorCallback(object):
    def __init__(self, args):
        self.args = args
        self.val_data_dict = dict()

    def reset(self):
        self.val_data_dict = dict()

    def update(self, out_dict):
        for v_n, v_o in out_dict.items():
            val_data = np.concatenate([np.argmax(v_d[0].numpy(), -1).astype(np.uint8) for v_d in v_o])
            val_data = np.concatenate((self.val_data_dict[v_n], val_data)) if v_n in self.val_data_dict else val_data
            self.val_data_dict[v_n] = val_data
            if len(v_o[0]) > 1:
                val_room_bbox = np.concatenate([v_d[1].numpy() for v_d in v_o])
                if f'{v_n}Bbox' in self.val_data_dict:
                    val_room_bbox = np.concatenate((self.val_data_dict[f'{v_n}Bbox'], val_room_bbox))
                self.val_data_dict[f'{v_n}Bbox'] = val_room_bbox
            if len(v_o[0]) > 2:
                val_room_bbox = np.concatenate([v_d[2].numpy() for v_d in v_o])
                if f'{v_n}Z' in self.val_data_dict:
                    val_room_bbox = np.concatenate((self.val_data_dict[f'{v_n}Z'], val_room_bbox))
                self.val_data_dict[f'{v_n}Z'] = val_room_bbox

    def analysis(self, epoch):
        val_dir = g_str.mkdir_automated(os.path.join(self.args.model_dir, 'eval', '%06d' % epoch))
        np.savez_compressed(os.path.join(val_dir, 'eval_meta'), **self.val_data_dict)

        self.reset()
