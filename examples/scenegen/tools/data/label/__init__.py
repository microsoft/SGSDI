# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


CACHED_LABEL_TYPE = None
CACHED_COLOR_MAP: Optional[np.ndarray] = None
CACHED_LABEL_LIST = None
CACHED_SHORT_LABEL_LIST = None


class CoreLabel(object):
    def __init__(self, label_type, label_path, color_path):
        global CACHED_LABEL_TYPE, CACHED_COLOR_MAP, CACHED_LABEL_LIST, CACHED_SHORT_LABEL_LIST
        if CACHED_LABEL_TYPE != label_type:
            CACHED_LABEL_TYPE = label_type
            CACHED_COLOR_MAP, CACHED_LABEL_LIST, CACHED_SHORT_LABEL_LIST = None, None, None

        self.label_path = label_path
        self.color_path = color_path

    def label_id_map_arr(self, return_short_label=False):
        global CACHED_LABEL_LIST, CACHED_SHORT_LABEL_LIST
        if CACHED_LABEL_LIST is None:
            with open(self.label_path, 'r') as fp:
                label_split_list = [n_t.split() for n_t in fp.readlines()]
                label_list = [l_s[1] for l_s in label_split_list]
                short_label_list = [l_s[2] for l_s in label_split_list] if len(label_split_list[0]) > 2 else label_list
            CACHED_LABEL_LIST = ['void'] + label_list
            CACHED_SHORT_LABEL_LIST = ['void'] + short_label_list
        if return_short_label:
            return CACHED_SHORT_LABEL_LIST
        return CACHED_LABEL_LIST

    def num_categories(self):
        return len(self.label_id_map_arr())

    def label_id_map(self, label):
        assert label in self.label_id_map_arr()
        return self.label_id_map_arr().index(label)

    def color_map_arr(self):
        global CACHED_COLOR_MAP
        if CACHED_COLOR_MAP is None:
            with open(self.color_path, 'r') as fp:
                colors_txt = fp.readlines()
            c_map = list([[255, 255, 255]])
            for c_t in colors_txt:
                c_t = c_t[1: c_t.find(')')]
                c = np.fromstring(c_t, sep=',', dtype=np.uint8)
                c_map.append(c)
            c_map = np.asarray(c_map, dtype=np.uint8)
            CACHED_COLOR_MAP = c_map
        return CACHED_COLOR_MAP

    def color_map(self, label_id):
        return self.color_map_arr()[label_id]

    def vis_color(self, color_show=True, out_path=None):
        labels = self.label_id_map_arr()
        colors = self.color_map_arr() / 255

        plt.pie(np.ones(len(labels)), labels=labels, colors=colors)
        if out_path:
            plt.savefig(out_path)
        if color_show:
            plt.show()


class NYU40(CoreLabel):
    def __init__(self):
        label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nyu', 'nyu40label.txt')
        color_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nyu', 'nyu40color.txt')
        super().__init__(self.__class__.__name__, label_path, color_path)


class Structured3DBedroom9(CoreLabel):
    def __init__(self):
        label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'structure3d', 'bedroom9label.txt')
        color_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'structure3d', 'bedroom9color.txt')
        super().__init__(self.__class__.__name__, label_path, color_path)


class Structured3DLiving11(CoreLabel):
    def __init__(self):
        label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'structure3d', 'living11label.txt')
        color_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'structure3d', 'living11color.txt')
        super().__init__(self.__class__.__name__, label_path, color_path)


class Structured3DKitchen5(CoreLabel):
    def __init__(self):
        label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'structure3d', 'kitchen5label.txt')
        color_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'structure3d', 'kitchen5color.txt')
        super().__init__(self.__class__.__name__, label_path, color_path)


class Matterport3DBedroom8(CoreLabel):
    def __init__(self):
        label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matterport3d', 'bedroom8label.txt')
        color_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matterport3d', 'bedroom8color.txt')
        super().__init__(self.__class__.__name__, label_path, color_path)
