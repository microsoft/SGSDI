# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from GraphicsDL.modules_v2.config import NetworkConfigurator, LossConfigurator, RunnerConfigurator


class CustomNetworkConfigurator(NetworkConfigurator):
    def __init__(self):
        super().__init__()
        self.flag = int(0)
        # size settings
        self.ndims = int(3)
        self.view_size = list([0])
        self.vox_size = list([0])
        # channels settings
        self.out_channels = int(0)
        self.net_channels = int(64)
        self.max_channels = int(512)
        self.d_latent = int(512)
        self.z_dims = int(128)
        # network structure settings
        self.up_sample = int(4)
        self.down_sample = int(4)
        self.progressive_iter = int(0)
        self.out_activation = str()
        # algorithm depended settings
        self.method = str()
        # view selection related settings
        self.room_height = float(0.)
        self.view_set = list([0])
        self.view_fov = list([0.])
        self.view_num = int(4)
        self.camera_num = int(4)
        # in & out settings
        self.with_depth = bool(True)
        # other settings
        self.seed = int(-1)
        self.random_view = int(0)
        self.shuffle_view = int(0)
        self.num_samples = int(0)


class CustomLossConfigurator(LossConfigurator):
    def __init__(self):
        super().__init__()


class CustomRunnerConfigurator(RunnerConfigurator):
    def __init__(self):
        super().__init__()
        self.nets = list([CustomNetworkConfigurator()])
        self.losses = [CustomLossConfigurator()]
        self.metrics = [CustomLossConfigurator()]
        self.solver_start = list([int(0)])
        self.solver_end = list([int(0)])
