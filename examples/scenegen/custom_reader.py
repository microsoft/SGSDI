# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import numpy as np
import tensorflow as tf
from scipy.stats import truncnorm

from GraphicsDL.modules_v2.reader import BaseReaderV2, DefaultTFReader, RandomReader


class RoomSizeReader(BaseReaderV2):
    def __init__(self, data_dir, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params, out_params,
                 w_params, prefix, rel_path, name=None, **kwargs):
        super().__init__(batch_size, num_devices, shuffle, split, infinite, in_params, out_params, w_params, name,
                         **kwargs)
        self.record_dir = os.path.join(data_dir, rel_path) if rel_path else data_dir
        prefix = prefix if prefix else 'room_size'
        self.room_size_files = os.path.join(self.record_dir, f'{prefix}_{num_samples}.npz')

        self.deterministic = None
        self.cur_samples = 0
        self.num_samples = num_samples

    @staticmethod
    def random_room_size(sample_num):
        room_rand_x = tf.random.normal([sample_num, 1], mean=4.53, stddev=0.98, dtype=tf.float32)
        room_rand_x = tf.clip_by_value(room_rand_x, 1.7, 6.4) / 2
        room_rand_z = tf.random.normal([sample_num, 1], mean=4.35, stddev=0.99, dtype=tf.float32)
        room_rand_z = tf.clip_by_value(room_rand_z, 1.5, 6.4) / 2
        room_rand_y = tf.random.normal([sample_num, 1], mean=2.74, stddev=0.05, dtype=tf.float32)
        room_rand_y = tf.clip_by_value(room_rand_y, 2.2, 3.2)

        box_max_x = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] + room_rand_x
        box_min_x = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] - room_rand_x
        box_max_z = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] + room_rand_z
        box_min_z = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] - room_rand_z
        box_max_y = tf.clip_by_value(room_rand_y, 2.2, 3.2)
        box_min_y = tf.zeros_like(box_max_y)

        room_rand = tf.concat([box_max_x, box_max_y, box_max_z, box_min_x, box_min_y, box_min_z], axis=-1)
        return room_rand

    def postprocess(self, inputs, post_str):
        post_str_split = np.split(np.reshape(np.asarray(post_str.split('-')), [-1, 2]), 2, axis=-1)
        p_method, p_args = [a_[..., 0] for a_ in post_str_split]
        for p_m, p_a in zip(p_method, p_args):
            if p_m == 'scale':
                inputs = inputs * float(p_a)
            else:
                raise NotImplementedError
        return inputs

    def next_stochastic(self):
        in_elem = list()
        for i_p in self.in_params:
            room_rand = self.random_room_size(self.batch_size)
            assert i_p.postprocess
            if i_p.postprocess:
                room_rand = self.postprocess(room_rand, i_p.postprocess)
            all_data = tf.split(room_rand, self.num_devices)
            in_elem.append(all_data)
        return dict(inputs=in_elem, outputs=list(), weights=list(), alias=list())

    def next_deterministic(self):
        if self.deterministic is None:
            if not os.path.exists(self.room_size_files):
                deterministic_data = dict()
                self.deterministic = list()
                for i_p in self.in_params:
                    rand_nd = self.random_room_size(self.num_samples)
                    assert i_p.postprocess
                    if i_p.postprocess:
                        rand_nd = self.postprocess(rand_nd, i_p.postprocess)
                    self.deterministic.append(rand_nd)
                    deterministic_data[i_p.name] = rand_nd
                np.savez_compressed(self.room_size_files, **deterministic_data)
            else:
                random_reader_meta = np.load(self.room_size_files)
                self.deterministic = list()
                for i_p in self.in_params:
                    rand_nd = random_reader_meta[i_p.name].astype(np.float32)
                    assert rand_nd.shape == (self.num_samples, 6)
                    self.deterministic.append(rand_nd)
        try:
            in_elem = list()
            if self.cur_samples > self.num_samples - self.batch_size:
                raise StopIteration
            for d in self.deterministic:
                all_data = tf.split(d[self.cur_samples: self.cur_samples + self.batch_size], self.num_devices, axis=0)
                in_elem.append(all_data)
            self.cur_samples += self.batch_size
            return dict(inputs=in_elem, outputs=list(), weights=list(), alias=list())
        except StopIteration:
            self.cur_samples = 0
            raise StopIteration

    def next(self):
        try:
            if self.shuffle:
                return self.next_stochastic()
            else:
                return self.next_deterministic()
        except StopIteration:
            if self.infinite:
                return self.next()
            else:
                raise StopIteration


class RandomReaderV1(RandomReader):
    def __init__(self, data_dir, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params, out_params,
                 w_params, prefix, rel_path, compress='', name=None, **kwargs):
        super().__init__(batch_size, num_samples, num_devices, shuffle, split, infinite, in_params, out_params,
                         w_params, name, **kwargs)
        self.record_dir = os.path.join(data_dir, rel_path) if rel_path else data_dir
        prefix = prefix if prefix else 'custom_random'
        self.random_reader_files = os.path.join(self.record_dir, f'{prefix}_{num_samples}.npz')

    def next_deterministic(self):
        if self.deterministic is None:
            if not os.path.exists(self.random_reader_files):
                deterministic_data = dict()
                self.deterministic = list()
                for i_p in self.in_params:
                    rand_nd = truncnorm.rvs(-1, 1, size=[self.num_samples, *i_p.raw_shape]).astype(np.float32)
                    # rand_nd = np.random.normal(size=[self.num_samples, *i_p.raw_shape]).astype(np.float32)
                    self.deterministic.append(rand_nd)
                    deterministic_data[i_p.name] = rand_nd
                np.savez_compressed(self.random_reader_files, **deterministic_data)
            else:
                random_reader_meta = np.load(self.random_reader_files)
                self.deterministic = list()
                for i_p in self.in_params:
                    rand_nd = random_reader_meta[i_p.name].astype(np.float32)
                    assert rand_nd.shape == (self.num_samples, *i_p.raw_shape)
                    self.deterministic.append(rand_nd)
        return super().next_deterministic()


class Str3DRoomSizeReader(RoomSizeReader):
    def __init__(self, data_dir, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params, out_params,
                 w_params, prefix, rel_path, name=None, **kwargs):
        super().__init__(data_dir, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params,
                         out_params, w_params, prefix, rel_path, name, **kwargs)

    @staticmethod
    def random_room_size(sample_num):
        room_rand_x = tf.random.normal([sample_num, 1], mean=3.98, stddev=1.14, dtype=tf.float32)
        room_rand_x = tf.clip_by_value(room_rand_x, 2.2, 6.4) / 2
        room_rand_z = tf.random.normal([sample_num, 1], mean=3.98, stddev=1.14, dtype=tf.float32)
        room_rand_z = tf.clip_by_value(room_rand_z, 2.2, 6.4) / 2
        room_rand_y = tf.random.normal([sample_num, 1], mean=2.74, stddev=0.05, dtype=tf.float32)
        room_rand_y = tf.clip_by_value(room_rand_y, 2.2, 3.2)

        box_max_x = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] + room_rand_x
        box_min_x = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] - room_rand_x
        box_max_z = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] + room_rand_z
        box_min_z = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] - room_rand_z
        box_max_y = tf.clip_by_value(room_rand_y, 2.2, 3.2)
        box_min_y = tf.zeros_like(box_max_y)

        room_rand = tf.concat([box_max_x, box_max_y, box_max_z, box_min_x, box_min_y, box_min_z], axis=-1)
        return room_rand


class Str3DLivingRoomSizeReader(RoomSizeReader):
    def __init__(self, data_dir, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params, out_params,
                 w_params, prefix, rel_path, name=None, **kwargs):
        super().__init__(data_dir, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params,
                         out_params, w_params, prefix, rel_path, name, **kwargs)

    @staticmethod
    def random_room_size(sample_num):
        room_rand_x = tf.random.normal([sample_num, 1], mean=8.44, stddev=1.70, dtype=tf.float32)
        room_rand_x = tf.clip_by_value(room_rand_x, 4.0, 9.6) / 2
        room_rand_z = tf.random.normal([sample_num, 1], mean=8.44, stddev=1.70, dtype=tf.float32)
        room_rand_z = tf.clip_by_value(room_rand_z, 4.0, 9.6) / 2
        room_rand_y = tf.random.normal([sample_num, 1], mean=2.80, stddev=0.06, dtype=tf.float32)
        room_rand_y = tf.clip_by_value(room_rand_y, 2.6, 3.0)

        box_max_x = tf.convert_to_tensor([4.8], dtype=tf.float32)[tf.newaxis, ...] + room_rand_x
        box_min_x = tf.convert_to_tensor([4.8], dtype=tf.float32)[tf.newaxis, ...] - room_rand_x
        box_max_z = tf.convert_to_tensor([4.8], dtype=tf.float32)[tf.newaxis, ...] + room_rand_z
        box_min_z = tf.convert_to_tensor([4.8], dtype=tf.float32)[tf.newaxis, ...] - room_rand_z
        box_max_y = tf.clip_by_value(room_rand_y, 2.6, 3.0)
        box_min_y = tf.zeros_like(box_max_y)

        room_rand = tf.concat([box_max_x, box_max_y, box_max_z, box_min_x, box_min_y, box_min_z], axis=-1)
        return room_rand


class Str3DKitchenRoomSizeReader(RoomSizeReader):
    def __init__(self, data_dir, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params, out_params,
                 w_params, prefix, rel_path, name=None, **kwargs):
        super().__init__(data_dir, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params,
                         out_params, w_params, prefix, rel_path, name, **kwargs)

    @staticmethod
    def random_room_size(sample_num):
        room_rand_x = tf.random.normal([sample_num, 1], mean=3.32, stddev=0.74, dtype=tf.float32)
        room_rand_x = tf.clip_by_value(room_rand_x, 2.0, 6.4) / 2
        room_rand_z = tf.random.normal([sample_num, 1], mean=3.32, stddev=0.74, dtype=tf.float32)
        room_rand_z = tf.clip_by_value(room_rand_z, 2.0, 6.4) / 2
        room_rand_y = tf.random.normal([sample_num, 1], mean=2.80, stddev=0.06, dtype=tf.float32)
        room_rand_y = tf.clip_by_value(room_rand_y, 2.5, 3.2)

        box_max_x = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] + room_rand_x
        box_min_x = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] - room_rand_x
        box_max_z = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] + room_rand_z
        box_min_z = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] - room_rand_z
        box_max_y = tf.clip_by_value(room_rand_y, 2.5, 3.2)
        box_min_y = tf.zeros_like(box_max_y)

        room_rand = tf.concat([box_max_x, box_max_y, box_max_z, box_min_x, box_min_y, box_min_z], axis=-1)
        return room_rand


class Mat3DBedroomRoomSizeReader(RoomSizeReader):
    def __init__(self, data_dir, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params, out_params,
                 w_params, prefix, rel_path, name=None, **kwargs):
        super().__init__(data_dir, batch_size, num_samples, num_devices, shuffle, split, infinite, in_params,
                         out_params, w_params, prefix, rel_path, name, **kwargs)

    @staticmethod
    def random_room_size(sample_num):
        room_rand_x = tf.random.normal([sample_num, 1], mean=4.164, stddev=0.973, dtype=tf.float32)
        room_rand_x = tf.clip_by_value(room_rand_x, 2.2, 6.4) / 2
        room_rand_z = tf.random.normal([sample_num, 1], mean=4.265, stddev=0.955, dtype=tf.float32)
        room_rand_z = tf.clip_by_value(room_rand_z, 2.2, 6.4) / 2
        room_rand_y = tf.random.normal([sample_num, 1], mean=2.387, stddev=0.425, dtype=tf.float32)
        room_rand_y = tf.clip_by_value(room_rand_y, 2.2, 3.2)

        box_max_x = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] + room_rand_x
        box_min_x = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] - room_rand_x
        box_max_z = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] + room_rand_z
        box_min_z = tf.convert_to_tensor([3.2], dtype=tf.float32)[tf.newaxis, ...] - room_rand_z
        box_max_y = tf.clip_by_value(room_rand_y, 1.7, 3.2)
        box_min_y = tf.zeros_like(box_max_y)

        room_rand = tf.concat([box_max_x, box_max_y, box_max_z, box_min_x, box_min_y, box_min_z], axis=-1)
        return room_rand
