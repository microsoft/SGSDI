# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import trimesh
import logging
import numpy as np
from PIL import Image
from ..data.base import AxisAlignBoundingBox


trimesh.util.attach_to_log(level=logging.ERROR)


class VisualizationUtilize(object):
    @staticmethod
    def vol_top_view_visualization(vis_path, vol_data, colors_map, image_size=256):
        vol_arange = np.tile(np.arange(vol_data.shape[1]), vol_data.shape[0] * vol_data.shape[2])
        vol_index_t = vol_arange.reshape([vol_data.shape[0], vol_data.shape[2], vol_data.shape[1]])
        vol_index = vol_index_t.transpose((0, 2, 1))
        vol_data_index = np.where(vol_data > 0, vol_index, vol_data)
        top_view_index = np.argmax(vol_data_index, axis=1)
        plane_index = np.tile(np.arange(top_view_index.shape[0]), top_view_index.shape[1]).reshape(top_view_index.shape)
        top_view_label = vol_data[plane_index.T, top_view_index, plane_index].T
        top_view_color = colors_map[top_view_label]
        top_view_image = Image.fromarray(top_view_color.astype('uint8')).convert('RGB')
        top_view_image = top_view_image.resize((image_size, image_size), resample=Image.NEAREST)
        top_view_image.save(vis_path + '.png')

    @staticmethod
    def get_render_camera_parameter(room_size=3.2, cam_height=1.3):
        camera_rotation_y_list = np.asarray([-np.pi / 4, -np.pi * 3 / 4, np.pi * 3 / 4, np.pi / 4])
        camera_position_room_list = np.asarray([[-room_size, room_size, room_size], [-room_size, room_size, -room_size],
                                                [room_size, room_size, -room_size], [room_size, room_size, room_size]])
        camera_position_list = camera_position_room_list * 10 + [room_size, 0, room_size]
        camera_distance = np.sqrt(np.sum(np.square(camera_position_list[-1][[0, 2]]))) - np.sqrt(room_size**2 * 2)
        camera_rotation_x = np.arctan(camera_distance / (camera_position_list[-1][1] - cam_height)) - np.pi / 2
        camera_matrix_list = list()
        for c_i, cam_pos in enumerate(camera_position_list):
            cam_m = trimesh.transformations.euler_matrix(camera_rotation_x, camera_rotation_y_list[c_i], 0)
            cam_m[:3, 3] = cam_pos
            camera_matrix_list.append(cam_m)
        return np.asarray(camera_matrix_list)

    @staticmethod
    def add_wall_floor(scene_mesh, floor_mesh_path, scene_bbox: AxisAlignBoundingBox, wall_flag=15):
        scene_center_floor = scene_bbox.center_floor()
        scene_size = scene_bbox.box_size()

        floor_mesh = trimesh.load_mesh(floor_mesh_path)
        floor_scale = scene_size / floor_mesh.bounding_box.extents
        floor_scale[1] = 1
        floor_mesh.apply_transform(np.identity(4) * np.append(floor_scale, 1))
        floor_mesh_center_floor = np.mean(floor_mesh.bounding_box.bounds, axis=0)
        floor_mesh_center_floor[1] = floor_mesh.bounding_box.bounds[1, 1]
        floor_mesh.apply_translation(scene_center_floor - floor_mesh_center_floor)
        scene_mesh.add_geometry(floor_mesh)

        wall_height, wall_thick = scene_size[1], 0.1
        wall_mesh_x = trimesh.creation.box(extents=[scene_size[0] + 2 * wall_thick, wall_height, wall_thick])
        wall_mesh_x.visual.face_colors = [112, 128, 144, 255]
        wall_mesh_x.apply_translation([0, wall_height / 2, 0])
        wall_mesh_x_max = wall_mesh_x.copy()

        wall_x = scene_center_floor.copy()
        if wall_flag & 1:
            wall_x[2] = scene_bbox.min[2] - wall_thick / 2
            wall_mesh_x.apply_translation(wall_x)
            scene_mesh.add_geometry(wall_mesh_x)
        if wall_flag & 2:
            wall_x[2] = scene_bbox.max[2] + wall_thick / 2
            wall_mesh_x_max.apply_translation(wall_x)
            scene_mesh.add_geometry(wall_mesh_x_max)

        wall_mesh_z = trimesh.creation.box(extents=[wall_thick, wall_height, scene_size[2] + 2 * wall_thick])
        wall_mesh_z.visual.face_colors = [112, 128, 144, 255]
        wall_mesh_z.apply_translation([0, wall_height / 2, 0])
        wall_mesh_z_max = wall_mesh_z.copy()

        wall_z = scene_center_floor.copy()
        if wall_flag & 4:
            wall_z[0] = scene_bbox.min[0] - wall_thick / 2
            wall_mesh_z.apply_translation(wall_z)
            scene_mesh.add_geometry(wall_mesh_z)
        if wall_flag & 8:
            wall_z[0] = scene_bbox.max[0] + wall_thick / 2
            wall_mesh_z_max.apply_translation(wall_z)
            scene_mesh.add_geometry(wall_mesh_z_max)
        return scene_mesh

    @staticmethod
    def mesh_visualization(scene_mesh, vis_path, cam_m, camera_fov, render_resolution):
        scene_mesh.camera.fov = camera_fov
        scene_mesh.graph[scene_mesh.camera.name] = cam_m
        try:
            scene_vox_mesh_png = scene_mesh.save_image(resolution=render_resolution, visible=True)
            with open(vis_path, 'wb') as f:
                f.write(scene_vox_mesh_png)
                f.close()
        except ZeroDivisionError as E:
            print("unable to save image", str(E))
        except OSError as E:
            print("unable to save image", str(E))

    @staticmethod
    def scene_vox_visualization(vox_path, vis_path, camera_matrix_list, camera_fov, render_resolution):
        scene_vox_mesh = trimesh.load_mesh(vox_path)
        if not isinstance(scene_vox_mesh, trimesh.scene.scene.Scene):
            scene_vox_mesh = scene_vox_mesh.scene()

        for c_i, cam_m in enumerate(camera_matrix_list):
            view_path = f'{vis_path}_view{c_i}.png'
            VisualizationUtilize.mesh_visualization(scene_vox_mesh, view_path, cam_m, camera_fov, render_resolution)

    @staticmethod
    def mesh_scene_visualization(vis_path, scene_mesh, scene_bbox, camera_matrix_list, floor_mesh_path,
                                 wall_flag_list, camera_fov, render_resolution):
        scene_with_wall = trimesh.Scene(scene_mesh.geometry)
        if floor_mesh_path is not None:
            scene_with_wall = VisualizationUtilize.add_wall_floor(scene_with_wall, floor_mesh_path, scene_bbox)
        scene_with_wall.export(vis_path + '_mesh.glb')

        for c_i, cam_m in enumerate(camera_matrix_list):
            scene_half_wall = trimesh.Scene(scene_mesh.geometry)
            if floor_mesh_path is not None:
                scene_half_wall = VisualizationUtilize.add_wall_floor(scene_half_wall, floor_mesh_path, scene_bbox,
                                                                      wall_flag=wall_flag_list[c_i])
            view_path = f'{vis_path}_view{c_i}.png'
            VisualizationUtilize.mesh_visualization(scene_half_wall, view_path, cam_m, camera_fov, render_resolution)