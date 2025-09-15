import bpy
import os
from mathutils import Vector, geometry, Matrix
import bmesh
import math
import json
import numpy as np
import heapq

def get_object_center(obj):
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    center = sum(verts, Vector()) / len(verts)
    return center

def get_object_world_center(obj):
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    center = sum(verts, Vector()) / len(verts)
    return center

def normalize_object(obj):
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_corner = Vector((min(v.x for v in verts), min(v.y for v in verts), min(v.z for v in verts)))
    max_corner = Vector((max(v.x for v in verts), max(v.y for v in verts), max(v.z for v in verts)))
    center = (min_corner + max_corner) / 2
    size = max((max_corner - min_corner).length, 1e-6)

    obj.location -= center
    obj.scale /= size
    bpy.context.view_layer.update()

def move_objects_to_origin(objects, move_vector):
    T = Matrix.Translation(-move_vector)
    
    for obj in objects:
        obj.matrix_world = T @ obj.matrix_world
    
    bpy.context.view_layer.update()

def normalize_for_gif(objs, target_size=5.0):
    x_max = y_max = z_max = -float("inf")
    x_min = y_min = z_min =  float("inf")

    for obj in objs:
        for v in obj.bound_box:
            world_co = obj.matrix_world @ Vector(v)
            x_max, y_max, z_max = max(x_max, world_co.x), max(y_max, world_co.y), max(z_max, world_co.z)
            x_min, y_min, z_min = min(x_min, world_co.x), min(y_min, world_co.y), min(z_min, world_co.z)

    max_dim = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_dim <= 1e-6:
        return                    

    scale_factor = target_size / max_dim

    S = Matrix.Scale(scale_factor, 4)      
    for obj in objs:
        obj.matrix_world = S @ obj.matrix_world

    bpy.context.view_layer.update()




def is_in_target_collection(obj, target_collection_name):
    for col in obj.users_collection:
        if col.name in target_collection_name:
            return True
        else:
            return False

# PIP: point in polygon: ray casting algorithm
def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def get_first_spawn_asset_location():
    for obj in bpy.data.objects:
        for col in obj.users_collection:
            if col.name == "unique_assets" and ".spawn_asset" in obj.name:
                print(f"Found spawn_asset object: {obj.name}")
                center = get_object_world_center(obj)
                # center = obj.location
                print(f"{obj}'s center is {center}")
                return center
    raise ValueError("No '.spawn_asset' object found in scene.")

def copy_objects_for_render():
    copied_objects = []
    for obj in bpy.data.objects:
        if not obj.hide_render:
            obj_copy = obj.copy()
            if obj.data:
                obj_copy.data = obj.data.copy()
            bpy.context.collection.objects.link(obj_copy)
            copied_objects.append(obj_copy)
    return copied_objects

def cleanup_copied_objects(objects):
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    print(f"[Cleanup] Deleted {len(objects)} copied objects.")

def render_scene():
    bpy.ops.render.render(write_still=True)
    print(f"[Render] Scene rendered to {bpy.context.scene.render.filepath}") 

def generate_semisphere_camera(x, y, z_height, angle):
    radius = z_height  

    xy_offset = radius * math.cos(math.radians(angle))  
    z_offset = radius * math.sin(math.radians(angle))  

    cam1 = Vector((x + xy_offset, y, z_offset)) 
    cam2 = Vector((x - xy_offset, y, z_offset))  

    cam3 = Vector((x, y + xy_offset, z_offset))  
    cam4 = Vector((x, y - xy_offset, z_offset))  

    return [cam1, cam2, cam3, cam4]

def compute_lookat_rotation(cam_location, target_location, up_axis=Vector((0, 0, 1))):

    forward = (cam_location - Vector(target_location)).normalized()

    right = up_axis.cross(forward).normalized()

    up = forward.cross(right).normalized()

    rot_mat = Matrix((right, up, forward)).transposed()

    return rot_mat

    return rot_matrix

def create_persp_camera(cam_location, target_location, cam_name="PerspCam"):
    cam_data = bpy.data.cameras.new(name=cam_name)
    cam_data.type = 'PERSP'
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians(80)

    cam_obj = bpy.data.objects.new(name=cam_name, object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)

    cam_obj.location = Vector(cam_location)

    rot_matrix = compute_lookat_rotation(cam_obj.location, Vector(target_location))
    cam_obj.rotation_euler = rot_matrix.to_euler()

    return cam_obj


def compute_dynamic_fov(cam_location, cam_target, bbox_corners, margin_ratio=1.05):

    cam_forward = (cam_target - cam_location).normalized()
    cam_right = cam_forward.cross(Vector((0,0,1))).normalized()
    cam_up = cam_right.cross(cam_forward).normalized()

    R = Matrix((cam_right, cam_up, cam_forward)).transposed()

    max_angle = 0

    for corner in bbox_corners:
        vec = corner - cam_location
        local_vec = R @ vec

        x = local_vec.x
        y = local_vec.y
        z = local_vec.z

        if z <= 0:
            continue

        horizontal_angle = math.atan(abs(x) / z)
        vertical_angle = math.atan(abs(y) / z)

        angle = max(horizontal_angle, vertical_angle)

        if angle > max_angle:
            max_angle = angle

    fov = 2 * max_angle * margin_ratio

    return fov


def get_obb_corners(obj):
    vertices = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if len(vertices) == 0:
        return None
    points = np.array([[v.x, v.y, v.z] for v in vertices])
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    cov = np.cov(points_centered.T)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    obb_coords = points_centered @ eig_vecs
    min_obb = np.min(obb_coords, axis=0)
    max_obb = np.max(obb_coords, axis=0)
    obb_corners_local = np.array([
        [min_obb[0], min_obb[1], min_obb[2]],
        [min_obb[0], min_obb[1], max_obb[2]],
        [min_obb[0], max_obb[1], min_obb[2]],
        [min_obb[0], max_obb[1], max_obb[2]],
        [max_obb[0], min_obb[1], min_obb[2]],
        [max_obb[0], min_obb[1], max_obb[2]],
        [max_obb[0], max_obb[1], min_obb[2]],
        [max_obb[0], max_obb[1], max_obb[2]],
    ])
    obb_corners_world = obb_corners_local @ eig_vecs.T + centroid
    return obb_corners_world.tolist()

def get_world_aabb(obj):
    local_bbox_corners = [Vector(corner) for corner in obj.bound_box]
    world_bbox_corners = [obj.matrix_world @ corner for corner in local_bbox_corners]
    min_corner = Vector(map(min, zip(*world_bbox_corners)))
    max_corner = Vector(map(max, zip(*world_bbox_corners)))
    return [list(min_corner), list(max_corner)]

def compute_bbox_stable(obj):
    """
    计算单个物体的稳定bounding box，处理极端变换情况
    """
    if obj.type != 'MESH':
        return None, None, None, None
    
    try:
        # 方法1：使用Blender内置的bound_box（更稳定）
        bound_box = obj.bound_box
        if bound_box:
            # bound_box是8个角点的本地坐标，需要转换到世界坐标
            world_coords = [obj.matrix_world @ Vector(corner) for corner in bound_box]
            
            # 检查结果是否合理（所有坐标都应该是有限数值）
            if all(all(np.isfinite([v.x, v.y, v.z])) for v in world_coords):
                xs = [v.x for v in world_coords]
                ys = [v.y for v in world_coords]
                zs = [v.z for v in world_coords]
                
                min_corner = Vector((min(xs), min(ys), min(zs)))
                max_corner = Vector((max(xs), max(ys), max(zs)))
                size = max_corner - min_corner
                center = (min_corner + max_corner) / 2
                
                # 检查尺寸是否合理（不能为负数或无穷大）
                if all(s >= 0 and np.isfinite(s) for s in [size.x, size.y, size.z]):
                    return min_corner, max_corner, size, center
        
        # 方法2：分别处理变换组件（备用方法）
        print(f"[BBox] Warning: Using fallback method for {obj.name}")
        
        # 获取本地空间的顶点
        local_verts = [v.co.copy() for v in obj.data.vertices]
        if not local_verts:
            return None, None, None, None
            
        # 手动应用变换
        location = obj.location
        rotation = obj.rotation_euler
        scale = obj.scale
        
        # 检查变换参数是否合理
        if any(abs(s) < 1e-6 for s in scale):  # 避免除零或极小缩放
            print(f"[BBox] Warning: Extreme scale detected for {obj.name}: {scale}")
            scale = Vector((max(abs(s), 1e-3) * (1 if s >= 0 else -1) for s in scale))
        
        # 构建变换矩阵（更可控）
        scale_matrix = Matrix.Scale(scale.x, 4, (1, 0, 0)) @ \
                      Matrix.Scale(scale.y, 4, (0, 1, 0)) @ \
                      Matrix.Scale(scale.z, 4, (0, 0, 1))
        
        rotation_matrix = rotation.to_matrix().to_4x4()
        translation_matrix = Matrix.Translation(location)
        
        transform_matrix = translation_matrix @ rotation_matrix @ scale_matrix
        
        # 应用变换
        world_verts = [transform_matrix @ v for v in local_verts]
        
        # 计算bounding box
        xs = [v.x for v in world_verts]
        ys = [v.y for v in world_verts]
        zs = [v.z for v in world_verts]
        
        min_corner = Vector((min(xs), min(ys), min(zs)))
        max_corner = Vector((max(xs), max(ys), max(zs)))
        size = max_corner - min_corner
        center = (min_corner + max_corner) / 2
        
        return min_corner, max_corner, size, center
        
    except Exception as e:
        print(f"[BBox] Error computing bbox for {obj.name}: {e}")
        # 最后的备用方法：使用物体位置和一个最小bounding box
        location = obj.location
        min_size = 0.01  # 最小尺寸
        min_corner = location - Vector((min_size, min_size, min_size))
        max_corner = location + Vector((min_size, min_size, min_size))
        size = max_corner - min_corner
        center = location
        return min_corner, max_corner, size, center

def compute_bbox(wall_objs, scale=1.0):
    """
    改进的bbox计算函数，使用更稳定的算法
    """
    if not wall_objs:
        raise ValueError("No objects provided for bbox computation.")
    
    valid_boxes = []
    for obj in wall_objs:
        bbox_result = compute_bbox_stable(obj)
        if bbox_result[0] is not None:  # 检查是否有效
            valid_boxes.append(bbox_result)
        else:
            print(f"[BBox] Warning: Failed to compute bbox for {obj.name}")
    
    if not valid_boxes:
        raise ValueError("No valid bounding boxes found.")
    
    # 合并所有有效的bounding box
    all_min_corners = [box[0] for box in valid_boxes]
    all_max_corners = [box[1] for box in valid_boxes]
    
    # 计算总体的min和max
    global_min = Vector((
        min(corner.x for corner in all_min_corners),
        min(corner.y for corner in all_min_corners),
        min(corner.z for corner in all_min_corners)
    ))
    
    global_max = Vector((
        max(corner.x for corner in all_max_corners),
        max(corner.y for corner in all_max_corners),
        max(corner.z for corner in all_max_corners)
    ))
    
    size = global_max - global_min
    center = (global_min + global_max) / 2
    
    # 应用scale参数（如果需要）
    if scale != 1.0:
        global_min = center + (global_min - center) * scale
        global_max = center + (global_max - center) * scale
        size = global_max - global_min
    
    # 最终稳定性检查
    if not all(np.isfinite([global_min.x, global_min.y, global_min.z, 
                           global_max.x, global_max.y, global_max.z])):
        print("[BBox] Warning: Invalid bbox result, using default")
        center = Vector((0, 0, 0))
        size = Vector((1, 1, 1))
        global_min = center - size/2
        global_max = center + size/2
    
    print(f"[BBox] Final bbox: min={global_min}, max={global_max}, size={size}, center={center}")
    
    return global_min, global_max, size, center

def compute_bbox_volume(min_corner, max_corner):
    length = max_corner.x - min_corner.x
    width = max_corner.y - min_corner.y
    height = max_corner.z - min_corner.z
    volume = length * width * height
    return volume
def compute_bbox_area(min_corner, max_corner):
    length = max_corner.x - min_corner.x
    width = max_corner.y - min_corner.y
    height = max_corner.z - min_corner.z
    area1 = length * width
    area2 = length * height
    area3 = width * height
    return area1, area2, area3

def top_ranked_objects_by_bbox_volume_area(objs, rank_num):
    heap_volume = []
    heap_area = []
    obj_info = {}  # record (volume, area)

    for obj in objs:
        name_low = obj.name.lower()
        if "door" in name_low or "window" in name_low:
            continue
        elif "table" not in name_low and "cabinet" not in name_low and "shelf" not in name_low and "bookcase" not in name_low:
            continue
        elif "trinkets" in name_low:
            continue
        try:
            min_corner, max_corner = compute_bbox([obj])[:2]
            volume = compute_bbox_volume(min_corner, max_corner)
            area = max(compute_bbox_area(min_corner, max_corner))
        except Exception as e:
            print(f"[Warning] Failed to compute bbox for {obj.name}: {e}")
            continue

        obj_info[obj] = (volume, area)

        if len(heap_volume) < rank_num:
            heapq.heappush(heap_volume, (volume, id(obj), obj))
        else:
            heapq.heappushpop(heap_volume, (volume, id(obj), obj))

        if len(heap_area) < rank_num:
            heapq.heappush(heap_area, (area, id(obj), obj))
        else:
            heapq.heappushpop(heap_area, (area, id(obj), obj))

    top_volume_objs = set(obj for _, __, obj in heap_volume)
    top_area_objs = set(obj for _, __, obj in heap_area)

    combined_objs = top_volume_objs.union(top_area_objs)

    result = []
    for obj in combined_objs:
        volume, area = obj_info[obj]
        result.append([obj, volume, area])

    result.sort(key=lambda x: (-x[1], -x[2]))
    return result

def add_captions_for_objects(objs_with_info, font_size=0.3, z_offset=10.0):
    """Add 3D captions to Blender scene for given objects"""
    created_captions = []

    for obj, volume, area in objs_with_info:
        min_corner, max_corner, _, center = compute_bbox([obj])

        object_height = max_corner.z - min_corner.z
        caption_z = max_corner.z + z_offset

        text_data = bpy.data.curves.new(name=f"Text_{obj.name}", type='FONT')
        text_data.body = obj.name.split("Factory")[0]
        text_data.size = font_size
        text_data.align_x = 'CENTER'
        text_data.align_y = 'CENTER'

        text_obj = bpy.data.objects.new(name=f"Caption_{obj.name}", object_data=text_data)
        text_obj.location = Vector((center.x, center.y, caption_z))
        text_obj.rotation_euler = (0, 0, 0) 
        text_obj.hide_render = False
        text_obj.hide_viewport = False

        bpy.context.collection.objects.link(text_obj)

        mat_name = "Caption_Red"
        if mat_name not in bpy.data.materials:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            for node in nodes:
                nodes.remove(node)
            
            emission = nodes.new(type='ShaderNodeEmission')
            emission.inputs[0].default_value = (1, 0, 0, 1) 
            emission.inputs[1].default_value = 5.0 
            
            output = nodes.new(type='ShaderNodeOutputMaterial')
            
            links.new(emission.outputs[0], output.inputs[0])
        else:
            mat = bpy.data.materials[mat_name]

        if text_obj.data.materials:
            text_obj.data.materials[0] = mat
        else:
            text_obj.data.materials.append(mat)

        created_captions.append(text_obj)

    return created_captions

def remove_all_captions():
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Caption_") and obj.type == 'FONT':
            bpy.data.objects.remove(obj, do_unlink=True)