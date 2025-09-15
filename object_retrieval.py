import bpy
import json
import os
import math
import sys
from mathutils import Vector, Euler, Matrix
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from util import compute_bbox
from glb_indexer import load_glb_index, find_best_matching_glb



SCENE_SCALE = 1.0

ROTATION_INSENSITIVE_CATEGORIES = {
    'bag', 'box', 'balloon', 'bowl', 'bottle', 'can', 'cup', 'jar', 'plate',
    'pot', 'pan', 'spoon', 'fork', 'chopsticks', 'wineglass', 'food_bag',
    'food_box', 'container', 'clutter', 'hardware',
    'cushion', 'decoration', 'statue', 'vase', 'picture', 'wall_art',
    'nature_shelf_trinkets',
    'plant', 'large_plant_container', 'plant_container',
    'toiletry', 'towel', 'toilet_paper',
    'handle', 'light_switch', 'vent', 'fan', 'lighting', 'appliances','basket', 'clothes', 'wine_cabinet', 'ceiling_lamp', 'desk_lamp', 'floor_lamp', 'fruit_container', 'trashcan',
    'gym_equipment'
}

# ========= 新增：按“类别”定义 Trellis → Infinigen 的“预对齐旋转” =========
PRE_ROTATION_BY_CATEGORY = {
    'toilet':  0
    # future objects
}

def rotate_object_world_z(obj, angle_rad: float):
    """在世界坐标系下绕 Z 轴旋转物体（左乘世界矩阵），稳且不受 rotation_mode 影响。"""
    if abs(angle_rad) < 1e-8:
        return
    Rz = Matrix.Rotation(angle_rad, 4, 'Z')
    obj.matrix_world = Rz @ obj.matrix_world
    bpy.context.view_layer.update()

def norm_cat(category: str) -> str:
    return (category or "").strip().lower()

def get_pre_rotation_for_category(category: str) -> Euler:
    cat = (category or "").lower()
    rz = PRE_ROTATION_BY_CATEGORY.get(cat, 0.0)
    return Euler((0.0, 0.0, rz), 'XYZ')

def clear_scene():
    print("[Clear Scene] Starting scene cleanup...")
    bpy.context.scene.camera = None
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.textures):
        bpy.data.textures.remove(block)
    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)
    for block in list(bpy.data.lights):
        bpy.data.lights.remove(block)
    for block in list(bpy.data.cameras):
        bpy.data.cameras.remove(block)
    print("[Clear Scene] Scene cleared successfully")

def import_glb_object(glb_path):
    if not os.path.exists(glb_path):
        print(f"Error: GLB file {glb_path} does not exist")
        return None

    objects_before = set(bpy.context.scene.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=glb_path)
    except Exception as e:
        print(f"Error importing {glb_path}: {e}")
        return None

    objects_after = set(bpy.context.scene.objects)
    new_objects = objects_after - objects_before
    if not new_objects:
        print(f"Warning: No objects imported from {glb_path}")
        return None

    useful_objects = []
    for obj in new_objects:
        if obj.type == 'MESH':
            useful_objects.append(obj)
        else:
            bpy.data.objects.remove(obj, do_unlink=True)

    if not useful_objects:
        print(f"Warning: No mesh objects imported from {glb_path}")
        return None

    if len(useful_objects) == 1:
        single_mesh = useful_objects[0]
        print(f"  - Single mesh imported: {single_mesh.name}")
        bpy.context.view_layer.objects.active = single_mesh
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        return single_mesh
    else:
        print(f"  - Multiple meshes found, merging {len(useful_objects)} objects")
        bpy.ops.object.select_all(action='DESELECT')
        for obj in useful_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = useful_objects[0]
        bpy.ops.object.join()
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
        merged_obj = bpy.context.active_object
        print(f"  - Merged into single object: {merged_obj.name}")
        return merged_obj

def apply_object_transformations(obj, location, size, rotation, category=None):
    """
    应用物体的变换（顺序：预旋转 -> JSON旋转(可选) -> 统一缩放 -> 平移）
    - 旋转：在世界坐标系下左乘 Rz，避免被 rotation_mode/父子关系等吞掉。
    - 对“旋转不敏感”类别：跳过旋转（预旋转 + JSON旋转），只做缩放 + 平移。
    """
    try:
        print(f"  - Applying transformations to {obj.name}")

        cat_lower = norm_cat(category)
        rotation_sensitive = cat_lower not in ROTATION_INSENSITIVE_CATEGORIES

        print(f"    Category='{category}' -> '{cat_lower}'")
        print(f"    JSON location: [{location[0]:.6f}, {location[1]:.6f}, {location[2]:.6f}]")
        print(f"    JSON size    : [{size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}]")
        print(f"    JSON rotation: {rotation:.6f} rad ({rotation*180/math.pi:.1f}°)")


        def get_pre_rz(cat: str) -> float:
            c = norm_cat(cat)
            if c in PRE_ROTATION_BY_CATEGORY:
                return PRE_ROTATION_BY_CATEGORY[c]
            return 0.0

        # ---------- Step 1: 预旋转（仅旋转敏感） ----------
        if rotation_sensitive:
            rz_pre = float(get_pre_rz(cat_lower))
            if abs(rz_pre) > 1e-6:
                before = obj.matrix_world.copy()
                rotate_object_world_z(obj, rz_pre)
                print(f"    Pre-rotation('{cat_lower}'): Rz={rz_pre:.6f} rad ({rz_pre*180/math.pi:.1f}°)")
            else:
                print(f"    No pre-rotation matched for '{cat_lower}'")
        else:
            print(f"    '{cat_lower}' is rotation-insensitive: skip pre-rotation & JSON rotation")

        # ---------- Step 2: JSON 旋转（叠加；仅旋转敏感） ----------
        if rotation_sensitive and abs(rotation) > 1e-6:
            rotate_object_world_z(obj, float(rotation))
            print(f"    JSON rotation applied: +{rotation:.6f} rad ({rotation*180/math.pi:.1f}°)")
        elif rotation_sensitive:
            print(f"    JSON rotation too small, skipped")

        # ---------- Step 3: 旋转后的 bbox（用于缩放） ----------
        rotated_min, rotated_max, rotated_size, rotated_center = compute_bbox([obj])
        print(f"    Rotated bbox: size={rotated_size}, center={rotated_center}")

        # === 关键：JSON 的 size 是 half-extents；且 (w,h,d) 在世界坐标映射为 (X,Y,Z) = (w,d,h) ===
        target_size_x = 2.0 * float(size[0])  # width  (w) -> X
        target_size_y = 2.0 * float(size[2])  # depth  (d) -> Y  ← 与之前相反
        target_size_z = 2.0 * float(size[1])  # height (h) -> Z  ← 与之前相反
        print(f"    Target FULL size (W,D,H→X,Y,Z): "
            f"[{target_size_x:.6f}, {target_size_y:.6f}, {target_size_z:.6f}]")

        # ---------- Step 4: 缩放（按类别决定：逐轴 or 等比） ----------
        # 逐轴缩放的类别（桌/柜/架类等）：长度、宽度、高度分别匹配
        ANISOTROPIC_SCALE_CATEGORIES = {
            'cabinet', 'large_shelf', 'cell_shelf',
            'kitchen_cabinet',  'children_cabinet'
            'dining_table', 'dressing_table', 'coffee_table',
            'console_table', 'corner_side_table', 'round_end_table',
            'table', 'bookshelf', 'shelf', 'rug'
        }
        use_anisotropic = cat_lower in ANISOTROPIC_SCALE_CATEGORIES
        eps = 1e-12
        
        if use_anisotropic:
            # 各轴分别缩放，正好贴合目标 bbox
            sx = target_size_x / max(rotated_size.x, eps)
            sy = target_size_y / max(rotated_size.y, eps)
            sz = target_size_z / max(rotated_size.z, eps)
            before_scale = tuple(obj.scale)
            obj.scale = (sx, sy, sz)
            bpy.context.view_layer.update()
            print(f"    Anisotropic scale applied for '{cat_lower}': "
                  f"sx={sx:.6f}, sy={sy:.6f}, sz={sz:.6f}  (scale before={before_scale})")
        else:
            # 等比缩放（不超尺寸，保持原比例）
            scale_ratio_x = target_size_x / max(rotated_size.x, eps)
            scale_ratio_y = target_size_y / max(rotated_size.y, eps)
            scale_ratio_z = target_size_z / max(rotated_size.z, eps)
            uniform_scale = min(scale_ratio_x, scale_ratio_y, scale_ratio_z)
            before_scale = tuple(obj.scale)
            obj.scale = (uniform_scale, uniform_scale, uniform_scale)
            bpy.context.view_layer.update()
            print(f"    Uniform scale applied: {uniform_scale:.6f}  scale(before)={before_scale}")
            print(f"    Scale ratios: X={scale_ratio_x:.6f}, Y={scale_ratio_y:.6f}, Z={scale_ratio_z:.6f}")


        # ---------- Step 5: 缩放后的 bbox ----------
        current_min, current_max, current_size, current_center = compute_bbox([obj])
        print(f"    Scaled bbox: size={current_size}, center={current_center}")


        # ---------- Step 6: 目标中心（你的 [x,z,y] 映射保持不变） ----------
        target_center = Vector((
            location[0] * SCENE_SCALE,  # x
            location[2] * SCENE_SCALE,  # z -> y
            location[1] * SCENE_SCALE   # y -> z
        ))
        print(f"    Target center: {target_center} (axis map [x,z,y])")

        # ---------- Step 7: 平移到目标中心 ----------
        move_distance = target_center - current_center
        before_loc = tuple(obj.location)
        obj.location = obj.location + move_distance
        bpy.context.view_layer.update()
        print(f"    Translation applied: Δ={move_distance}  loc(before)={before_loc} -> (after)={tuple(obj.location)}")

        
        # ---------- Step 8: 校验 ----------
        final_min, final_max, final_size, final_center = compute_bbox([obj])
        alignment_error = (final_center - target_center).length
        print(f"    Final bbox: size={final_size}, center={final_center}")
        print(f"    Alignment error: {alignment_error:.8f}")

        size_fit_x = final_size.x <= target_size_x + 1e-3
        size_fit_y = final_size.y <= target_size_y + 1e-3
        size_fit_z = final_size.z <= target_size_z + 1e-3
        print(f"    Size fit: X={size_fit_x}, Y={size_fit_y}, Z={size_fit_z}")

        actual_scale_x = final_size.x / target_size_x if target_size_x > 0 else 1.0
        actual_scale_y = final_size.y / target_size_y if target_size_y > 0 else 1.0
        actual_scale_z = final_size.z / target_size_z if target_size_z > 0 else 1.0
        print(f"    Actual vs Target scale: X={actual_scale_x:.6f}, "
              f"Y={actual_scale_y:.6f}, Z={actual_scale_z:.6f}")

        return True

    except Exception as e:
        print(f"    Error applying transformations: {e}")
        return False


def setup_topdown_camera(objects_to_render, z_margin=5.0, resolution=(1024, 1024)):
    print("[Camera Setup] Setting up top-down camera...")
    min_corner, max_corner, size, center = compute_bbox(objects_to_render)
    print(f"  Scene bounds: min={min_corner}, max={max_corner}")
    print(f"  Scene size: {size}")
    print(f"  Scene center: {center}")
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "TopDownCamera"
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = max(size.x, size.y) * 1.2
    camera.location = (center.x, center.y, center.z + max(size.z, 1.0) + z_margin)
    camera.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = camera
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    print(f"  Camera created: {camera.name}")
    print(f"  Camera location: {camera.location}")
    print(f"  Camera ortho scale: {camera.data.ortho_scale}")
    print(f"  Render resolution: {resolution[0]}x{resolution[1]}")
    return camera

def setup_render_settings():
    print("[Render Setup] Configuring render settings...")
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128
    bpy.context.scene.view_layers[0].use_pass_z = True
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_nodes.clear()
    background = world_nodes.new(type='ShaderNodeBackground')
    background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # 纯白色背景
    background.inputs[1].default_value = 1.0
    output = world_nodes.new(type='ShaderNodeOutputWorld')
    bpy.context.scene.world.node_tree.links.new(background.outputs[0], output.inputs[0])
    print("  Render engine: Cycles")
    print("  Samples: 128")
    print("  World lighting: Configured")

def setup_lighting():
    print("[Lighting Setup] Adding lights...]")
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.name = "SunLight"
    sun.data.energy = 3.0
    sun.rotation_euler = (0, 0, 0)
    bpy.ops.object.light_add(type='AREA', location=(5, 5, 8))
    area_light = bpy.context.active_object
    area_light.name = "FillLight"
    area_light.data.energy = 2.0
    area_light.data.size = 5.0
    print(f"  Sun light: {sun.name} (energy: {sun.data.energy})")
    print(f"  Fill light: {area_light.name} (energy: {area_light.data.energy})")

def apply_random_colors_to_objects():
    """为场景中的所有物体应用随机纯色材质"""
    print("[Color Setup] Applying random colors to objects...")
    
    # 预定义的颜色列表（深色高对比度）
    colors = [
        # 深红色系
        (0.6, 0.0, 0.0, 1.0),    # 深红色
        (0.8, 0.1, 0.1, 1.0),    # 暗红色
        (0.4, 0.0, 0.0, 1.0),    # 极深红色
        
        # 深绿色系
        (0.0, 0.5, 0.0, 1.0),    # 深绿色
        (0.0, 0.7, 0.0, 1.0),    # 暗绿色
        (0.0, 0.3, 0.0, 1.0),    # 极深绿色
        
        # 深蓝色系
        (0.0, 0.0, 0.6, 1.0),    # 深蓝色
        (0.0, 0.0, 0.8, 1.0),    # 暗蓝色
        (0.0, 0.0, 0.4, 1.0),    # 极深蓝色
        
        # 深紫色系
        (0.4, 0.0, 0.6, 1.0),    # 深紫色
        (0.6, 0.0, 0.8, 1.0),    # 暗紫色
        (0.2, 0.0, 0.4, 1.0),    # 极深紫色
        
        # 深橙色系
        (0.7, 0.3, 0.0, 1.0),    # 深橙色
        (0.8, 0.4, 0.0, 1.0),    # 暗橙色
        (0.5, 0.2, 0.0, 1.0),    # 极深橙色
        
        # 深青色系
        (0.0, 0.5, 0.5, 1.0),    # 深青色
        (0.0, 0.7, 0.7, 1.0),    # 暗青色
        (0.0, 0.3, 0.3, 1.0),    # 极深青色
        
        # 深黄色系
        (0.6, 0.6, 0.0, 1.0),    # 深黄色
        (0.8, 0.8, 0.0, 1.0),    # 暗黄色
        (0.4, 0.4, 0.0, 1.0),    # 极深黄色
        
        # 深洋红色系
        (0.6, 0.0, 0.6, 1.0),    # 深洋红色
        (0.8, 0.0, 0.8, 1.0),    # 暗洋红色
        (0.4, 0.0, 0.4, 1.0),    # 极深洋红色
        
        # 深棕色系
        (0.4, 0.2, 0.0, 1.0),    # 深棕色
        (0.6, 0.3, 0.0, 1.0),    # 暗棕色
        (0.2, 0.1, 0.0, 1.0),    # 极深棕色
        
        # 深灰色系
        (0.2, 0.2, 0.2, 1.0),    # 深灰色
        (0.3, 0.3, 0.3, 1.0),    # 暗灰色
        (0.1, 0.1, 0.1, 1.0),    # 极深灰色
        
        # 深橄榄色系
        (0.3, 0.4, 0.0, 1.0),    # 深橄榄色
        (0.4, 0.5, 0.0, 1.0),    # 暗橄榄色
        (0.2, 0.3, 0.0, 1.0),    # 极深橄榄色
        
        # 深青绿色系
        (0.0, 0.4, 0.3, 1.0),    # 深青绿色
        (0.0, 0.6, 0.4, 1.0),    # 暗青绿色
        (0.0, 0.2, 0.2, 1.0),    # 极深青绿色
        
        # 深粉红色系
        (0.6, 0.2, 0.4, 1.0),    # 深粉红色
        (0.8, 0.3, 0.5, 1.0),    # 暗粉红色
        (0.4, 0.1, 0.2, 1.0),    # 极深粉红色
        
        # 深靛蓝色系
        (0.2, 0.0, 0.6, 1.0),    # 深靛蓝色
        (0.3, 0.0, 0.8, 1.0),    # 暗靛蓝色
        (0.1, 0.0, 0.4, 1.0),    # 极深靛蓝色
        
        # 深珊瑚色系
        (0.8, 0.3, 0.2, 1.0),    # 深珊瑚色
        (0.9, 0.4, 0.3, 1.0),    # 暗珊瑚色
        (0.6, 0.2, 0.1, 1.0),    # 极深珊瑚色
        
        # 深薄荷色系
        (0.0, 0.6, 0.4, 1.0),    # 深薄荷色
        (0.0, 0.8, 0.5, 1.0),    # 暗薄荷色
        (0.0, 0.4, 0.3, 1.0),    # 极深薄荷色
        
        # 深金色系
        (0.6, 0.5, 0.0, 1.0),    # 深金色
        (0.8, 0.6, 0.0, 1.0),    # 暗金色
        (0.4, 0.3, 0.0, 1.0),    # 极深金色
        
        # 深银灰色系
        (0.4, 0.4, 0.4, 1.0),    # 深银灰色
        (0.5, 0.5, 0.5, 1.0),    # 暗银灰色
        (0.3, 0.3, 0.3, 1.0),    # 极深银灰色
    ]
    
    # 获取所有网格物体
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        print("  No mesh objects found to color")
        return
    
    print(f"  Found {len(mesh_objects)} mesh objects to color")
    
    # 为每个物体分配随机颜色
    import random
    colored_count = 0
    
    for obj in mesh_objects:
        try:
            # 选择随机颜色
            color = random.choice(colors)
            
            # 创建新材质
            material_name = f"RandomColor_{obj.name}"
            material = bpy.data.materials.new(name=material_name)
            material.use_nodes = True
            
            # 清除默认节点
            material.node_tree.nodes.clear()
            
            # 添加输出节点
            output_node = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
            
            # 添加原理化BSDF节点
            bsdf_node = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
            
            # 设置颜色
            bsdf_node.inputs['Base Color'].default_value = color
            
            # 连接节点
            material.node_tree.links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
            
            # 将材质分配给物体
            if obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)
            
            # 确保物体使用材质
            obj.active_material = material
            
            colored_count += 1
            print(f"  ✓ {obj.name} -> {color[:3]} (RGB)")
            
        except Exception as e:
            print(f"  ✗ Failed to color {obj.name}: {e}")
    
    print(f"  Successfully colored {colored_count}/{len(mesh_objects)} objects")
    return colored_count

def render_topdown_scene(filepath):
    print("[Render] Starting top-down render...")
    try:
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        print(f"  RGB render saved: {filepath}")
        return True
    except Exception as e:
        print(f"  Error during rendering: {e}")
        return False

def save_scene_file(filepath):
    print("[Save] Saving scene file...")
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=filepath)
        print(f"  Scene saved: {filepath}")
        return True
    except Exception as e:
        print(f"  Error saving scene: {e}")
        return False

def process_scene(scene_data, scene_index, output_base_dir, save_blend=True):
    """处理单个场景"""
    print("=" * 80)
    print(f"Processing scene {scene_index}")
    print("=" * 80)
    
    clear_scene()
    glb_index_path = "/root/autodl-tmp/zyh/retriever/glb_index.json"
    glb_index = load_glb_index(glb_index_path) or None

    # 解析新的JSON格式
    class_names = scene_data.get('class_names', [])
    translations_raw = scene_data.get('translations', [])
    sizes_raw = scene_data.get('sizes', [])
    angles_raw = scene_data.get('angles', [])
    
    # 处理嵌套数组格式 - 取第一个元素（如果存在）
    translations = translations_raw[0] if translations_raw and len(translations_raw) > 0 else []
    sizes = sizes_raw[0] if sizes_raw and len(sizes_raw) > 0 else []
    angles = angles_raw[0] if angles_raw and len(angles_raw) > 0 else []
    
    print(f"Number of objects: {len(class_names)}")
    print(f"Translations length: {len(translations)}")
    print(f"Sizes length: {len(sizes)}")
    print(f"Angles length: {len(angles)}")
    
    # 确保所有数组长度一致
    min_length = min(len(class_names), len(translations), len(sizes), len(angles))
    if min_length == 0:
        print("No objects found in scene")
        return False
    
    successful_imports = 0
    obj_folder = "/root/autodl-tmp/zyh/retriever/obj"

    for i in range(min_length):
        category = class_names[i]
        location = translations[i]
        size = sizes[i]
        rotation_raw = angles[i]
        
        # 处理rotation可能是列表的情况
        if isinstance(rotation_raw, list) and len(rotation_raw) > 0:
            rotation = rotation_raw[0]  # 取第一个元素
        else:
            rotation = rotation_raw

        print(f"\n[Object {i+1}/{min_length}] Processing {category}")
        print(f"  Location: {location}")
        print(f"  Size: {size}")
        print(f"  Rotation: {math.degrees(rotation):.1f}°")

        best_glb = None
        if glb_index:
            best_glb = find_best_matching_glb(category, size, glb_index)
        if not best_glb:
            print(f"  - Using random selection for {category}")
            class_folder = os.path.join(obj_folder, category)
            if os.path.exists(class_folder):
                glb_files = [f for f in os.listdir(class_folder) if f.endswith('.glb')]
                if glb_files:
                    import random
                    best_glb = os.path.join(class_folder, random.choice(glb_files))
        if not best_glb:
            print(f"  ✗ No GLB file found for {category}")
            continue

        print(f"  - Selected GLB: {os.path.basename(best_glb)}")
        obj = import_glb_object(best_glb)
        if not obj:
            print(f"  ✗ Failed to import GLB file")
            continue

        if apply_object_transformations(obj, location, size, rotation, category):
            successful_imports += 1
            print(f"  ✓ Object {category} imported and positioned successfully")
        else:
            print(f"  ✗ Failed to position object {category}")

    print(f"\n[Summary] Successfully imported {successful_imports}/{min_length} objects")
    if successful_imports == 0:
        print("No objects imported, cannot proceed with rendering")
        return False

    all_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    camera = setup_topdown_camera(all_objects)
    setup_lighting()
    setup_render_settings()
    
    # 为所有物体应用随机纯色材质
    apply_random_colors_to_objects()

    # 设置输出路径
    output_folder = os.path.join(output_base_dir, f"scene_{scene_index}")
    os.makedirs(output_folder, exist_ok=True)
    scene_file_path = os.path.join(output_folder, f"scene_{scene_index}.blend")
    render_output_path = os.path.join(output_folder, f"scene_{scene_index}.png")

    # 根据开关决定是否保存.blend文件
    if save_blend:
        if save_scene_file(scene_file_path):
            print("✓ Scene file saved successfully")
        else:
            print("✗ Failed to save scene file")
    else:
        print("⚠ Scene file saving disabled (save_blend=False)")

    if render_topdown_scene(render_output_path):
        print("✓ Top-down render completed successfully")
    else:
        print("✗ Failed to render top-down view")

    print(f"\n[Final Summary]")
    print(f"  Objects imported: {successful_imports}/{min_length}")
    if save_blend:
        print(f"  Scene file: {scene_file_path}")
    print(f"  Render output: {render_output_path}")
    print(f"  Output folder: {output_folder}")
    return True

def process_scenes_batch(scenes_data, output_base_dir, save_blend=True):
    """批量处理多个场景"""
    print("=" * 80)
    print(f"Starting batch processing of {len(scenes_data)} scenes")
    print(f"Save .blend files: {'Yes' if save_blend else 'No'}")
    print("=" * 80)
    
    os.makedirs(output_base_dir, exist_ok=True)
    success_count = 0
    
    for i, scene_data in enumerate(scenes_data):
        try:
            print(f"\n{'='*20} Processing Scene {i+1}/{len(scenes_data)} {'='*20}")
            if process_scene(scene_data, i, output_base_dir, save_blend):
                success_count += 1
                print(f"✓ Scene {i} processed successfully")
            else:
                print(f"✗ Scene {i} failed to process")
        except Exception as e:
            print(f"✗ Scene {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Batch processing completed: {success_count}/{len(scenes_data)} scenes successful")
    print(f"{'='*80}")
    
    return success_count

def process_legacy_scene():
    """保持原有的单场景处理函数以兼容性"""
    print("=" * 80)
    print("Processing 765a3932.json scene")
    print("=" * 80)
    clear_scene()
    glb_index_path = "/root/autodl-tmp/zyh/retriever/glb_index.json"
    glb_index = load_glb_index(glb_index_path) or None

    json_path = "/root/autodl-tmp/zyh/retriever/json/20c17bd9.json"
    try:
        with open(json_path, 'r') as f:
            layout_data = json.load(f)
        print(f"Loaded layout file: {json_path}")
    except Exception as e:
        print(f"Error loading layout file: {e}")
        return False

    if 'scenes' not in layout_data or not layout_data['scenes']:
        print("No scenes found in layout file")
        return False

    scene = layout_data['scenes'][0]
    scene_id = scene.get('scene_id', 'unknown')
    objects = scene.get('objects', [])
    print(f"Scene ID: {scene_id}")
    print(f"Number of objects: {len(objects)}")

    successful_imports = 0
    obj_folder = "/root/autodl-tmp/zyh/retriever/obj"

    for i, obj_data in enumerate(objects):
        category = obj_data.get('category', 'unknown')
        location = obj_data.get('location', [0, 0, 0])
        size = obj_data.get('size', [1, 1, 1])
        rotation = obj_data.get('rotation', 0.0)

        print(f"\n[Object {i+1}/{len(objects)}] Processing {category}")
        print(f"  Location: {location}")
        print(f"  Size: {size}")
        print(f"  Rotation: {math.degrees(rotation):.1f}°")

        best_glb = None
        if glb_index:
            best_glb = find_best_matching_glb(category, size, glb_index)
        if not best_glb:
            print(f"  - Using random selection for {category}")
            class_folder = os.path.join(obj_folder, category)
            if os.path.exists(class_folder):
                glb_files = [f for f in os.listdir(class_folder) if f.endswith('.glb')]
                if glb_files:
                    import random
                    best_glb = os.path.join(class_folder, random.choice(glb_files))
        if not best_glb:
            print(f"  ✗ No GLB file found for {category}")
            continue

        print(f"  - Selected GLB: {os.path.basename(best_glb)}")
        obj = import_glb_object(best_glb)
        if not obj:
            print(f"  ✗ Failed to import GLB file")
            continue

        if apply_object_transformations(obj, location, size, rotation, category):
            successful_imports += 1
            print(f"  ✓ Object {category} imported and positioned successfully")
        else:
            print(f"  ✗ Failed to position object {category}")

    print(f"\n[Summary] Successfully imported {successful_imports}/{len(objects)} objects")
    if successful_imports == 0:
        print("No objects imported, cannot proceed with rendering")
        return False

    all_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    camera = setup_topdown_camera(all_objects)
    setup_lighting()
    setup_render_settings()
    
    # 为所有物体应用随机纯色材质
    apply_random_colors_to_objects()

    # 使用默认输出路径
    output_folder = "/root/autodl-tmp/zyh/retriever/output_765a3932_1"
    os.makedirs(output_folder, exist_ok=True)
    scene_file_path = os.path.join(output_folder, "scene_765a3932.blend")
    render_output_path = os.path.join(output_folder, "topdown_render.png")

    if save_scene_file(scene_file_path):
        print("✓ Scene file saved successfully")
    else:
        print("✗ Failed to save scene file")

    if render_topdown_scene(render_output_path):
        print("✓ Top-down render completed successfully")
    else:
        print("✗ Failed to render top-down view")

    print(f"\n[Final Summary]")
    print(f"  Objects imported: {successful_imports}/{len(objects)}")
    print(f"  Scene file: {scene_file_path}")
    print(f"  Render output: {render_output_path}")
    print(f"  Output folder: {output_folder}")
    return True

def main_legacy():
    print("Scene Processor")
    print("=" * 80)
    success = process_legacy_scene()
    if success:
        print("\n✓ Scene processing completed successfully!")
    else:
        print("\n✗ Scene processing failed!")
    return success

def test_rotation_logic():
    print("Testing rotation logic...")
    test_cases = [
        ("chair", True),
        ("table", True),
        ("bag", False),
        ("box", False),
        ("cup", False),
        ("vase", False),
        ("plant", False),
        ("cushion", False),
        ("lighting", False),
        ("sofa", True),
    ]
    print("\nRotation sensitivity test results:")
    print("-" * 50)
    all_passed = True
    for category, expected_sensitive in test_cases:
        is_sensitive = category.lower() not in ROTATION_INSENSITIVE_CATEGORIES
        status = "✓" if is_sensitive == expected_sensitive else "✗"
        sensitive_text = "rotation-sensitive" if is_sensitive else "rotation-insensitive"
        print(f"{status} {category:<15} -> {sensitive_text}")
        if is_sensitive != expected_sensitive:
            all_passed = False
    print("-" * 50)
    print("✓ All test cases passed!" if all_passed else "✗ Some test cases failed!")
    return all_passed

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_rotation_logic()
    else:
        main_legacy()
