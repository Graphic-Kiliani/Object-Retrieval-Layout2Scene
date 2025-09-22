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
# path to your colors_mapping.json
COLORS_JSON_PATH = "path_to_mapping_color_json"

# Objects which is robust to rotation
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

# ========= Trellis → Infinigen: Pre rotation =========
PRE_ROTATION_BY_CATEGORY = {
    'toilet':  0
    # future objects
}

# ====== Import category->RGB from json======


def rotate_object_world_z(obj, angle_rad: float):
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
    Apply object transformations in the following order: pre-rotation → JSON rotation (optional) → uniform scaling → translation.
    Rotation: Left-multiply R_z in world coordinates to avoid issues caused by rotation_mode or parent-child relationships.
    For "rotation-insensitive" categories: Skip all rotations (both pre-rotation and JSON rotation), and only perform scaling and translation.
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

        # ---------- Step 1: Pre Rotation (For rotation sensitive objects) ----------
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

        # ---------- Step 2: JSON rotation（For rotation sensitive objects） ----------
        if rotation_sensitive and abs(rotation) > 1e-6:
            rotate_object_world_z(obj, float(rotation))
            print(f"    JSON rotation applied: +{rotation:.6f} rad ({rotation*180/math.pi:.1f}°)")
        elif rotation_sensitive:
            print(f"    JSON rotation too small, skipped")

        # ---------- Step 3: bbox after rotation (For scaling) ----------
        rotated_min, rotated_max, rotated_size, rotated_center = compute_bbox([obj])
        print(f"    Rotated bbox: size={rotated_size}, center={rotated_center}")

        # === JSON's size is half-extentt; and (w,h,d) is mapped to (X,Y,Z) = (w,d,h) in world coordinate system ===
        target_size_x = 2.0 * float(size[0])  # width  (w) -> X
        target_size_y = 2.0 * float(size[2])  # depth  (d) -> Y  
        target_size_z = 2.0 * float(size[1])  # height (h) -> Z  
        print(f"    Target FULL size (W,D,H→X,Y,Z): "
            f"[{target_size_x:.6f}, {target_size_y:.6f}, {target_size_z:.6f}]")

        # ---------- Step 4: scaling（By category：by axis or by proportion） ----------
        # By axis scaling
        ANISOTROPIC_SCALE_CATEGORIES = {
            'cabinet', 'large_shelf', 'cell_shelf',
            'kitchen_cabinet',  'children_cabinet',
            'dining_table', 'dressing_table', 'coffee_table',
            'console_table', 'corner_side_table', 'round_end_table',
            'table', 'shelf','rug', 'bookshelf', 'kitchen_space'
        }

        use_anisotropic = cat_lower in ANISOTROPIC_SCALE_CATEGORIES
        eps = 1e-12
        
        if use_anisotropic:
            # By axis
            sx = target_size_x / max(rotated_size.x, eps)
            sy = target_size_y / max(rotated_size.y, eps)
            sz = target_size_z / max(rotated_size.z, eps)
            before_scale = tuple(obj.scale)
            obj.scale = (sx, sy, sz)
            bpy.context.view_layer.update()
            print(f"    Anisotropic scale applied for '{cat_lower}': "
                  f"sx={sx:.6f}, sy={sy:.6f}, sz={sz:.6f}  (scale before={before_scale})")

        else:
            # By proportion
            scale_ratio_x = target_size_x / max(rotated_size.x, eps)
            scale_ratio_y = target_size_y / max(rotated_size.y, eps)
            scale_ratio_z = target_size_z / max(rotated_size.z, eps)
            uniform_scale = min(scale_ratio_x, scale_ratio_y, scale_ratio_z)
            before_scale = tuple(obj.scale)
            obj.scale = (uniform_scale, uniform_scale, uniform_scale)
            bpy.context.view_layer.update()
            print(f"    Uniform scale applied: {uniform_scale:.6f}  scale(before)={before_scale}")
            print(f"    Scale ratios: X={scale_ratio_x:.6f}, Y={scale_ratio_y:.6f}, Z={scale_ratio_z:.6f}")


        # ---------- Step 5: Scaled bbox ----------
        current_min, current_max, current_size, current_center = compute_bbox([obj])
        print(f"    Scaled bbox: size={current_size}, center={current_center}")


        # ---------- Step 6: Target center ----------
        target_center = Vector((
            location[0] * SCENE_SCALE,  # x
            location[2] * SCENE_SCALE,  # z -> y
            location[1] * SCENE_SCALE   # y -> z
        ))
        print(f"    Target center: {target_center} (axis map [x,z,y])")

        # ---------- Step 7: Move to center ----------
        move_distance = target_center - current_center
        before_loc = tuple(obj.location)
        obj.location = obj.location + move_distance
        bpy.context.view_layer.update()
        print(f"    Translation applied: Δ={move_distance}  loc(before)={before_loc} -> (after)={tuple(obj.location)}")

        
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
    bpy.context.scene.cycles.use_adaptive_sampling = True

    bpy.context.scene.render.film_transparent = False

    vs = bpy.context.scene.view_settings
    vs.view_transform = 'Standard'   
    vs.look = 'None'
    vs.exposure = 0.0
    vs.gamma = 1.0
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.sequencer_colorspace_settings.name = 'sRGB'

    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    links = bpy.context.scene.world.node_tree.links
    world_nodes.clear()
    bg = world_nodes.new(type='ShaderNodeBackground')
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  
    bg.inputs[1].default_value = 1.0                  
    out = world_nodes.new(type='ShaderNodeOutputWorld')
    links.new(bg.outputs[0], out.inputs[0])

    print("  Render engine: Cycles")
    print("  Samples: 128")
    print("  World: pure white, film_transparent=False, view_transform=Standard")

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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def load_category_colors(json_path: str = COLORS_JSON_PATH) -> dict:
    try:
        with open(json_path, "r") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[ColorMap] Failed to load {json_path}: {e}")
        return {}

    cmap = {}
    for k, v in raw.items():
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            continue
        r, g, b = _clamp01(v[0]), _clamp01(v[1]), _clamp01(v[2])
        cmap[(k or "").strip().lower()] = (r, g, b, 1.0)
    print(f"[ColorMap] Loaded {len(cmap)} entries from {json_path}")
    return cmap

def make_solid_material(mat_name: str, rgba) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out_node = nt.nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = rgba
    nt.links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    return mat


def apply_color_by_category(obj: bpy.types.Object, category: str, color_map: dict):
    """unify material slot, set material_index 0"""
    cat = (category or "").strip().lower()
    rgba = color_map.get(cat)

    if rgba is None:
        print(f"  ! No color for '{category}' in color_map; skip coloring")
        return 

    mat_name = f"CatColor_{cat}"
    mat = make_solid_material(mat_name, rgba)

    me = obj.data
    if me is None:
        return
    if len(me.materials) == 0:
        me.materials.append(mat)
    else:
        for i in range(len(me.materials)):
            me.materials[i] = mat

    if hasattr(me, "polygons"):
        for poly in me.polygons:
            poly.material_index = 0

    obj.active_material = mat

    print(f"  ✓ Color applied for '{category}': {rgba[:3]}  (slots={len(me.materials)})")


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

def process_scene(scene_data, scene_index, output_base_dir, obj_folder, glb_index_path, save_blend=False, colorize=False):
    """Single scene"""
    print("=" * 80)
    print(f"Processing scene {scene_index}")
    print("=" * 80)

    clear_scene()
    glb_index = load_glb_index(glb_index_path) or None

    class_names = scene_data.get('class_names', [])
    translations_raw = scene_data.get('translations', [])
    sizes_raw = scene_data.get('sizes', [])
    angles_raw = scene_data.get('angles', [])

    translations = translations_raw[0] if translations_raw and len(translations_raw) > 0 else []
    sizes = sizes_raw[0] if sizes_raw and len(sizes_raw) > 0 else []
    angles = angles_raw[0] if angles_raw and len(angles_raw) > 0 else []

    print(f"Number of objects: {len(class_names)}")
    print(f"Translations length: {len(translations)}")
    print(f"Sizes length: {len(sizes)}")
    print(f"Angles length: {len(angles)}")

    min_length = min(len(class_names), len(translations), len(sizes), len(angles))
    if min_length == 0:
        print("No objects found in scene")
        return False

    successful_imports = 0

    color_map = load_category_colors(COLORS_JSON_PATH) if colorize else {}

    for i in range(min_length):
        category = class_names[i]
        location = translations[i]
        size = sizes[i]
        rotation_raw = angles[i]

        if isinstance(rotation_raw, list) and len(rotation_raw) > 0:
            rotation = rotation_raw[0]
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
            if colorize:
                apply_color_by_category(obj, category, color_map)
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

    output_folder = os.path.join(output_base_dir, f"scene_{scene_index}")
    os.makedirs(output_folder, exist_ok=True)
    scene_file_path = os.path.join(output_folder, f"scene_{scene_index}.blend")
    render_output_path = os.path.join(output_folder, f"scene_{scene_index}.png")

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


def process_scenes_batch(scenes_data, output_base_dir, obj_folder, glb_index_path, save_blend=False, selected_indices=None, colorize=False):
    """Batch scenes"""
    print("=" * 80)
    total = len(scenes_data)
    if selected_indices is None:
        indices = list(range(total))
        print(f"Starting batch processing of {total} scenes")
    else:
        indices = list(selected_indices)
        print(f"Starting batch processing of {len(indices)} selected scenes (from total {total})")
        print(f"Selected indices: {indices}")
    print(f"Save .blend files: {'Yes' if save_blend else 'No'}")
    print(f"Apply category colors: {'Yes' if colorize else 'No'}")  
    print("=" * 80)

    os.makedirs(output_base_dir, exist_ok=True)
    success_count = 0

    for k, idx in enumerate(indices):
        try:
            print(f"\n{'='*20} Processing Scene {k+1}/{len(indices)} (scene_id={idx}) {'='*20}")
            scene_data = scenes_data[idx]
            if process_scene(scene_data, idx, output_base_dir, obj_folder, glb_index_path, save_blend, colorize=colorize):
                success_count += 1
                print(f"✓ Scene {idx} processed successfully")
            else:
                print(f"✗ Scene {idx} failed to process")
        except Exception as e:
            print(f"✗ Scene {idx} failed with error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"Batch processing completed: {success_count}/{len(indices)} scenes successful")
    print(f"{'='*80}")

    return success_count


