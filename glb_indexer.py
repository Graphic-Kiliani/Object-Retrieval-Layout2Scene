import bpy
import json
import os
import math
from mathutils import Vector
from util import compute_bbox

def clear_scene_for_indexing():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)

def compute_glb_bbox_ratios(glb_path):
    if not os.path.exists(glb_path):
        return None

    clear_scene_for_indexing()
    objects_before = set(bpy.context.scene.objects)

    try:
        bpy.ops.import_scene.gltf(filepath=glb_path)
    except Exception as e:
        print(f"Error importing {glb_path}: {e}")
        return None

    objects_after = set(bpy.context.scene.objects)
    new_objects = objects_after - objects_before
    if not new_objects:
        return None

    mesh_objects = [obj for obj in new_objects if obj.type == 'MESH']
    if not mesh_objects:
        return None

    try:
        min_corner, max_corner, size, center = compute_bbox(mesh_objects)
        if size.x == 0 or size.y == 0 or size.z == 0:
            return None

        max_dim = max(size.x, size.y, size.z)
        ratios = {
            'width_ratio':  size.x / max_dim,
            'height_ratio': size.y / max_dim,
            'depth_ratio':  size.z / max_dim,
            'absolute_size': [size.x, size.y, size.z],
            'aspect_ratios': {
                'width_height': size.x / size.y,
                'width_depth':  size.x / size.z,
                'height_depth': size.y / size.z
            }
        }
        return ratios

    except Exception as e:
        print(f"Error computing bbox for {glb_path}: {e}")
        return None

def build_glb_index(obj_folder, index_file_path):
    print(f"[GLB Indexer] Building index for {obj_folder}")
    print(f"[GLB Indexer] Index will be saved to {index_file_path}")

    if not os.path.exists(obj_folder):
        print(f"Error: Path {obj_folder} does not exist")
        return None

    direct_glb_files = [f for f in os.listdir(obj_folder) if f.endswith('.glb')]

    if direct_glb_files:
        print("[GLB Indexer] Detected single category mode")
        return _build_single_category_index(obj_folder, index_file_path, direct_glb_files)
    else:
        print("[GLB Indexer] Detected multi-category mode")
        return _build_multi_category_index(obj_folder, index_file_path)

def _build_single_category_index(category_path, index_file_path, glb_files):
    """
    Only update single category's glb objects' index in glb_index.json
    """
    category_name = os.path.basename(category_path)
    print(f"\n[GLB Indexer] Processing single category: {category_name}")
    print(f"  Found {len(glb_files)} GLB files")

    existing_index = {}
    if os.path.exists(index_file_path):
        try:
            with open(index_file_path, 'r') as f:
                existing_index = json.load(f)
            print(f"  Loaded existing index with {len(existing_index)} categories")
        except Exception as e:
            print(f"  Warning: Could not load existing index: {e}")
            existing_index = {}

    index = existing_index
    index[category_name] = {}  

    total_files = 0
    successful_files = 0

    for glb_file in glb_files:
        glb_path = os.path.join(category_path, glb_file)
        total_files += 1
        print(f"  Processing: {glb_file}")

        ratios = compute_glb_bbox_ratios(glb_path)
        if ratios is not None:
            index[category_name][glb_file] = {
                'path': glb_path,
                'ratios': ratios
            }
            successful_files += 1
            print(f"    ✓ Ratios: W={ratios['width_ratio']:.3f}, H={ratios['height_ratio']:.3f}, D={ratios['depth_ratio']:.3f}")
        else:
            print(f"    ✗ Failed to compute ratios")

    return _save_index_and_report(index, index_file_path, total_files, successful_files, len(index))

def _build_multi_category_index(obj_folder, index_file_path):
    """
    Update glb_index.json completely
    """
    index = {}
    total_files = 0
    successful_files = 0

    for category in os.listdir(obj_folder):
        category_path = os.path.join(obj_folder, category)
        if not os.path.isdir(category_path):
            continue

        print(f"\n[GLB Indexer] Processing category: {category}")
        index[category] = {}

        glb_files = [f for f in os.listdir(category_path) if f.endswith('.glb')]
        if not glb_files:
            print(f"  No GLB files found in {category}")
            continue

        print(f"  Found {len(glb_files)} GLB files")
        for glb_file in glb_files:
            glb_path = os.path.join(category_path, glb_file)
            total_files += 1
            print(f"  Processing: {glb_file}")

            ratios = compute_glb_bbox_ratios(glb_path)
            if ratios is not None:
                index[category][glb_file] = {
                    'path': glb_path,
                    'ratios': ratios
                }
                successful_files += 1
                print(f"    ✓ Ratios: W={ratios['width_ratio']:.3f}, H={ratios['height_ratio']:.3f}, D={ratios['depth_ratio']:.3f}")
            else:
                print(f"    ✗ Failed to compute ratios")

    return _save_index_and_report(index, index_file_path, total_files, successful_files, len(index))

def _save_index_and_report(index, index_file_path, total_files, successful_files, category_count):
    try:
        with open(index_file_path, 'w') as f:
            json.dump(index, f, indent=2)
        print(f"\n[GLB Indexer] Index saved successfully!")
        print(f"[GLB Indexer] Total files processed: {total_files}")
        print(f"[GLB Indexer] Successfully indexed: {successful_files}")
        print(f"[GLB Indexer] Categories: {category_count}")
        if total_files > 0:
            print(f"[GLB Indexer] Success rate: {successful_files/total_files*100:.1f}%")
        return index
    except Exception as e:
        print(f"Error saving index: {e}")
        return None

def load_glb_index(index_file_path):
    try:
        with open(index_file_path, 'r') as f:
            index = json.load(f)
        print(f"[GLB Indexer] Loaded index with {len(index)} categories")
        return index
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

def calculate_shape_similarity(target_ratios, glb_ratios):
    try:
        w_diff = abs(target_ratios['width_ratio'] - glb_ratios['width_ratio'])
        h_diff = abs(target_ratios['height_ratio'] - glb_ratios['height_ratio'])
        d_diff = abs(target_ratios['depth_ratio'] - glb_ratios['depth_ratio'])
        shape_distance = math.sqrt(w_diff**2 + h_diff**2 + d_diff**2)

        aspect_weight = 0.3
        try:
            aspect_diff = (
                abs(target_ratios['aspect_ratios']['width_height'] - glb_ratios['aspect_ratios']['width_height']) +
                abs(target_ratios['aspect_ratios']['width_depth']  - glb_ratios['aspect_ratios']['width_depth'])  +
                abs(target_ratios['aspect_ratios']['height_depth'] - glb_ratios['aspect_ratios']['height_depth'])
            ) / 3.0
            total_similarity = shape_distance + aspect_weight * aspect_diff
        except:
            total_similarity = shape_distance

        return total_similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return float('inf')

def find_best_matching_glb(class_name, target_bbox_size, glb_index):
    if class_name not in glb_index:
        print(f"Warning: Category {class_name} not found in index")
        return None

    category_files = glb_index[class_name]
    if not category_files:
        print(f"Warning: No files found for category {class_name}")
        return None

    max_dim = max(target_bbox_size[0], target_bbox_size[1], target_bbox_size[2])
    if max_dim == 0:
        return None

    target_ratios = {
        'width_ratio':  target_bbox_size[0] / max_dim,
        'height_ratio': target_bbox_size[1] / max_dim,
        'depth_ratio':  target_bbox_size[2] / max_dim,
        'aspect_ratios': {
            'width_height': target_bbox_size[0] / target_bbox_size[1] if target_bbox_size[1] != 0 else 1.0,
            'width_depth':  target_bbox_size[0] / target_bbox_size[2] if target_bbox_size[2] != 0 else 1.0,
            'height_depth': target_bbox_size[1] / target_bbox_size[2] if target_bbox_size[2] != 0 else 1.0
        }
    }

    candidates = []
    for glb_file, file_data in category_files.items():
        similarity = calculate_shape_similarity(target_ratios, file_data['ratios'])
        candidates.append((similarity, glb_file, file_data['path']))

    candidates.sort(key=lambda x: x[0])
    if candidates:
        best_similarity, best_file, best_path = candidates[0]
        print(f"  Best match for {class_name}: {best_file} (similarity: {best_similarity:.4f})")
        return best_path
    return None

if __name__ == "__main__":
    obj_folder = "/root/autodl-tmp/zyh/retriever/obj"
    index_file = "/root/autodl-tmp/zyh/retriever/glb_index.json"
    build_glb_index(obj_folder, index_file)
