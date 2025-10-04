"""
<path_to_blender> --background --python main.py -- \
    --input_json <path_to_room_json> \
    --out_dir    <output_dir_path> \
    --obj_folder <retrieval_assets_dir_path> \
    --glb_index  <glb_json_path> \
    --save_blend \
    --scene_id <num> \
    --colorize  

/root/autodl-tmp/zyh/Infinigen-Dataset-Toolkit/blender-4.5.2-linux-x64/blender --background --python main.py -- \
    --input_json /root/autodl-tmp/zyh/retriever/test.json \
    --out_dir    /root/autodl-tmp/zyh/retriever/colorful \
    --obj_folder /root/autodl-tmp/zyh/retriever/obj \
    --glb_index  /root/autodl-tmp/zyh/retriever/glb_index.json \
    --colorize  
"""

import bpy
import sys
import os
import json
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from object_retrieval import process_scenes_batch

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender Batch Scene Processor for InfinigenV3")
    parser.add_argument("--input_json", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--obj_folder", type=str)
    parser.add_argument("--glb_index", type=str)
    parser.add_argument("--save_blend", action="store_true", help="Save .blend files per scene")
    parser.add_argument("--scene_id", type=str, default=None, help="Process specific scene(s). Accepts a single index '7' or comma list '3,12,22'")
    parser.add_argument("--colorize", action="store_true", help="Apply solid color materials by category")
    return parser.parse_args(argv)

def main():
    args = parse_args()

    input_json_path = args.input_json
    output_base_dir = args.out_dir
    obj_folder = args.obj_folder
    glb_index_path = args.glb_index
    SAVE_BLEND_FILES = bool(args.save_blend)
    COLORIZE_OBJECTS = bool(args.colorize)  

    selected_indices = None
    if args.scene_id is not None:
        try:
            selected_indices = _parse_scene_id_list(args.scene_id)
            print(f"[Mode] Only processing scene_id(s) = {selected_indices}")
        except ValueError as e:
            print(f"Error: invalid --scene_id value '{args.scene_id}'. Use '7' or '3,12,22'.")
            return False

    try:
        with open(input_json_path, 'r') as f:
            scenes_data = json.load(f)
        print(f"Loaded {len(scenes_data)} scenes from {input_json_path}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return False

    if selected_indices:
        max_idx = len(scenes_data) - 1
        bad = [i for i in selected_indices if not (0 <= i <= max_idx)]
        if bad:
            print(f"Error: scene_id out of range. Valid range: [0, {max_idx}], got {bad}")
            return False


    success_count = process_scenes_batch(
        scenes_data,
        output_base_dir,
        obj_folder,
        glb_index_path,
        SAVE_BLEND_FILES,
        selected_indices=selected_indices,
        colorize=COLORIZE_OBJECTS,           
    )

    print("\n" + "=" * 80)
    print(f"✓ BATCH PROCESSING COMPLETED!")
    print("=" * 80)
    processed_total = len(selected_indices) if selected_indices else len(scenes_data)
    print(f"Successfully processed: {success_count}/{processed_total} scenes")
    print(f"Output directory: {output_base_dir}")
    print(f"Save .blend files: {'Yes' if SAVE_BLEND_FILES else 'No'}")
    print(f"Applied category colors: {'Yes' if COLORIZE_OBJECTS else 'No'}") 
    return success_count > 0


if __name__ == "__main__":
    main()