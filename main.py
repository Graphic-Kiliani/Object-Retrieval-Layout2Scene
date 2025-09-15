"""
/root/autodl-tmp/zyh/Infinigen-Dataset-Toolkit/blender-4.5.2-linux-x64/blender --background --python main.py
"""

import bpy
import sys
import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from object_retrieval import process_scenes_batch

def main():
    print("=" * 80)
    print("Blender Batch Scene Processor for InfinigenV3")
    print("=" * 80)
    
    if not hasattr(bpy, 'context'):
        print("Error: This script must be run in Blender")
        return False
    
    print("Blender environment detected")
    print(f"Blender version: {bpy.app.version_string}")
    print(f"Current scene: {bpy.context.scene.name}")
    
    input_json_path = "/root/autodl-tmp/zyh/retriever/mlayout_generated_format/generated_layouts_3dfront_train.json"
    output_base_dir = "/root/autodl-tmp/zyh/retriever/rendered_scenes_3dfront_test"
    
    # False to skip saving blend files
    SAVE_BLEND_FILES = False  
    
    try:
        with open(input_json_path, 'r') as f:
            scenes_data = json.load(f)
        print(f"Loaded {len(scenes_data)} scenes from {input_json_path}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return False
    
    try:
        success_count = process_scenes_batch(scenes_data, output_base_dir, SAVE_BLEND_FILES)
        
        print("\n" + "=" * 80)
        print(f"✓ BATCH PROCESSING COMPLETED!")
        print("=" * 80)
        print(f"Successfully processed: {success_count}/{len(scenes_data)} scenes")
        print(f"Output directory: {output_base_dir}")
        print(f"Save .blend files: {'Yes' if SAVE_BLEND_FILES else 'No'}")
        print("\nYou can now:")
        print("  1. View the rendered top-down images")
        if SAVE_BLEND_FILES:
            print("  2. Check individual scene files if needed")
        else:
            print("  2. .blend files were not saved (SAVE_BLEND_FILES=False)")
        
        return success_count > 0
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
