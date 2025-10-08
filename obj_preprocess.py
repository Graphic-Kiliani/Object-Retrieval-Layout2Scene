# <path_to_blender> --background --python obj_preprocess.py
import os 
import math
import bpy
from mathutils import Matrix

PRE_ROTATION_BY_CATEGORY = {
    # 'toilet':  math.pi/2,
    # 'standing_sink': math.pi/2,
    # future objects
    # 'desk": -math.pi/2,
    'bookshelf': (3*math.pi)/2,
    # 'chair': math.pi/2,
    # 'cell_shelf': -math.pi/2,
}
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def rotate_glb(obj, rotation):
    if abs(rotation)<1e-8:
        return
    R = Matrix.Rotation(rotation, 4, 'Z')
    obj.matrix_world = R @ obj.matrix_world

def import_glb(glb_path):
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=glb_path)

def export_glb(glb_path):
    bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB')

# your retrieved dataset folder
obj_folder="./obj"
# your rotated output folder, you can view your rotation logic, and then move it to your dataset folder and then use build_glb_index.py to build your new glb index
output_folder="./obj_preprocess"
for category in os.listdir(obj_folder):
    if category in PRE_ROTATION_BY_CATEGORY:
        rotation = PRE_ROTATION_BY_CATEGORY[category]
        print(f"Category: {category}, Rotation: {rotation}")
        category_folder = os.path.join(obj_folder, category)
        for file in os.listdir(category_folder):
            if file.endswith(".glb"):
                glb_path = os.path.join(category_folder, file)
                import_glb(glb_path)
                output_path = os.path.join(output_folder, category, file);
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                for obj in bpy.context.scene.objects:
                    rotate_glb(obj, rotation)
                    export_glb(output_path)
                    print(f"Exported:{output_path}")



                
