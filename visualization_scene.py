import bpy
import os
from mathutils import Vector, geometry, Matrix
import bmesh
import math
import json
from PIL import Image
from util import *
from render import *

scene_path = "path_to_scene.blend"
output_dir = "path_to_output_dir"

# we have provided a sample parameters.json file in the assets folder, you can check our render.py for more details on how to use it as a reference
json_path = "path_to_your_parameters.json"
render_gif(scene_path, output_dir, json_path, use_scene_copy=True, radius=10, elevation=-10, start_azimuth=-40, end_azimuth=40, step=4, duration=100)