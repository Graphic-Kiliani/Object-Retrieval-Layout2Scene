import cv2
import numpy as np
import json
import os
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation


@dataclass
class SceneObject:
    """Represents a 3D object in the scene"""
    name: str
    location: List[float]  # [x, y, z]
    size: List[float]  # [width, height, depth]
    rotation: float  # [roll, pitch, yaw] in radians
    color: Optional[List[int]] = None  # [R, G, B]

    def get_bounding_box(self) -> np.ndarray:
        """Calculate the 8 vertices of the rotated 3D bounding box"""
        w, h, d = self.size
        cx, cz, cy = self.location
        angle_z = self.rotation  # Use Z-axis rotation
        
        # Calculate the 8 vertices of the bounding box
        vertices = np.array([
            [-w, -d, -h], [w, -d, -h],
            [w, d, -h], [-w, d, -h],
            [-w, -d, h], [w, -d, h],
            [w, d, h], [-w, d, h]
        ])
        
        # Apply rotation (around Z-axis only)
        R = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])
        vertices = (R @ vertices.T).T
        
        # Translate to world coordinate system
        vertices[:, 0] += cx
        vertices[:, 1] += cy
        vertices[:, 2] += cz
        
        return vertices

class SceneLayout:
    """Encapsulates scene layout, provides scene statistics and normalization functions"""
    def __init__(self, objects: List[SceneObject]):
        self.objects = objects
        self._calculate_bounds()
    
    def _calculate_bounds(self) -> None:
        """Calculate scene boundaries and center"""
        all_vertices = []
        for obj in self.objects:
            bbox = obj.get_bounding_box()
            all_vertices.append(bbox)
        
        if not all_vertices:
            self.min_bounds = np.array([0, 0, 0])
            self.max_bounds = np.array([0, 0, 0])
            self.center = np.array([0, 0, 0])
            return
        
        all_vertices = np.vstack(all_vertices)
        self.min_bounds = np.min(all_vertices, axis=0)
        self.max_bounds = np.max(all_vertices, axis=0)
        self.center = (self.min_bounds + self.max_bounds) / 2
    
    def get_scene_center(self) -> np.ndarray:
        """Get scene center coordinates"""
        return self.center.copy()
    
    def get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get minimum and maximum bounds of the scene"""
        return self.min_bounds.copy(), self.max_bounds.copy()
    
    def get_scene_size(self) -> np.ndarray:
        """Get XYZ dimensions of the scene"""
        return self.max_bounds - self.min_bounds
    
    def normalize_to_origin(self) -> None:
        """Normalize the entire scene to the origin"""
        if len(self.objects) == 0:
            return
        
        # Calculate offset needed for movement
        offset = -self.center
        
        # Calculate floor level by filtering out extreme minimum values
        all_z_min = []
        for obj in self.objects:
            bbox = obj.get_bounding_box()
            z_min = np.min(bbox[:, 2])  # Get minimum Z coordinate of bounding box
            all_z_min.append(z_min)
        
        # Filter out extreme outliers (bottom 5% of values)
        all_z_min = np.array(all_z_min)
        sorted_z_min = np.sort(all_z_min)
        percentile_5 = np.percentile(sorted_z_min, 5)
        floor_level = np.min(all_z_min[all_z_min >= percentile_5])
        
        # Adjust Z offset to align with floor
        offset[2] = -floor_level
        
        # Move all objects
        for obj in self.objects:
            obj.location = [
                obj.location[0] + offset[0],
                obj.location[1] + offset[1],
                obj.location[2] + offset[2]
            ]
        
        # Recalculate bounds
        self._calculate_bounds()
    
    def scale_to_unit_cube(self) -> None:
        """Scale the scene to fit in a unit cube"""
        if len(self.objects) == 0:
            return
        
        # First normalize to origin
        self.normalize_to_origin()
        
        # Get current scene size
        scene_size = self.get_scene_size()
        max_dim = np.max(scene_size)
        if max_dim <= 1e-6:  # Avoid division by zero
            return
        
        # Calculate scale factor
        scale_factor = 1.0 / max_dim * 5
        
        # Scale all objects
        for obj in self.objects:
            obj.location = [
                obj.location[0] * scale_factor,
                obj.location[1] * scale_factor,
                obj.location[2] * scale_factor
            ]
            obj.size = [
                obj.size[0] * scale_factor,
                obj.size[1] * scale_factor,
                obj.size[2] * scale_factor
            ]
        
        # Recalculate bounds
        self._calculate_bounds()
    
    def __len__(self) -> int:
        return len(self.objects)
    
    def __getitem__(self, idx: int) -> SceneObject:
        return self.objects[idx]
    
    def __iter__(self):
        return iter(self.objects)

class Camera:
    """Represents a virtual camera for 3D to 2D projection"""
    def __init__(self, img_width: int = 1100, img_height: int = 700, focal_length: int = 1300):
        self.img_width = img_width
        self.img_height = img_height
        self.focal_length = focal_length
        self.matrix = np.array([
            [focal_length, 0, img_width//2],
            [0, focal_length, img_height//2],
            [0, 0, 1]
        ], dtype=np.float32)
        
    def project_point(self, point_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D image plane"""
        point_3d = point_3d.reshape(1, 3)
        rotation_mat, _ = cv2.Rodrigues(rvec)
        point_camera = (rotation_mat @ point_3d.T + tvec.reshape(3, 1)).T
        
        # Fix type error
        distortion_coeffs = np.zeros((4, 1), dtype=np.float32)
        point_2d, _ = cv2.projectPoints(
            point_camera.astype(np.float32), 
            np.zeros(3, dtype=np.float32), 
            np.zeros(3, dtype=np.float32), 
            self.matrix, 
            distortion_coeffs
        )
        return point_2d.squeeze()

class SceneVisualizer:
    """Scene visualizer responsible for rendering 3D scenes"""
    def __init__(self, scene_layout: SceneLayout):
        self.layout = scene_layout
        self.camera = Camera()
        self.scene_center = self.layout.get_scene_center() 

    def _get_camera_pose(self, radius: float, elevation: float, azimuth: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate camera position and pose"""
        position = self.scene_center + np.array([
            -radius * np.cos(np.deg2rad(elevation)) * np.sin(np.deg2rad(azimuth)),
            -radius * np.cos(np.deg2rad(elevation)) * np.cos(np.deg2rad(azimuth)),
            -radius * np.sin(np.deg2rad(elevation))
        ])
        
        # Build camera rotation matrix
        R1 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        R2 = np.array([
            [np.cos(np.deg2rad(azimuth)), 0, -np.sin(np.deg2rad(azimuth))],
            [0, 1, 0],
            [np.sin(np.deg2rad(azimuth)), 0, np.cos(np.deg2rad(azimuth))]
        ], dtype=np.float32)
        R3 = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(elevation)), np.sin(np.deg2rad(elevation))],
            [0, -np.sin(np.deg2rad(elevation)), np.cos(np.deg2rad(elevation))]
        ], dtype=np.float32)
        R = R3 @ R2 @ R1
        
        rvec, _ = cv2.Rodrigues(R)
        tvec = -(R @ position.reshape(-1, 1)).reshape(-1)
        
        return rvec, tvec
    
    def _draw_bounding_box(self, img: np.ndarray, vertices: np.ndarray, rvec: np.ndarray, 
                          tvec: np.ndarray, color: List[int]) -> None:
        """Draw 2D projection of 3D bounding box on image"""
        vertices_2d = []
        for vertex in vertices:
            point_2d = self.camera.project_point(vertex, rvec, tvec)
            vertices_2d.append(point_2d)
        
        vertices_2d = np.array(vertices_2d, dtype=np.int32)
        
        # Draw bounding box edges
        for i in range(4):
            # Bottom rectangle
            self._draw_line_if_visible(img, vertices_2d[i], vertices_2d[(i+1)%4], color)
            # Top rectangle
            self._draw_line_if_visible(img, vertices_2d[i+4], vertices_2d[(i+1)%4+4], color)
            # Vertical lines connecting top and bottom
            self._draw_line_if_visible(img, vertices_2d[i], vertices_2d[i+4], color)
    
    def _draw_line_if_visible(self, img: np.ndarray, pt1: np.ndarray, pt2: np.ndarray, 
                            color: List[int], thickness: int = 1) -> None:
        """Draw line if it's within image bounds"""
        img_w, img_h = self.camera.img_width, self.camera.img_height
        if (0 <= pt1[0] <= img_w and 0 <= pt1[1] <= img_h) or \
           (0 <= pt2[0] <= img_w and 0 <= pt2[1] <= img_h):
            cv2.line(img, tuple(pt1), tuple(pt2), color, thickness, cv2.LINE_AA)
    
    def text_to_2d_points(self, text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, thickness=1):
        # Step 1: Render text to grayscale image
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        img = np.zeros((h + baseline + 5, w + 5), dtype=np.uint8)
        cv2.putText(img, text, (0, h), font, scale, (255,), thickness, cv2.LINE_AA)

        # Step 2: Find all non-zero points (text pixels)
        points = np.column_stack(np.where(img > 0))  # (y, x) order
        # Convert to (x, y), and flip y coordinate direction (image origin is top-left)
        points = np.array([[x, img.shape[0] - y] for y, x in points], dtype=np.float32)
        return points, (w, h)

    def transform_to_3d(self, points_2d, position, right=np.array([1, 0, 0]), up=np.array([0, 0, 1]), scale=0.005):
        # Step 3: Map 2D points to a plane in front of bbox, position is bbox center or front center point
        right = right / np.linalg.norm(right)
        up = up / np.linalg.norm(up)
        normal = np.cross(right, up)
        
        # Transform points from pixel coordinates to 3D space
        centered = points_2d - np.mean(points_2d, axis=0)  # Center
        points_3d = position + (centered[:, 0:1] * right + centered[:, 1:2] * up) * scale
        return points_3d

    def project_and_draw(self, img, points_3d, rvec, tvec, color=(0, 255, 0)):
        # Step 4: Use camera intrinsics to project 3D points back to 2D image
        # Assumes no distortion and no rotation transform (can be extended)
        points_2d = []
        for point_3d in points_3d:
            point_2d = self.camera.project_point(point_3d, rvec, tvec)
            points_2d.append(point_2d)
        points_2d = np.array(points_2d, dtype=np.int32)

        for pt in points_2d:
            cv2.circle(img, tuple(pt), 1, color, -1)

        return img

    def render_scene(self, radius: float = 10, elevation: float = -10, 
                    azimuth: float = 0) -> np.ndarray:
        """Render a single view of the scene"""
        img = np.ones((self.camera.img_height, self.camera.img_width, 3), np.uint8) * 255
        rvec, tvec = self._get_camera_pose(radius, elevation, azimuth)
        
        for obj in self.layout.objects:
            vertices = obj.get_bounding_box()
            color = obj.color if obj.color is not None else [0, 0, 255]
            self._draw_bounding_box(img, vertices, rvec, tvec, color)
            # Draw object name
            if max(obj.size) > 0.5:
                text = obj.name
                text_points, (w, h) = self.text_to_2d_points(text)
                text_position = np.mean(vertices, axis=0)
                text_points = self.transform_to_3d(text_points, text_position)
                text_position = self.project_and_draw(img, text_points, rvec, tvec, color)

        
        return img
    
    def generate_rotation_animation(self, output_path: str, radius: float = 10, 
                                  elevation: float = -10, start_azimuth: float = -40, 
                                  end_azimuth: float = 40, step: float = 4, 
                                  duration: int = 100) -> None:
        """Generate scene rotation animation"""
        frames = []
        for azimuth in tqdm(np.arange(start_azimuth, end_azimuth, step)):
            frame = self.render_scene(radius, elevation, azimuth)
            frames.append(Image.fromarray(frame))
        for azimuth in tqdm(np.arange(end_azimuth, start_azimuth, -step)):
            frame = self.render_scene(radius, elevation, azimuth)
            frames.append(Image.fromarray(frame))
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        frames[0].save(output_path, format="GIF", append_images=frames[1:],
                     save_all=True, duration=duration, loop=0)

# def load_scene_from_json(json_path: str) -> Dict[str, SceneLayout]:
#     """Load scene objects from JSON file (adapted for mlayout format)"""
#     with open(json_path, 'r') as f:
#         data = json.load(f)
    
#     scenes = {}
    
#     # Generate random colors for each object category
#     color_map = {}
    
#     for scene_data in data["scenes"]:
#         scene_name = scene_data["scene_id"]
#         scene_objects = []
        
#         for obj in scene_data["objects"]:
#             obj_name = obj["category"]
            
#             # Skip structural objects
#             if obj_name in ["wall", "floor", "ceiling", "door", "window", 
#                            "railing", "column", "beam", "stairs", "frame"]:
#                 continue
            
#             # Generate color for new categories
#             if obj_name not in color_map:
#                 color_map[obj_name] = (np.random.rand(3) * 255).astype(np.int32).tolist()

#             scene_objects.append(SceneObject(
#                 name=obj["category"],
#                 location=obj["location"],
#                 size=obj["size"],
#                 rotation=obj["rotation"],
#                 color=color_map[obj_name]
#             ))

#         scenes[scene_name] = SceneLayout(scene_objects)
    
#     return scenes

def load_scene_from_json(json_path: str) -> Dict[str, SceneLayout]:
    """Load scene objects from JSON file (supports both list and {'scenes': [...]})"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        scene_list = data
    elif isinstance(data, dict) and isinstance(data.get("scenes"), list):
        scene_list = data["scenes"]
    else:
        raise ValueError(f"Unsupported JSON top-level type: {type(data)}. Expect list or dict with 'scenes'.")

    scenes: Dict[str, SceneLayout] = {}
    color_map: Dict[str, list] = {}

    def first_or_same(x):
        if isinstance(x, list) and x and isinstance(x[0], list):
            return x[0]
        return x or []

    for idx, scene_data in enumerate(scene_list):
        scene_name = scene_data.get("scene_id") or scene_data.get("id") or scene_data.get("name") or f"scene_{idx}"

        class_names = scene_data.get("class_names", [])
        translations = first_or_same(scene_data.get("translations"))
        sizes        = first_or_same(scene_data.get("sizes"))
        angles       = first_or_same(scene_data.get("angles"))

        n = min(len(class_names), len(translations), len(sizes), len(angles))
        if n == 0:
            scenes[scene_name] = SceneLayout([])
            continue

        objs: list[SceneObject] = []
        for i in range(n):
            cat  = class_names[i]
            loc  = translations[i]
            size = sizes[i]
            ang  = angles[i]
            if isinstance(ang, (list, tuple)) and ang:
                ang = ang[0]
            if cat not in color_map:
                color_map[cat] = (np.random.rand(3) * 255).astype(np.int32).tolist()
            objs.append(SceneObject(
                name=cat,
                location=loc,
                size=size,
                rotation=float(ang),
                color=color_map[cat]
            ))
        scenes[scene_name] = SceneLayout(objs)

    return scenes


def main(input_path: str, output_path: Optional[str] = None, max_scenes: int = 5):
    
    scenes = load_scene_from_json(input_path)
    
    print(f"Loaded {len(scenes)} scenes total, will process first {min(max_scenes, len(scenes))} scenes")

    for i, (scene_name, scene_layout) in enumerate(scenes.items()):
        if i >= max_scenes:
            break
        
        os.makedirs(os.path.join(os.path.dirname(input_path), "vis_diningroom"), exist_ok=True)
        output_path = os.path.join(os.path.dirname(input_path), "vis_dininingroom", f"{scene_name}.gif")

        min_bounds, max_bounds = scene_layout.get_scene_bounds()
        print(f"\n=== Scene {i+1}/{min(max_scenes, len(scenes))}: {scene_name} ===")
        print(f"Scene bounds - Min: {min_bounds}, Max: {max_bounds}")
        print(f"Scene center: {scene_layout.get_scene_center()}")
        print(f"Scene size: {scene_layout.get_scene_size()}")
        
        # Normalize scene to origin
        scene_layout.normalize_to_origin()
        print("\nAfter normalization:")
        print(f"New scene center: {scene_layout.get_scene_center()}")
        
        # Optional: scale to unit cube
        scene_layout.scale_to_unit_cube()
        print("\nAfter scaling to unit cube:")
        print(f"Scene size: {scene_layout.get_scene_size()}")

        # Count objects
        object_count = len(scene_layout)
        print(f"Number of objects in the scene: {object_count}")
        
        # Create a SceneVisualizer instance and generate the animation
        visualizer = SceneVisualizer(scene_layout)
        visualizer.generate_rotation_animation(output_path)
        
        print(f"Animation saved to: {output_path}")

def types_stastic(input_path: str, output_path: Optional[str] = None):

    types = []
    
    scenes = load_scene_from_json(input_path)

    for scene_name, scene_layout in scenes.items():
        for obj in scene_layout.objects:
            types.append(obj.name)

    return types

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MP3D scene layout')
    parser.add_argument('input_file', nargs='?', default="mlayout/infinigen_train.json", 
                       help='Input JSON file path')
    parser.add_argument('--max_scenes', type=int, default=1e6, 
                       help='Maximum number of scenes to process (default: 1e6)')
    
    args = parser.parse_args()
    main(args.input_file, max_scenes=args.max_scenes)

    # python visualization_mlayout.py /root/autodl-tmp/zyh/retriever/mlayout_diffusion_infinigen_280_30000_prompt_gpt/diningroom.json
    # python visualization_mlayout.py mlayout/mp3d_train.json
    # python visualization_mlayout.py mlayout/3dfront_train.json