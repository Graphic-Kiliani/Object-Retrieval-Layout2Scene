# Layout2Scene Object Retrieval

This repository provides a simple and scalable pipeline for **layout-to-scene object retrieval**, focusing on matching 2D room layouts with relevant 3D objects and rendering them for visualization.

## Method Overview

Our approach follows a straightforward workflow:

1. **Layout Parsing** – Read the input layout file and extract room and object placement information.
2. **Object Retrieval** – For each object category in the layout, retrieve candidate 3D models from a pre-built object index for shape similarity.
3. **Scene Composition** – Assemble the retrieved 3D objects into a coherent scene according to the layout constraints.
4. **Rendering** – Render the composed scene using Blender with specified resolution and camera settings.

This pipeline is designed to be **lightweight, reproducible, and extensible**, enabling researchers to quickly test layout-based scene generation and retrieval tasks.

## Dataset

We use a curated dataset containing 3D objects categorized by type, designed to support layout-based retrieval tasks. 

- **Dataset link:** [Download Here](https://pan.baidu.com/s/1Xxopue8EjIelQxhDkSoK6Q?pwd=p4cj)

You can also add your desired object category and 3D assets according to our dataset format. However, in order to correctly integrate with our object retrieval process, you should **first uniform the orientation within specific category under your model's/platform's coordinate system**, and **index the size information** for calculating shape similarity with following code:

```bash
# Processing the whole root dir
python build_glb_index.py --input-path <path_to_dataset_dir>

# Processing Single Category
python build_glb_index.py --input-path <path_to_added_category_dir>
```
Afterwards, you will obtain or see changes in glb_index.json file.

## Test Input

We provide an example JSON file containing room layout information for testing and reproducing results.

- **Test JSON link:** [Download Here](https://pan.baidu.com/s/1qW1-OXeabD8Uf3z9QCZCHA?pwd=9sx2)

## Usage

### Scene
After downloading the dataset and the test JSON file, modify the json path and dataset path in main.py, you can run, see arguments'definition in main.py:

```bash
<path_to_blender> --background --python main.py -- \
    --input_json <path_to_room_json> \
    --out_dir    <output_dir_path> \
    --obj_folder <retrieval_assets_dir_path> \
    --glb_index  <glb_json_path> \
    --save_blend \
    --scene_id <num>
    --colorize
```
Besides, you can palette your favourite color in `colors_mapping.json` when you turn on  `--colorize` to get pure colorful topdown images.

Afterwards, you will obtain corresponding topdown.png and scene.blend according to your input scene info json.

### Layout vis
We provide tools to visualize layout in image or gif form. Before that, you only need to convert your scene info json format into ours. (Check it in Test JSON link)
see arguments'definition in visualization_mlayout.py
```bash
python visualization_mlayout.py <path_to_room_json> --scene_id <num> --label_small --small_thresh <num> --flipover --azimuth_offset <num>
```




