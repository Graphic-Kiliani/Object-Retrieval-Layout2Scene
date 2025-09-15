# Layout2Scene Object Retrieval

This repository provides a simple pipeline for **layout-to-scene object retrieval**, focusing on matching 2D room layouts with relevant 3D objects and rendering them for visualization.

## Method Overview

Our approach follows a straightforward workflow:

1. **Layout Parsing** – Read the input layout file and extract room and object placement information.
2. **Object Retrieval** – For each object category in the layout, retrieve candidate 3D models from a pre-built object index for shape similarity.
3. **Scene Composition** – Assemble the retrieved 3D objects into a coherent scene according to the layout constraints.
4. **Rendering** – Render the composed scene using Blender with specified resolution and camera settings.

This pipeline is designed to be **lightweight, reproducible, and extensible**, enabling researchers to quickly test layout-based scene generation and retrieval tasks.

## Dataset

We use a curated dataset containing 3D objects categorized by type, designed to support layout-based retrieval tasks.

- **Dataset link:** [Download Here](https://pan.baidu.com/s/1HSNx3WmHBmBf2wo5anpS9g?pwd=fx62)

## Test Input

We provide an example JSON file containing room layout information for testing and reproducing results.

- **Test JSON link:** [Download Here](https://pan.baidu.com/s/1qW1-OXeabD8Uf3z9QCZCHA?pwd=9sx2)

## Usage

After downloading the dataset and the test JSON file, modify the json path and dataset path in main.py, you can run:

```bash
<path_to_blender> --background --python main.py
