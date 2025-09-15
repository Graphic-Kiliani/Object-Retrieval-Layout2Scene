'''

pip install torch torchvision
pip install pillow
pip install tqdm
pip install git+https://github.com/openai/CLIP.git

'''

import os
import json
import numpy as np
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
import clip


def load_clip_model(model_name: str = "ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading CLIP model: {model_name}")
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device


def compute_clip_scores(
    json_path: str,
    images_dir: str,
    model_name: str = "ViT-B/32",
    batch_size: int = 8,  
):
    model, preprocess, device = load_clip_model(model_name)

    with open(json_path, "r") as f:
        scenes_data = json.load(f)

    print(f"[INFO] Loading {len(scenes_data)} scene descriptions")

    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"[ERROR] Image Dir not exists: {images_dir}")

    scores = []
    all_pairs = []

    existing_scenes = []
    for scene_dir in sorted(images_dir.glob("scene_*")):
        if scene_dir.is_dir():
            png_files = list(scene_dir.glob("*.png"))
            if png_files:
                scene_index = int(scene_dir.name.split("_")[1])
                existing_scenes.append(scene_index)
    
    print(f"[INFO] Found {len(existing_scenes)} rendered scenes: {existing_scenes}")
    
    for scene_index in tqdm(existing_scenes, desc="Processing scenes"):
        if scene_index >= len(scenes_data):
            print(f"[WARN] Scene {scene_index} not found in JSON data (only {len(scenes_data)} scenes available)")
            continue
            
        scene = scenes_data[scene_index]
        text_description = scene.get("text", "")
        if not text_description:
            print(f"[WARN] Scene {scene_index} has no text description")
            continue
        
        # 截断过长的文本描述（CLIP有77个token的限制）
        if len(text_description) > 300:
            text_description = text_description[:300] + "..."
            print(f"[INFO] Scene {scene_index} text truncated to 300 characters")

        image_path = images_dir / f"scene_{scene_index}" / f"scene_{scene_index}.png"
        if not image_path.exists():
            print(f"[WARN] Image file not found: {image_path}")
            continue

        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            image_tensor = preprocess(image).unsqueeze(0).to(device)
            text = clip.tokenize([text_description]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).item()

        except Exception as e:
            print(f"[WARN] Error when processing {image_path}: {e}")
            continue

        scores.append(similarity)
        all_pairs.append(
            {
                "index": scene_index,
                "text": text_description,
                "image": str(image_path),
                "clip_score": similarity,
            }
        )

    mean_score = float(np.mean(scores)) if scores else 0.0
    return scores, mean_score, all_pairs


def save_results(scores, mean_score, pairs, output_file: str):
    sorted_pairs = sorted(pairs, key=lambda x: x["clip_score"], reverse=True)

    with open(output_file, "w") as f:
        f.write(f"Average CLIP score: {mean_score:.4f}\n")

        if scores:
            f.write(f"The highest CLIP score: {max(scores):.4f}\n")
            f.write(f"The lowest CLIP score: {min(scores):.4f}\n")
            f.write(f"The median CLIP score: {np.median(scores):.4f}\n")
            f.write(f"The standard deviation: {np.std(scores):.4f}\n\n")
        else:
            f.write("No valid scores.\n\n")

        f.write("The CLIP scores of each scene (sorted in descending order):\n")
        f.write("-" * 80 + "\n")

        for pair in sorted_pairs:
            index = pair["index"]
            score = pair["clip_score"]
            text = pair["text"]
            image = pair["image"]

            if len(text) > 60:
                text = text[:57] + "..."

            f.write(f"Scene {index:03d}: {score:.4f} - {text}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write(f"Total number of scene-picture pairs: {len(scores)}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate CLIP Score between text descriptions and images"
    )
    parser.add_argument(
        "--json",
        default="/root/autodl-tmp/zyh/retriever/mlayout_generated_format/generated_layouts_infinigen_train.json",
        help="JSON file path",
    )
    parser.add_argument(
        "--images",
        default="/root/autodl-tmp/zyh/retriever/rendered_scenes_infinigenv3_test",
        help="Rendered image dir",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/zyh/retriever/clip_score_infinigenv3_test.txt",
        help="Output path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="(reserved) batch size"
    )
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP model edition",
    )

    args = parser.parse_args()

    print("[INFO] Starting calculating CLIP score")
    scores, mean_score, pairs = compute_clip_scores(
        args.json, args.images, model_name=args.model, batch_size=args.batch_size
    )

    print(f"[INFO] Evaluating {len(scores)} scene-image pairs")
    print(f"[INFO] Average CLIP score: {mean_score:.4f}")

    save_results(scores, mean_score, pairs, args.output)
    print(f"[INFO] Results saved to {args.output}")


if __name__ == "__main__":
    main()
