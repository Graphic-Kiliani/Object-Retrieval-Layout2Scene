"""
# Processing the whole root dir
python build_glb_index.py --input-path /root/autodl-tmp/zyh/retriever/obj

# Processing Single Category
python build_glb_index.py --input-path /root/autodl-tmp/zyh/retriever/obj/oven

"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from glb_indexer import build_glb_index

def setup_gpu_device(gpu_id=None):
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"[GPU Setup] Set CUDA_VISIBLE_DEVICES to {gpu_id}")
    else:
        print("[GPU Setup] Using default GPU device")

def parse_arguments():
    parser = argparse.ArgumentParser(description='GLB Index Builder with GPU Selection')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID to use (0, 1, 2, etc.). If not specified, uses default device.')
    parser.add_argument('--input-path', type=str, default=None,
                        help='Input path: either a root folder with multiple categories, '
                             'or a single category folder containing .glb files. '
                             'Default: /root/autodl-tmp/zyh/retriever/obj')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output index file path. Default: /root/autodl-tmp/zyh/retriever/glb_index.json')
    
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv[1:]
    return parser.parse_args(argv)

def main():
    args = parse_arguments()
    setup_gpu_device(args.gpu)

    obj_folder = args.input_path if args.input_path else "/root/autodl-tmp/zyh/retriever/obj"
    index_file = args.output_file if args.output_file else "/root/autodl-tmp/zyh/retriever/glb_index.json"

    print("=" * 60)
    print("GLB Index Builder")
    print("=" * 60)
    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")
    print("=" * 60)

    if not os.path.exists(obj_folder):
        print(f"Error: Object folder {obj_folder} does not exist")
        return

    print(f"Building GLB index...")
    print(f"Source folder: {obj_folder}")
    print(f"Output file:  {index_file}")
    print("=" * 60)

    index = build_glb_index(obj_folder, index_file)

    if not index:
        print("\n" + "=" * 60)
        print("Index building failed!")
        print("Please check the error messages above and ensure:")
        print("1. All GLB files are valid")
        print("2. Blender can access the files")
        print("3. You have write permission for the output directory")

if __name__ == "__main__":
    main()
