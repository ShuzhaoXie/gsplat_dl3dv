
# Download 480P resolution images and poses, 0~1K subset, output to DL3DV-10K directory   
python download.py --odir 960P --subset 1K --resolution 960P --file_type images+poses --clean_cache

HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=yourtoken python download_lost.py --odir 960P --subset 8K --resolution 960P --file_type images+poses --clean_cache

# Download 480P resolution images and poses, 1K~2K subset, output to DL3DV-10K directory   
python download.py --odir DL3DV-10K --subset 2K --resolution 480P --file_type images+poses --clean_cache

python download.py --odir 960P --subset 1K --resolution 960P --file_type images+poses --clean_cache

python download.py --odir  --subset full --only_level4 --clean_cache

python download.py --odir ALL-ColmapCache --subset 1K --resolution 480P --file_type colmap_cache --clean_cache


CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default \
    --data_dir 960P-unzip/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3 --data_factor 4 \
    --result_dir ./results/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/2025-01-22-21-10-default \
    --init_type random

CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default \
    --dataset_type COLMAP \
    --data_dir ~/nerf_data/360_v2/garden \
    --result_dir ./results/360_v2/garden/2025-01-22-19-37 \
    --init_type random