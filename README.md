# A gsplat-based trainer for the DL3DV dataset
This reposity provides the code to
* Download the [DL3DV](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/tree/main) dataset;
* Construct 3D Gaussians from the DL3DV scenes based on the [gsplat](https://github.com/nerfstudio-project/gsplat/tree/main).

Note that using [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio) can be a better choice.
## Install

```bash 
conda create -n gdl python=3.10
conda activate gdl
pip3 install ninja numpy jaxtyping rich
pip3 install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip3 install gsplat --index-url https://docs.gsplat.studio/whl/pt21cu121
pip3 install imageio opencv-python pyyaml scikit-learn matplotlib tensorly tensorboard
pip install --upgrade imageio-ffmpeg
pip3 install nerfview==0.0.2
git clone https://github.com/rahul-goel/fused-ssim
pip install fused-ssim/
```

## Download DL3DV (960P)
1. Prepare you `HF_TOKEN`.
2. Then, run the following code to download the zip files of the `1K` subset of DL3DV:
    ```bash
    HF_TOKEN='your hf token' python download.py --odir 960P --subset 1K --resolution 960P --file_type images+poses --clean_cache
    ```
    Use `HF_ENDPOINT=https://hf-mirror.com` if your network is not good.
    ```bash
    HF_ENDPOINT=https://hf-mirror.com HF_TOKEN='your hf token' python download.py --odir 960P --subset 1K --resolution 960P --file_type images+poses --clean_cache
    ```
3. Change the `1K` config in the above command to `2K`...`11K` to download the full 960P DL3DV dataset.


## Run the trainer
Check `run.sh`.
```bash
#!/bin/bash

split=$1
scene=$2
time=$(date "+%Y-%m-%d_%H:%M:%S")

CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --data_dir 960P-unzip/$split/$scene --data_factor 4 \
    --result_dir ./results/$split/$scene/$time \
    --dataset_type DL3DV \
    --init_type random
```