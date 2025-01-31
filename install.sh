CUVER=cu121
conda create -n gdl python=3.10
conda activate gdl
pip3 install ninja numpy jaxtyping rich
pip3 install torch==2.1.0+$CUVER --index-url https://download.pytorch.org/whl/$CUVER
pip3 install torchvision==0.16.0+$CUVER torchaudio==2.1.0+$CUVER --index-url https://download.pytorch.org/whl/$CUVER
pip3 install gsplat --index-url https://docs.gsplat.studio/whl/pt21$CUVER
pip3 install imageio opencv-python pyyaml scikit-learn matplotlib tensorly tensorboard torchmetrics
pip install --upgrade imageio-ffmpeg
pip3 install nerfview==0.0.2
# git clone https://github.com/rahul-goel/fused-ssim
pip install fused-ssim/
pip install git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e
pip install numpy==1.26.4