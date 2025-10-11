# A Review of Recent Progress in Seismic Waves Propagation Modeling Using Machine Learning Based Methods

 

## Installation

We recommend setting up a new Python environment with conda. You can do this by running the following commands:

```
conda env create -f ml-seismic-waves-env.yml
conda activate ml-seismic-waves-env
```

### Install PyTorch with CUDA support

After activating the environment, install PyTorch, TorchVision, and TorchAudio with CUDA 12.8 support (adjust if your nvidia-smi shows a different CUDA version):

 ```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
 ```

Make sure your system’s NVIDIA driver and CUDA toolkit are properly installed.
You can check your CUDA version with:

 ```
nvidia-smi
 ```

Example output: 

 ```
CUDA Version: 12.8
 ```

To confirm that PyTorch detects your GPU and CUDA correctly, run:

 ```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
 ```

Example output:

 ```
2.8.0+cu128 12.8 True NVIDIA RTX 2000 Ada Generation Laptop GPU
 ```

 2.8.0+cu128   →  PyTorch version 2.8.0 compiled with CUDA 12.8

12.8          →  CUDA runtime version recognized by PyTorch

True          →  GPU is available and correctly detected

NVIDIA RTX 2000 Ada Generation Laptop GPU  →  Your GPU model

To verify the packages installed in your `ml-seismic-waves-env-env` conda environment, you can use the following command:

 ```
conda list -n ml-seismic-waves-env
 ```
