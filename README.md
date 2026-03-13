# HDINO

This is the official implementation of **HDINO**.

Please feel free to reach out if you have any questions or suggestions.

## Updates
* **[2026-03-13]**: Repository initialized and model weights uploaded.

### 1. Create a virtual environment
```bash
conda create -n hdino python=3.10 -y
conda activate hdino


### 2. Install PyTorch and CUDA
The following command installs the specific versions used in our development environment (PyTorch 2.1.0 + CUDA 12.1):
    torch                         2.1.0+cu121
    torchvision                   0.16.0+cu121

### 🎨 Demo
You can run our interactive demo locally to experience **HDINO**:
```bash
python gradio_demo.py

## 📜 Acknowledgement
We express our sincere gratitude to the authors for their contributions to the community:
* [DINO](https://github.com/IDEA-Research/DINO)
* [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
* [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino)
