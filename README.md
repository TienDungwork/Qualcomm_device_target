# Qualcomm Device Target Setup Guide

## 1. Trên thiết bị Target

Copy file model vừa được convert `model.bin` từ Host sang Target.

### Tải SDK

Tải SDK tương tự từ: https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK

```bash
unzip v2.40.0.251030.zip
cd qairt/v2.40.0.251030
```

## 2. Thiết lập môi trường

### Tạo và kích hoạt môi trường ảo

```bash
python -m venv "<venv_path>"
source <venv_path>/bin/activate
```

### Cài đặt các dependencies cơ bản

```bash
pip install --upgrade pip setuptools wheel
pip install pybind11==2.13.6
apt-get update && apt-get install -y python3.10-dev
```

### Cài đặt AI Engine Direct Helper

```bash
cd ai-engine-direct-helper
export QNN_SDK_ROOT=`pwd\`
python setup.py bdist_wheel
pip install dist/qai_appbuilder-2.40.0-cp310-cp310-linux_aarch64.whl --force-reinstall
```
### Cài đặt các thư viện bổ sung

```bash
pip install requests py3-wget tqdm importlib-metadata qai-hub Pillow
pip install torch torchvision opencv-python
```

## 3. Test

Chạy script test YOLOv8:

```bash
bash run_yolov8.sh
```

## 4. Hàm infer đã viết sẵn
detect_qnn.py và infer_qnn.py
