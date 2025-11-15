export QNN_SDK_ROOT=/home/ntiendung/qairt/2.40.0.251030
export ADSP_LIBRARY_PATH=$QNN_SDK_ROOT/lib/hexagon-v73/unsigned
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2
 
python ai-engine-direct-helper/samples/python/yolov8_det/yolov8_det.py
