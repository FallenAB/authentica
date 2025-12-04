# ...existing code...
import os
# suppress oneDNN informational message and lower TF log level (0=ALL,1=ERROR,2=WARNING,3=FATAL)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
print(f"✓ OpenCV: {cv2.__version__}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except:
    print("✗ PyTorch not found")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except:
    print("✗ TensorFlow not found")

try:
    import dlib
    print(f"✓ dlib available")
except:
    print("✗ dlib not available (will use simpler methods)")

try:
    import librosa
    print(f"✓ librosa: {librosa.__version__}")
except:
    print("✗ librosa not found")