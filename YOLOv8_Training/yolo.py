from ultralytics import YOLO
import torch

# Load a model
model = YOLO("yolov8s.yaml") 

# Use the model
def run():
    model.train(data="data.yaml", epochs=1000)  # train the model

# Ensure CUDA is available
if torch.cuda.is_available():
    run()
else:
    print("CUDA is not available on this device.")
