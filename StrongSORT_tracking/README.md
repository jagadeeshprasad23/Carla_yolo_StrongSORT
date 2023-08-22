# StrongSORT Algorithm

### The code used under sort folder is used from the original strongSORT repository. 
##### Ground Truth, Yolo detection and Ground Truth Scores will have copies both in python file(.py) and jupyter notebook(.ipynb). The license is included.

##### Configs folder contains the stronSORT configuration yaml file.

##### Weights folder contains the pretrained yolo models: yolo nano model(best_n.pt) and yolo small model(best_s.pt) also the pretrained weights for the strongSORT algorithm from osnet. And included yolo pretrained weights of nano and small models(yolov8n.pt and yolov8s.pt)

##### To change the yolo models download navigate to carla_tracker.py and select the desired model.

##### Note: yolo pretrained weights contain 80 classes. From those classes the vehicles are filtered for detection.