# Real Time Vehicle Tracking in Carla Simulator with Yolo and StrongSORT

# Requirements 

Install Carla simulator [documentation](https://carla.readthedocs.io/en/latest/start_quickstart/)

Install Cuda
```
https://developer.nvidia.com/cuda-downloads
```
Install Cuda DNN
```
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
```
Install Tensorrt
```
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
```
create a new virtual environment in conda 
```
conda create --name carla-sim python=3.8
```
install carla in conda virtual environment
```
pip install carla
```
install Linear Assignment Problem
```
conda install -c conda-forge lap
```
Run CarlaE4.exe and Run any of the following in the command prompt 
```
tracking/tracker.py
```
Or Open Jupyter notebook/ Jupyter-lab and run the following notebook
```
tracking/tracker.ipynb
```
