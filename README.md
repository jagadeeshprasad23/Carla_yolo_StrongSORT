# Real Time Vehicle Tracking in Carla Simulator with Yolo and StrongSORT

# Requirements 

Download and Install Carla simulator [Carla Installation link](https://carla.readthedocs.io/en/latest/start_quickstart/)

Download and Install Cuda [cuda download Link](https://developer.nvidia.com/cuda-downloads)
Install Cuda DNN [cudeDNN download Link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

create a new virtual environment in conda 
```
conda create --name <env-name> python=3.8
# example : conda create --name carla-sim python=3.8

#activate the virtual environnment
conda activate <env-name>   
# example : conda activate carla-sim

```
install carla in conda virtual environment
```
pip install carla
```
Run CarlaE4.exe and Run any of the following in the command prompt 
```
StrongSORT/tracker.py
```
Or Open Jupyter notebook/ Jupyter-lab and run the following notebook
```
StrongSORT/tracker.ipynb
```
