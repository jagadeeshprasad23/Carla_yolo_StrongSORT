import carla
import math
import random
import time
import numpy as np
import cv2

# Trained models
YOLO_PATH = 'weights/best_n.pt'
#YOLO_PATH = 'weights/best_s.pt'

#yolo Pretrained models
#YOLO_PATH = 'weights/yolov8n.pt'
#YOLO_PATH = 'weights/yolov8s.pt'

#The local Host for carla simulator is 2000
client = carla.Client('localhost', 2000)
# world has methods to access all things in simulator(vehicles, buildings, etc and spawning)
client.set_timeout(20.0)
world = client.get_world() 

#blueprint will acess to all bps to create objects( like vehicles, people)
bp_lib = world.get_blueprint_library()
#spawn points to spawn
spawn_points = world.get_map().get_spawn_points()

#List of vehicles in the blueprint
#all_veh = [i for i in bp_lib if "vehicle" in i.tags]

#From vehicle blueprint getting the information of specific vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

#Shifting the view to the spectator of the car(camera)
spectator = world.get_spectator()
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x = -4, z = 2.5)), vehicle.get_transform().rotation)
spectator.set_transform(transform)

#Spawning the Vehicles as npc
spawn_num = 30
for i in range(spawn_num):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    
#autopiloting the vehicles
for v in world.get_actors().filter('*vehicle*'):
    v.set_autopilot(True)
    
#Attaching camera sensor to the vehicle
camera_bp = bp_lib.find('sensor.camera.rgb')

#Setting the width and height to display
IM_WIDTH =  256*4
IM_HEIGHT =  256*3

camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
camera_bp.set_attribute('fov', '110')

#transforming the carla into a desirable position and attaching to the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.6, x = 0.4))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to = vehicle)

#To save images in the drive
#camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

#To stop saving the images
#camera.stop()

#converting the image to use yolo
def camera_callback(image, data_dict):
    #data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    image_data = np.array(image.raw_data)
    image_rgb = image_data.reshape((image.height, image.width, 4))[:, :, :3]  # Extract RGB channels
    data_dict['image'] = image_rgb

image_w = camera_bp.get_attribute('image_size_x').as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()

camera_data = {'image': np.zeros((image_h, image_w, 4))}

camera.listen(lambda image:camera_callback(image, camera_data))

vehicle.set_autopilot(True)

img = camera_data['image']

classes_of_interest = ['car', 'bus', 'truck']

from ultralytics import YOLO

#use the best model for object detection

#model = YOLO('yolov8n.pt')
model = YOLO(YOLO_PATH)

while True:
    frame = camera_data['image']
    results = model(frame ,show = True) 

    # Display the frame
    cv2.imshow("YOLOv8", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
        
cv2.destroyAllWindows()
camera.stop()
camera.destroy()
vehicle.destroy()