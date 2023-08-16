# Source link https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/#bounding-boxes

import json
from pascal_voc_writer import Writer
from typing import Any
import torch
import numpy as np
import cv2
from ultralytics import YOLO
import carla
import queue
import random
from strong_sort import StrongSORT
from sort.parser import get_config

# Part 1
image_w = 256*4
image_h = 256*3

# for yolo pretrained
# class_id = [2, 5, 7]
# YOLO_PATH = 'weights/yolov8s.pt'
# class_name = {1: 'bicycle' , 2: 'car', 3: 'motorcycle', 5: 'bus' ,7: 'truck'}

# for trained model
class_id = [1]
YOLO_PATH = 'weights/best_s.pt'
output_path = "output.mp4"

#strongSORT weights
cfg = get_config()
cfg.merge_from_file('configs/strong_sort.yaml')
strongsort_weights = "weights/osnet_x0_25_market1501.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#initialising strongSORT
strongsort = StrongSORT(
    strongsort_weights,
    device,
    max_dist=0.2,
    max_iou_distance=0.7,
    max_age=70, n_init=3,
    nn_budget=100,
    mc_lambda=0.995,
    ema_alpha=0.9)

#connecting CARLA Simulator
client = carla.Client('localhost', 2000)
world = client.get_world()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True  # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Get the world spectator
spectator = world.get_spectator()

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# spawn vehicle
bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

camera_bp = bp_lib.find('sensor.camera.rgb')

camera_bp.set_attribute('image_size_x', f'{image_w}')
camera_bp.set_attribute('image_size_y', f'{image_h}')
camera_bp.set_attribute('fov', '110')
fov = 110

# spawn camera
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
'''
def camera_callback(image, data_dict):
            image_data = np.array(image.raw_data)
            image_rgb = image_data.reshape((image.height, image.width, 4))[:, :, :3]
            data_dict['image'] = image_rgb
'''
image_w = camera_bp.get_attribute('image_size_x').as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()

# camera_data = {'image': np.zeros((image_h, image_w, 4))}
# camera.listen(lambda image: camera_callback(image, camera_data))

vehicle.set_autopilot(True)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    point_camera = np.array(
        [point_camera[1], -point_camera[2], point_camera[0]]).T

    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img


# Remember the edge pairs
edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
         [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

# Get the world to camera matrix
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

for i in range(50):
    vehicle_bp = bp_lib.filter('vehicle')

    # Exclude bicycle
    car_bp = [bp for bp in vehicle_bp if int(
        bp.get_attribute('number_of_wheels')) == 4]
    npc = world.try_spawn_actor(random.choice(
        car_bp), random.choice(spawn_points))

    if npc:
        npc.set_autopilot(True)

# Retrieve all the objects of the level
car_objects = world.get_environment_objects(carla.CityObjectLabel.Car)
truck_objects = world.get_environment_objects(carla.CityObjectLabel.Truck)
bus_objects = world.get_environment_objects(carla.CityObjectLabel.Bus)

env_object_ids = []

for obj in (car_objects + truck_objects + bus_objects):
    env_object_ids.append(obj.id)

# Disable all static vehicles
world.enable_environment_objects(env_object_ids, False)

edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
         [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

def get_vanishing_point(p1, p2, p3, p4):

    k1 = (p4[1] - p3[1]) / (p4[0] - p3[0])
    k2 = (p2[1] - p1[1]) / (p2[0] - p1[0])

    vp_x = (k1 * p3[0] - k2 * p1[0] + p1[1] - p3[1]) / (k1 - k2)
    vp_y = k1 * (vp_x - p3[0]) + p3[1]

    return [vp_x, vp_y]

def clear():
    settings = world.get_settings()
    settings.synchronous_mode = False  # Disables synchronous mode
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    camera.stop()

    for npc in world.get_actors().filter('*vehicle*'):
        if npc:
            npc.destroy()

    print("Vehicles Destroyed.")

# Main Loop
vehicle.set_autopilot(True)

edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
         [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

carla_annotations = []
yolo_annotations = []

while True:
    try:
        world.tick()

        # Move the spectator to the top of the vehicle
        transform = carla.Transform(vehicle.get_transform().transform(
            carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
        spectator.set_transform(transform)

        # Retrieve and reshape the image
        image = image_queue.get()
        # img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        img = np.reshape(np.copy(image.raw_data),
                         (image.height, image.width, 4))[:, :, :3]

        timestamp_sec = image.timestamp

        # Get the camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the image frame from the image queue
        frame = np.copy(img)  # Use the image from the queue
        carla_gt_annotations = []
        yolo_gt_annotations = []

        # Perform YOLO object detection
        model = YOLO(YOLO_PATH)
        preds = model(frame)

        bbox_xyxy = []
        conf_score = []
        cls_id = []
        outputs = []

        # Iterate through the detected objects and their bounding boxes
        for box in preds:
            for r in box.boxes.data.tolist():
                x_min, y_min, x_max, y_max, conf, class_ids = r
                id = int(class_ids)
                if id in class_id:
                    bbox_xyxy.append(
                        [int(x_min), int(y_min), int(x_max), int(y_max)])
                    conf_score.append(conf)
                    cls_id.append(int(id))
                else:
                    continue
                outputs = strongsort.update(
                    bbox_xyxy, conf_score, cls_id, frame)
                for output, conf, id in zip(outputs, conf_score, cls_id):
                    yolo_gt_annotations.append({

                        "height": output[3] - output[1],
                        "width": output[2] - output[0],
                        "id": "vehicle",  # Replace with actual class name
                        "y": output[1],
                        "x": output[0]
                    })

        yolo_annotations.append({
            "timestamp": timestamp_sec,
            "num": image.frame,
            "class": "frame",
            "hypotheses": yolo_gt_annotations
        })

        yolo_output = [{
            "frames": yolo_annotations,
            "class": "video",
            "filename": "yolo_gt.json"
        }]

        for npc in world.get_actors().filter('*vehicle*'):

            # Filter out the ego vehicle
            if npc.id != vehicle.id:

                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                # Filter for the vehicles within 50m
                if dist < 50:

                    forward_vec = vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - vehicle.get_transform().location

                    if forward_vec.dot(ray) > 0:

                        verts = [v for v in bb.get_world_vertices(
                            npc.get_transform())]

                        points_image = []

                        for vert in verts:
                            ray0 = vert - camera.get_transform().location
                            cam_forward_vec = camera.get_transform().get_forward_vector()

                            if (cam_forward_vec.dot(ray0) > 0):
                                p = get_image_point(vert, K, world_2_camera)
                            else:
                                p = get_image_point(vert, K_b, world_2_camera)

                            points_image.append(p)

                        x_min, x_max = 10000, -10000
                        y_min, y_max = 10000, -10000

                        for edge in edges:
                            p1 = points_image[edge[0]]
                            p2 = points_image[edge[1]]

                            p1_in_canvas = point_in_canvas(
                                p1, image_h, image_w)
                            p2_in_canvas = point_in_canvas(
                                p2, image_h, image_w)

                            # Both points are out of the canvas
                            if not p1_in_canvas and not p2_in_canvas:
                                continue

                            # Draw 2D Bounding Boxes
                            p1_temp, p2_temp = (p1.copy(), p2.copy())

                            # One of the point is out of the canvas
                            if not (p1_in_canvas and p2_in_canvas):
                                p = [0, 0]

                                # Find the intersection of the edge with the window border
                                p_in_canvas, p_not_in_canvas = (
                                    p1, p2) if p1_in_canvas else (p2, p1)
                                k = (
                                    p_not_in_canvas[1] - p_in_canvas[1]) / (p_not_in_canvas[0] - p_in_canvas[0])

                                x = np.clip(p_not_in_canvas[0], 0, image.width)
                                y = k * (x - p_in_canvas[0]) + p_in_canvas[1]

                                if y >= image.height:
                                    p[0] = (image.height - p_in_canvas[1]
                                            ) / k + p_in_canvas[0]
                                    p[1] = image.height - 1
                                elif y <= 0:
                                    p[0] = (0 - p_in_canvas[1]) / \
                                        k + p_in_canvas[0]
                                    p[1] = 0
                                else:
                                    p[0] = image.width - \
                                        1 if x == image.width else 0
                                    p[1] = y

                                p1_temp, p2_temp = (p, p_in_canvas)

                            # Find the rightmost vertex
                            x_max = p1_temp[0] if p1_temp[0] > x_max else x_max
                            x_max = p2_temp[0] if p2_temp[0] > x_max else x_max

                            # Find the leftmost vertex
                            x_min = p1_temp[0] if p1_temp[0] < x_min else x_min
                            x_min = p2_temp[0] if p2_temp[0] < x_min else x_min

                            # Find the highest vertex
                            y_max = p1_temp[1] if p1_temp[1] > y_max else y_max
                            y_max = p2_temp[1] if p2_temp[1] > y_max else y_max

                            # Find the lowest vertex
                            y_min = p1_temp[1] if p1_temp[1] < y_min else y_min
                            y_min = p2_temp[1] if p2_temp[1] < y_min else y_min

                        # Exclude very small bounding boxes
                        if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                            if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                                img = np.array(img, dtype=np.uint8)
                                cv2.line(img, (int(x_min), int(y_min)), (int(
                                    x_max), int(y_min)), (0, 0, 255, 255), 1)
                                cv2.line(img, (int(x_min), int(y_max)), (int(
                                    x_max), int(y_max)), (0, 0, 255, 255), 1)
                                cv2.line(img, (int(x_min), int(y_min)), (int(
                                    x_min), int(y_max)), (0, 0, 255, 255), 1)
                                cv2.line(img, (int(x_max), int(y_min)), (int(
                                    x_max), int(y_max)), (0, 0, 255, 255), 1)

                                # timestamp_sec = image.timestamp.elapsed_seconds

                            carla_gt_annotations.append({
                                "dco": True,
                                "height": y_max - y_min,
                                "width": x_max - x_min,
                                "id": "vehicle",
                                "y": y_min,
                                "x": x_min
                            })
        carla_annotations.append({
            "timestamp": timestamp_sec,
            "num": image.frame,
            "class": "frame",
            "annotations": carla_gt_annotations
        })

        gt_output = [{
            "frames": carla_annotations,
            "class": "video",
            "filename": "carla_gt.json"
        }]

        with open('carla_gt.json', 'w') as json_file:
            json.dump(gt_output, json_file)

        with open('yolo_gt.json', 'w') as json_file:
            json.dump(yolo_output, json_file)

        cv2.imshow('Ground Truth', img)

        if cv2.waitKey(1) == ord('q'):
            clear()
            break

    except KeyboardInterrupt as e:
        clear()
        break

camera.stop()
camera.destroy()
vehicle.destroy()

cv2.destroyAllWindows()