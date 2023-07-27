from typing import Any
import torch
import numpy as np
import cv2
from time import perf_counter
from ultralytics import YOLO
import yaml
from pathlib import Path
import supervision as sv
from types import SimpleNamespace
import carla
import math
import random
import time
from strongsort.strong_sort import StrongSORT        

SAVE_VIDEO = True
TRACKER = 'strongsort'
IM_WIDTH = 256*4
IM_HEIGHT = 256*3

YOLO_PATH = 'weights/yolov8n.pt'
YOLO_PATH = 'weights/best.pt'

class ObjectTracking:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('using device: ', self.device)
        
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
        
        # reid weights
        reid_weights = Path('weights/osnet_x0_25_msmt17.pt')
        
        if SAVE_VIDEO:
            self.video_save_path = "output_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_save_path, fourcc, 20.0, (IM_WIDTH, IM_HEIGHT))
        
        if TRACKER == 'strongsort':
            tracker_config = 'configs/strongsort.yaml' 
            with open(tracker_config, 'r') as file:
                cfg = yaml.load(file, Loader=yaml.FullLoader)
            cfg = SimpleNamespace(**cfg)
            
            self.tracker = StrongSORT(
                reid_weights,
                torch.device(self.device),
                False,
                max_dist=cfg.max_dist,
                max_iou_dist=cfg.max_iou_dist,
                max_age=cfg.max_age,
                max_unmatched_preds=cfg.max_unmatched_preds,
                n_init=cfg.n_init,
                nn_budget=cfg.nn_budget,
                mc_lambda=cfg.mc_lambda,
                ema_alpha=cfg.ema_alpha,
            )
    def load_model(self):
        model = YOLO(YOLO_PATH)
        
        return model

    def __call__(self):
        # The local Host for carla simulator is 2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world() 

        # blueprint will access to all blueprints to create objects (vehicles, people, etc.)
        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        spectator = world.get_spectator()
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)), vehicle.get_transform().rotation)
        spectator.set_transform(transform)

        spawn_num = 30
        for i in range(spawn_num):
            vehicle_bp = random.choice(bp_lib.filter('vehicle'))
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            
        for v in world.get_actors().filter('*vehicle*'):
            v.set_autopilot(True)
            
        camera_bp = bp_lib.find('sensor.camera.rgb')
        
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '110')

        camera_init_trans = carla.Transform(carla.Location(z=1.6, x=0.4))
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

        def camera_callback(image, data_dict):
            image_data = np.array(image.raw_data)
            image_rgb = image_data.reshape((image.height, image.width, 4))[:, :, :3]
            data_dict['image'] = image_rgb

        image_w = camera_bp.get_attribute('image_size_x').as_int()
        image_h = camera_bp.get_attribute('image_size_y').as_int()

        camera_data = {'image': np.zeros((image_h, image_w, 4))}
        camera.listen(lambda image: camera_callback(image, camera_data))

        vehicle.set_autopilot(True)
        
        curr_frames, prev_frames = None, None
        
        #model = YOLO('yolov8n.pt')
        
        while True:
            details = []
            start_time = perf_counter()
            frame = camera_data['image']
            results = self.model(frame)
            
            for box in results:  
                for r in box.boxes.data.tolist():
                    x1, y1, x2, y2, conf, id = r
                    '''
                    x1 = torch.tensor(x1)
                    y1 = torch.tensor(y1)
                    x2 = torch.tensor(x2)
                    y2 = torch.tensor(y2)
                    conf = torch.tensor(conf)
                    details.append([x1.cpu().numpy(), y1.cpu().numpy(), x2.cpu().numpy(), y2.cpu().numpy(), conf.cpu().numpy(),id])
                    '''
                    #else
                    details.append([int(x), int(y), int(w), int(h), conf,id])

            np_details = np.array(details) #only send nmupy array 

            if hasattr(self.tracker, 'tracker'):
                if prev_frames in locals() and prev_frames is not None and curr_frames in locals() and curr_frames is not None:
                    self.tracker.tracker.camera_update(prev_frames, curr_frames)

            outputs = [None]
            #for detail in np_details:
            outputs[0] = self.tracker.update(np_details, frame)
            for i, output in enumerate(outputs[0]):
                bbox = output[0:4]
                tracked_id = output[4]
                top_left = (int(bbox[-2] - 100), int(bbox[1]))
                frame = cv2.UMat(frame)
                cv2.putText(frame, f"ID: {tracked_id}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            end_time = perf_counter()
            fps = 1 / np.round(end_time - start_time, 2)

            # Convert the frame to a compatible format (e.g., cv::UMat) if needed
            frame = cv2.UMat(frame)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('Yolov8 StrongSORT', frame)

            if cv2.waitKey(1) == ord('q'):
                break
        if SAVE_VIDEO:
            self.video_writer.release()

        cv2.destroyAllWindows()
        camera.stop()
        camera.destroy()
        vehicle.destroy()

tracker = ObjectTracking()
tracker()