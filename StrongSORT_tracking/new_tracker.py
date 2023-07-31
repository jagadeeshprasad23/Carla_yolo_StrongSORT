from typing import Any
import torch
import numpy as np
import cv2
from time import perf_counter
from ultralytics import YOLO
import yaml
from pathlib import Path
from types import SimpleNamespace
import carla
import random
import os
from strong_sort.strong_sort import StrongSORT 
from strong_sort.utils.parser import get_config
from PIL import Image, ImageDraw, ImageFont

#For Pretrained Weights
#YOLO_PATH = 'weights/best.pt'
#CLASS_IDS = [0,1,2]
#CLASS_NAMES = {0: 'bicycle', 1: 'motorcycle', 2: 'vehicle',}

YOLO_PATH = 'weights/yolov8n.pt'
CLASS_IDS = [1, 2, 3, 5, 7]
CLASS_NAMES = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

IM_WIDTH = 256*4
IM_HEIGHT = 256*3

class main:
    def __init__(self):
        
        self.model = self.load_model()
        self.save_vid = True
        
        # initialize StrongSORT
        self.cfg = get_config()
        self.cfg.merge_from_file('strong_sort/configs/strong_sort.yaml')
        self.strong_sort_weights = 'strong_sort/deep/checkpoint/osnet_x0_25_market1501.pth'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')

        self.strongsort = StrongSORT(
            self.strong_sort_weights,
            self.device,
            max_dist=self.cfg.STRONGSORT.MAX_DIST,
            max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
            max_age=self.cfg.STRONGSORT.MAX_AGE,
            n_init=self.cfg.STRONGSORT.N_INIT,
            nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
            mc_lambda=self.cfg.STRONGSORT.MC_LAMBDA,
            ema_alpha=self.cfg.STRONGSORT.EMA_ALPHA,
        )
        
    def load_model(self):
        # Change path accordingly
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
        
        fps = 0
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('output.mp4',fmt, fps, (IM_WIDTH, IM_HEIGHT))
        
        curr_frames, prev_frames = None, None
        
        while True:
            
            start_time = perf_counter()
            
            '''
            if self.cfg.STRONGSORT.ECC:
                self.strongsort.tracker.camera_update(prev_frames, curr_frames)
            '''    
            if hasattr(self.strongsort, 'tracker'):
                if prev_frames in locals() and prev_frames is not None and curr_frames in locals() and curr_frames is not None:
                    self.strongsort.tracker.camera_update(prev_frames, curr_frames)
            
            frame = camera_data['image']
            outputs,confs = main.perf_track(self,frame)
            frame = np.array(frame)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                        frame = main.annotation(self, frame, output, conf)
            #save
            end_time = perf_counter()
            fps = 1 / np.round(end_time - start_time, 2)
            
            frame = cv2.UMat(frame)
            
            prev_frames = curr_frames
            
            cv2.imshow('Yolov8 StrongSORT', frame)
            
            if self.save_vid:
                writer.write(frame)
                
            if cv2.waitKey(1) == ord('q'):
                break
            
        cv2.destroyAllWindows()
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        
    def perf_track(self, frame):
        
        preds = self.model(frame)
        
        bbox_xyxy = []
        confs = []
        clss = []
        outputs = []
        
        for box in preds:  
            for r in box.boxes.data.tolist():
                x1, y1, x2, y2, conf, id = r
                
                if int(id) in CLASS_IDS:
                    bbox_xyxy.append([int(x1), int(y1), int(x2), int(y2)])
                    confs.append(conf)
                    clss.append(int(id))
                    
                else:
                        continue
                    
            print('Results: ', bbox_xyxy, confs, clss)        
            outputs = self.strongsort.update(bbox_xyxy, confs, clss, frame)
            print('Output values: ', outputs)
        return outputs,confs
    
    def annotation(self, frame, output, conf):
        x1, y1, x2, y2 = map(int,output[0:4])
        id = int(output[4])
        clss = int(output[5])
        label = None
        if clss in CLASS_NAMES:
            label = CLASS_NAMES[clss]  # Make the object name change to match the clss number
        

        # Convert the frame to a NumPy array (if it's not already)
        frame = frame if isinstance(frame, np.ndarray) else np.array(frame)

        rectcolor = (0, 188, 68)
        linewidth = 8
        cv2.rectangle(frame, (x1, y1), (x2, y2), rectcolor, linewidth)

        textcolor = cv2.FONT_HERSHEY_SIMPLEX
        textsize = 40

        # Specify font style by path
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = f'{id} {label} {conf:.2f}'

        txpos = (x1, y1 - 10)  # Coordinates to start drawing text

        # Draw the filled rectangle as background for the text
        cv2.rectangle(frame, txpos, (int(output[0]) + len(text) * 20, int(output[1])), rectcolor, -1)

        # Draw the text on the frame
        cv2.putText(frame, text, txpos, font, 1, textcolor, 2)

        return frame
    
if __name__ == '__main__':
    run = main()