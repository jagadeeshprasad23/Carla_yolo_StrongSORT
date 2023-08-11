import torch
import numpy as np
import cv2
from time import perf_counter
from ultralytics import YOLO
import carla
import random
from strong_sort import StrongSORT 
from utils.parser import get_config

# For Custom models
trained_n = True 
trained_s = False

#For Pretrained yolo weights
yolov8n = False
yolov8s = False

# Based on the configuration it will load the weights

# Trained Models    
if trained_n:
    YOLO_PATH = 'weights/best_n.pt'
    CLASS_IDS = [0, 1]
    CLASS_NAMES = {0:'bike', 1: 'vehicle'}
    model_type = 'train_n'
    print('The tracker is using detection model trained on roboflow dataset "nano"')

if trained_s:
    YOLO_PATH = 'weights/best_s.pt'
    CLASS_IDS = [0, 1]
    CLASS_NAMES = {0:'bike', 1: 'vehicle'}
    model_type = 'train_s'
    print('The tracker is using detection model trained on roboflow dataset "small"')
    
# yolo pretrained models
if yolov8n:
    YOLO_PATH = 'weights/yolov8n.pt'
    CLASS_IDS = [1, 2, 3, 5, 7]
    CLASS_NAMES = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    model_type = 'yolov8n'
    print('The Tracker is using detection model trained on yolov8n')
    
if yolov8s:
    YOLO_PATH = 'weights/yolov8s.pt'
    CLASS_IDS = [1, 2, 3, 5, 7]
    CLASS_NAMES = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    model_type = 'yolov8s'
    print('The Tracker is using detection model trained on yolov8s')


#The image height and width should be mained in 256 multiplier format for yolo
IM_WIDTH = 256*4
IM_HEIGHT = 256*3

class main:
    def __init__(self):
        
        self.model = self.load_model() #loads Yolo model
        self.save_vid = True
        self.initialize_strongsort() 
        
    def initialize_strongsort(self):
        self.cfg = get_config()
        self.cfg.merge_from_file('configs/strong_sort.yaml')
        self.strong_sort_weights = 'weights/osnet_x0_25_market1501.pth'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        #From vehicle blueprint getting the information of specific vehicle
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

        #Shifting the view to the spectator of the car(camera)
        spectator = world.get_spectator()
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)), vehicle.get_transform().rotation)
        spectator.set_transform(transform)

        # change the spawn numbers accordingly
        spawn_num = 40
        for i in range(spawn_num):
            vehicle_bp = random.choice(bp_lib.filter('vehicle'))
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        
        #Set vehicles to Auto pilot    
        for v in world.get_actors().filter('*vehicle*'):
            v.set_autopilot(True)
            
        # Extract sensor data
        camera_bp = bp_lib.find('sensor.camera.rgb')
        
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '110') #field of view

        #moved camera towards hood
        camera_init_trans = carla.Transform(carla.Location(x = 1.5, z = 1.6 )) 
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
        
        fps = 3
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter('output_'+ model_type+ '.mp4',fmt, fps, (IM_WIDTH, IM_HEIGHT))
        
        curr_frames, prev_frames = None, None
        
        while True:
            
            start_time = perf_counter()
               
            if hasattr(self.strongsort, 'tracker'):
                if prev_frames in locals() and prev_frames is not None and curr_frames in locals() and curr_frames is not None:
                    self.strongsort.tracker.camera_update(prev_frames, curr_frames)
            
            frame = camera_data['image']
            outputs,confs = main.perf_track(self,frame)
            frame = np.array(frame)
            
            end_time = perf_counter()
            fps = 1 / np.round(end_time - start_time, 2)
            
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                        frame = main.annotation(self, frame, output, conf, fps)
            #save
            frame = cv2.UMat(frame)
            
            prev_frames = curr_frames
            
            cv2.imshow('Yolov8 StrongSORT', frame)
            
            if self.save_vid:
                writer.write(frame)
                
            if cv2.waitKey(1) == ord('q'):
                break
            
        writer.release()  # Release the VideoWriter  
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
    
    def annotation(self, frame, output, conf, fps):
        x1, y1, x2, y2 = map(int,output[0:4])
        id = int(output[4])
        clss = int(output[5])
        label = None
        if clss in CLASS_NAMES:
            label = CLASS_NAMES[clss]  # Make the object name change to match the clss number
        
        # Convert the frame to a NumPy array (if it's not already)
        frame = frame if isinstance(frame, np.ndarray) else np.array(frame)

        rectcolor = (0, 188, 68)
        linewidth = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), rectcolor, linewidth)

        textcolor = cv2.FONT_HERSHEY_SIMPLEX
        
        # Specify font style by path
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'{id} {label} {conf:.2f}'

        txpos = (x1, y1 - 10)  # Coordinates to start drawing text

        # Draw the text on the frame
        cv2.putText(frame, text, txpos, font, 1, textcolor, 2)
        
        # Add the fps information to the frame
        fps_text = f'FPS: {fps:.2f}'
        fps_position = (10, 50)  # Coordinates to place the fps text (top-left corner)
        cv2.putText(frame, fps_text, fps_position, font, 1, textcolor, 2)

        return frame
    
if __name__ == '__main__':
    run = main()