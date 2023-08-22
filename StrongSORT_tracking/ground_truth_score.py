#source : https://github.com/cheind/py-motmetrics

import motmetrics as mm
import json

# carla, tracked JSON files
with open('carla_gt.json', 'r') as f:
    carla_value = json.load(f)
with open('yolo_gt.json', 'r') as f:
    track_value = json.load(f)

acc = mm.MOTAccumulator(auto_id=True)

# Iterate through each frame of data
for frame in range(len(carla_value[0]['frames'])):
    carla_frame = carla_value[0]['frames'][frame]
    track_frame = track_value[0]['frames'][frame]

    gt_boxes = [[d['x'], d['y'], d['width'], d['height']] for d in carla_frame['annotations']]           
    ds_boxes = [[d['x'], d['y'], d['width'], d['height']] for d in track_frame['hypotheses']]

    # IOU
    distances = mm.distances.iou_matrix(gt_boxes, ds_boxes, max_iou=0.5)

    # Update the accumulator with the calculated distances
    acc.update([d['id'] for d in carla_frame['annotations']],
                [d['id'] for d in track_frame['hypotheses']], distances)

# Calculate the MOTA and MOTP metrics from the accumulator data
mh = mm.metrics.create()
results = mh.compute(acc, metrics=['mota', 'motp'], name='acc')
print(results)