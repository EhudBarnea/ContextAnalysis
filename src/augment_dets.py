from attach_det_windows_to_gt import *
import time
from tqdm import tqdm


def augment_dets(data, params):
    # add information to data:
    # num_positives - total number of positives (for each label)
    # data['dets'], data['dets_pi'] - a list of all dataset detections with:
    #     fp/tp, dont_care, false_type

    # false types:
    # -1 = correct detection
    #  0 = confusion with background
    #  1 = localization error
    #  2 = confusion with another class

    overlap_threshold = params['overlap']

    print('augmenting detections', flush=True)
    time.sleep(0.1)

    num_positives = {x: 0 for x in params['object_labels']}
    for img_id in tqdm(data['imgs'].keys()):
        objects = data['objects_pi'][img_id]
        detections = data['dets_pi'][img_id]

        # count total number of positives
        for o in objects:
            if not o['dont_care']:
                num_positives[o['label']] += 1

        # attach correctness to detections
        det_correct = attach_det_windows_to_gt(detections, objects, overlap_threshold)
        for di, d in enumerate(detections):
            d['correct'] = det_correct[di, 0]
            d['dont_care'] = det_correct[di, 2]
            d['fp_type'] = det_correct[di, 3]

    time.sleep(0.1)
    print('done augmenting detections', flush=True)

    return num_positives
