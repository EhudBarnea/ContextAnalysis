import numpy as np
import json
from collections import defaultdict


def read_data_coco_faster_rcnn(params, subset, num_imgs=-1):
    # subset = 'train'/'test'/'both'
    # if 'both' - instead of all the validation set
    # use the 8k minival used for testing in Google's 'speed/accuracy tradeoff' paper.

    # num_imgs - limit the number of images to use (set to -1 for all images)

    # min_conf = 0.1
    min_conf = -1

    # num_imgs = 1000

    print('Loading objects and detections')

    # load dataset
    if subset == 'train':
        dataset = json.load(open(params['train_anno_fname'], 'r'))
        dets_json = json.load(open(params['dets_folder'] + 'dets_train.json', 'r'))
    elif subset == 'test' or subset == 'both':
        dataset = json.load(open(params['test_anno_fname'], 'r'))
        dets_json = json.load(open(params['dets_folder'] + 'dets_val.json', 'r'))

        # keep only images from the minival 8k test set (speed/accurate paper)
        if subset == 'test':
            with open(params['test_minival_fname'], "r") as infile:
                minival_imgs = [int(line) for line in infile]
            dataset['images'] = [img for img in dataset['images'] if img['id'] in minival_imgs]
    else:
        raise RuntimeError('Wrong subset')

    # get categories and their IDs
    cats = {cat['id']: cat['name'] for cat in dataset['categories']}

    # index dataset
    if num_imgs < 1:
        img_set = dataset['images']
    else:
        img_set = dataset['images'][: min(len(dataset['images']), num_imgs)]
    imgs = {img['id']: img for img in img_set}

    objects = []  # list of all objects
    objects_pi = defaultdict(list)  # objects arrange per image
    for obj_id, obj in enumerate(dataset['annotations']):
        label = cats[obj['category_id']]
        if obj['image_id'] not in imgs or label not in params['object_labels']:
            continue
        bb = obj['bbox']
        x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
        if obj['iscrowd']:
            dont_care = True
        else:
            dont_care = False
        obj_new = {'label': label, 'dont_care': dont_care, 'id': obj_id,
                   'img_id': obj['image_id'], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

        if 'ignore' in obj.keys():
            print('now')

        objects.append(obj_new)
        objects_pi[obj_new['img_id']].append(obj_new)

    # load detections
    dets = []
    dets_pi = defaultdict(list)
    det_id = 0
    for img_name, img_dets in dets_json.items():
        img_id = int(img_name[-16:-4])
        if img_id not in imgs:
            continue
        for det_i, det in enumerate(img_dets):
            if det['conf'] > min_conf and det['category'] in params['object_labels']:
                bbox = np.array([det['x1'], det['y1'], det['x2'], det['y2']])
                bbox[bbox < 0] = 0
                if bbox[2] >= imgs[img_id]['width']:
                    bbox[2] = imgs[img_id]['width'] - 1
                if bbox[3] >= imgs[img_id]['height']:
                    bbox[3] = imgs[img_id]['height'] - 1

                det_new = {'label': det['category'], 'conf': det['conf'], 'id': det_id, 'id_in_img': det_i,
                           'img_id': img_id, 'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3]}
                dets.append(det_new)
                dets_pi[img_id].append(det_new)
                det_id += 1

    data = {'imgs': imgs, 'objects': objects, 'objects_pi': objects_pi, 'dets': dets, 'dets_pi': dets_pi}
    return data
