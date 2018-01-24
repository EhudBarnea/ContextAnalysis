import get_image_size
import numpy as np
from collections import defaultdict


def read_data_kitti_ssd(params, subset):
    # subset = 'train' / 'test / 'both'

    # min_conf = 0.1
    min_conf = -1

    print('Loading objects and detections')

    # load splits
    with open(params['train_split_fname']) as f:
        content = f.readlines()
    train_imgs = [int(x.strip()) - 1 for x in content]  # remove '\n' and convert to int
    with open(params['test_split_fname']) as f:
        content = f.readlines()
    test_imgs = [int(x.strip()) - 1 for x in content]  # remove '\n' and convert to int

    # choose working split
    if subset == 'train':
        img_nums = train_imgs
    elif subset == 'test':
        img_nums = test_imgs
    elif subset == 'both':
        img_nums = train_imgs + test_imgs
    else:
        raise RuntimeError('Wrong subset')

    # read test detections, objects, etc
    imgs = {}
    objects = []  # list of all objects
    objects_pi = defaultdict(list)  # objects arrange per image
    obj_id = 0
    dets = []
    dets_pi = defaultdict(list)
    det_id = 0
    for img_num in img_nums:
        # get image size
        img_name_base = str(img_num).rjust(6, '0')
        img_name = img_name_base + '.png'
        fname = params['imgs_folder'] + img_name
        width, height = get_image_size.get_image_size(fname)
        # index image
        imgs[img_num] = {'id': img_num, 'width': width, 'height': height, 'file_name': img_name}

        # read objects
        fname = params['objects_folder'] + img_name_base + '.txt'
        with open(fname) as f:
            content = f.readlines()
        content = [x.strip() for x in content]  # remove '\n'
        for line in content:
            l = line.split(' ')

            # Handle labels and dont_cares
            label = l[0].lower()
            if label in params['object_labels']:
                dont_care = False
            elif label == 'dontcare' or label == 'van':
                dont_care = True
                label = 'dontcare'
            else:
                continue

            x1 = int(round(float(l[4])))
            y1 = int(round(float(l[5])))
            x2 = int(round(float(l[6])))
            y2 = int(round(float(l[7])))

            obj_new = {'label': label, 'x1': x1, 'y1': y1, 'x2': x2,
                       'y2': y2, 'dont_care': dont_care, 'id': obj_id, 'img_id': img_num}
            objects.append(obj_new)
            objects_pi[img_num].append(obj_new)
            obj_id += 1

        # read detections
        fname = params['dets_folder'] + str(img_num).rjust(6, '0') + '.txt'
        with open(fname) as f:
            content = f.readlines()
        content = [x.strip() for x in content]  # remove '\n'
        for line in content:
            l = line.split(' ')
            conf = float(l[15])

            bbox = np.array([int(l[4]), int(l[5]), int(l[6]), int(l[7])])
            bbox[bbox < 0] = 0
            if bbox[2] >= width:
                bbox[2] = width - 1
            if bbox[3] >= height:
                bbox[3] = height - 1

            if conf > min_conf:
                det_new = {'label': l[0], 'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3],
                           'conf': conf, 'id': det_id, 'img_id': img_num}
                dets.append(det_new)
                dets_pi[img_num].append(det_new)
                det_id += 1

    data = {'imgs': imgs, 'objects': objects, 'objects_pi': objects_pi, 'dets': dets, 'dets_pi': dets_pi}
    return data
