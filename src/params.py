from mkdir import *


def get_params():
    hostname = os.uname()[1]
    # print('hostname - ' + hostname)

    params = dict()
    # params['dataset'] = 'kitti'
    params['dataset'] = 'coco'

    # number of processes to use
    params['num_proc'] = 1
    # params['num_proc'] = 35
    # params['num_proc'] = 15

    params['out_folder'] = '../out/nn_big_rect/'

    if params['dataset'] == 'coco':
        # notice - the index of the following categories does not correspond to COCO's category id
        # params['object_labels'] = ['person', 'bicycle', 'hair drier', 'cup', 'fork', 'knife', 'bottle']
        params['object_labels'] = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                                   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                                   'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                                   'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                                   'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                                   'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                                   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                                   'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                                   'toothbrush']
        # params['object_labels'] = ['person', 'bicycle']
        if hostname[0:5] == 'acomp':
            params['train_anno_fname'] = '/Users/ehud/Work/Datasets/coco/annotations/instances_train2014.json'
            params['test_anno_fname'] = '/Users/ehud/Work/Datasets/coco/annotations/instances_val2014.json'
            params['test_minival_fname'] = \
                '/Users/ehud/Work/Datasets/coco/annotations/tensorflor_detectors_coco_8k_minival.txt'
            params['dets_folder'] = '/Users/ehud/Work/detectors/coco_faster_rcnn_res/'
            params['imgs_folder'] = '/Users/ehud/Work/Datasets/coco/images/'
        elif hostname[0:3] == 'sge':
            params['train_anno_fname'] = '/fastspace/users/barneaeh/datasets/coco/annotations/instances_train2014.json'
            params['test_anno_fname'] = '/fastspace/users/barneaeh/datasets/coco/annotations/instances_val2014.json'
            params['test_minival_fname'] = \
                '/fastspace/users/barneaeh/datasets/coco/annotations/tensorflor_detectors_coco_8k_minival.txt'
            params['dets_folder'] = '/fastspace/users/barneaeh/detres/'
            params['imgs_folder'] = ''
        elif hostname == 'edd90cabbcf8':
            params['train_anno_fname'] = '/root/mnt_ws/data/coco/annotations/instances_train2014.json'
            params['test_anno_fname'] = '/root/mnt_ws/data/coco/annotations/instances_val2014.json'
            params['test_minival_fname'] = \
                '/root/mnt_ws/data/coco/annotations/tensorflor_detectors_coco_8k_minival.txt'
            params['dets_folder'] = '/root/mnt_ws//det_res/coco_faster_rcnn/'
            params['imgs_folder'] = '/root/mnt_ws/data/coco/images/'
        else:
            raise RuntimeError('Unknown hostname ' + hostname)
    elif params['dataset'] == 'kitti':
        params['object_labels'] = ['car', 'pedestrian', 'cyclist']
        if hostname[0:5] == 'acomp':
            params['train_split_fname'] = \
                '/Users/ehud/Work/ssd_detector_over_kitti/src/data/KITTI/splits/split1_train.txt'
            params['test_split_fname'] = \
                '/Users/ehud/Work/ssd_detector_over_kitti/src/data/KITTI/splits/split1_test.txt'
            params['dets_folder'] = '/Users/ehud/Work/ssd_detector_over_kitti/out_all_dets/'
            params['objects_folder'] = '/Users/ehud/Work/Datasets/KITTI/training/label_2/'
            params['imgs_folder'] = '/Users/ehud/Work/Datasets/KITTI/training/image_2/'
        elif hostname[0:3] == 'sge':
            params['train_split_fname'] = \
                '/fastspace/users/barneaeh/m2c/kitti_data/splits/split1_train.txt'
            params['test_split_fname'] = \
                '/fastspace/users/barneaeh/m2c/kitti_data/splits/split1_test.txt'
            params['dets_folder'] = '/fastspace/users/barneaeh/m2c/kitti_data/out_all_dets/'
            params['objects_folder'] = '/fastspace/users/barneaeh/m2c/kitti_data/label_2/'
            params['imgs_folder'] = '/fastspace/users/barneaeh/KITTI_imgs/'
        else:
            raise RuntimeError('Unknown hostname ' + hostname)
    else:
        raise RuntimeError('Unknown dataset ' + params['dataset'])

    params['spatial_context'] = {}
    # number of bins in the radius of each detection
    params['spatial_context']['rad_num_bins'] = 3
    # size of each spatial bin relative to detection size
    params['spatial_context']['bin_size_factor'] = 1.3

    # relative size discretization params.
    # bin the relative size between rs_limit times bigger and 1/rs_limit times smaller
    params['spatial_context']['rs_use'] = False  # whether to consider relative scale
    params['spatial_context']['rs_limit'] = 1.7
    params['spatial_context']['rs_rad_num_bins'] = 3

    # overlap threshold for detections to be considered correct
    params['overlap'] = 0.5
    # params['overlap'] = 0.75

    # fix detection errors due to bad localization.
    # this applies only to the main improvement analysis (in function 'eval_by_bin_pr')
    params['fix_loc_fps'] = False

    # number of best context types to draw pairs from. For a value of x the number of pairs is x*(x-1)/2
    params['num_best_context_types'] = 50

    # use neural network based context
    params['nn_context'] = True

    # find best AP by scanning all permutations
    params['scan_permutations'] = False

    return params
