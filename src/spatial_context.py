import numpy as np
from vis_img import *
from attach_det_windows_to_gt import box_overlap


def get_spatial_context(det, objects, params, get_types=True, img_num=-1):
    # Get the spatial context (based on the ground-truth objects) relative to the detection det.
    # To do so, we examine where objects fall in a grid centered on det, with bin sizes defined relative to its size.

    # For each location and object type this sets 0 to mark 'empty', 1 for 'has objects'.
    # 2 is used for missing/truncated locations or for the central bin

    # get_types - get list of context types

    object_labels = params['object_labels']

    # bin size relative to the detection size
    bin_size_factor = float(params['spatial_context']['bin_size_factor'])
    # number of bins in the radius
    frame = dict()
    frame['rad_num_bins'] = float(params['spatial_context']['rad_num_bins'])

    # set bin size relative to the detection's size
    det_size = np.abs(float(det['y2']) - float(det['y1']))
    frame['bin_size'] = det_size * bin_size_factor

    # set vars for relative size discretization
    if params['spatial_context']['rs_use']:
        rs_num_bins = 1 + 2 * params['spatial_context']['rs_rad_num_bins']
    else:
        rs_num_bins = 1
    rs_end = np.log(params['spatial_context']['rs_limit'])
    rs_start = -rs_end
    rs_bin_size = (rs_end - rs_start) / rs_num_bins

    # calculate frame limits
    frame['center'] = np.array([det['x1'] + det['x2'], det['y1'] + det['y2']]) / 2
    frame['start'] = frame['center'] - frame['bin_size'] * (frame['rad_num_bins'] + 0.5)
    num_bins_from_start = np.floor(frame['rad_num_bins'] * 2 + 1)
    frame['end'] = frame['start'] + frame['bin_size'] * num_bins_from_start

    # prepare frames to make object locations per type
    if params['spatial_context']['rs_use']:
        frame['has_obj'] = np.zeros([len(object_labels), int(num_bins_from_start), int(num_bins_from_start), int(rs_num_bins)])
    else:
        frame['has_obj'] = np.zeros([len(object_labels), int(num_bins_from_start), int(num_bins_from_start)])

    # list context types (strings)
    context_types = []
    if get_types:
        for x in np.ndindex(frame['has_obj'].shape):
            lbl = object_labels[x[0]]
            context_types.append('has_' + lbl + '_loc' + str(x[1:]))

    # insert objects to frame
    for oi, o in enumerate(objects):
        if o['label'] == 'dontcare':
            continue

        # don't use overlapping objects as context
        if box_overlap(det, o) > 0.1:
            continue

        # get object location in frame
        lbl = o['label']
        o_center = np.array([o['x1'] + o['x2'], o['y1'] + o['y2']]) / 2
        # check if object falls outside frame
        if any(o_center < frame['start']) or any(o_center >= frame['end']):
            continue
        o_bin = ((o_center - frame['start']) / frame['bin_size']).astype(int)
        o_bin[o_bin >= num_bins_from_start] = num_bins_from_start - 1

        # get (log of) relative size
        obj_size = np.abs(o['y2'] - o['y1'])
        if obj_size < 1:  # a few objects have zero or nearly zero size
            continue
        rel_size = np.log(obj_size / det_size)
        if rs_start < rel_size < rs_end:
            rs_bin = int((rel_size - rs_start) / rs_bin_size)
        elif rel_size <= rs_start:
            rs_bin = 0
        else:
            rs_bin = rs_num_bins - 1

        if params['spatial_context']['rs_use']:
            frame['has_obj'][object_labels.index(lbl), o_bin[1], o_bin[0], rs_bin] = 1
        else:
            frame['has_obj'][object_labels.index(lbl), o_bin[1], o_bin[0]] = 1

    # prepare output context as indices of locations with objects
    context = np.flatnonzero(frame['has_obj'])

    # visualization / debugging
    if img_num >= 0:
        if not params['spatial_context']['rs_use']:
            raise NotImplementedError  # to do - update to support frame's 3rd dimension (relative size)
        fig = plt.figure()
        for i, lbl in enumerate(object_labels):
            fig.add_subplot(1, 3, i)
            plt.imshow(frame['has_obj'][i], interpolation='nearest')
            plt.title(lbl)

        vis_img(img_num, [det], objects, params)
        plt.show()

    return context, context_types

