import numpy as np


def attach_det_windows_to_gt(dets, gt, overlap_threshold):
    # Attach detections det to ground-truth object gt if their IoU>overlap_threshold.
    # Outputs an array of size [len(dets),3]:
    # Column 0 - if correct detection
    # Column 1 - index of attached gt object
    # Column 2 - if attached to dont_care object
    # Column 3 - type of error (if incorrect)
    # error types:
    # -1 = correct detection
    #  0 = confusion with background
    #  1 = localization error
    #  2 = confusion with another class

    out_det = np.zeros([len(dets), 4])

    if len(dets) < 1 or len(gt) < 1:
        return out_det

    # sort detections by decreasing confidence (don't change their order)
    dets_order = np.argsort([-d['conf'] for d in dets]).tolist()

    # for each object hold the closest detection
    objects_detected = np.zeros([len(gt), 1])

    # for each gt object find the most confident associated / overlapping detection
    # for d in xrange(len(dets)):
    for di in dets_order:
        d = dets[di]
        # find closest (non dont_care) object
        ovmax = -1
        closest = -1
        for oi, o in enumerate(gt):
            if o['dont_care']:
                continue

            # calculate window overlap
            ov = box_overlap(d, o)
            if ov > ovmax:
                ovmax = ov
                closest = oi

        # determine if tp/fp
        correct = False
        close_enough = ovmax > overlap_threshold
        if close_enough and d['label'] == gt[closest]['label']:
            if not objects_detected[closest]:
                correct = True
                objects_detected[closest] = True
                out_det[di, 0] = True
                out_det[di, 1] = closest

        # determine fp type
        fp_type = 0
        if correct:
            fp_type = -1
        elif ovmax > 0.1 and dets[di]['label'] == gt[closest]['label']:
            fp_type = 1
        elif ovmax > 0.1 and dets[di]['label'] != gt[closest]['label']:
            fp_type = 2
        out_det[di, 3] = fp_type

    # attach unassigned detections to dont_care regions
    for di, d in enumerate(dets):
        if out_det[di, 0]:  # detection is correct (assigned to non dont_care object)
            continue

        is_dont_care = False
        for oi, o in enumerate(gt):
            if not o['dont_care']:
                continue

            # calculate window overlap (not IOU)
            ov = box_overlap(d, o, dont_care=True)
            close_enough = ov > overlap_threshold
            if close_enough:
                is_dont_care = True

        if is_dont_care:
            out_det[di, 2] = True

    return out_det


def box_overlap(o1, o2, dont_care=False):
    # calculate box overlap (IoU) for two objects/detections.
    # this also accepts two bouding boxes.
    # if dont_care is true o2 is treated as a dont_care region
    # and instead of IoU the overlap is calculated as intersection over the area of the detection o1

    if type(o1) == np.ndarray:
        o1_bbox = o1
        o2_bbox = o2
    else:
        o1_bbox = np.array([o1['x1'], o1['y1'], o1['x2'], o1['y2']])
        o2_bbox = np.array([o2['x1'], o2['y1'], o2['x2'], o2['y2']])

    # the intersection of the two boxes
    bi = [
        max(o1_bbox[0], o2_bbox[0]),
        max(o1_bbox[1], o2_bbox[1]),
        min(o1_bbox[2], o2_bbox[2]),
        min(o1_bbox[3], o2_bbox[3])]  # this is the intersection of the two boxes
    iw = bi[2] - bi[0] + 1  # intersection width
    ih = bi[3] - bi[1] + 1  # intersection height
    if iw > 0 and ih > 0:
        if not dont_care:
            # compute overlap as area of intersection / area of union
            ua = \
                (o1_bbox[2] - o1_bbox[0] + 1) * (o1_bbox[3] - o1_bbox[1] + 1) + \
                (o2_bbox[2] - o2_bbox[0] + 1) * (o2_bbox[3] - o2_bbox[1] + 1) - \
                iw * ih  # union area
            ov = float(iw * ih) / float(ua)  # overlap
        else:
            # compute overlap as area of intersection / area of detection (not union)
            ov = float(iw * ih) / float((o1_bbox[2] - o1_bbox[0] + 1) * (o1_bbox[3] - o1_bbox[1] + 1))  # overlap
    else:
        ov = -1

    return ov
