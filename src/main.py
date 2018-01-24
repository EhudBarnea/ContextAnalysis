from read_data_kitti_ssd import *
from read_data_coco_faster_rcnn import *
from eval_detections import *
from spatial_context import *
from augment_dets import *
from params import *
from mkdir import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings
import pickle
from joblib import Parallel, delayed
import itertools
import gc
import glob
import shutil
from operator import itemgetter
from itertools import permutations


def main():
    # run_experiments()
    test_locfp_effect()
    # test_bin_precision_heuristic()

    # learn_params()

    print('done')


def run_experiments():
    # load parameters
    params = get_params()
    # prepare test folder
    mkdir(params['out_folder'])
    copy_src_files(params)

    # ---- run experiments without context
    # load dataset and detections
    # data, _, _ = prep_data(params, no_extra=True)
    # gen_pr_curves(params, data)
    # test_strong_loc_errors(params, data)
    # test_fix_by_type(params, data)
    # ----

    # ---- run experiments with context
    # load dataset and add correctness and context info to detections
    data, context_types, num_positives = prep_data(params)
    test_det_improvement(params, data, context_types, num_positives)
    test_separation_tpfp(params, data, context_types)
    # ----


def test_locfp_effect():
    # load parameters
    params = get_params()

    base_out_folder = params['out_folder']

    params['out_folder'] = base_out_folder + '/coco_ov=0.5/'
    params['overlap'] = 0.5
    params['fix_loc_fps'] = False
    mkdir(params['out_folder'])
    copy_src_files(params)
    data, context_types, num_positives = prep_data(params)
    test_det_improvement(params, data, context_types, num_positives)
    test_separation_tpfp(params, data, context_types)

    params['out_folder'] = base_out_folder + '/coco_ov=0.5_fix_loc_fps/'
    params['overlap'] = 0.5
    params['fix_loc_fps'] = True
    mkdir(params['out_folder'])
    copy_src_files(params)
    data, context_types, num_positives = prep_data(params)
    test_det_improvement(params, data, context_types, num_positives)
    test_separation_tpfp(params, data, context_types)

    params['out_folder'] = base_out_folder + '/coco_ov=0.75/'
    params['overlap'] = 0.75
    params['fix_loc_fps'] = False
    mkdir(params['out_folder'])
    copy_src_files(params)
    data, context_types, num_positives = prep_data(params)
    test_det_improvement(params, data, context_types, num_positives)
    test_separation_tpfp(params, data, context_types)

    # params['out_folder'] = base_out_folder + '/coco_ov=0.75_fix_loc_fps/'
    # params['overlap'] = 0.75
    # params['fix_loc_fps'] = True
    # mkdir(params['out_folder'])
    # copy_src_files(params)
    # data, context_types, num_positives = prep_data(params)
    # test_det_improvement(params, data, context_types, num_positives)
    # test_separation_tpfp(params, data, context_types)

    params['out_folder'] = base_out_folder


def learn_params():
    # determine best parameters for spatial context discretization

    # load parameters
    params = get_params()
    # prepare test folder
    mkdir(params['out_folder'])
    copy_src_files(params)

    with open(params['out_folder'] + "best_param.txt", "w") as out_file:
        for x in np.arange(0.2, 4, 0.1):
            params['spatial_context']['bin_size_factor'] = x
            data, context_types, num_positives = prep_data(params)

            sum_aps = test_det_improvement(params, data, context_types, num_positives)
            print('Param %f - %f' % (x, sum_aps))
            out_file.write('%f - %f\n' % (x, sum_aps))
            out_file.flush()


def copy_src_files(params):
    # prepare folder for current test and copy current src to it (py files)
    mkdir(params['out_folder'] + 'src/')

    # copy source file
    src_folder = '.'
    for filename in glob.glob(os.path.join(src_folder, '*.py')):
        shutil.copy(filename, params['out_folder'] + 'src/')


def prep_data(params, no_extra=False):
    # load data add information about correctness and context
    # no_extra - do not include context and correctness

    # load data (image data, objects, detections)
    if params['dataset'] == 'kitti':
        data = read_data_kitti_ssd(params, 'both')
    elif params['dataset'] == 'coco':
        data = read_data_coco_faster_rcnn(params, 'both')
        # data = read_data_coco_faster_rcnn(params, 'test')
    else:
        raise RuntimeError('Wrong dataset')

    if no_extra:
        context_types = None
        num_positives = None
    else:
        # add context information to detections
        context_types = add_context(data, params)
        # add correctness and FP type to detections
        num_positives = augment_dets(data, params)

    return data, context_types, num_positives


def add_context(data, params):
    # add co-occurrence information, whether there is an object with some label at the same scene

    print('adding context to detections', flush=True)
    time.sleep(0.1)

    # load nn context-based predictions
    nn_data = {}
    if params['nn_context']:
        fname = '../out/coco_nn_predictions.txt'
        with open(fname, 'r') as file:
            lines = file.readlines()
        for line in lines:
            linesplit = line[:-1].split(sep=',')
            id_in_img = int(linesplit[0])
            img_id = int(linesplit[1])
            conf = float(linesplit[2])
            nn_data[(img_id, id_in_img)] = conf

    # get context types, separate to dense and sparse
    num_rand_contexts = 10
    context_types = list()
    context_types.append('no_context')  # no context
    for k in range(num_rand_contexts):
        context_types.append('random' + str(k))  # random context
    for lbl in params['object_labels']:
        context_types.append('has_' + lbl)  # co-occurence context
    nn_context_idx = len(context_types)
    if params['nn_context']:
        context_types.append('nn')  # neural network based context
    # spatial context
    _, spatial_context_types = get_spatial_context(data['dets'][0], [], params)
    spatial_context_start_idx = len(context_types)
    context_types += spatial_context_types

    for img_id in tqdm(data['imgs'].keys()):
        objects = data['objects_pi'][img_id]
        dets = data['dets_pi'][img_id]

        for di, d in enumerate(dets):
            d_con = dict()

            # no context - always 0

            # random context - add several so we can take the mean
            for k in range(num_rand_contexts):
                if np.random.randint(0, 2) == 1:
                    d_con[k+1] = 1

            # co-occurence context.
            # see if there are *other* objects in the scene
            for lbli, lbl in enumerate(params['object_labels']):
                has_label = 0
                for o in objects:
                    if o['dont_care'] or o['label'] != lbl:
                        continue
                    ov = box_overlap(d, o)
                    if ov < 0.1:
                        has_label = 1
                if has_label:
                    d_con[num_rand_contexts+1+lbli] = 1

            # neural network context
            if params['nn_context']:
                key = (d['img_id'], d['id_in_img'])
                if key in nn_data:
                    nn_context = nn_data[key]
                    d_con[nn_context_idx] = float(nn_context)

            # spatial context
            spatial_context, _ = get_spatial_context(d, objects, params, get_types=False)

            # merge context types
            for k in spatial_context:
                d_con[k + spatial_context_start_idx] = 1
            d['context'] = d_con

    time.sleep(0.1)
    print('done adding context')

    return context_types


def eval_by_bin_pr(dets, num_positives, test_label, context_name, context_type, params, verbose=False, inner=None):
    # evaluate detections by precision in bins (incorporating context).

    # context_type - index of context type
    # inner - to hold stuff that can be calculated outside (faster), but is more suited to remain inside
    # ap_diff - difference between ap based on ordering vs iteratively finding the best ap

    bin_by_recall = True  # bin confidence into equaly-sized recall bins
    plot_pr = False
    num_conf_bins = 10
    if params['scan_permutations']:
        num_conf_bins = 5
    fix_loc_fps = params['fix_loc_fps']

    # num_real_context_bins = 5  # number of bins to discretize real-valued context
    num_real_context_bins = 2
    num_binary_context_bins = 2
    real_context_bin_size = 1 / float(num_real_context_bins)
    if context_name == 'nn':
        num_context_bins = num_real_context_bins
    else:
        num_context_bins = num_binary_context_bins

    if type(inner) is dict:
        dets = inner['dets']
        conf_bins = inner['conf_bins']
        corrects = inner['corrects']
    else:
        # prepare detections
        dets = [d for d in dets if d['label'] == test_label and not d['dont_care']]
        if fix_loc_fps:
            dets = [d for d in dets if not d['fp_type'] == 1]
        dets.sort(key=itemgetter('conf'), reverse=True)

        # calculate recall
        corrects = np.array([d['correct'] for d in dets])
        recalls = np.cumsum(corrects)
        recalls = recalls / (float(num_positives[test_label]))

        # get detection's recall bins
        conf_bin_size = float(1) / num_conf_bins
        if bin_by_recall:
            conf_bins = (np.array(recalls) / conf_bin_size).astype(np.int32)
        else:
            conf_bins = (np.array([d['conf'] for d in dets]) / conf_bin_size).astype(np.int32)
        conf_bins[conf_bins >= num_conf_bins] = num_conf_bins - 1

        inner = {'dets': dets, 'conf_bins': conf_bins, 'corrects': corrects}

    # get detection's context bins
    if type(context_type) == int and context_name != 'nn':  # binary context
        d_cont = np.array([context_type in d['context'] for d in dets])
    elif context_name == 'nn':  # real-valued context
        # get context value
        d_cont = []
        for d in dets:
            # get value
            context_val = 0
            if context_type in d['context']:
                context_val = d['context'][context_type]

            # discretize
            # context_bin = int(context_val / real_context_bin_size)
            # if context_bin == num_real_context_bins:
            #     context_bin = num_real_context_bins - 1

            d_cont.append(context_val)
        d_cont = np.array(d_cont)

        # discretize context into bins of equal recall
        for i in range(num_conf_bins):
            # get detections in confidence bin
            cb_dets = [d for di, d in enumerate(dets) if conf_bins[di] == i]
            cb_dets_context = d_cont[conf_bins == i]
            # sort by context value
            cb_order = np.argsort(cb_dets_context)
            cb_dets = [cb_dets[idx] for idx in cb_order.tolist()]
            cb_corrects = np.array([d['correct'] for d in cb_dets])
            # split into bins
            cb_recalls = np.cumsum(cb_corrects) / (float(sum(cb_corrects)))
            cb_context_bins = (cb_recalls / real_context_bin_size).astype(np.int32)
            cb_context_bins[cb_context_bins >= num_real_context_bins] = num_real_context_bins - 1
            for di, d in enumerate(cb_dets):
                d['real_context'] = cb_context_bins[di]
        d_cont = np.array([d['real_context'] for d in dets])

    else:  # tuple of several types
        op = context_type[0]
        t1 = context_type[1]
        t2 = context_type[2]
        t1c = np.array([t1 in d['context'] for d in dets])
        t2c = np.array([t2 in d['context'] for d in dets])
        if op == 'or':
            d_cont = np.logical_or(t1c, t2c)
        else:
            d_cont = np.logical_and(t1c, t2c)

    # count number of detections and TPs in each bin
    bin_num_dets = np.zeros([num_conf_bins, num_context_bins])
    bin_num_corrects = np.zeros([num_conf_bins, num_context_bins])
    for i in range(num_conf_bins):
        for j in range(num_context_bins):
            bin_dets = np.logical_and(conf_bins == i, d_cont == j)
            bin_num_dets[i, j] = np.sum(bin_dets)
            bin_num_corrects[i, j] = np.sum(np.logical_and(bin_dets, corrects))

    # calc AP by bin precisions
    bin_num_corrects = np.reshape(bin_num_corrects, bin_num_corrects.size)
    bin_num_dets = np.reshape(bin_num_dets, bin_num_dets.size)
    bin_num_fps = bin_num_dets - bin_num_corrects
    bin_rec = bin_num_corrects / (float(num_positives[test_label]))
    with warnings.catch_warnings():  # has division by zero, but the code works
        warnings.simplefilter("ignore")
        bin_prec = bin_num_corrects / (bin_num_dets.astype(float))
        bin_tps_to_fps = bin_num_corrects / (bin_num_fps.astype(float))
    # order = np.argsort(-bin_prec)  # order by decreasing bin-precision
    order = np.argsort(-bin_tps_to_fps)  # order by decreasing bin ratio to TPs to FPs

    # calculate AP
    with warnings.catch_warnings():  # has division by zero, but the code works
        warnings.simplefilter("ignore")
        bin_prec_cs = np.cumsum(bin_num_corrects[order]) / np.cumsum(bin_num_dets[order])
    ap = sum(bin_prec_cs * bin_rec[order])

    # find best order by going over all permutations
    ap_diff = 0
    if params['scan_permutations'] and context_name != 'nn':
        ap_orig = ap
        ap_max = 0
        for x in permutations(range(0, len(order))):
            order_new = np.array(x)

            # calculate AP
            with warnings.catch_warnings():  # has division by zero, but the code works
                warnings.simplefilter("ignore")
                bin_prec_cs = np.cumsum(bin_num_corrects[order_new]) / np.cumsum(bin_num_dets[order_new])
            ap = sum(bin_prec_cs * bin_rec[order_new])

            if ap > ap_max:
                ap_max = ap
        ap_diff = ap_max - ap_orig
        ap = ap_orig
        print(test_label)
        print(ap_orig)
        print(ap_max)
        print(ap_diff)

    # plot precision-recall
    if plot_pr:
        bin_rec_cs = np.cumsum(bin_rec[order])
        ap_str = "{0:.4f}".format(ap)
        plt.plot(bin_rec_cs, bin_prec_cs)
        plt.title('AP=' + ap_str)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.grid(True)
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.show()

    # output data to help understand what is going on
    if verbose:
        # num_misses = num_positives[test_label] - np.sum(bin_num_corrects)
        # reverse reshapes
        bin_num_corrects = np.reshape(bin_num_corrects, [num_conf_bins, num_context_bins])
        bin_num_dets = np.reshape(bin_num_dets, [num_conf_bins, num_context_bins])

        # add a column for the detector without context
        bin_num_corrects = np.hstack(
            [bin_num_corrects, np.reshape(bin_num_corrects[:, 0] + bin_num_corrects[:, 1], [num_conf_bins, 1])])
        bin_num_dets = np.hstack(
            [bin_num_dets, np.reshape(bin_num_dets[:, 0] + bin_num_dets[:, 1], [num_conf_bins, 1])])

        # recalculate precision and recall (because of the added column)
        bin_rec = bin_num_corrects / (float(num_positives[test_label]))
        with warnings.catch_warnings():  # has division by zero, but the code works
            warnings.simplefilter("ignore")
            bin_prec = bin_num_corrects / (bin_num_dets.astype(float))

        print('%s: %s - %f' % (test_label, context_type, ap))
        for i in range(bin_num_corrects.shape[0]):
            j = -1
            out_str = '%d: (%d,%d) -> ' % (i, bin_num_corrects[i, j], bin_num_dets[i, j]-bin_num_corrects[i, j])
            out_str2 = '%d: (%.4f,%.4f) -> ' % (i, bin_prec[i, j], bin_rec[i, j])
            for j in range(num_context_bins):
                out_str += '(%d,%d),' % (bin_num_corrects[i, j], bin_num_dets[i, j]-bin_num_corrects[i, j])
                out_str2 += '(%.4f,%.4f),' % (bin_prec[i, j], bin_rec[i, j])
            print(out_str)
            print(out_str2)
        print('')

    return ap, inner, ap_diff


def accuracy(gt, pred):
    # gt and pred are binary np arrays
    correct = np.equal(gt, pred)
    return np.sum(correct)/len(gt)


def test_det_improvement(params, data, context_types, num_positives):
    # test improvement for the detection with various types of context
    print('test_det_improvement')

    # run in parallel
    if params['num_proc'] <= 1:
        res = []
        for lbl in tqdm(params['object_labels']):
            res_lbl = test_det_improvement_lbl(data['dets'], context_types, num_positives, lbl, params)
            res.append(res_lbl)
    else:
        # to save memory during parallelization save detections to file and load them in each process
        dets = None
        mkdir(params['out_folder'] + 'tmp/')
        for lbl in params['object_labels']:
            print('saving dets ' + lbl)
            dets = [d for d in data['dets'] if d['label'] == lbl]
            pickle.dump(dets, open(params['out_folder'] + "tmp/dets_" + lbl + ".p", "wb"))
        del dets
        del data
        gc.collect()
        # run processes
        res = Parallel(n_jobs=params['num_proc'], verbose=100)(delayed(
            test_det_improvement_lbl)(None, context_types, num_positives, lbl, params)
                                                               for lbl in params['object_labels'])

    # print results
    with open(params['out_folder'] + "out_improve.txt", "w") as out_file:
        for res_lbl in res:
            lbl = res_lbl[0]['lbl']
            for r in res_lbl:
                out_str = lbl + ':' + r['context']
                out_str += ' - ' + str(r['ap'])
                # print(out_str)
                out_file.write(out_str + '\n')
                out_file.flush()

    # save results
    pickle.dump(res, open(params['out_folder'] + "out_improve.p", "wb"))

    # calculate sum of maximal APs of all categories (for parameter learning)
    max_aps = []
    for res_lbl in res:
        max_ap = 0
        for r in res_lbl:
            if r['ap'] > max_ap:
                max_ap = r['ap']
        max_aps.append(max_ap)
    sum_max_aps = sum(max_aps)

    # get max of ap_diff
    if params['scan_permutations']:
        max_ap_diff = 0
        for res_lbl in res:
            for r in res_lbl:
                if r['ap_diff'] > max_ap_diff:
                    max_ap_diff = r['ap_diff']
        print('maximal ap difference after iterating = %f' % max_ap_diff)
        with open(params['out_folder'] + "max_ap_diff.txt", "w") as out_file:
            out_file.write('maximal ap difference after iterating = %f\n' % max_ap_diff)

    return sum_max_aps


def test_det_improvement_lbl(dets, context_types, num_positives, lbl, params):
    # test improvement for the detection of some label lbl with various types of context
    # print(lbl)

    if dets is None:
        file_name = params['out_folder'] + 'tmp/dets_' + lbl + '.p'
        dets = pickle.load(open(file_name, "rb"))
        os.remove(file_name)

    res = []
    inner = None
    for cti, ct in enumerate(context_types):
        ap, inner, ap_diff = eval_by_bin_pr(dets, num_positives, lbl, ct, cti, params, inner=inner)

        res.append({'lbl': lbl, 'context': ct, 'context_val': cti, 'ap': ap, 'ap_diff': ap_diff})

        out_str = lbl + ':' + context_types[cti]
        out_str += ' - ' + str(ap)
        # print(out_str)

    # calculate AP of average random context
    # random_ap = np.mean([x['ap'] for x in res if x['context'][0:6] == 'random'])

    # generate pairs of context types that improve the most
    pair_names, pair_vals = get_context_pairs(res, 'ap', params)

    # run analysis over pairs
    for cti, ct in enumerate(pair_vals):
        ap, _, ap_diff = eval_by_bin_pr(dets, num_positives, lbl, ct, ct, params, inner=inner)
        res.append({'lbl': lbl, 'context': pair_names[cti], 'context_val': ct, 'ap': ap, 'ap_diff': ap_diff})

        out_str = lbl + ':' + pair_names[cti]
        out_str += ' - ' + str(ap)
        # print(out_str)

    # print(lbl + ' done')
    return res


def get_context_pairs(res, m, params):
    # given results res of employing contexts in context_types,
    # generate pairs of most improving context types
    # m - 'ap'/'accuracy'

    # remove no_context, random context, and nn context
    best_res = [x for x in res if x['context'] != 'no_context' and
                x['context'][0:6] != 'random' and x['context'] != 'nn']

    # sort by most improving context types
    best_res.sort(key=itemgetter(m), reverse=True)

    pair_names = []
    pair_vals = []
    num_best_context_types = min(len(best_res), params['num_best_context_types'])
    for x in itertools.combinations(range(num_best_context_types), 2):
        name1 = best_res[x[0]]['context']
        name2 = best_res[x[1]]['context']
        val1 = best_res[x[0]]['context_val']
        val2 = best_res[x[1]]['context_val']

        op = 'or'
        name = '%s %s %s' % (name1, op, name2)
        val = (op, val1, val2)
        pair_names.append(name)
        pair_vals.append(val)
        op = 'and'
        name = '%s %s %s' % (name1, op, name2)
        val = (op, val1, val2)
        pair_names.append(name)
        pair_vals.append(val)
    return pair_names, pair_vals


def test_separation_tpfp(params, data, context_types):
    # test ability of context to separate TPs from FPs (by FP type)
    print('test_separation_tpfp')

    if params['num_proc'] <= 1:
        res = []
        for lbl in tqdm(params['object_labels']):
            res_lbl = test_separation_tpfp_lbl(data['dets'], context_types, lbl, params)
            res.append(res_lbl)
    else:
        # to save memory during parallelization save detections to file and load them in each process
        dets = None
        mkdir(params['out_folder'] + 'tmp/')
        for lbl in params['object_labels']:
            print('dets ' + lbl)
            dets = [d for d in data['dets'] if d['label'] == lbl]
            pickle.dump(dets, open(params['out_folder'] + "tmp/dets_" + lbl + ".p", "wb"))
        del dets
        del data
        gc.collect()
        # run processes
        res = Parallel(n_jobs=params['num_proc'], verbose=100)(delayed(
            test_separation_tpfp_lbl)(None, context_types, lbl, params)
                                                               for lbl in params['object_labels'])

    # print results
    with open(params['out_folder'] + "out_tpfp.txt", "w") as out_file:
        for res_lbl in res:
            lbl = res_lbl[0]['lbl']
            for r in res_lbl:
                out_str = '%s, %s, %s: %f' % (lbl, r['fp_type'], r['context'], r['accuracy'])
                # print(out_str)
                out_file.write(out_str + '\n')

    # save results
    pickle.dump(res, open(params['out_folder'] + "out_tpfp.p", "wb"))


def test_separation_tpfp_lbl(dets, context_types, lbl, params):
    # print(lbl)

    if dets is None:
        file_name = params['out_folder'] + "tmp/dets_" + lbl + ".p"
        dets = pickle.load(open(file_name, "rb"))
        os.remove(file_name)

    # separate FPs to types
    fp_bg = [x for x in dets if x['label'] == lbl and not x['dont_care'] and x['fp_type'] == 0]
    fp_loc = [x for x in dets if x['label'] == lbl and not x['dont_care'] and x['fp_type'] == 1]
    fp_oth = [x for x in dets if x['label'] == lbl and not x['dont_care'] and x['fp_type'] == 2]
    tp = [x for x in dets if x['label'] == lbl and x['correct'] == 1]
    tp.sort(key=itemgetter('conf'), reverse=True)  # sort by decreasing confidence
    fps = [fp_bg, fp_loc, fp_oth]
    fps_str = ['fp_bg', 'fp_loc', 'fp_oth']

    # see how each context type separates fps from tps
    res_all_fps = []
    for fpi, fp in enumerate(fps):
        # create two equaly sized groups of TPs and FPs
        fp.sort(key=itemgetter('conf'), reverse=True)  # sort by decreasing confidence
        # make sure both categories have the same size
        size = min(len(tp), len(fp))
        tp_smaller = tp[:size]
        fp_smaller = fp[:size]
        gt = np.concatenate((np.ones(len(tp_smaller)), np.zeros(len(fp_smaller))))

        res = []
        for cti, ct in enumerate(context_types):
            # classify by context
            if ct != 'nn':  # binary context
                pred = np.array([cti in x['context'] for x in tp_smaller + fp_smaller])
                acc = accuracy(gt, pred)
                acc = np.max((acc, 1 - acc))
            else:  # real-valued context
                # try different thresholds and take the maximal accuracy
                acc = 0
                for th in np.arange(0.05, 1, 0.05):
                    pred = []
                    for x in tp_smaller + fp_smaller:
                        # get value
                        context_val = 0
                        if cti in x['context']:
                            context_val = x['context'][cti]
                        pred.append(int(context_val <= th))
                    pred = np.array(pred)
                    th_acc = accuracy(gt, pred)
                    th_acc = np.max((th_acc, 1 - th_acc))
                    if th_acc > acc:
                        acc = th_acc
            res.append({'lbl': lbl, 'fp_type': fps_str[fpi], 'context': ct, 'context_val': cti, 'accuracy': acc})
            # print('%s, %s, %s: %f' % (lbl, fps_str[fpi], ct, acc))

        # calculate AP of average random context
        # random_acc = np.mean([x['accuracy'] for x in res if x['context'][0:6] == 'random'])

        # generate pairs of context types that improve more than the random
        pair_names, pair_vals = get_context_pairs(res, 'accuracy', params)

        # run analysis over pairs
        for cti, ct in enumerate(pair_vals):
            # classify by context
            op = ct[0]
            t1 = ct[1]
            t2 = ct[2]
            if op == 'or':
                pred = np.array([t1 in x['context'] or t2 in x['context']
                                 for x in tp_smaller + fp_smaller])
            else:
                pred = np.array([t1 in x['context'] and t2 in x['context']
                                 for x in tp_smaller + fp_smaller])
            acc = accuracy(gt, pred)
            acc = np.max((acc, 1 - acc))
            res.append({'lbl': lbl, 'fp_type': fps_str[fpi],
                        'context': pair_names[cti], 'context_val': ct, 'accuracy': acc})
            # print('%s, %s, %s: %f' % (lbl, fps_str[fpi], pair_names[cti], acc))

        res_all_fps += res

    # print(lbl + ' done')
    return res_all_fps


def test_fix_by_type(params, data):
    # test the improvement in detections when fixing FPs by type
    print('test_fix_by_type')

    with open(params['out_folder'] + "out_fix_by_type.txt", "w") as out_file:
        for lbl in tqdm(params['object_labels']):
            # print(lbl + ':')
            fix_types = []
            ap = eval_detections(lbl, data, fix_types)
            fix_types = [1]
            ap_fix_loc = eval_detections(lbl, data, fix_types)
            fix_types = [0, 2]
            ap_fix_bg_other = eval_detections(lbl, data, fix_types)
            fix_types = [0, 1, 2]
            ap_fix_all = eval_detections(lbl, data, fix_types)

            out_file.write(lbl + ':' + '\n')
            out_file.write(str(ap) + '\n')
            out_file.write(str(ap_fix_loc) + '\n')
            out_file.write(str(ap_fix_bg_other) + '\n')
            out_file.write(str(ap_fix_all) + '\n')
            out_file.write(str(ap_fix_bg_other - ap) + '\n')
            out_file.write(str(ap_fix_all - ap_fix_loc) + '\n')

    return


def test_strong_loc_errors(params, data):
    # get the amount of localization errors among the most confident FPs

    # group of strong FPs to test
    # num_strong_fps = 10

    print('get_num_strong_loc_errors')
    for num_strong_fps in [2, 5, 10, 15, 20, 30, 50, 100, 200, 500]:
        with open(params['out_folder'] + "out_strong_loc_errors.txt", "w") as out_file:
            print('analyzing detections for overlap=0.5')
            params['overlap'] = 0.5
            augment_dets(data, params)
            out_file.write('Overlap = 0.5' + '\n')
            nums1 = []
            for lbl in tqdm(params['object_labels']):
                # print(lbl + ':')
                fps = [x for x in data['dets'] if x['label'] == lbl and not x['dont_care'] and not x['correct']]
                fps.sort(key=lambda x: x['conf'], reverse=True)  # sort by decreasing confidence
                strong_fps = fps[0:num_strong_fps]
                num_strong_loc_fps = sum(1 for x in strong_fps if x['fp_type'] == 1)
                nums1.append(num_strong_loc_fps/num_strong_fps)
                out_file.write(lbl + ': ' + str(num_strong_loc_fps/num_strong_fps) + '\n')

            print('analyzing detections for overlap=0.75')
            params['overlap'] = 0.75
            augment_dets(data, params)
            out_file.write('\n')
            out_file.write('Overlap = 0.75' + '\n')
            nums2 = []
            for lbl in tqdm(params['object_labels']):
                # print(lbl + ':')
                fps = [x for x in data['dets'] if x['label'] == lbl and not x['dont_care'] and not x['correct']]
                fps.sort(key=lambda x: x['conf'], reverse=True)  # sort by decreasing confidence
                strong_fps = fps[0:num_strong_fps]
                num_strong_loc_fps = sum(1 for x in strong_fps if x['fp_type'] == 1)
                nums2.append(num_strong_loc_fps / num_strong_fps)
                out_file.write(lbl + ': ' + str(num_strong_loc_fps / num_strong_fps) + '\n')

            print(num_strong_fps)
            print(sum([1 for x in nums1 if x > 0.5]))
            print(sum([1 for x in nums2 if x > 0.5]))


def gen_pr_curves(params, data):

    curves_folder = params['out_folder'] + 'pr_curves/'
    mkdir(curves_folder)

    aps = []
    for lbl in params['object_labels']:
        print(lbl)
        ap = eval_detections(lbl, data, plot_fig=True)
        aps.append(ap)
        plt.savefig(curves_folder + lbl + '.png')
        plt.close()
        ap_str = "{0:.3f}".format(ap)
        print(ap_str)
    print('Average mAP (over all categories) = %f' % np.mean(aps))


def test_bin_precision_heuristic():
    # test the heuristic of employing the bin-precision by considering the AP of all permutations

    params = get_params()
    base_out_folder = params['out_folder']
    params['out_folder'] = base_out_folder + '/coco_ov=0.5/'
    params['scan_permutations'] = True

    # load data
    data, _, num_positives = prep_data(params)
    # load best context types
    res_file = params['out_folder'] + 'out_improve.p'
    res = pickle.load(open(res_file, "rb"))

    # test the higher scoring context type for each object label
    for lbl_res in res:
        lbl = lbl_res[0]['lbl']

        # get best binary context
        lbl_res_no_rand = [x for x in lbl_res if x['context'][0:6] != 'random' and x['context'] != 'nn']
        max_context_id = int(np.argmax([x['ap'] for x in lbl_res_no_rand]))
        context_name = lbl_res_no_rand[max_context_id]['context']
        context_val = lbl_res_no_rand[max_context_id]['context_val']

        # re-run context evaluation
        eval_by_bin_pr(data['dets'], num_positives, lbl, context_name, context_val, params)

    params['scan_permutations'] = False
    params['out_folder'] = base_out_folder


if __name__ == "__main__":
    main()
