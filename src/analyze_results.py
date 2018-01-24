from main import *
import pickle
import numpy as np
import matplotlib.pyplot as plt


def analyze_imp(params):
    # analyze detection improvement
    print('analyze_imp')

    # load data
    res_file = params['out_folder'] + 'out_improve.p'
    res = pickle.load(open(res_file, "rb"))

    # find contexts that improve more than the random
    max_imps = []
    with open(params['out_folder'] + "analyze_imp.txt", "w") as out_file:
        for lbl_res in res:
            lbl = lbl_res[0]['lbl']

            # AP without context
            base = [x['ap'] for x in lbl_res if x['context'] == 'no_context'][0]

            # calculate random context
            random_imps = [x['ap'] for x in lbl_res if x['context'][0:6] == 'random']
            random_imp = np.mean(random_imps)

            # show max improvement
            lbl_res_no_rand = [x for x in lbl_res if x['context'][0:6] != 'random']
            max_imp_id = np.argmax([x['ap'] for x in lbl_res_no_rand])
            # noinspection PyTypeChecker
            max_imp_name = lbl_res_no_rand[max_imp_id]['context']
            # noinspection PyTypeChecker
            max_imp_val = lbl_res_no_rand[max_imp_id]['context_val']
            # noinspection PyTypeChecker
            max_imp = lbl_res_no_rand[max_imp_id]['ap']

            # max_imp = 100 * (max_imp - random_imp)
            max_imp = 100 * (max_imp - base)
            random_imp = 100 * (random_imp - base)
            max_imps.append({'label': lbl, 'imp': max_imp, 'random_imp': random_imp,
                             'name': max_imp_name, 'val': max_imp_val})
            out_file.write('%s: %s = %f\n' % (lbl, max_imp_name, max_imp))

            if max_imp < random_imp:
                out_file.write('***  %s: max improvement is less than random\n' % lbl)

        out_file.write('\n')
        out_file.write('Median imp over all categories: %s\n' % str(np.median([x['imp'] for x in max_imps])))
        out_file.write('Mean imp over all categories: %s\n' % str(np.mean([x['imp'] for x in max_imps])))
        out_file.write('Max imp over all categories: %s\n' % str(max([x['imp'] for x in max_imps])))

    plt.hist([x['imp'] for x in max_imps], bins=np.arange(10))
    plt.ylim((0, 58))
    plt.savefig(params['out_folder'] + "analyze_imp.png")
    plt.close()
    return max_imps


def analyze_imp_verbose(max_imps, params):
    # analyze a single object type with a single context type.
    # this outputs more information to better understand what context does.
    # Specifically, it prints the number of TPs and FP in eace bin
    print('analyze_imp_verbose')

    # load detections and attach context and correctness
    data, context_types, num_positives = prep_data(params)

    for x in max_imps:
        lbl = x['label']
        con = x['val']
        con_name = x['name']

        # sort lbl detections by decreasing confidence
        lbl_dets = [d for d in data['dets'] if d['label'] == lbl and not d['dont_care']]
        lbl_dets.sort(key=itemgetter('conf'), reverse=True)

        eval_by_bin_pr(lbl_dets, num_positives, lbl, con, params, verbose=True)

        vis = True  # or lbl == 'kite'
        lbl_dets_tp = [d for d in lbl_dets if d['correct']][0:100]
        lbl_dets_fp = [d for d in lbl_dets if d['fp_type'] == 1][0:100]
        lbl_dets = lbl_dets_tp + lbl_dets_fp
        if vis:
            out_dir = '../out/tmp/det_imgs_%s_%s/' % (lbl, con_name)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            for di, d in enumerate(lbl_dets[0:200]):
                # get detection context
                if type(con) == int:
                    d_cont = con in d['context']
                else:  # tuple of several types
                    op = con[0]
                    t1 = con[1]
                    t2 = con[2]
                    if op == 'or':
                        d_cont = t1 in d['context'] or t2 in d['context']
                    else:
                        d_cont = t1 in d['context'] and t2 in d['context']

                summ = (d['conf'], d['correct'], d_cont)
                # print(summ)
                img_name = data['imgs'][d['img_id']]['file_name']
                out_name = '%s/%d_%s' % (out_dir, di, str(summ))
                vis_img(img_name, [d], data['objects_pi'][d['img_id']], params, save=True, out_name=out_name)


def analyze_tpfp(params):
    # analyze separation of TPs and FPs
    print('analyze_tpfp')

    # load data
    res_file = params['out_folder'] + 'out_tpfp.p'
    res = pickle.load(open(res_file, "rb"))
    # print(res)

    # for each FP type keep all the maximally improve accuracies
    max_imps = {'fp_bg': [], 'fp_loc': [], 'fp_oth': []}
    with open(params['out_folder'] + "analyze_tpfp.txt", "w") as out_file:
        for lbl_res in res:
            lbl = lbl_res[0]['lbl']
            for fp_type in ['fp_bg', 'fp_loc', 'fp_oth']:
                rel_res = [x for x in lbl_res if x['fp_type'] == fp_type]

                # calculate random context
                random_imps = [x['accuracy'] for x in rel_res if x['context'][0:6] == 'random']
                random_imp = np.mean(random_imps)

                # get improvements stronger than random
                imps = [x for x in rel_res if x['accuracy'] > random_imp and x['context'][0:6] != 'random']

                # show max improvements
                rel_res_no_rand = [x for x in rel_res if x['context'][0:6] != 'random']
                max_imp_id = np.argmax([x['accuracy'] for x in rel_res_no_rand])
                # noinspection PyTypeChecker
                max_imp_name = rel_res_no_rand[max_imp_id]['context']
                # noinspection PyTypeChecker
                max_imp = rel_res_no_rand[max_imp_id]['accuracy']
                if max_imp < random_imp:
                    out_file.write('%s: max improvement is less than random\n' % lbl)
                max_imp *= 100
                out_file.write('%s,%s: %s = %f\n' % (lbl, fp_type, max_imp_name, max_imp))
                max_imps[fp_type].append({'label': lbl, 'imp': max_imp, 'name': max_imp_name})
            out_file.write('\n')

    # plt.hist([max_imps['fp_bg'], max_imps['fp_loc'], max_imps['fp_oth']], bins=np.arange(0.5, 1.01, 0.1))
    # plt.legend(['Background FPs', 'Localization FPs', 'Other FPs'])
    # plt.show()
    return max_imps


def gen_table():
    params = get_params()

    base = params['out_folder']
    res05_folder = base + 'coco_ov=0.5/'
    res05_fix_locfp_folder = base + 'coco_ov=0.5_fix_loc_fps/'
    res075_folder = base + 'coco_ov=0.75/'
    # res075_fix_locfp_folder = base + 'coco_ov=0.75_fix_loc_fps/'

    # analyze different results
    params['out_folder'] = res05_folder
    imp05 = analyze_imp(params)
    tpfp05 = analyze_tpfp(params)
    params['out_folder'] = res075_folder
    imp075 = analyze_imp(params)
    tpfp075 = analyze_tpfp(params)
    params['out_folder'] = res05_fix_locfp_folder
    imp05_fix = analyze_imp(params)
    # params['out_folder'] = res075_fix_locfp_folder
    # imp075_fix = analyze_imp(params)

    # load results of fixing different FP types
    # fix_by_type = []
    # filename = params['out_folder'] + res05_folder + '/out_fix_by_type.txt'
    # with open(filename, 'r') as file:
    #     lines = file.readlines()
    # for i in range(len(params['object_labels'])):
    #     lbl = lines[i * 7][0:-2]
    #     x = float(lines[i * 7 + 5])
    #     y = float(lines[i * 7 + 6])
    #     fix_by_type.append((lbl, x, y))

    column_titles = ['Category',
                     'Random improvement ov=0.5',
                     'Max improvement ov=0.5', 'context',
                     'Max improvement ov=0.75', 'context',
                     'Max improvement ov=0.5 fixed loc FPs', 'context',
                     # 'Max improvement ov=0.75 fixed loc FPs', 'context',
                     'Max separation Bg FP ov=0.5', 'context',
                     'Max separation Loc FP ov=0.5', 'context',
                     'Max separation Oth FP ov=0.5', 'context',
                     'Max separation Bg FP ov=0.75', 'context',
                     'Max separation Loc FP ov=0.75', 'context',
                     'Max separation Oth FP ov=0.75', 'context',
                     ]

    # prepare lists to hold all data per category
    data = dict()
    for lbl in params['object_labels']:
        data[lbl] = []

    # add category names
    for lbl in params['object_labels']:
        data[lbl].append(lbl)

    # add random improvements
    for x in imp05:
        data[x['label']].append(x['random_imp'])

    # add max improvements
    for x in imp05:
        data[x['label']].append(x['imp'])
        data[x['label']].append(x['name'])
    for x in imp075:
        data[x['label']].append(x['imp'])
        data[x['label']].append(x['name'])
    for x in imp05_fix:
        data[x['label']].append(x['imp'])
        data[x['label']].append(x['name'])
    # for x in imp075_fix:
    #     data[x['label']].append(x['imp'])
    #     data[x['label']].append(x['name'])

    # add max separability
    for x in tpfp05['fp_bg']:
        data[x['label']].append(x['imp'])
        data[x['label']].append(x['name'])
    for x in tpfp05['fp_loc']:
        data[x['label']].append(x['imp'])
        data[x['label']].append(x['name'])
    for x in tpfp05['fp_oth']:
        data[x['label']].append(x['imp'])
        data[x['label']].append(x['name'])
    for x in tpfp075['fp_bg']:
        data[x['label']].append(x['imp'])
        data[x['label']].append(x['name'])
    for x in tpfp075['fp_loc']:
        data[x['label']].append(x['imp'])
        data[x['label']].append(x['name'])
    for x in tpfp075['fp_oth']:
        data[x['label']].append(x['imp'])
        data[x['label']].append(x['name'])

    # write CSV file
    with open(base + 'summary.csv', 'w') as out_file:
        for c in column_titles:
            out_file.write(c + ',')
        out_file.write('\n')
        for lbl in params['object_labels']:
            for x in data[lbl]:
                out_file.write(str(x).replace(',', '') + ',')
            out_file.write('\n')


def analyze_single_res():
    params = get_params()

    extra_analysis = True
    max_imps = analyze_imp(params)
    if extra_analysis:
        analyze_imp_verbose(max_imps, params)
    analyze_tpfp(params)


def main():
    # analyze results of a single experiment
    # analyze_single_res()

    # analyze results of multiple experiment and summarize as a table
    gen_table()


if __name__ == "__main__":
    main()
