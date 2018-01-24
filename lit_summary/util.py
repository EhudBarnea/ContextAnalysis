import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter


def xlsx(fname):
    import zipfile
    from xml.etree.ElementTree import iterparse
    z = zipfile.ZipFile(fname)
    strings = [el.text for e, el in iterparse(z.open('xl/sharedStrings.xml')) if el.tag.endswith('}t')]
    rows = []
    row = {}
    value = ''
    for e, el in iterparse(z.open('xl/worksheets/sheet1.xml')):
        if el.tag.endswith('}v'):  # <v>84</v>
            value = el.text
        if el.tag.endswith('}c'):  # <c r="A3" t="s"><v>84</v></c>
            if el.attrib.get('t') == 's':
                value = strings[int(value)]
            letter = el.attrib['r']  # AZ22
            while letter[-1].isdigit():
                letter = letter[:-1]
            row[letter] = value
            value = ''
        if el.tag.endswith('}row'):
            rows.append(row)
            row = {}
    return rows


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def gen_summaries():
    # generate summaries to all algorithms

    file_name = '/Users/ehud/Work/measure2context/lit_summary/lit_summary.xlsx'

    # define important constants
    num_cats = 161  # number of object categories in the XLSX file
    row_alg = 15
    row_avg = 17
    row_start_cat = 19
    # col_cat_name = 'A'
    cols_alg_names = ['B', 'F', 'J', 'N', 'R', 'V', 'Z', 'AD', 'AH', 'AL', 'AP', 'AT', 'AX']
    cols_ap_add = ['D', 'H', 'L', 'P', 'T', 'X', 'AB', 'AF', 'AJ', 'AN', 'AR', 'AV', 'AZ']
    cols_ap_perc = ['E', 'I', 'M', 'Q', 'U', 'Y', 'AC', 'AG', 'AK', 'AO', 'AS', 'AW', 'BA']

    by_percent = False

    # read xlsx file
    rows = xlsx(file_name)

    # collect best and worst results of each algorithm
    best = []
    worst = []

    for alg in range(len(cols_alg_names)):
        alg_name = rows[row_alg][cols_alg_names[alg]]
        orig_avg = float(rows[row_avg][cols_alg_names[alg]])

        if by_percent:
            col = cols_ap_perc[alg]
        else:
            col = cols_ap_add[alg]

        # get average result
        res_avg = float(rows[row_avg][col])

        # collect category results
        res = []
        for i in range(num_cats):
            r = rows[row_start_cat + i]
            # cat = r[col_cat_name]
            if col not in r.keys():
                continue

            val = r[col]
            if not is_number(val):
                continue
            res.append(float(val))
        res.sort(reverse=True)

        # print(res)
        plt.axhline(y=2, color='r')
        plt.bar(np.arange(0, len(res)), res)
        # plt.plot(np.arange(0, len(res)), res)
        plt.title('%s: %f + %f' % (alg_name, orig_avg, res_avg))
        if by_percent:
            plt.ylim((-100, 100))
        else:
            plt.ylim((-15, 15))
        plt.savefig('out/%d.png' % alg)
        plt.close()

        # collect best and worst results
        if res[-1] < 0:
            best.append(res[0])
            worst.append(res[-1])
            best.append(res[1])
            worst.append(res[-2])
            best.append(res[2])
            worst.append(res[-3])
        

    # best and worst results figure
    # plt.scatter(best,worst)
    # plt.xlim((0, 15))
    # plt.ylim((0, -15))
    # plt.show()



def gen_literature_summary():
    # generate summaries over several algorithms (for the paper)

    file_name = '/Users/ehud/Work/measure2context/lit_summary/lit_summary.xlsx'

    # define important constants
    num_cats = 161  # number of object categories in the XLSX file
    row_alg = 15
    row_avg = 17
    row_start_cat = 19
    # col_cat_name = 'A'
    cols_alg_names = ['R', 'V', 'AD']
    cols_ap_add =    ['T', 'X', 'AF']
    cols_ap_perc =   ['U', 'Y', 'AG']
    labels = ['Choi et al. (PASCAL 2007)','Choi et al. (SUN 09)','Chen et al. (PASCAL 2007)']

    by_percent = False

    # read xlsx file
    rows = xlsx(file_name)

    # prepare figure
    fig = plt.figure()
    ax = plt.gca()

    left, width = .55, .5
    bottom, height = .38, .5
    right = left + width
    top = bottom + height
    ax.text(right, 0.5 * (bottom + top), 'Improved\n Results',
            horizontalalignment='center',
            verticalalignment='center',
            rotation='vertical',
            transform=ax.transAxes,
            fontsize=12)

    left, width = .55, .5
    bottom, height = -.11, .5
    right = left + width
    top = bottom + height
    ax.text(right, 0.5 * (bottom + top), 'Diminished\n Results',
            horizontalalignment='center',
            verticalalignment='center',
            rotation='vertical',
            transform=ax.transAxes,
            fontsize=12)

    plt.axhline(y=2, color='r') 
    plt.axhline(y=0, color='black')        
    plt.grid()

    for alg in range(len(cols_alg_names)):
        alg_name = rows[row_alg][cols_alg_names[alg]]
        orig_avg = float(rows[row_avg][cols_alg_names[alg]])

        if by_percent:
            col = cols_ap_perc[alg]
        else:
            col = cols_ap_add[alg]

        # get average result
        res_avg = float(rows[row_avg][col])

        # collect category results
        res = []
        for i in range(num_cats):
            r = rows[row_start_cat + i]
            # cat = r[col_cat_name]
            if col not in r.keys():
                continue

            val = r[col]
            if not is_number(val):
                continue
            res.append(float(val))
        res.sort(reverse=True)

        # print(res)

        x_len = len(res)
        x_len_fix = 1
        if x_len > 20:
            x_len_fix = 19/106

        
        plt.plot(np.arange(0, len(res))*x_len_fix, res, label=labels[alg])
        

    if by_percent:
        plt.ylim((-100, 100))
    else:
        plt.ylim((-5, 12))
    plt.xlim((0, 19))
    plt.xticks([])
    plt.xlabel('Object categories', fontsize=12)
    plt.ylabel('Change in average precision (AP)', fontsize=12)
    plt.title('Detection improvement with context')
    plt.legend()
    plt.savefig('out/%d.eps' % alg, bbox_inches='tight')
    plt.close()


def per_category():
    # test improvement per category over PASCAL dataset

    file_name = '/Users/ehud/Work/measure2context/lit_summary/lit_summary.xlsx'

    # define important constants
    num_cats = 20  # number of object categories in the XLSX file
    row_alg = 15
    row_avg = 17
    row_start_cat = 19
    col_cat_name = 'A'
    cols_alg_names = ['B', 'F', 'J', 'R', 'Z', 'AD', 'AL', 'AP', 'AT']
    cols_ap_add = ['D', 'H', 'L', 'T', 'AB', 'AF', 'AN', 'AR', 'AV']

    # read xlsx file
    rows = xlsx(file_name)

    cats = []
    res = []
    for i in range(num_cats):
        r = rows[row_start_cat + i]
        cat = r[col_cat_name]
        # print(cat)
        res_cat = []
        for alg in range(len(cols_alg_names)):
            alg_name = rows[row_alg][cols_alg_names[alg]]
            col = cols_ap_add[alg]
            # print(alg_name)
            # print(r[col])
            # print(len(r[col]))
            val = float(r[col])
            res_cat.append(val)

        # print(cat)
        cat = cat[0:-1]
        if cat == 'motorbike':
            cat = 'bike'
        if cat == 'potted plant':
            cat = 'plant'
        if cat == 'tv/monito':
            cat = 'tv'
        if cat == 'bicycl':
            cat = 'cycle'
        cat = cat.capitalize()
        cats.append(cat)
        res.append(res_cat)

    # sort by mean
    meds = np.array([np.median(x) for x in res])
    order = np.argsort(-meds).tolist()
    cats = [cats[i] for i in order]
    res = [res[i] for i in order]
    meds = np.array([np.median(x) for x in res])

    fig = plt.figure()
    ax = plt.gca()

    left, width = .55, .5
    bottom, height = .38, .5
    right = left + width
    top = bottom + height
    ax.text(right, 0.5 * (bottom + top), 'Improved\n Results',
            horizontalalignment='center',
            verticalalignment='center',
            rotation='vertical',
            transform=ax.transAxes,
            fontsize=12)

    left, width = .55, .5
    bottom, height = -.11, .5
    right = left + width
    top = bottom + height
    ax.text(right, 0.5 * (bottom + top), 'Diminished\n Results',
            horizontalalignment='center',
            verticalalignment='center',
            rotation='vertical',
            transform=ax.transAxes,
            fontsize=12)

    plt.axhline(y=0, color='black')
    plt.axhline(y=2, color='r')
    plt.boxplot(res, labels=cats, whis=999999)
    # plt.ylim((-15, 15))
    plt.xticks(rotation=45)
    plt.ylabel('Change in average precision (AP)', fontsize=12)
    plt.title('Summary of improvement by different models')
    # plt.show()
    plt.savefig('out/summary_per_cat_pascal.eps', bbox_inches='tight')
    plt.savefig('out/summary_per_cat_pascal.png', bbox_inches='tight')
    plt.close()


def main():
    gen_summaries()
    gen_literature_summary()
    per_category()


if __name__ == "__main__":
    main()
