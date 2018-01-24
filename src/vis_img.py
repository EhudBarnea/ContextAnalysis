from matplotlib import pyplot as plt
import matplotlib.patches as patches


def vis_img(img_name, dets, objects, params, save=False, out_name=''):
    width = 1

    img = plt.imread(params['imgs_folder'] + img_name)

    figsize = (20, 6)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)

    for o in objects:
        color = 'blue'
        if o['dont_care']:
            color = 'white'
        rect = patches.Rectangle((o['x1'], o['y1']), o['x2']-o['x1'], o['y2']-o['y1'],
                                 linewidth=width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    for d in dets:
        color = 'red'
        rect = patches.Rectangle((d['x1'], d['y1']), d['x2']-d['x1'], d['y2']-d['y1'],
                                 linewidth=width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    if save:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.savefig(out_name + ".png", bbox_inches='tight')
        plt.close()

    return
