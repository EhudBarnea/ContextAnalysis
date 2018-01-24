import numpy as np

def gen_rand_data(params):
    img_data = []

    num_imgs = 5
    max_num_objects = 10

    # define (virtual) image size
    img_size = 500

    # define objects and detections of fixed window size
    window_size = 10

    label = 'car'
    dont_care = False

    # generate data
    objects = []
    for img_num in range(num_imgs):

        # generate objects
        num_objects = np.random.randint(max_num_objects)
        for i in range(num_objects):
            center_x = np.random.randint(img_size)
            center_y = np.random.randint(img_size)
            x1 = center_x - window_size / 2
            y1 = center_x - window_size / 2
            x2 = center_y + window_size / 2
            y2 = center_y + window_size / 2
            objects.append({'label': label, 'x1': x1, 'y1': y1, 'x2': x2,
                            'y2': y2, 'dont_care': dont_care})

        # generate detections

        img_data.append({'img_num': img_num, 'img_size': img_size, 'dets': dets, 'objects': objects})



    return img_data