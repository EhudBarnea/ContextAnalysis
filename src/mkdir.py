import os


def mkdir(name):
    if not os.path.exists(name):
        os.mkdir(name)
