import os
from os.path import dirname, abspath


def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')


def get_graph_path(model_name):
    dyn_graph_path = {
        'cmu': 'graph/cmu/graph_opt.pb',
        'mobilenet_thin': 'graph/mobilenet_thin/graph_opt.pb'
    }

    base_data_dir = dirname(dirname(abspath(__file__)))
    if os.path.exists(os.path.join(base_data_dir, 'models')):
        base_data_dir = os.path.join(base_data_dir, 'models')
    else:
        base_data_dir = os.path.join(base_data_dir, 'tf_pose_data')

    graph_path = os.path.join(base_data_dir, dyn_graph_path[model_name])
    if os.path.isfile(graph_path):
        return graph_path

    raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)
