from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import setuptools
from distutils.core import Extension, setup

import numpy as np

_VERSION = '0.1'

setup(name='openpose-py-tf',
      version=_VERSION,
      description='OpenPose implemented in TensorFlow',
      py_modules=[
          'openpose_py_tf'
      ])


cwd = os.path.dirname(os.path.abspath(__file__))
subprocess.check_output(["bash", "models/graph/cmu/download.sh"], cwd=cwd)

POSE_DIR = os.path.realpath(os.path.dirname(__file__))

REQUIRED_PACKAGES = [
    "numpy>=1.14.5",
    "opencv-python>=3.4.2",
    "scipy>=1.1.0",
    "slidingwindow>=0.0.13",
    "tensorflow>=1.9.0"
]

EXT = Extension('_pafprocess',
                sources=[
                    'openpose_py_tf/pafprocess/pafprocess_wrap.cpp',
                    'openpose_py_tf/pafprocess/pafprocess.cpp',
                ],
                swig_opts=['-c++'],
                include_dirs=[np.get_include()])

setuptools.setup(
    name='openpose-tf-py',
    version=_VERSION,
    description=
    'Deep Pose Estimation implemented using Tensorflow with Custom Architectures for fast inference.',
    install_requires=REQUIRED_PACKAGES,
    url='https://github.com/mgrinshpon/openpose-py-tf/',
    license='Apache License 2.0',
    package_dir={'tf_pose_data': 'models'},
    packages=['tf_pose_data'] +
             [pkg_name for pkg_name in setuptools.find_packages()  # main package
              if 'openpose_py_tf' in pkg_name],
    ext_modules=[EXT],
    package_data={'tf_pose_data': ['graph/cmu/graph_opt.pb',
                                   'graph/mobilenet_thin/graph_opt.pb']},
    py_modules=[
        "pafprocess"
    ],
    zip_safe=False)
