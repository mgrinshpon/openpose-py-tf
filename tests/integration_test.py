import cv2
import numpy as np
import pytest
from tf_pose import Estimator
from tf_pose.common import CocoPart


_epsilon = 1e-2


def test_integration():
    image_path = "resources/friendly_test_picture.png"
    image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)  # Color image size of 1080x1920x3

    graph_path = "../models/graph/cmu/graph_opt.pb"
    target_size = (1312, 736)  # Default largest size in width and height

    estimator = Estimator(graph_path, target_size=target_size)
    humans = estimator.inference(np.array(image), resize_to_default=True,
                                 upsample_size=4.0)  # TODO why is this set to a default of 4.0?

    # retval = Estimator.draw_humans(np.array(image), humans)
    # cv2.imwrite("output.png", retval)

    assert 3 == len(humans)

    monica = humans[0]
    assert 11 == monica.part_count()
    monica_nose = monica.body_parts.get(CocoPart.Nose.value)
    assert monica_nose is not None
    assert 0.90 == pytest.approx(monica_nose.score, abs=_epsilon)
    assert 0.54 == pytest.approx(monica_nose.x, abs=_epsilon)
    assert 0.39 == pytest.approx(monica_nose.y, abs=_epsilon)

    chandler = humans[1]
    assert 8 == chandler.part_count()
    chandler_nose = chandler.body_parts.get(CocoPart.Nose.value)
    assert chandler_nose is not None
    assert 0.83 == pytest.approx(chandler_nose.score, abs=_epsilon)
    assert 0.70 == pytest.approx(chandler_nose.x, abs=_epsilon)
    assert 0.26 == pytest.approx(chandler_nose.y, abs=_epsilon)

    joey = humans[2]
    assert 10 == joey.part_count()
    joey_nose = joey.body_parts.get(CocoPart.Nose.value)
    assert joey_nose is not None
    assert 0.80 == pytest.approx(joey_nose.score, abs=_epsilon)
    assert 0.31 == pytest.approx(joey_nose.x, abs=_epsilon)
    assert 0.28 == pytest.approx(joey_nose.y, abs=_epsilon)
