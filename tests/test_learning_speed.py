import numpy as np

from scripts.return_tests import SMOOTH_WINDOW, _first_reach


def test_reports_end_of_first_complete_qualifying_window():
    curve = np.r_[np.zeros(3), np.ones(SMOOTH_WINDOW)]
    assert _first_reach(curve, 0.99) == 3 + SMOOTH_WINDOW - 1


def test_first_window_and_unreached_target():
    assert _first_reach(np.ones(30), 0.5) == SMOOTH_WINDOW - 1
    assert _first_reach(np.zeros(30), 0.5) is None
