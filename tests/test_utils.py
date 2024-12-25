from unittest import TestCase

import numpy as np

from energymanagementrl.simulation import shuffle_array_blocks


class Test(TestCase):
    def test_shuffle_array_blocks(self):
        arrays = [np.arange(100)[::-1] for _ in range(3)]
        shuffled_arrays = shuffle_array_blocks(arrays, 10, 4, .5)
        assert all(np.array_equal(shuffled_arrays[0], arr) for arr in shuffled_arrays[1:]), "Arrays are not equal."
