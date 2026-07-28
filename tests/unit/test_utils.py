import numpy as np
import pytest

from energymanagementrl.simulation.utils import shuffle_array_blocks, create_gaussian_under_sampling_matrix, sparse_matrix


class TestShuffleArrayBlocks:
    def test_basic_shuffle(self):
        arr = np.arange(100)
        result = shuffle_array_blocks([arr], 10, 4, 0.5, np.random.RandomState(42))
        assert len(result) == 1
        assert len(result[0]) == 100

    def test_multiple_arrays_same_length(self):
        arr1 = np.arange(100)
        arr2 = np.arange(100, 200)
        result = shuffle_array_blocks([arr1, arr2], 10, 4, 0.5, np.random.RandomState(42))
        assert len(result) == 2
        assert len(result[0]) == len(result[1]) == 100

    def test_mixed_arrays_different_values(self):
        arr1 = np.arange(100)[::-1]
        arr2 = np.arange(100)[::-1]
        result = shuffle_array_blocks([arr1, arr2], 10, 4, 0.5, np.random.RandomState(42))
        assert all(np.array_equal(result[0], arr) for arr in result[1:])

    def test_zero_mix_preserves_order(self):
        arr = np.arange(100)
        result = shuffle_array_blocks([arr], 10, 4, 0.0, np.random.RandomState(42))
        assert np.array_equal(result[0], arr)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            shuffle_array_blocks([np.arange(100)], 0, 4, 0.5)

    def test_invalid_mix_probability(self):
        with pytest.raises(ValueError, match="mix_probability"):
            shuffle_array_blocks([np.arange(100)], 10, 4, 1.5)

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            shuffle_array_blocks([np.arange(100), np.arange(50)], 10, 4, 0.5)


class TestGaussianMatrix:
    def test_shape(self):
        matrix = create_gaussian_under_sampling_matrix(48, 2)
        assert matrix.shape[1] == 48

    def test_rows_normalized(self):
        matrix = create_gaussian_under_sampling_matrix(48, 2)
        for row in matrix:
            assert pytest.approx(row.sum(), abs=1e-6) == 1.0

    def test_sparse_matrix(self):
        assert sparse_matrix.shape[1] == 18
