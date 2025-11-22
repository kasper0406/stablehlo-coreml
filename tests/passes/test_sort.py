import unittest
import numpy as np
from stablehlo_coreml.sort_utils import stable_argsort

class TestStableArgsort(unittest.TestCase):

    def spy_argsort(self, arr, axis=-1):
        """
        A mock argsort that doesn't just sort, but INSPECTS the input.
        It asserts that the input 'arr' has NO DUPLICATES.
        """
        # Flatten along the axis to check uniqueness per row/list
        if axis == -1:
            # Simple check for 1D case or last axis
            # For ND arrays, this check loops over the sorting axis
            # To keep the test simple, we will verify global uniqueness for 1D inputs
            if arr.ndim == 1:
                unique_elements = np.unique(arr)
                if len(unique_elements) != len(arr):
                    # This is the core failure condition.
                    # If stable_argsort logic is flawed, duplicates will exist here.
                    raise AssertionError(f"Collisions detected in composite keys! Input to op was: {arr}")

        # Return a valid sort so the function completes
        return np.argsort(arr, axis=axis)

    def test_integer_stability_ascending(self):
        """Test standard integer stability (Ascending)."""
        # Input: [10, 5, 10, 5]
        # Indices: 0, 1, 2, 3
        # Expected:
        #   5 (idx 1), 5 (idx 3), 10 (idx 0), 10 (idx 2)
        data = np.array([10, 5, 10, 5])

        result = stable_argsort(data, ascending=True, argsort_op=self.spy_argsort)

        expected = np.array([1, 3, 0, 2])
        np.testing.assert_array_equal(result, expected)
        print("\n[Pass] Integer Ascending: Uniqueness verified, order correct.")

    def test_integer_stability_descending(self):
        """
        Test descending stability.
        CRITICAL: Indices must still be ascending (0, 1, 2) for stability.
        If the implementation inverted the index bits, this test would fail.
        """
        # Input: [10, 5, 10, 5]
        # Expected:
        #   10 (idx 0), 10 (idx 2), 5 (idx 1), 5 (idx 3)
        data = np.array([10, 5, 10, 5])

        result = stable_argsort(data, ascending=False, argsort_op=self.spy_argsort)

        expected = np.array([0, 2, 1, 3])
        np.testing.assert_array_equal(result, expected)
        print("[Pass] Integer Descending: Uniqueness verified, order correct.")

    def test_float_negative_values(self):
        """Test that float bit-casting correctly handles negative numbers and order."""
        # -5 is smaller than -2.
        # Input: [-2.0, -5.0, -2.0, 0.0]
        # Expected Ascending: -5.0 (idx 1), -2.0 (idx 0), -2.0 (idx 2), 0.0 (idx 3)
        data = np.array([-2.0, -5.0, -2.0, 0.0], dtype=np.float32)

        result = stable_argsort(data, ascending=True, argsort_op=self.spy_argsort)

        expected = np.array([1, 0, 2, 3])
        np.testing.assert_array_equal(result, expected)
        print("[Pass] Float Negatives: Bit-casting preserved correct numeric order.")

    def test_float_descending_stability(self):
        """Test that descending floats maintain index stability."""
        # Input: [1.5, 1.5, 1.5]
        # Expected Descending: 1.5 (idx 0), 1.5 (idx 1), 1.5 (idx 2)
        data = np.array([1.5, 1.5, 1.5], dtype=np.float32)

        result = stable_argsort(data, ascending=False, argsort_op=self.spy_argsort)

        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(result, expected)
        print("[Pass] Float Descending Identity: Tie-breakers worked correctly.")

    def test_ensure_unstable_behavior_without_fix(self):
        """
        Sanity Check: Verify that passing raw duplicates to our spy
        WOULD fail if we didn't use the composite key logic.
        """
        data = np.array([10, 10])
        try:
            self.spy_argsort(data)
        except AssertionError as e:
            print(f"[Pass] Sanity Check: Spy correctly catches duplicates in raw data. ({e})")
            return

        self.fail("Spy function failed to detect duplicates in raw data!")
