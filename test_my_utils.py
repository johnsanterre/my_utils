import unittest
import numpy as np
class TestArrayEmitter(unittest.TestCase):

    def test_zero(self):
        A= 0
        B= 0
        self.assertTrue(np.array_equal(A, B))

if __name__ == '__main__':
    unittest.main()