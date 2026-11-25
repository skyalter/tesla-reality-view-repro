import os
import tempfile
import unittest

import _util  # noqa: F401
import numpy as np

from realityview.data.lidar_depth import SparseDepth, seethrough_flags, zbuffer
from realityview.data.store import LogStore, depth_from_bytes, depth_to_bytes, pack_log, write_bytes


def random_depth(n=5000, hw=(392, 518), seed=0):
    rng = np.random.default_rng(seed)
    pix = np.sort(rng.choice(hw[0] * hw[1], n, replace=False))
    vu = np.stack([pix // hw[1], pix % hw[1]], 1).astype(np.uint16)
    z = rng.uniform(0.5, 250, n).astype(np.float16)
    flag = rng.integers(0, 8, n).astype(np.uint8)
    return SparseDepth(vu, z, flag)


class TestDepthFile(unittest.TestCase):
    def test_round_trip(self):
        d = random_depth()
        b = depth_to_bytes(d)
        e = depth_from_bytes(b)
        np.testing.assert_array_equal(d.vu, e.vu)
        np.testing.assert_array_equal(d.z, e.z)
        np.testing.assert_array_equal(d.flag, e.flag)
        self.assertEqual(e.vu.dtype, np.uint16)
        self.assertEqual(e.z.dtype, np.float16)
        self.assertEqual(e.flag.dtype, np.uint8)

    def test_deterministic_bytes(self):
        d = random_depth()
        self.assertEqual(depth_to_bytes(d), depth_to_bytes(random_depth()))

    def test_empty(self):
        d = SparseDepth(np.zeros((0, 2), np.uint16), np.zeros(0, np.float16), np.zeros(0, np.uint8))
        e = depth_from_bytes(depth_to_bytes(d))
        self.assertEqual(len(e), 0)
        dense, flag = e.densify((392, 518))
        self.assertFalse(dense.any())

    def test_densify(self):
        d = random_depth()
        dense, flag = d.densify((392, 518))
        self.assertEqual(int((dense > 0).sum()), len(d))
        v, u = d.vu[:, 0].astype(int), d.vu[:, 1].astype(int)
        np.testing.assert_array_equal(dense[v, u], d.z.astype(np.float32))
        np.testing.assert_array_equal(flag[v, u], d.flag)

    def test_float16_precision(self):
        z = np.array([0.5, 10.0, 100.0, 200.0, 250.0])
        rel = np.abs(z.astype(np.float16).astype(np.float64) - z) / z
        self.assertLess(rel.max(), 1e-3)


class TestZBuffer(unittest.TestCase):
    def test_nearest_wins(self):
        pix = np.array([5, 5, 7, 5, 7, 9])
        z = np.array([3.0, 1.0, 2.0, 1.0, 4.0, 8.0])
        win = zbuffer(pix, z, 20)
        self.assertEqual(sorted(win.tolist()), [1, 2, 5])  # tie at pixel 5 -> lowest index

    def test_seethrough(self):
        vu = np.array([[10, 10], [10, 11], [10, 20]])
        z = np.array([5.0, 6.5, 6.5], np.float32)
        st = seethrough_flags(vu, z, (40, 40), 5, 1.2)
        np.testing.assert_array_equal(st, [False, True, False])


class TestLogStore(unittest.TestCase):
    def test_dir_and_zip_read_the_same(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "log")
            write_bytes(os.path.join(d, "a.txt"), b"hello")
            write_bytes(os.path.join(d, "depth", "acc", "cam", "1.npz"), depth_to_bytes(random_depth()))
            z = os.path.join(tmp, "log.zip")
            self.assertEqual(pack_log(d, z), 2)
            a, b = LogStore(d), LogStore(z)
            self.assertEqual(a.read("a.txt"), b.read("a.txt"))
            self.assertEqual(len(a.depth("acc", "cam", 1)), len(b.depth("acc", "cam", 1)))
            self.assertTrue(b.exists("a.txt") and not b.exists("nope"))
            z2 = os.path.join(tmp, "log2.zip")
            pack_log(d, z2)
            with open(z, "rb") as f1, open(z2, "rb") as f2:
                self.assertEqual(f1.read(), f2.read())


if __name__ == "__main__":
    unittest.main()
