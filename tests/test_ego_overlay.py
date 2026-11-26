"""The ego-vehicle overlay: the built-in sedan, reading and placing an OBJ, the mesh rasteriser, laying the
overlay over an image, the mask of the car, and the soft confidence."""
from __future__ import annotations

import os
import tempfile
import unittest

import _util  # noqa: F401  (sets sys.path)
import numpy as np
import torch

from realityview.render.camera import chase_camera
from realityview.render.ego_overlay import (SEDAN, Mesh, composite, load_obj, load_overlay, place_mesh, render_mesh,
                                            save_overlay, sedan_mesh)
from realityview.render.targets import EGO_BOX, ego_box_mask, overlay_mask, soft_confidence


def camera(back=10.0, up=6.0, pitch=14.0, hfov=80.0, width=320, height=180):
    T, K = chase_camera(torch.eye(4), back, up, pitch, hfov, width, height)
    return T.double().numpy(), K.double().numpy()


def quad(z: float, half: float, color) -> Mesh:
    """A square of side 2 * half at depth z in front of an identity camera, facing it."""
    v = np.asarray([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]], np.float64)
    return Mesh(v, np.asarray([[0, 2, 1], [0, 3, 2]], np.int64), np.tile(np.asarray(color, np.float64), (2, 1)))


class Sedan(unittest.TestCase):
    def test_size_and_place(self):
        m = sedan_mesh()
        lo, hi = m.vertices.min(0), m.vertices.max(0)
        self.assertAlmostEqual(lo[0], -SEDAN["rear_overhang"], delta=0.02)
        self.assertAlmostEqual(hi[0] - lo[0], SEDAN["length"], delta=0.02)
        self.assertAlmostEqual(hi[1] - lo[1], SEDAN["width"], delta=0.02)
        self.assertAlmostEqual(lo[2], -SEDAN["axle_height"], delta=0.01)             # the tyres touch the road
        self.assertAlmostEqual(hi[2] - lo[2], SEDAN["height"], delta=0.03)
        self.assertAlmostEqual(lo[1], -hi[1], delta=1e-6)                            # symmetric
        self.assertTrue(np.isfinite(m.vertices).all())
        self.assertEqual(m.faces.max(), len(m.vertices) - 1)

    def test_inside_ego_box(self):
        m = sedan_mesh()
        for axis, (a, b) in enumerate(EGO_BOX):
            self.assertGreaterEqual(m.vertices[:, axis].min(), a)
            self.assertLessEqual(m.vertices[:, axis].max(), b)

    def test_faces_point_outwards(self):
        m = sedan_mesh()
        a, b, c = (m.vertices[m.faces[:, k]] for k in range(3))
        volume = float((a * np.cross(b, c)).sum() / 6.0)
        self.assertGreater(volume, 4.0)             # a closed surface with outward faces has a positive volume
        self.assertLess(volume, SEDAN["length"] * SEDAN["width"] * SEDAN["height"])


class Rasteriser(unittest.TestCase):
    K = np.asarray([[100.0, 0, 31.5], [0, 100.0, 23.5], [0, 0, 1]])

    def test_area_and_colour(self):
        # a 0.2 m square at 1 m: 20 x 20 pixels around the centre
        ov = render_mesh(quad(1.0, 0.1, (0.2, 0.4, 0.6)), np.eye(4), self.K, 48, 64, shadow=0.0,
                         light=(0, 0, -1), ambient=1.0)
        self.assertAlmostEqual(float(ov[..., 3].sum()), 400.0, delta=12.0)
        self.assertEqual(float(ov[0, 0, 3]), 0.0)
        centre = ov[24, 32]
        np.testing.assert_allclose(centre[:3], (0.2, 0.4, 0.6), atol=0.26)          # colour plus the highlight
        self.assertEqual(float(centre[3]), 1.0)
        self.assertGreater(centre[2], centre[1])
        self.assertGreater(centre[1], centre[0])

    def test_nearer_surface_wins(self):
        far, near = quad(2.0, 0.4, (1.0, 0.0, 0.0)), quad(1.0, 0.05, (0.0, 0.0, 1.0))
        both = Mesh(np.concatenate([far.vertices, near.vertices]), np.concatenate([far.faces, near.faces + 4]),
                    np.concatenate([far.colors, near.colors]))
        for mesh in (both, Mesh(both.vertices, both.faces[::-1].copy(), both.colors[::-1].copy())):
            ov = render_mesh(mesh, np.eye(4), self.K, 48, 64, shadow=0.0, ambient=1.0)
            self.assertGreater(ov[24, 32, 2], ov[24, 32, 0])                         # blue in the middle
            self.assertGreater(ov[24, 44, 0], ov[24, 44, 2])                         # red around it

    def test_behind_the_camera(self):
        with self.assertRaises(ValueError):
            render_mesh(quad(-1.0, 0.1, (1, 1, 1)), np.eye(4), self.K, 48, 64)

    def test_supersampling_gives_soft_edges(self):
        ov = render_mesh(quad(1.0, 0.103, (1, 1, 1)), np.eye(4), self.K, 48, 64, shadow=0.0, supersample=4)
        a = ov[..., 3]
        self.assertTrue(((a > 0.05) & (a < 0.95)).any())


class SedanInTheChaseCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T, cls.K = camera()
        cls.body = render_mesh(sedan_mesh(), cls.T, cls.K, 180, 320, shadow=0.0)
        cls.full = render_mesh(sedan_mesh(), cls.T, cls.K, 180, 320)

    def test_inside_the_box_mask(self):
        T, K = chase_camera(torch.eye(4), 10.0, 6.0, 14.0, 80.0, 320, 180)
        box = ego_box_mask(torch.eye(4), T, K, 180, 320).numpy()
        car = self.body[..., 3] > 0
        self.assertEqual(int((car & ~box).sum()), 0)
        self.assertGreater(car.sum() / box.sum(), 0.35)

    def test_left_right_symmetric(self):
        a = self.body[..., 3]
        self.assertLess(float(np.abs(a - a[:, ::-1]).mean()), 2e-3)

    def test_shadow_only_adds(self):
        self.assertTrue((self.full[..., 3] >= self.body[..., 3] - 1e-6).all())
        self.assertGreater(float(self.full[..., 3].sum()), float(self.body[..., 3].sum()))
        opaque = self.body[..., 3] > 0.999
        np.testing.assert_allclose(self.full[opaque][:, :3], self.body[opaque][:, :3], atol=1e-5)

    def test_mask_of_the_car(self):
        tight = overlay_mask(self.body, 0)
        grown = overlay_mask(self.body, 3)
        self.assertEqual(tight.dtype, torch.bool)
        self.assertTrue(bool((grown | ~tight).all()))
        self.assertGreater(int(grown.sum()), int(tight.sum()))


class Composite(unittest.TestCase):
    def test_float_and_uint8(self):
        img = np.full((4, 6, 3), 0.5, np.float32)
        ov = np.zeros((4, 6, 4), np.float32)
        ov[1, 1] = (1.0, 0.0, 0.0, 1.0)
        ov[2, 2] = (0.0, 1.0, 0.0, 0.5)
        out = composite(img, ov)
        np.testing.assert_allclose(out[0, 0], 0.5)
        np.testing.assert_allclose(out[1, 1], (1.0, 0.0, 0.0))
        np.testing.assert_allclose(out[2, 2], (0.25, 0.75, 0.25))
        out8 = composite((img * 255).astype(np.uint8), ov)
        self.assertEqual(out8.dtype, np.uint8)
        np.testing.assert_array_equal(out8[1, 1], (255, 0, 0))
        with self.assertRaises(ValueError):
            composite(img[:2], ov)

    def test_png_round_trip(self):
        ov = np.zeros((4, 6, 4), np.float32)
        ov[1, 2] = (1.0, 0.5, 0.0, 0.25)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o.png")
            save_overlay(path, ov)
            back = load_overlay(path)
        self.assertEqual(back.dtype, np.uint8)
        np.testing.assert_array_equal(back[1, 2], (255, 128, 0, 64))
        self.assertEqual(int(back[0, 0, 3]), 0)


class ObjFiles(unittest.TestCase):
    def test_read_and_place(self):
        # a box 2 long (file -z is the front), 1 wide (file x), 0.5 high (file y is up); one quad per side
        obj = ["mtllib car.mtl"]
        for x in (-0.5, 0.5):
            for y in (3.0, 3.5):
                for z in (-1.0, 1.0):
                    obj.append(f"v {x} {y} {z}")
        obj += ["usemtl paint", "f 1 2 4 3", "f 5 7 8 6", "usemtl glass", "f 1/1/1 5/1/1 6/1/1 2/1/1",
                "f -6 -5 -1 -2", "f 1 3 7 5", "f 2 6 8 4"]
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "car.mtl"), "w") as f:
                f.write("newmtl paint\nKd 0.9 0.1 0.1\nnewmtl glass\nKd 0.1 0.1 0.2\n")
            with open(os.path.join(d, "car.obj"), "w") as f:
                f.write("\n".join(obj) + "\n")
            mesh = load_obj(os.path.join(d, "car.obj"))
        self.assertEqual(mesh.vertices.shape, (8, 3))
        self.assertEqual(mesh.faces.shape, (12, 3))
        np.testing.assert_allclose(mesh.colors[0], (0.9, 0.1, 0.1))
        np.testing.assert_allclose(mesh.colors[4], (0.1, 0.1, 0.2))
        placed = place_mesh(mesh, forward="-z", up="+y", length=4.0, rear_overhang=1.0, axle_height=0.3)
        lo, hi = placed.vertices.min(0), placed.vertices.max(0)
        np.testing.assert_allclose(lo, (-1.0, -1.0, -0.3), atol=1e-9)
        np.testing.assert_allclose(hi, (3.0, 1.0, 0.7), atol=1e-9)
        # the corner that was front (z = -1), file +x, bottom is now front, right (ego y < 0), bottom
        np.testing.assert_allclose(placed.vertices[4], (3.0, -1.0, -0.3), atol=1e-9)
        with self.assertRaises(ValueError):
            place_mesh(mesh, forward="+x", up="-x")


class SoftConfidence(unittest.TestCase):
    def test_values(self):
        alpha = torch.tensor([1.0, 1.0, 1.0, 0.8, 0.4])
        vis = torch.tensor([1.0, 0.5, 0.0, 0.25, 1.0])
        dcons = torch.tensor([1.0, 0.25, 1.0, 1.0, 1.0])
        out = soft_confidence(alpha, vis, dcons)
        torch.testing.assert_close(out, torch.tensor([1.0, 0.5, 0.5, 0.8 * 0.75, 0.0]))


if __name__ == "__main__":
    unittest.main()
