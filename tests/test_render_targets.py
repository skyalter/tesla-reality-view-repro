"""Chase-camera targets: moving Gaussians between frames, counting observations by depth consistency,
merging moments, the ego mask, and the confidence of a chase render (alpha x visibility x depth consistency)."""
from __future__ import annotations

import math
import unittest

import torch
from _models_common import need_cuda

from realityview.render.camera import chase_camera, rotmat_to_quat
from realityview.render.gaussians import FLAG_MOVER, Gaussians, unproject
from realityview.render.targets import count_observations, ego_box_mask, merge_moments, transform_gaussians

H, W = 48, 64
F = 50.0


def rot(axis: str, deg: float) -> torch.Tensor:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
         "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
    return torch.tensor(m, dtype=torch.float64)


def pose(R: torch.Tensor, t) -> torch.Tensor:
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = R
    T[:3, 3] = torch.tensor(t, dtype=torch.float64)
    return T


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def intrinsics(n: int) -> torch.Tensor:
    K = torch.tensor([[F, 0, (W - 1) / 2], [0, F, (H - 1) / 2], [0, 0, 1]])
    return K[None].repeat(n, 1, 1)


def plane_views(xs, z: float = 10.0):
    """Cameras at (x, 0, 0) for x in ``xs``, all looking along +z at the plane z = ``z``: their depth maps,
    K and ref_T_cam."""
    n = len(xs)
    T = torch.stack([pose(torch.eye(3, dtype=torch.float64), [x, 0.0, 0.0]) for x in xs]).float()
    return torch.full((n, H, W), z), intrinsics(n), T


def gaussians_from_depth(depth, K, T, opacity=0.99, scale_px=0.8, flags=0) -> Gaussians:
    """One small Gaussian per pixel of each view, at its depth."""
    n = depth.shape[0]
    pts = unproject(depth, K, T).reshape(-1, 3)
    footprint = (depth * 0.5 * (1 / K[:, 0, 0] + 1 / K[:, 1, 1])[:, None, None]).reshape(-1)
    G = len(pts)
    return Gaussians(means=pts, quats=torch.tensor([[1.0, 0, 0, 0]]).repeat(G, 1),
                     scales=(scale_px * footprint)[:, None].repeat(1, 3), opacities=torch.full((G,), opacity),
                     colors=torch.full((G, 3), 0.5), footprint=footprint,
                     flags=torch.full((G,), flags, dtype=torch.uint8),
                     view=torch.arange(n)[:, None, None].expand(n, H, W).reshape(-1))


def random_gaussians(n: int, seed: int = 0) -> Gaussians:
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
    return Gaussians(means=torch.randn(n, 3, generator=g) * 5, quats=q, scales=torch.rand(n, 3, generator=g),
                     opacities=torch.rand(n, generator=g), colors=torch.rand(n, 3, generator=g),
                     footprint=torch.rand(n, generator=g), flags=(torch.rand(n, generator=g) > 0.5).to(torch.uint8),
                     view=torch.randint(0, 4, (n,), generator=g))


class Transform(unittest.TestCase):
    def test_roundtrip_positions_and_rotations(self):
        g = random_gaussians(200)
        T = pose(rot("z", 33) @ rot("x", -70), [120.5, -40.25, 3.0])
        moved = transform_gaussians(g, T)
        back = transform_gaussians(moved, torch.linalg.inv(T))
        self.assertTrue(torch.allclose(back.means, g.means, atol=1e-4))
        # q and -q are the same rotation
        same = torch.minimum((back.quats - g.quats).abs().amax(-1), (back.quats + g.quats).abs().amax(-1))
        self.assertLess(float(same.max()), 1e-5)
        expected = (g.means.double() @ T[:3, :3].T + T[:3, 3]).float()
        self.assertTrue(torch.allclose(moved.means, expected, atol=1e-4))
        R = quat_to_rotmat(moved.quats.double())
        self.assertTrue(torch.allclose(R, T[:3, :3] @ quat_to_rotmat(g.quats.double()), atol=1e-5))
        for name in ("scales", "opacities", "colors", "footprint", "flags", "view"):
            self.assertTrue(torch.equal(getattr(moved, name), getattr(g, name)), name)


class Observations(unittest.TestCase):
    def test_two_views_of_a_plane(self):
        depth, K, T = plane_views([0.0, 2.0])         # 2 m apart at 10 m: 10 pixels of parallax
        g = gaussians_from_depth(depth, K, T)
        n = count_observations(g, depth, K, T).reshape(2, H, W)
        u = torch.arange(W)
        # view 0 pixel u lands on view 1 pixel u - 10; view 1 pixel u on view 0 pixel u + 10
        self.assertTrue(torch.equal(n[0], (u >= 10).float().expand(H, W)))
        self.assertTrue(torch.equal(n[1], (u <= W - 11).float().expand(H, W)))
        # a wrong depth in view 1: nothing of view 0 is confirmed there (and view 1's own Gaussians are not
        # compared with view 1)
        bad = depth.clone()
        bad[1] *= 1.3
        n_bad = count_observations(g, bad, K, T).reshape(2, H, W)
        self.assertEqual(float(n_bad[0].sum()), 0.0)
        self.assertTrue(torch.equal(n_bad[1], n[1]))
        # within the tolerance still counts
        close = depth.clone()
        close[1] *= 1.03
        self.assertTrue(torch.equal(count_observations(g, close, K, T).reshape(2, H, W)[0], n[0]))

    def test_source_view_does_not_count(self):
        depth, K, T = plane_views([0.0])
        g = gaussians_from_depth(depth, K, T)
        self.assertEqual(float(count_observations(g, depth, K, T).sum()), 0.0)

    def test_behind_the_camera_does_not_count(self):
        depth, K, T = plane_views([0.0, 0.0])
        T[1, :3, :3] = rot("y", 180).float()              # view 1 looks the other way
        g = gaussians_from_depth(depth[:1], K[:1], T[:1])
        self.assertEqual(float(count_observations(g, depth, K, T).sum()), 0.0)


class Merge(unittest.TestCase):
    def test_movers_only_from_the_current_moment(self):
        a = random_gaussians(100, 1)
        b = random_gaussians(80, 2)
        na, nb = torch.arange(100).float(), torch.arange(80).float()
        T = pose(rot("z", 10), [5.0, 0.0, 0.0])
        g, n, current = merge_moments([(a, na, None, True), (b, nb, T, False)])
        static_b = (b.flags & FLAG_MOVER) == 0
        self.assertEqual(len(g), 100 + int(static_b.sum()))
        self.assertTrue(torch.equal(current, torch.cat([torch.ones(100, dtype=torch.bool),
                                                        torch.zeros(int(static_b.sum()), dtype=torch.bool)])))
        self.assertTrue(torch.equal(g.flags[:100], a.flags))            # movers of the current moment stay
        self.assertTrue(bool(((g.flags[100:] & FLAG_MOVER) == 0).all()))
        self.assertTrue(torch.equal(n, torch.cat([na, nb[static_b]])))
        self.assertTrue(torch.allclose(g.means[100:], transform_gaussians(b, T).means[static_b]))
        self.assertTrue(torch.equal(g.means[:100], a.means))


class EgoMask(unittest.TestCase):
    def test_default_chase_camera(self):
        ref_T_ego = pose(rot("z", 30), [5.0, 1.0, -1.4]).float()
        ref_T_cam, K = chase_camera(ref_T_ego, 7.0, 4.0, 12.0, 70.0, 1280, 720)
        m = ego_box_mask(ref_T_ego, ref_T_cam, K, 720, 1280)
        self.assertEqual(m.shape, (720, 1280))
        share = float(m.float().mean())
        self.assertGreater(share, 0.03)
        self.assertLess(share, 0.4)
        ys, xs = torch.nonzero(m, as_tuple=True)
        self.assertAlmostEqual(float(xs.float().mean()), 639.5, delta=5.0)      # centred left-right
        self.assertGreater(float(ys.float().mean()), 360.0)                    # lower half
        self.assertTrue(bool(m[700, 640]))                                     # reaches the bottom middle
        self.assertFalse(bool(m[:200].any()))
        self.assertFalse(bool(m[:, :250].any()) or bool(m[:, -250:].any()))


class ChaseRendering(unittest.TestCase):
    """Needs a GPU (gsplat rasterizes with CUDA)."""

    @classmethod
    def setUpClass(cls):
        cls.dev = need_cuda()
        from realityview.render.targets import render_chase
        cls.render_chase = staticmethod(render_chase)

    def to(self, g: Gaussians) -> Gaussians:
        return Gaussians(**{k: v.to(self.dev) for k, v in g.__dict__.items()})

    def chase(self, x: float = 1.0):
        """A camera between the inputs, slightly behind them, also looking along +z."""
        T = pose(torch.eye(3, dtype=torch.float64), [x, 0.0, -1.0]).float()
        return T.to(self.dev), intrinsics(1)[0].to(self.dev)

    def check_product(self, out):
        self.assertTrue(torch.allclose(out["confidence"], out["alpha"] * out["visibility"] * out["depth_consistency"]))

    def test_plane_seen_by_two_and_three_views(self):
        for xs, expected in (([0.0, 2.0], 0.5), ([0.0, 1.0, 2.0], 1.0)):
            depth, K, T = plane_views(xs)
            g = gaussians_from_depth(depth, K, T)
            n = count_observations(g, depth, K, T)
            ref_T_cam, Kc = self.chase(1.0)
            out = self.render_chase(self.to(g), n.to(self.dev), ref_T_cam, Kc, H, W,
                                    background=torch.tensor([0.0, 0.0, 1.0]))
            centre = (slice(12, 36), slice(20, 44))       # well inside the region every view sees
            self.assertGreater(float(out["alpha"][centre].min()), 0.95)
            self.assertLess(float((out["visibility"][centre] - expected).abs().max()), 0.02, f"views {xs}")
            self.assertGreater(float(out["depth_consistency"][centre].min()), 0.97)
            self.assertLess(float((out["depth"][centre] - 11.0).abs().max()), 0.05)
            self.assertTrue(torch.allclose(out["rgb"][:, centre[0], centre[1]],
                                           torch.full_like(out["rgb"][:, centre[0], centre[1]], 0.5), atol=0.03))
            self.assertLess(float(out["mover"].max()), 1e-6)
            self.check_product(out)

    def test_one_surface_is_consistent_from_a_rotated_camera(self):
        """The depth used for D is the depth in the chase camera itself, also when that camera is rotated
        against the reference frame (a rotation that is not its own transpose)."""
        depth, K, T = plane_views([0.0, 1.0, 2.0])
        g = gaussians_from_depth(depth, K, T)
        n = torch.full((len(g),), 2.0)
        ref_T_cam = pose(rot("y", 25) @ rot("x", 20) @ rot("z", 10), [-2.0, -1.5, -1.0]).float().to(self.dev)
        out = self.render_chase(self.to(g), n.to(self.dev), ref_T_cam, intrinsics(1)[0].to(self.dev), H, W)
        solid = out["alpha"] > 0.95
        self.assertGreater(int(solid.sum()), 200)
        # not 1: the plane is slanted in this coarse camera, so one pixel blends slightly different depths.
        # With the depth of another frame instead, D drops to 0 in places.
        self.assertGreater(float(out["depth_consistency"][solid].min()), 0.8)
        # and the expected depth is the plane's depth along this camera's optical axis
        cam_T_ref = torch.linalg.inv(ref_T_cam)
        pts = self.to(g).means
        z = pts @ cam_T_ref[2, :3] + cam_T_ref[2, 3]
        self.assertGreater(float(out["depth"][solid].min()), float(z.min()) - 0.2)
        self.assertLess(float(out["depth"][solid].max()), float(z.max()) + 0.2)

    def test_two_layers_lower_depth_consistency(self):
        depth, K, T = plane_views([0.0, 2.0])
        # a half-transparent layer at z = 10 in front of an opaque one at z = 15 (flagged, to read off how
        # much of each pixel comes from the far layer)
        near = gaussians_from_depth(depth[:1], K[:1], T[:1], opacity=0.3)
        far = gaussians_from_depth(depth * 1.5, K, T, opacity=0.99, flags=FLAG_MOVER)
        g = Gaussians.cat([near, far])
        n = torch.full((len(g),), 2.0)
        ref_T_cam, Kc = self.chase(1.0)                   # 1 m behind the inputs: z = 11 and 16
        out = self.render_chase(self.to(g), n.to(self.dev), ref_T_cam, Kc, H, W)
        centre = (slice(12, 36), slice(20, 44))
        self.assertGreater(float(out["alpha"][centre].min()), 0.95)
        self.assertLess(float(out["depth_consistency"][centre].max()), 0.2)
        p = out["mover"][centre]                          # weight of the far layer
        self.assertGreater(float(p.min()), 0.1)
        self.assertLess(float(p.max()), 0.9)
        # two depths 11 and 16 with weights 1 - p and p: sigma = 5 sqrt(p (1 - p)), E[z] = 11 + 5 p
        expected = torch.exp(-5.0 * torch.sqrt(p * (1 - p)) / (0.05 * (11.0 + 5.0 * p)))
        self.assertLess(float((out["depth_consistency"][centre] - expected).abs().max()), 0.02)
        self.assertLess(float((out["depth"][centre] - (11.0 + 5.0 * p)).abs().max()), 0.05)
        self.check_product(out)
        # one layer alone is consistent
        alone = self.render_chase(self.to(far), n[:len(far)].to(self.dev), ref_T_cam, Kc, H, W)
        self.assertGreater(float(alone["depth_consistency"][centre].min()), 0.97)

    def test_no_confidence_where_nothing_is(self):
        depth, K, T = plane_views([0.0, 2.0])
        g = gaussians_from_depth(depth, K, T)
        keep = g.means[:, 0].abs() < 1.0                  # a strip of the plane
        g = Gaussians(**{k: v[keep] for k, v in g.__dict__.items()})
        n = torch.full((len(g),), 3.0)
        ref_T_cam, Kc = self.chase(0.0)
        out = self.render_chase(self.to(g), n.to(self.dev), ref_T_cam, Kc, H, W)
        empty = out["alpha"] == 0
        self.assertGreater(int(empty.sum()), 100)
        for key in ("confidence", "visibility", "depth_consistency", "mover"):
            self.assertEqual(float(out[key][empty].abs().max()), 0.0, key)
        self.assertGreater(float(out["confidence"].max()), 0.9)
        self.check_product(out)


if __name__ == "__main__":
    unittest.main()
