"""Gaussians from per-pixel predictions and their rendering: positions, the half-pixel convention of the
renderer, rotations in the reference frame, expected depth, coverage, the chase camera."""
from __future__ import annotations

import math
import unittest

import torch
from _models_common import need_cuda

from realityview.render.camera import chase_camera, half_pixel_K, pixel_rays, project, quat_mul, rotmat_to_quat
from realityview.render.gaussians import FLAG_MOVER, RAW_CHANNELS, Gaussians, assemble, unproject

H, W = 64, 96


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


def camera(dtype=torch.float32):
    K = torch.tensor([[80.0, 0, 46.3], [0, 82.0, 30.6], [0, 0, 1]], dtype=dtype)
    return K


def initial_raw(n: int, h: int = H, w: int = W) -> torch.Tensor:
    """The raw outputs of an untrained appearance branch: a = -2, q = (1, 0, 0, 0), f = 0, o = 0."""
    raw = torch.zeros(n, RAW_CHANNELS, h, w)
    raw[:, 0:3] = -2.0
    raw[:, 3] = 1.0
    return raw


def blob_moments(alpha: torch.Tensor):
    """Centroid (x, y) and covariance of an image used as weights; pixel centres at integer coordinates."""
    a = alpha.double()
    ys, xs = torch.meshgrid(torch.arange(a.shape[0], dtype=torch.float64, device=a.device),
                            torch.arange(a.shape[1], dtype=torch.float64, device=a.device), indexing="ij")
    s = a.sum()
    cx, cy = (a * xs).sum() / s, (a * ys).sum() / s
    cxx, cyy = (a * (xs - cx) ** 2).sum() / s, (a * (ys - cy) ** 2).sum() / s
    cxy = (a * (xs - cx) * (ys - cy)).sum() / s
    return float(cx), float(cy), float(cxx), float(cyy), float(cxy)


class Geometry(unittest.TestCase):
    def test_rotmat_quat_roundtrip(self):
        mats = [torch.eye(3, dtype=torch.float64), rot("x", 180), rot("y", 180), rot("z", 180),
                rot("z", 25) @ rot("y", 40), rot("x", -100) @ rot("z", 170), rot("y", 179.9) @ rot("x", 30)]
        R = torch.stack(mats)
        q = rotmat_to_quat(R)
        self.assertTrue(torch.allclose(q.norm(dim=-1), torch.ones(len(mats), dtype=torch.float64)))
        self.assertTrue(torch.allclose(quat_to_rotmat(q), R, atol=1e-9))
        # product of quaternions = product of rotations, in the same order
        q_ab = quat_mul(rotmat_to_quat(mats[4]), rotmat_to_quat(mats[5]))
        self.assertTrue(torch.allclose(quat_to_rotmat(q_ab), mats[4] @ mats[5], atol=1e-9))

    def test_unproject_gives_depth_and_pixel(self):
        g = torch.Generator().manual_seed(0)
        K = torch.stack([camera(torch.float64), camera(torch.float64) * torch.tensor([[1.3], [0.9], [1.0]])])
        T = torch.stack([pose(rot("y", 40) @ rot("z", 25), [1.0, -2.0, 0.5]), pose(rot("x", -15), [0.0, 0.3, -4.0])])
        depth = 2.0 + 28.0 * torch.rand(2, H, W, generator=g, dtype=torch.float64)
        pts = unproject(depth, K, T)
        v, u = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
        for n in range(2):
            uv, z = project(pts[n], T[n], K[n])
            self.assertLess(float((uv[..., 0] - u).abs().max()), 1e-8)
            self.assertLess(float((uv[..., 1] - v).abs().max()), 1e-8)
            self.assertLess(float((z - depth[n]).abs().max()), 1e-9)
        # the rays have z = 1: depth is along the optical axis, not the distance to the camera
        self.assertTrue(torch.all(pixel_rays(K, H, W)[..., 2] == 1))

    def test_assemble_initial_values(self):
        g = torch.Generator().manual_seed(1)
        K = camera()[None].repeat(2, 1, 1)
        T = torch.stack([pose(rot("y", 40) @ rot("z", 25), [1.0, -2.0, 0.5]), pose(torch.eye(3, dtype=torch.float64), [0, 0, 0])]).float()
        images = torch.rand(2, 3, H, W, generator=g)
        depth = 2.0 + 28.0 * torch.rand(2, H, W, generator=g)
        keep = torch.rand(2, H, W, generator=g) > 0.3
        mover = torch.rand(2, H, W, generator=g) > 0.9
        gs = assemble(initial_raw(2), images, depth, K, T, keep, mover)
        self.assertEqual(len(gs), int(keep.sum()))
        self.assertTrue(torch.allclose(gs.means, unproject(depth, K, T)[keep], atol=1e-5))
        footprint = (depth * 0.5 * (1 / 80.0 + 1 / 82.0))[keep]
        self.assertTrue(torch.allclose(gs.footprint, footprint, rtol=1e-6))
        size = 0.1 + 7.9 * torch.sigmoid(torch.tensor(-2.0))            # about 1.04 pixel footprints
        self.assertAlmostEqual(float(size), 1.0417, places=3)
        self.assertTrue(torch.allclose(gs.scales, (size * footprint)[:, None].expand(-1, 3), rtol=1e-5))
        self.assertTrue(torch.allclose(gs.opacities, torch.sigmoid(torch.tensor(-2.0)).expand(len(gs))))
        self.assertTrue(torch.allclose(gs.colors, images.clamp(0.01, 0.99).permute(0, 2, 3, 1)[keep], atol=1e-5))
        # an untrained Gaussian has the orientation of its camera
        R = quat_to_rotmat(gs.quats.double())
        self.assertTrue(torch.allclose(R, T[:, :3, :3].double()[gs.view], atol=1e-6))
        self.assertTrue(torch.equal((gs.flags & FLAG_MOVER) > 0, mover[keep]))
        self.assertTrue(torch.equal(gs.view, torch.arange(2)[:, None, None].expand(2, H, W)[keep]))

    def test_size_is_capped_relative_to_distance(self):
        K = camera()[None]
        T = torch.eye(4)[None]
        raw = initial_raw(1)
        raw[:, 0:3] = 20.0                                              # the largest size: 8 footprints
        depth = torch.full((1, H, W), 10.0)
        keep = torch.ones(1, H, W, dtype=torch.bool)
        gs = assemble(raw, torch.rand(1, 3, H, W), depth, K, T, keep, max_scale_rel=0.02)
        distance = (depth[..., None] * pixel_rays(K, H, W)).norm(dim=-1)[keep]
        self.assertTrue(torch.allclose(gs.scales[:, 0], 0.02 * distance, rtol=1e-5))   # 8 footprints = 1 m > cap
        gs = assemble(raw, torch.rand(1, 3, H, W), depth, K, T, keep, max_scale_rel=1.0)
        self.assertTrue(torch.allclose(gs.scales[:, 0], 8.0 * gs.footprint, rtol=1e-4))

    def test_chase_camera(self):
        ref_T_ego = pose(rot("z", 30), [5.0, 1.0, -1.4]).float()
        ref_T_cam, K = chase_camera(ref_T_ego, back_m=7.0, up_m=4.0, pitch_down_deg=12.0, hfov_deg=70.0,
                                    width=1280, height=720)
        ego_T_cam = torch.linalg.inv(ref_T_ego) @ ref_T_cam
        self.assertTrue(torch.allclose(ego_T_cam[:3, 3], torch.tensor([-7.0, 0.0, 4.0]), atol=1e-5))
        self.assertTrue(torch.allclose(ego_T_cam[:3, 2], torch.tensor([math.cos(math.radians(12)), 0.0,
                                                                        -math.sin(math.radians(12))]), atol=1e-6))
        self.assertTrue(torch.allclose(ego_T_cam[:3, 0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6))   # right
        self.assertLess(float(ego_T_cam[2, 1]), 0)                       # image y points down
        self.assertAlmostEqual(float(torch.linalg.det(ego_T_cam[:3, :3])), 1.0, places=5)
        self.assertAlmostEqual(float(K[0, 0]), 640.0 / math.tan(math.radians(35.0)), places=3)
        # a point on the optical axis lands on the image centre; the ego origin lands below it, in the middle
        on_axis = ref_T_cam[:3, 3] + 10.0 * ref_T_cam[:3, 2]
        uv, z = project(on_axis[None], ref_T_cam, K)
        self.assertTrue(torch.allclose(uv[0], torch.tensor([639.5, 359.5]), atol=1e-3))
        uv, z = project(ref_T_ego[:3, 3][None], ref_T_cam, K)
        self.assertAlmostEqual(float(uv[0, 0]), 639.5, places=2)
        self.assertGreater(float(uv[0, 1]), 359.5)
        self.assertGreater(float(z[0]), 0)
        # a point to the left of the car (ego +y) lands in the left half of the image
        left = (ref_T_ego @ torch.tensor([10.0, 3.0, 0.0, 1.0]))[:3]
        self.assertLess(float(project(left[None], ref_T_cam, K)[0][0, 0]), 639.5)


class Rendering(unittest.TestCase):
    """Needs a GPU (gsplat rasterizes with CUDA)."""

    @classmethod
    def setUpClass(cls):
        cls.dev = need_cuda()
        from realityview.render.rasterize import coverage, render
        cls.render, cls.coverage = staticmethod(render), staticmethod(coverage)

    def _points(self, uv, z, K, T, scale_px=0.3, opacity=0.99):
        """Small opaque Gaussians at the 3D points of given pixels of the camera (K, T)."""
        uv = torch.tensor(uv, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)
        cam = torch.stack([(uv[:, 0] - K[0, 2]) / K[0, 0] * z, (uv[:, 1] - K[1, 2]) / K[1, 1] * z, z], -1)
        means = cam @ T[:3, :3].T + T[:3, 3]
        n = len(uv)
        footprint = z * 0.5 * (1 / K[0, 0] + 1 / K[1, 1])
        g = Gaussians(means=means, quats=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
                      scales=(scale_px * footprint)[:, None].repeat(1, 3), opacities=torch.full((n,), opacity),
                      colors=torch.ones(n, 3), footprint=footprint, flags=torch.zeros(n, dtype=torch.uint8),
                      view=torch.zeros(n, dtype=torch.int64))
        return Gaussians(**{k: v.to(self.dev) for k, v in g.__dict__.items()})

    def test_pixel_alignment(self):
        """A Gaussian at the 3D point of pixel (u, v) renders centred on (u, v): the renderer is given
        cx, cy + 0.5 and everything else keeps the integer pixel-centre convention."""
        K = camera()
        T = pose(rot("y", 20) @ rot("x", -10), [0.5, -0.2, 1.0]).float()
        pixels = [(12, 10), (40, 12), (70, 15), (20, 40), (55, 33), (83, 50)]
        g = self._points(pixels, [5.0, 8.0, 12.0, 6.0, 20.0, 9.0], K, T)
        for mode in ("classic", "antialiased"):
            out = self.render(g, T[None].to(self.dev), K[None].to(self.dev), H, W, mode=mode)
            alpha = out["alpha"][0]
            for u, v in pixels:
                window = torch.zeros_like(alpha)
                window[v - 5:v + 6, u - 5:u + 6] = alpha[v - 5:v + 6, u - 5:u + 6]
                cx, cy, *_ = blob_moments(window)
                self.assertLess(abs(cx - u), 0.1, f"{mode}: x of pixel {(u, v)}")
                self.assertLess(abs(cy - v), 0.1, f"{mode}: y of pixel {(u, v)}")
        # without the half-pixel shift the image would be off by 0.5 pixel
        from gsplat import rasterization
        _, alpha, _ = rasterization(g.means, g.quats, g.scales, g.opacities, g.colors,
                                    torch.linalg.inv(T)[None].to(self.dev), K[None].to(self.dev), W, H)
        u, v = pixels[0]
        window = torch.zeros_like(alpha[0, ..., 0])
        window[v - 5:v + 6, u - 5:u + 6] = alpha[0, v - 5:v + 6, u - 5:u + 6, 0]
        cx, cy, *_ = blob_moments(window)
        self.assertAlmostEqual(cx - u, -0.5, delta=0.1)
        self.assertTrue(torch.equal(half_pixel_K(K)[:2, 2], K[:2, 2] + 0.5))

    def _plane(self, z0: float, T, K, opacity_raw: float = 4.0, keep=None, mover=None):
        raw = initial_raw(1)
        raw[:, 10] = opacity_raw
        images = torch.rand(1, 3, H, W, generator=torch.Generator().manual_seed(3))
        depth = torch.full((1, H, W), z0)
        keep = torch.ones(1, H, W, dtype=torch.bool) if keep is None else keep
        args = [x.to(self.dev) for x in (raw, images, depth, K[None], T[None], keep)]
        g = assemble(*args, None if mover is None else mover.to(self.dev))
        return g, images

    def test_plane_renders_back_with_its_depth_and_colour(self):
        K = camera()
        T = pose(rot("y", 20) @ rot("z", 10), [0.5, -0.2, 1.0]).float()
        g, images = self._plane(10.0, T, K)
        out = self.render(g, T[None].to(self.dev), K[None].to(self.dev), H, W)
        inner = (slice(4, H - 4), slice(4, W - 4))
        self.assertGreater(float(out["alpha"][0][inner].min()), 0.95)
        self.assertLess(float((out["depth"][0][inner] - 10.0).abs().max()), 0.05)
        # colours: every Gaussian has the colour of its pixel; the render is a blurred copy of the image
        blurred = torch.nn.functional.avg_pool2d(images, 5, 1, 2).to(self.dev)
        self.assertLess(float((out["rgb"][0][:, inner[0], inner[1]] - blurred[0][:, inner[0], inner[1]]).abs().mean()), 0.08)
        # a camera 2 m further along the optical axis sees the plane 2 m closer
        T2 = T.clone()
        T2[:3, 3] += 2.0 * T[:3, 2]
        out2 = self.render(g, T2[None].to(self.dev), K[None].to(self.dev), H, W)
        self.assertAlmostEqual(float(out2["depth"][0][H // 2, W // 2]), 8.0, delta=0.05)

    def test_background_and_mover_channel(self):
        K = camera()
        T = torch.eye(4)
        keep = torch.zeros(1, H, W, dtype=torch.bool)
        keep[:, :, : W // 2] = True                                      # only the left half produces Gaussians
        mover = torch.zeros(1, H, W, dtype=torch.bool)
        mover[:, : H // 2] = True                                        # the upper half is "moving"
        g, _ = self._plane(10.0, T, K, keep=keep, mover=mover)
        bg = torch.tensor([0.2, 0.5, 0.8], device=self.dev)
        out = self.render(g, T[None].to(self.dev), K[None].to(self.dev), H, W, background=bg)
        self.assertLess(float(out["alpha"][0][:, W // 2 + 6:].max()), 0.01)
        self.assertTrue(torch.allclose(out["rgb"][0][:, 10, W - 5], bg, atol=1e-3))
        self.assertGreater(float(out["mover"][0][5:H // 2 - 5, 5:W // 2 - 5].min()), 0.9)
        self.assertLess(float(out["mover"][0][H // 2 + 5:, 5:W // 2 - 5].max()), 0.05)
        self.assertEqual(float(out["depth"][0][10, W - 5]), 0.0)         # nothing hit: depth 0

    def test_coverage_ignores_learned_opacity(self):
        K = camera()
        T = torch.eye(4)
        keep = torch.zeros(1, H, W, dtype=torch.bool)
        keep[:, :, : W // 2] = True
        g_low, _ = self._plane(10.0, T, K, opacity_raw=-6.0, keep=keep)   # nearly transparent
        g_high, _ = self._plane(10.0, T, K, opacity_raw=6.0, keep=keep)
        cams = (T[None].to(self.dev), K[None].to(self.dev))
        c_low, c_high = self.coverage(g_low, *cams, H, W), self.coverage(g_high, *cams, H, W)
        self.assertTrue(torch.allclose(c_low, c_high))
        self.assertGreater(float(c_low[0][4:H - 4, 4:W // 2 - 4].min()), 0.95)
        self.assertLess(float(c_low[0][:, W // 2 + 6:].max()), 0.01)
        self.assertLess(float(self.render(g_low, *cams, H, W)["alpha"].max()), 0.2)   # while the render is faint
        self.assertFalse(c_low.requires_grad)

    def test_rotation_in_the_reference_frame(self):
        """An elongated Gaussian keeps its orientation: along the image x axis of its own (rotated) camera,
        and at the matching angle when seen from the reference camera."""
        K = camera()
        R = rot("z", 25) @ rot("y", 15)                                   # roll and yaw of the source camera
        T = pose(R, [0.3, 0.1, -0.5]).float()
        raw = initial_raw(1, 9, 9)
        raw[:, 0] = 20.0                                                   # 8 footprints along the camera x axis
        raw[:, 1:3] = -20.0                                                # 0.1 footprint along y and z
        raw[:, 10] = 6.0
        Kc = torch.tensor([[80.0, 0, 4.0], [0, 80.0, 4.0], [0, 0, 1]])     # a 9 x 9 image, centre pixel (4, 4)
        keep = torch.zeros(1, 9, 9, dtype=torch.bool)
        keep[0, 4, 4] = True

        def one(raw_values):
            args = [x.to(self.dev) for x in (raw_values, torch.ones(1, 3, 9, 9), torch.full((1, 9, 9), 10.0), Kc[None], T[None], keep)]
            return assemble(*args, max_scale_rel=1.0)

        big = torch.tensor([[160.0, 0, 79.5], [0, 160.0, 59.5], [0, 0, 1]])   # view it with a 160 x 120 camera
        g = one(raw)
        self.assertEqual(len(g), 1)
        own = self.render(g, T[None].to(self.dev), big[None].to(self.dev), 120, 160, mode="classic")["alpha"][0]
        cx, cy, cxx, cyy, cxy = blob_moments(own)
        self.assertGreater(cxx, 20 * cyy)                                  # long along x, thin along y
        self.assertLess(abs(cxy), 0.02 * cxx)
        # the head's own rotation is applied in the camera frame: 90 degrees about the optical axis
        raw90 = raw.clone()
        raw90[:, 3:7] = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])[None, :, None, None]
        own90 = self.render(one(raw90), T[None].to(self.dev), big[None].to(self.dev), 120, 160, mode="classic")["alpha"][0]
        _, _, cxx, cyy, cxy = blob_moments(own90)
        self.assertGreater(cyy, 20 * cxx)
        # seen from the reference camera, the long axis is the source camera's x axis
        ref = torch.eye(4)
        centre = g.means[0].cpu()
        axis = T[:3, 0]
        ends, _ = project(torch.stack([centre - axis, centre + axis]), ref, big)
        expected = math.degrees(math.atan2(float(ends[1, 1] - ends[0, 1]), float(ends[1, 0] - ends[0, 0]))) % 180.0
        seen = self.render(g, ref[None].to(self.dev), big[None].to(self.dev), 120, 160, mode="classic")["alpha"][0]
        _, _, cxx, cyy, cxy = blob_moments(seen)
        angle = math.degrees(0.5 * math.atan2(2 * cxy, cxx - cyy)) % 180.0
        self.assertGreater(abs(expected - 90.0), 10.0)                     # a tilted axis, so the test is meaningful
        self.assertLess(min(abs(angle - expected), 180.0 - abs(angle - expected)), 3.0)


if __name__ == "__main__":
    unittest.main()
