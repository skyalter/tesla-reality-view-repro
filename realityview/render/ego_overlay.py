"""The ego vehicle of the chase view: a car mesh rendered once into an RGBA image that is laid over every frame.

The chase camera is fixed in the ego frame, so the car looks the same in every frame: ``render_mesh`` draws
it once (a small z-buffer rasteriser, numpy only) and ``composite`` lays the result over a chase image. The
training targets keep the ego area empty (confidence 0); the overlay is for display.

Meshes are in the ego frame (origin at the centre of the rear axle, x forward, y left, z up, metres):

- ``sedan_mesh()`` builds a plain three-box sedan from a few numbers (default: the outline of the Argoverse 2
  vehicle, a Ford Fusion). It needs no file.
- ``load_obj(path)`` reads a Wavefront OBJ (with the diffuse colours of its MTL file, if any); ``place_mesh``
  turns and scales a mesh from its own axes into the ego frame.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

# Ford Fusion (2013-2020): length 4.87 m, width 1.85 m, height 1.48 m, wheelbase 2.85 m, rear overhang about
# 1.10 m; the rear-axle centre is about 0.33 m above the road
SEDAN = {"length": 4.87, "width": 1.85, "height": 1.48, "wheelbase": 2.85, "rear_overhang": 1.10,
         "axle_height": 0.33, "wheel_radius": 0.33, "wheel_width": 0.24}

BODY = (0.80, 0.81, 0.83)
GLASS = (0.10, 0.12, 0.15)
TYRE = (0.06, 0.06, 0.07)
LAMP = (0.75, 0.08, 0.08)


@dataclass
class Mesh:
    vertices: np.ndarray        # (V, 3) float64, ego frame, metres
    faces: np.ndarray           # (F, 3) int64, counter-clockwise seen from outside
    colors: np.ndarray          # (F, 3) float64 0..1, one colour per face


def _loft(rings: np.ndarray, colors: np.ndarray) -> Mesh:
    """Rings (S, N, 3) of N points each, joined into a closed tube with flat caps at both ends.
    ``colors`` (S - 1, N, 3): colour of the quad between ring s, s + 1 and points n, n + 1."""
    S, N, _ = rings.shape
    v = rings.reshape(-1, 3)
    faces, cols = [], []
    for s in range(S - 1):
        for n in range(N):
            a, b = s * N + n, s * N + (n + 1) % N
            c, d = a + N, b + N
            faces += [(a, c, b), (b, c, d)]
            cols += [colors[s, n], colors[s, n]]
    centres = np.stack([rings[0].mean(0), rings[-1].mean(0)])
    base = len(v)
    v = np.concatenate([v, centres])
    for n in range(N):
        faces.append((base, n, (n + 1) % N))
        cols.append(colors[0, n])
        faces.append((base + 1, (S - 1) * N + (n + 1) % N, (S - 1) * N + n))
        cols.append(colors[-1, n])
    faces = np.asarray(faces, np.int64)
    a, b, c = (v[faces[:, k]] - v.mean(0) for k in range(3))
    if float((a * np.cross(b, c)).sum()) < 0:               # make the faces counter-clockwise seen from outside
        faces = faces[:, ::-1].copy()
    return Mesh(v, faces, np.asarray(cols, np.float64))


def _interp(x, xs, ys):
    return np.interp(x, xs, ys)


def sedan_mesh(dims: dict | None = None, stations: int = 49, ring: int = 40) -> Mesh:
    """A plain three-box sedan in the ego frame: a lofted body with a glass band, tail lamps and four wheels."""
    p = {**SEDAN, **(dims or {})}
    L, Wd, Hh = p["length"], p["width"], p["height"]
    x0 = -p["rear_overhang"]                    # rear bumper
    x1 = x0 + L                                 # front bumper
    ground = -p["axle_height"]
    top = ground + Hh
    floor = ground + 0.20                       # underside of the body
    u = np.linspace(0.0, 1.0, stations)         # 0 = rear bumper, 1 = front bumper
    xs = x0 + u * L
    # side profile (fractions of the length from the rear): boot lid, rear window, roof, windscreen, bonnet
    belt = ground + _interp(u, [0, 0.03, 0.20, 0.70, 0.97, 1.0], [0.90, 1.00, 1.00, 0.95, 0.74, 0.60]) * (Hh * 0.69)
    roof = ground + _interp(u, [0, 0.14, 0.30, 0.42, 0.56, 0.74, 1.0], [0.0, 0.0, 0.97, 1.0, 0.98, 0.0, 0.0]) * Hh
    roof = np.maximum(roof, belt)
    half = 0.5 * Wd * _interp(u, [0, 0.03, 0.12, 0.85, 0.96, 1.0], [0.86, 0.96, 1.0, 1.0, 0.90, 0.70])
    low = floor + _interp(u, [0, 0.05, 0.95, 1.0], [0.22, 0.0, 0.0, 0.18])
    cabin = roof > belt + 0.02
    # one ring, counter-clockwise seen from the front (+x): parameters around the section
    t = np.linspace(0.0, 2.0 * math.pi, ring, endpoint=False)
    rings = np.zeros((stations, ring, 3))
    is_glass = np.zeros((stations, ring), bool)
    for s in range(stations):
        w, zb, zl, zr = half[s], belt[s], low[s], roof[s]
        wr = w * 0.74                           # the cabin is narrower than the body
        # rounded box below the belt (superellipse), cabin above it
        c, sn = np.cos(t), np.sin(t)
        ex = 0.35
        y = w * np.sign(c) * np.abs(c) ** ex
        zc, zh = 0.5 * (zb + zl), 0.5 * (zb - zl)
        z = zc + zh * np.sign(sn) * np.abs(sn) ** ex
        if cabin[s]:
            up = sn > 0
            k = np.clip((np.abs(sn) ** ex - 0.80) / 0.20, 0.0, 1.0)        # 0 at the belt, 1 on the roof
            k = np.where(up, k, 0.0)
            y = np.where(up, y * (1.0 - k) + np.clip(y, -wr, wr) * k, y)
            z = np.where(up, z + k * (zr - zb), z)
            is_glass[s] = up & (k > 0.08) & (k < 0.85)
        rings[s, :, 0], rings[s, :, 1], rings[s, :, 2] = xs[s], y, z
    colors = np.tile(np.asarray(BODY), (stations - 1, ring, 1))
    for s in range(stations - 1):
        both = cabin[s] and cabin[s + 1]
        pillar = both and (abs(u[s] - 0.50) < 0.012)                         # the pillar between the doors
        edge = (cabin[s] != cabin[s + 1]) or (both and (roof[s] < top - 0.25 * (top - belt[s])))
        g = (is_glass[s] | is_glass[s + 1]) & (both or edge) & (not pillar)
        if both and roof[s] < belt[s] + 0.9 * (top - belt[s]):               # front and rear windows: also the top
            g = g | ((np.sin(t) > 0) & (np.abs(rings[s, :, 1]) < half[s] * 0.66) & (rings[s, :, 2] > belt[s] + 0.03))
        colors[s][g] = GLASS
    parts = [_loft(rings, colors)]
    # tail lamps: two thin plates just behind the rear face
    for side in (-1.0, 1.0):
        ya, yb = sorted((side * half[0] * 0.42, side * half[0] * 0.97))
        za, zb = belt[0] - 0.24, belt[0] - 0.08
        xr = x0 - 0.004
        quad = np.asarray([[xr, ya, za], [xr, yb, za], [xr, yb, zb], [xr, ya, zb]])
        parts.append(Mesh(quad, np.asarray([[0, 2, 1], [0, 3, 2]], np.int64), np.tile(np.asarray(LAMP), (2, 1))))
    # wheels: short cylinders along y
    r, ww = p["wheel_radius"], p["wheel_width"]
    a = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
    for ax in (0.0, p["wheelbase"]):
        for side in (-1.0, 1.0):
            yc = side * (0.5 * Wd - 0.5 * ww - 0.02)
            ys = np.asarray([yc - 0.5 * ww, yc + 0.5 * ww])
            wr_ = np.zeros((2, len(a), 3))
            for i in range(2):
                wr_[i, :, 0] = ax + r * np.cos(a)
                wr_[i, :, 1] = ys[i]
                wr_[i, :, 2] = r * np.sin(a)
            wheel = _loft(wr_, np.tile(np.asarray(TYRE), (1, len(a), 1)))
            parts.append(wheel)
    return merge_meshes(parts)


def merge_meshes(parts) -> Mesh:
    offset, v, f, c = 0, [], [], []
    for m in parts:
        v.append(m.vertices)
        f.append(m.faces + offset)
        c.append(m.colors)
        offset += len(m.vertices)
    return Mesh(np.concatenate(v), np.concatenate(f), np.concatenate(c))


# ---------------------------------------------------------------------------------------------- files
def load_obj(path: str, default_color=BODY) -> Mesh:
    """A Wavefront OBJ as a triangle mesh in the axes of the file. Polygons are fanned into triangles; the
    colour of a face is the diffuse colour (Kd) of its material when the MTL file is next to the OBJ."""
    materials: dict[str, tuple] = {}
    vertices, faces, colors = [], [], []
    color = tuple(default_color)
    folder = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif key == "f":
                idx = [int(tok.split("/")[0]) for tok in parts[1:]]
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
                    colors.append(color)
            elif key == "mtllib":
                mtl = os.path.join(folder, line.split(None, 1)[1].strip())
                if os.path.isfile(mtl):
                    materials.update(_load_mtl(mtl))
            elif key == "usemtl":
                color = materials.get(line.split(None, 1)[1].strip(), tuple(default_color))
    if not faces:
        raise ValueError(f"{path}: no faces")
    return Mesh(np.asarray(vertices, np.float64), np.asarray(faces, np.int64), np.asarray(colors, np.float64))


def _load_mtl(path: str) -> dict:
    out, name = {}, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "newmtl":
                name = line.split(None, 1)[1].strip()
            elif parts[0] == "Kd" and name is not None:
                out[name] = tuple(float(x) for x in parts[1:4])
    return out


_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def place_mesh(mesh: Mesh, forward: str = "+x", up: str = "+z", length: float | None = SEDAN["length"],
               rear_overhang: float = SEDAN["rear_overhang"], axle_height: float = SEDAN["axle_height"]) -> Mesh:
    """Put a mesh into the ego frame. ``forward`` and ``up`` name the axes of the file that point to the front
    and to the top of the car ("+x", "-z", ...). The mesh is scaled so that it is ``length`` metres long (None:
    not scaled), centred across, its lowest point put on the road (``axle_height`` below the origin) and its
    rear end ``rear_overhang`` behind the origin."""
    f, u = np.asarray(_AXES[forward], np.float64), np.asarray(_AXES[up], np.float64)
    if abs(float(f @ u)) > 1e-9:
        raise ValueError("forward and up must be different axes")
    left = np.cross(u, f)
    R = np.stack([f, left, u])                               # file axes -> ego axes
    v = mesh.vertices @ R.T
    faces = mesh.faces
    used = np.unique(faces)
    lo, hi = v[used].min(0), v[used].max(0)
    if length is not None:
        scale = length / (hi[0] - lo[0])
        v, lo, hi = v * scale, lo * scale, hi * scale
    v = v - np.asarray([lo[0] + rear_overhang, 0.5 * (lo[1] + hi[1]), lo[2] + axle_height])
    return Mesh(v, faces, mesh.colors)


# ---------------------------------------------------------------------------------------------- drawing
def _vertex_normals(v: np.ndarray, faces: np.ndarray, crease_deg: float = 40.0) -> np.ndarray:
    """Per-corner normals (F, 3, 3): the area-weighted mean of the faces around a vertex that are within
    ``crease_deg`` of the face (so that sharp edges stay sharp)."""
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    acc = np.zeros_like(v)
    for k in range(3):
        np.add.at(acc, faces[:, k], fn)
    unit = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    smooth = acc[faces]                                                    # (F, 3, 3)
    smooth = smooth / np.maximum(np.linalg.norm(smooth, axis=2, keepdims=True), 1e-12)
    keep = (smooth * unit[:, None, :]).sum(2, keepdims=True) > math.cos(math.radians(crease_deg))
    return np.where(keep, smooth, unit[:, None, :])


def render_mesh(mesh: Mesh, ego_T_cam: np.ndarray, K: np.ndarray, height: int, width: int, *, supersample: int = 3,
                light=(0.25, 0.35, 1.0), ambient: float = 0.45, shadow: float = 0.45,
                ground_z: float = -SEDAN["axle_height"]) -> np.ndarray:
    """The mesh seen from a camera: (H, W, 4) float32 RGBA, straight (not premultiplied) alpha.

    ``ego_T_cam`` (4, 4) camera -> ego, OpenCV camera axes; ``K`` (3, 3) with integer pixel centres (as
    ``chase_camera`` returns them). Faces are lit by one directional light (``light``, in the ego frame,
    pointing from the car to the light) plus ``ambient``. ``shadow`` > 0 adds a soft dark patch on the road
    under the car (its largest opacity); 0 leaves it out."""
    s = int(supersample)
    Hs, Ws = height * s, width * s
    Ks = np.asarray(K, np.float64).copy()
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = (Ks[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (Ks[1, 2] + 0.5) * s - 0.5
    cam_T_ego = np.linalg.inv(np.asarray(ego_T_cam, np.float64))
    v = mesh.vertices
    pc = v @ cam_T_ego[:3, :3].T + cam_T_ego[:3, 3]
    if (pc[np.unique(mesh.faces), 2] <= 0.05).any():
        raise ValueError("the mesh reaches behind the camera")
    uv = np.stack([Ks[0, 0] * pc[:, 0] / pc[:, 2] + Ks[0, 2], Ks[1, 1] * pc[:, 1] / pc[:, 2] + Ks[1, 2]], 1)
    lvec = np.asarray(light, np.float64)
    lvec = lvec / np.linalg.norm(lvec)
    normals = _vertex_normals(v, mesh.faces)
    eye = cam_T_ego[:3, :3].T @ (-cam_T_ego[:3, 3])                         # camera centre in the ego frame
    shade = ambient + (1.0 - ambient) * np.clip(normals @ lvec, 0.0, 1.0)   # (F, 3)
    half = lvec + (eye - v[mesh.faces]) / np.linalg.norm(eye - v[mesh.faces], axis=2, keepdims=True)
    half = half / np.linalg.norm(half, axis=2, keepdims=True)
    gloss = 0.25 * np.clip((normals * half).sum(2), 0.0, 1.0) ** 40

    zbuf = np.full((Hs, Ws), np.inf)
    rgb = np.zeros((Hs, Ws, 3))
    tri = uv[mesh.faces]                                                    # (F, 3, 2)
    area = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
            - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    iz = 1.0 / pc[mesh.faces, 2]                                            # (F, 3)
    lo = np.floor(tri.min(1)).astype(np.int64)
    hi = np.ceil(tri.max(1)).astype(np.int64)
    for f in np.nonzero(np.abs(area) > 1e-9)[0]:
        u0, v0 = max(lo[f, 0], 0), max(lo[f, 1], 0)
        u1, v1 = min(hi[f, 0], Ws - 1), min(hi[f, 1], Hs - 1)
        if u0 > u1 or v0 > v1:
            continue
        px, py = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
        (ax, ay), (bx, by), (cx, cy) = tri[f]
        w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) / area[f]
        w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) / area[f]
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        q = w0 * iz[f, 0] + w1 * iz[f, 1] + w2 * iz[f, 2]                   # interpolated 1 / z
        z = 1.0 / np.maximum(q, 1e-12)
        view = zbuf[v0:v1 + 1, u0:u1 + 1]
        closer = inside & (z < view)
        if not closer.any():
            continue
        # perspective-correct shading
        b0, b1, b2 = w0 * iz[f, 0] * z, w1 * iz[f, 1] * z, w2 * iz[f, 2] * z
        lit = b0 * shade[f, 0] + b1 * shade[f, 1] + b2 * shade[f, 2]
        spec = b0 * gloss[f, 0] + b1 * gloss[f, 1] + b2 * gloss[f, 2]
        colour = np.clip(mesh.colors[f][None, None, :] * lit[..., None] + spec[..., None], 0.0, 1.0)
        view[closer] = z[closer]
        rgb[v0:v1 + 1, u0:u1 + 1][closer] = colour[closer]
    alpha = np.isfinite(zbuf).astype(np.float64)

    if shadow > 0:                                                          # soft patch on the road under the car
        used = v[np.unique(mesh.faces)]
        x0, x1, y0, y1 = used[:, 0].min(), used[:, 0].max(), used[:, 1].min(), used[:, 1].max()
        ys_, xs_ = np.mgrid[0:Hs, 0:Ws]
        ray = np.stack([(xs_ - Ks[0, 2]) / Ks[0, 0], (ys_ - Ks[1, 2]) / Ks[1, 1], np.ones_like(xs_, float)], -1)
        ray = ray @ cam_T_ego[:3, :3]                                       # directions in the ego frame
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ground_z - eye[2]) / ray[..., 2]
        gx, gy = eye[0] + t * ray[..., 0], eye[1] + t * ray[..., 1]
        soft = 0.25
        dx = np.maximum(np.maximum(x0 - gx, gx - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - gy, gy - y1), 0.0)
        sh = shadow * np.clip(1.0 - np.hypot(dx, dy) / soft, 0.0, 1.0) ** 2
        sh = np.where((t > 0) & np.isfinite(t), sh, 0.0)
        out_a = alpha + (1.0 - alpha) * sh                                  # car over shadow (black)
        rgb = rgb * alpha[..., None]                                        # premultiplied; the shadow adds no colour
        alpha = out_a
    else:
        rgb = rgb * alpha[..., None]
    # box filter down to the output size, then back to straight alpha
    rgb = rgb.reshape(height, s, width, s, 3).mean((1, 3))
    alpha = alpha.reshape(height, s, width, s).mean((1, 3))
    rgb = np.where(alpha[..., None] > 1e-6, rgb / np.maximum(alpha[..., None], 1e-6), 0.0)
    return np.concatenate([rgb, alpha[..., None]], -1).astype(np.float32)


def composite(image: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Lay an RGBA overlay (H, W, 4), float 0..1 or uint8, straight alpha, over an image (H, W, 3) of the same
    kind and channel order. Returns the type of ``image``."""
    if image.shape[:2] != overlay.shape[:2]:
        raise ValueError(f"image {image.shape[:2]} and overlay {overlay.shape[:2]} differ in size")
    is_u8 = image.dtype == np.uint8
    img = image.astype(np.float32) / 255.0 if is_u8 else image.astype(np.float32)
    ov = overlay.astype(np.float32) / 255.0 if overlay.dtype == np.uint8 else overlay.astype(np.float32)
    a = ov[..., 3:4]
    out = ov[..., :3] * a + img * (1.0 - a)
    return (out * 255.0).round().clip(0, 255).astype(np.uint8) if is_u8 else out


def save_overlay(path: str, overlay: np.ndarray) -> None:
    """Write an RGBA overlay (float 0..1, RGB order) as an 8-bit PNG."""
    import cv2

    u8 = (overlay.clip(0, 1) * 255.0).round().astype(np.uint8)
    cv2.imwrite(path, np.ascontiguousarray(u8[..., [2, 1, 0, 3]]))


def load_overlay(path: str) -> np.ndarray:
    """Read an RGBA PNG written by ``save_overlay``: (H, W, 4) uint8, RGB order."""
    import cv2

    x = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if x is None or x.ndim != 3 or x.shape[2] != 4:
        raise ValueError(f"{path}: not an RGBA image")
    return np.ascontiguousarray(x[..., [2, 1, 0, 3]])
