"""Renderer-backed queries Q1 (rendered view) and Q4 (expected info gain).

Both run on CUDA via the diff_gaussian_rasterization that the gs_planning conda
env already ships. The Inria repo on disk supplies the GaussianModel loader
and a MiniCam class which we lean on directly instead of re-implementing.

A pre-built GS model directory (Inria layout)::

    model_dir/
      point_cloud/iteration_<N>/point_cloud.ply
      cameras.json   (intrinsics/extrinsics dump from training)
      cfg_args       (training args; gives us active_sh_degree etc.)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

GS_REPO = Path("/home/xiaoming/Desktop/2026_sonar_GS/gaussian-splatting")
if str(GS_REPO) not in sys.path:
    sys.path.insert(0, str(GS_REPO))


def _to_torch_view_proj(world_from_cam_opencv: np.ndarray, fovx_rad: float, fovy_rad: float,
                        znear: float = 0.01, zfar: float = 100.0):
    """Convert an OpenCV world-from-camera 4x4 to Inria's transposed
    world_view_transform and full_proj_transform tensors.

    Inria reads R as the *camera-from-world rotation as columns* (see
    ``dataset_readers.readCamerasFromTransforms``), and getWorld2View2
    composes R, T into a column-major matrix that gets transposed before being
    pushed to GPU. We reproduce that path here to avoid drift.
    """
    import torch
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # type: ignore

    # Our input is already OpenCV world-from-camera (== c2w_CV).
    # The Inria reader's "c2w[:3, 1:3] *= -1" is specifically the
    # OpenGL->OpenCV conversion; do NOT apply it here.
    c2w = world_from_cam_opencv
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3, :3])
    T = w2c[:3, 3]

    wvt = torch.tensor(getWorld2View2(R, T, np.zeros(3), 1.0)).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx_rad, fovY=fovy_rad).transpose(0, 1).cuda()
    full = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    return wvt, full


class GSRenderer:
    """Wraps the Inria render() entry point so callers can render a synthetic
    view from any OpenCV pose and obtain (rgb, depth, alpha)."""

    def __init__(self, model_dir: str | Path, iteration: int | None = None):
        import torch
        from scene.gaussian_model import GaussianModel  # type: ignore
        self.model_dir = Path(model_dir)

        # Discover trained sh_degree from cfg_args
        sh_degree = 3
        cfg_args = self.model_dir / "cfg_args"
        if cfg_args.exists():
            txt = cfg_args.read_text()
            # crude parse: look for sh_degree=N
            for tok in txt.replace("(", " ").replace(")", " ").replace(",", " ").split():
                if tok.startswith("sh_degree="):
                    sh_degree = int(tok.split("=")[1])

        # Pick the latest iteration if not specified
        pc_dir = self.model_dir / "point_cloud"
        iters = sorted(int(p.name.split("_")[-1]) for p in pc_dir.iterdir() if p.is_dir())
        chosen = iteration if iteration is not None else iters[-1]
        ply = pc_dir / f"iteration_{chosen}" / "point_cloud.ply"

        self.gs = GaussianModel(sh_degree)
        self.gs.load_ply(str(ply))
        self.bg = torch.zeros(3, device="cuda")

        # Faux pipeline config -- the Inria render() inspects three booleans.
        class _Pipe:
            convert_SHs_python = False
            compute_cov3D_python = False
            debug = False
            antialiasing = False
        self.pipe = _Pipe()

    def render(self, world_from_cam_opencv: np.ndarray, width: int, height: int,
               hfov_deg: float):
        """Return (rgb [3,H,W] float in [0,1], depth [H,W] float meters)."""
        import torch
        from gaussian_renderer import render  # type: ignore

        fovx = math.radians(hfov_deg)
        # NeRF-Synthetic intrinsics: square pixels, fovy derived from aspect.
        fovy = 2.0 * math.atan(math.tan(fovx / 2.0) * height / width)
        wvt, full = _to_torch_view_proj(world_from_cam_opencv, fovx, fovy)

        # Inria's render() wants a viewpoint_camera with these attrs.
        class _Cam:
            pass
        cam = _Cam()
        cam.image_width = width
        cam.image_height = height
        cam.FoVx = fovx
        cam.FoVy = fovy
        cam.world_view_transform = wvt
        cam.full_proj_transform = full
        cam.camera_center = torch.tensor(world_from_cam_opencv[:3, 3], dtype=torch.float32, device="cuda")

        with torch.no_grad():
            out = render(cam, self.gs, self.pipe, self.bg)
        rgb = out["render"]
        depth = out.get("depth", None)
        if depth is None:
            # Some diff-gauss-raster builds don't emit depth; fall back to NaN.
            depth = torch.full((height, width), float("nan"), device="cuda")
        elif depth.dim() == 3:
            depth = depth[0]
        return rgb, depth


# ---------- Q1 ----------

def q1_render(renderer: GSRenderer, pose, width=640, height=480, hfov_deg=90.0):
    """Q1: rendered RGB from a candidate pose."""
    rgb, _ = renderer.render(np.asarray(pose), width, height, hfov_deg)
    return rgb  # CHW torch tensor on cuda


# ---------- Q4 ----------

def _rgb_entropy(rgb_chw, bins: int = 32) -> float:
    """Shannon entropy (nats) of the rendered RGB histogram. Higher == more
    photometric content (sharper / textured), so we use it as a 'photometric
    novelty' proxy: a candidate view that renders to a flat color is poorly
    constrained or boring.
    """
    import torch
    x = rgb_chw.clamp(0, 1).flatten()
    # Histogram per channel summed.
    H_total = 0.0
    for c in range(3):
        ch = rgb_chw[c].clamp(0, 1).flatten()
        h = torch.histc(ch, bins=bins, min=0.0, max=1.0)
        p = h / (h.sum() + 1e-9)
        H_total += -(p * (p + 1e-12).log()).sum().item()
    return H_total / 3.0


def q4_information_gain(renderer: GSRenderer, pose, under_rec_grid,
                        grid_min, cell: float, lam: float = 0.5,
                        width=640, height=480, hfov_deg=90.0,
                        depth_max: float = 6.0) -> float:
    """Q4: H(rendered) + lam * count(under-reconstructed voxels visible).

    Parameters
    ----------
    under_rec_grid : (Nx, Ny, Nz) bool, output of
        ``gs_queries.under_reconstructed_cells``.
    grid_min       : (3,) world-frame anchor of voxel [0,0,0].
    cell           : voxel size in meters.
    """
    rgb, depth = renderer.render(np.asarray(pose), width, height, hfov_deg)
    H = _rgb_entropy(rgb)

    # Cast each pixel ray to the rendered depth; count voxels along the ray
    # that fall in flagged cells. Cap depth to avoid runaway rays in empty
    # regions (where rendered depth -> very large).
    import torch
    d = depth.detach().cpu().numpy()
    finite = np.isfinite(d) & (d > 0) & (d < depth_max)
    if not finite.any():
        return H

    # Subsample to keep it cheap: ~4096 rays per query.
    H_img, W_img = d.shape
    ys, xs = np.where(finite)
    if ys.size > 4096:
        idx = np.random.default_rng(0).choice(ys.size, 4096, replace=False)
        ys, xs = ys[idx], xs[idx]

    fovx = math.radians(hfov_deg)
    fovy = 2.0 * math.atan(math.tan(fovx / 2.0) * H_img / W_img)
    fx = 0.5 * W_img / math.tan(fovx / 2.0)
    fy = 0.5 * H_img / math.tan(fovy / 2.0)
    cx_, cy_ = 0.5 * W_img, 0.5 * H_img

    cam_eye = pose[:3, 3]
    cam_R = pose[:3, :3]  # cam basis as columns in world
    n_hits = 0
    grid_min = np.asarray(grid_min, dtype=np.float32)
    dims = np.array(under_rec_grid.shape)

    # Step the ray in voxel-sized increments
    for y_pix, x_pix in zip(ys, xs):
        z_d = d[y_pix, x_pix]
        ray_cam = np.array([(x_pix - cx_) / fx, (y_pix - cy_) / fy, 1.0])
        ray_world = cam_R @ ray_cam
        ray_world = ray_world / (np.linalg.norm(ray_world) + 1e-9)
        n_steps = max(2, int(z_d / cell))
        for s in range(1, n_steps + 1):
            p = cam_eye + ray_world * (s * cell)
            idx_v = ((p - grid_min) / cell).astype(int)
            if np.all(idx_v >= 0) and np.all(idx_v < dims):
                if under_rec_grid[idx_v[0], idx_v[1], idx_v[2]]:
                    n_hits += 1
                    break  # only count once per ray
    return H + lam * (n_hits / max(1, ys.size))
