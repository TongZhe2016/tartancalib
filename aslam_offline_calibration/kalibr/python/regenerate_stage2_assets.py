#!/usr/bin/env python3
import argparse
import importlib.machinery
import importlib.util
import os
import pickle
import re

import numpy as np
import pylab as pl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

import kalibr_common as kc
import kalibr_camera_calibration as kcc


def _load_tartan_calibrate_module(script_path):
    loader = importlib.machinery.SourceFileLoader("tartan_calibrate_module", script_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _slugify(title):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title.strip().lower()).strip("_")
    return slug or "figure"


def _scatter_with_optional_lognorm(ax, uv, values, img_w, img_h, title, colorbar_label):
    vals = np.asarray(values, dtype=float).reshape(-1)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    finite = np.isfinite(vals) & np.isfinite(uv).all(axis=1)
    uv_f = uv[finite]
    vals_f = vals[finite]

    if len(vals_f) == 0:
        ax.text(0.5, 0.5, "No finite samples", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
    else:
        pos = vals_f[vals_f > 0]
        kwargs = dict(s=1.0, alpha=0.45, cmap="plasma")
        if len(pos) > 0:
            vmin = max(np.nanpercentile(pos, 1), 1e-3)
            vmax = np.nanpercentile(pos, 99)
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = max(vmin * 10.0, vmin + 1e-6)
            vals_plot = vals_f.copy()
            vals_plot[vals_plot <= 0] = vmin
            sc = ax.scatter(
                uv_f[:, 0], uv_f[:, 1], c=vals_plot,
                norm=LogNorm(vmin=vmin, vmax=vmax), **kwargs)
        else:
            sc = ax.scatter(uv_f[:, 0], uv_f[:, 1], c=vals_f, **kwargs)
        pl.colorbar(sc, ax=ax, label=colorbar_label)
        ax.set_title(title)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_aspect("equal")


def _load_valid_point_stats(log_pkl, cam_id):
    with open(log_pkl, "rb") as f:
        logs = pickle.load(f)

    if not logs:
        raise RuntimeError("No logger entries found in log pickle: {0}".format(log_pkl))

    final_logger = logs[-1]
    stats = [
        s for s in final_logger.stats_
        if getattr(s, "camid", None) == cam_id and bool(getattr(s, "valid", True))
    ]
    if not stats:
        raise RuntimeError("No valid point statistics for cam{0} in {1}".format(cam_id, log_pkl))

    uv = np.array([np.asarray(s.y, dtype=float) for s in stats], dtype=float)
    err = np.array([np.linalg.norm(np.asarray(s.e, dtype=float)) for s in stats], dtype=float)
    return uv, err


def _write_report_supplement(save_dir, report_base, cam_id, img_w, img_h, uv, err):
    title_reproj = "cam{0}: uv reprojection error map".format(cam_id)
    title_hessian = "cam{0}: uv hessian strength map".format(cam_id)
    png_reproj = os.path.join(save_dir, "{0}__{1}.png".format(report_base, _slugify(title_reproj)))
    png_hessian = os.path.join(save_dir, "{0}__{1}.png".format(report_base, _slugify(title_hessian)))
    pdf_path = os.path.join(save_dir, "{0}-supplement.pdf".format(report_base))

    fig1, ax1 = pl.subplots(figsize=(10, 6))
    _scatter_with_optional_lognorm(
        ax1, uv, err, img_w, img_h, title_reproj, "|reprojection error| [px]")
    fig1.tight_layout()
    fig1.savefig(png_reproj, dpi=150)

    fig2, ax2 = pl.subplots(figsize=(10, 6))
    _scatter_with_optional_lognorm(
        ax2, uv, err, img_w, img_h,
        title_hessian + " (fallback to reprojection map)",
        "|reprojection error| [px]")
    fig2.tight_layout()
    fig2.savefig(png_hessian, dpi=150)

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)

    pl.close(fig1)
    pl.close(fig2)
    return png_reproj, png_hessian, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate missing stage diagnostics and report supplement for a successful second-stage run.")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--final-camchain-yaml", required=True)
    parser.add_argument("--init-camchain-yaml", default=None)
    parser.add_argument("--log-pkl", default=None)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--bag-from-to", type=float, nargs=2, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    tc = _load_tartan_calibrate_module(
        os.path.join(os.path.dirname(__file__), "tartan_calibrate"))

    target_config = kc.CalibrationTargetParameters(args.target)
    dataset = tc.initBagDataset(args.bag, args.topic, args.bag_from_to)
    extraction_dir = os.path.join(args.save_dir, "calibration_corners_regen")
    cam = kcc.CameraGeometry(
        tc.cameraModels[args.model], target_config, dataset, verbose=False,
        cam_id=args.cam_id, extraction_dir=extraction_dir)

    observations = kc.extractCornersFromDataset(
        cam.dataset, cam.ctarget.detector, multithreading=True,
        clearImages=False, noTransformation=True)
    if len(observations) == 0:
        raise RuntimeError("No observations extracted from {0}".format(args.bag))

    img_w = int(observations[0].imCols())
    img_h = int(observations[0].imRows())

    init_params = tc._load_initial_camera_params(args.init_camchain_yaml) if args.init_camchain_yaml else {}
    final_params = tc._load_initial_camera_params(args.final_camchain_yaml)

    if args.init_camchain_yaml:
        tc._apply_initial_camera_geometry(cam, args.topic, args.model, init_params[args.topic])
        kcc.collectStageDiagnostics(cam, observations, mode_label="loaded_init")

    tc._apply_initial_camera_geometry(cam, args.topic, args.model, final_params[args.topic])
    kcc.collectStageDiagnostics(cam, observations, mode_label="final")
    tc._save_corner_diagnostics(
        cam, args.cam_id, args.topic, observations, img_w, img_h, args.save_dir)

    if args.log_pkl:
        uv, err = _load_valid_point_stats(args.log_pkl, args.cam_id)
        report_base = "log1-report-cam"
        png_reproj, png_hessian, pdf_path = _write_report_supplement(
            args.save_dir, report_base, args.cam_id, img_w, img_h, uv, err)
        print("Report supplement saved to: {0}".format(pdf_path))
        print("UV reprojection map saved to: {0}".format(png_reproj))
        print("UV hessian fallback map saved to: {0}".format(png_hessian))


if __name__ == "__main__":
    main()
