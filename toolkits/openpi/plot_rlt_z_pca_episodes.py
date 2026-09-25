# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Plot PCA of sampled Z vectors from selected fail and success episodes."""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
# Set the original episode_index values to plot from each dataset.
FAIL_EPISODE_INDICES = list(range(8))
SUCCESS_EPISODE_INDICES = list(range(15))
FRAME_STRIDE = 6  # 30 Hz -> 5 Hz (one frame in six)


def load_episode_frames(dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return episode and frame IDs in the order of the exported vectors."""
    files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data under {dataset_root}")
    tables = [
        pq.read_table(file, columns=["index", "episode_index", "frame_index"])
        for file in files
    ]
    indices = np.concatenate([table["index"].to_numpy() for table in tables])
    episodes = np.concatenate([table["episode_index"].to_numpy() for table in tables])
    frames = np.concatenate([table["frame_index"].to_numpy() for table in tables])
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError(f"Dataset rows are not in exported order: {dataset_root}")
    for episode in np.unique(episodes):
        positions = np.flatnonzero(episodes == episode)
        if not np.array_equal(frames[positions], np.arange(len(positions))):
            raise ValueError(f"Unexpected frame order in episode {episode}")
    return episodes, frames


def select_frames(
    vectors: np.ndarray,
    episodes: np.ndarray,
    frames: np.ndarray,
    selected: list[int],
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep selected episodes and every sixth frame of each episode."""
    if len(vectors) != len(episodes):
        raise ValueError(f"{label}: vector and parquet frame counts differ")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{label}: duplicate episode indices")
    missing = set(selected) - set(np.unique(episodes))
    if missing:
        raise ValueError(f"{label}: episode indices not found: {sorted(missing)}")
    mask = np.isin(episodes, selected) & (frames % FRAME_STRIDE == 0)
    return vectors[mask], episodes[mask], frames[mask]


def episode_color(index: int) -> tuple[float, float, float]:
    """Spread nearby legend entries around the hue wheel."""
    hue = (index * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.78, 0.88)


def plot_step(
    ax: plt.Axes,
    coordinates: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    outcome_ids: np.ndarray,
    selected_keys: list[tuple[str, int]],
    ratios: np.ndarray,
    step: int,
) -> list[Line2D]:
    """Draw sampled frames, with opacity increasing over episode time."""
    handles = []
    for color_index, (outcome, episode) in enumerate(selected_keys):
        outcome_id = 0 if outcome == "fail" else 1
        positions = np.flatnonzero(
            (outcome_ids == outcome_id) & (episode_ids == episode)
        )
        if not len(positions):
            continue
        positions = positions[np.argsort(frame_ids[positions])]
        rgb = episode_color(color_index)
        rgba = np.tile((*rgb, 1.0), (len(positions), 1))
        rgba[:, 3] = np.linspace(0.18, 0.95, len(positions))
        ax.scatter(
            coordinates[positions, 0],
            coordinates[positions, 1],
            c=rgba,
            marker="^" if outcome == "fail" else "o",
            s=17 if outcome == "fail" else 11,
            linewidths=0,
            rasterized=True,
        )
        handles.append(
            Line2D([], [], marker="s", linestyle="", color=rgb, markersize=7,
                   label=f"{outcome[0].upper()}{episode:02d}")
        )
    ax.set_xlabel(f"PC1 ({ratios[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({ratios[1]:.2%} variance)")
    ax.set_title(f"RLT Stage 1 z · selected episodes · step {step} · 5 Hz")
    ax.grid(alpha=0.15)
    return handles


def add_legends(fig: plt.Figure, ax: plt.Axes, episode_handles: list[Line2D]) -> None:
    """Add legends for outcome shape and episode color."""
    shape_handles = [
        Line2D([], [], marker="^", linestyle="", color="black", label="fail"),
        Line2D([], [], marker="o", linestyle="", color="black", label="success"),
    ]
    ax.add_artist(
        ax.legend(handles=shape_handles, title="Outcome / marker", loc="upper left")
    )
    fig.legend(
        handles=episode_handles,
        title="Episode / color",
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        ncol=2,
        fontsize=8,
        title_fontsize=9,
        columnspacing=0.5,
        handletextpad=0.2,
    )


def main() -> None:
    """Fit and plot PCA for each requested checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--steps", type=int, nargs="+", default=[15000, 18000, 21000])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if not FAIL_EPISODE_INDICES and not SUCCESS_EPISODE_INDICES:
        raise ValueError("Select at least one fail or success episode")
    repo_root = args.repo_root.expanduser().resolve()
    results = repo_root / "results"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results / "rlt_z_pca_selected_episodes"
    )
    fail_episodes, fail_frames = load_episode_frames(
        repo_root / "data/lerobot_data/plug_deploy_v1_merge_fail_vid"
    )
    success_episodes, success_frames = load_episode_frames(
        repo_root / "data/lerobot_data/plug_deploy_v1_merge_success_vid"
    )
    selected_keys = [("fail", i) for i in FAIL_EPISODE_INDICES] + [
        ("success", i) for i in SUCCESS_EPISODE_INDICES
    ]
    plt.rcParams["svg.fonttype"] = "none"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for step in args.steps:
        fail_dir = results / f"rlt_z_fail_step{step}"
        success_dir = results / f"rlt_z_success_step{step}"
        fail_meta = json.loads((fail_dir / "metadata.json").read_text())
        success_meta = json.loads((success_dir / "metadata.json").read_text())
        if fail_meta["checkpoint_step"] != step or success_meta["checkpoint_step"] != step:
            raise ValueError(f"Checkpoint step mismatch: {step}")
        if fail_meta["norm_stats"] != success_meta["norm_stats"]:
            raise ValueError(f"Normalization statistics differ: {step}")
        fail, fail_ids, fail_frame_ids = select_frames(
            np.load(fail_dir / "z_rl.npy", allow_pickle=False),
            fail_episodes, fail_frames, FAIL_EPISODE_INDICES, "fail"
        )
        success, success_ids, success_frame_ids = select_frames(
            np.load(success_dir / "z_rl.npy", allow_pickle=False),
            success_episodes, success_frames, SUCCESS_EPISODE_INDICES, "success"
        )
        vectors = np.concatenate([fail, success])
        if len(vectors) < 2:
            raise ValueError("At least two sampled frames are needed for PCA")
        # Retain the fitted model for the explained variance shown on the axes.
        pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
        coordinates = pca.fit_transform(vectors)
        episode_ids = np.concatenate([fail_ids, success_ids])
        frame_ids = np.concatenate([fail_frame_ids, success_frame_ids])
        outcome_ids = np.concatenate([
            np.zeros(len(fail), dtype=np.uint8),
            np.ones(len(success), dtype=np.uint8),
        ])
        fig, ax = plt.subplots(figsize=(12, 8))
        handles = plot_step(ax, coordinates, episode_ids, frame_ids, outcome_ids,
                            selected_keys, pca.explained_variance_ratio_, step)
        add_legends(fig, ax, handles)
        fig.subplots_adjust(right=0.75)
        svg_path = output_dir / f"step{step}.svg"
        png_path = output_dir / f"step{step}.png"
        fig.savefig(svg_path)
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        np.save(output_dir / f"step{step}_coordinates.npy", coordinates.astype(np.float32))
        np.save(output_dir / f"step{step}_episode_ids.npy", episode_ids.astype(np.int32))
        np.save(output_dir / f"step{step}_frame_ids.npy", frame_ids.astype(np.int32))
        np.save(output_dir / f"step{step}_outcome_ids.npy", outcome_ids)
        summary.append({
            "step": step,
            "svg": str(svg_path),
            "png": str(png_path),
            "fail_episodes": FAIL_EPISODE_INDICES,
            "success_episodes": SUCCESS_EPISODE_INDICES,
            "fail_frames": len(fail),
            "success_frames": len(success),
            "frame_stride": FRAME_STRIDE,
            "source_fps": 30,
            "sampled_fps": 30 / FRAME_STRIDE,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        })
        print(svg_path)

    (output_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
