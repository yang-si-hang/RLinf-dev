"""Plot sampled RLT latent vectors for selected fail and success episodes."""

import colorsys
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path("/root/autodl-tmp/workspace/RLinf-dev")
RESULTS = ROOT / "results"
STEPS = (9000, 21000)
# Edit these lists to choose the original episode_index values in each dataset.
FAIL_EPISODE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]
SUCCESS_EPISODE_INDICES = [0, 1, 2, 3, 4, 5]
FRAME_STRIDE = 6  # 30 Hz -> 5 Hz (one frame in six)
PYTHON_PARAMS = {
    "pca_components": 50,
    "perplexity": 30,
    "learning_rate": "auto",
    "max_iter": 1000,
    "method": "barnes_hut",
    "random_state": 42,
}


def get_episode_frames(label):
    """Return episode and frame indices in the same order as the saved vectors."""
    data = ROOT / "data/lerobot_data" / f"plug_deploy_v1_merge_{label}_vid" / "data"
    paths = sorted(data.glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {data}")
    tables = [pq.read_table(p, columns=["index", "episode_index", "frame_index"])
              for p in paths]
    indices = np.concatenate([t["index"].to_numpy() for t in tables])
    episodes = np.concatenate([t["episode_index"].to_numpy() for t in tables])
    frames = np.concatenate([t["frame_index"].to_numpy() for t in tables])
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError(f"{label}: parquet order does not match vector order")
    for episode in np.unique(episodes):
        positions = np.flatnonzero(episodes == episode)
        if not np.array_equal(frames[positions], np.arange(len(positions))):
            raise ValueError(f"{label}: unexpected frame order for episode {episode}")
    return episodes, frames


def select_frames(vectors, episodes, frames, selected, label):
    """Select requested episodes and every sixth frame within each episode."""
    if len(vectors) != len(episodes):
        raise ValueError(f"{label}: vector and parquet frame counts differ")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{label}: duplicate episode indices")
    missing = set(selected) - set(np.unique(episodes))
    if missing:
        raise ValueError(f"{label}: episode indices not found: {sorted(missing)}")
    mask = np.isin(episodes, selected) & (frames % FRAME_STRIDE == 0)
    return vectors[mask], episodes[mask], frames[mask]


def episode_color(index):
    """Spread adjacent legend colors around the hue wheel."""
    hue = (index * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.78, 0.88)


fail_episode, fail_frame = get_episode_frames("fail")
success_episode, success_frame = get_episode_frames("success")
if not FAIL_EPISODE_INDICES and not SUCCESS_EPISODE_INDICES:
    raise ValueError("Select at least one fail or success episode")
selected_keys = [("fail", i) for i in FAIL_EPISODE_INDICES] + [
    ("success", i) for i in SUCCESS_EPISODE_INDICES
]
out_dir = RESULTS / "rlt_z_tsne_selected_episodes"
out_dir.mkdir(parents=True, exist_ok=True)

for step in STEPS:
    fail_dir = RESULTS / f"rlt_z_fail_step{step}"
    success_dir = RESULTS / f"rlt_z_success_step{step}"
    fail_meta = json.loads((fail_dir / "metadata.json").read_text())
    success_meta = json.loads((success_dir / "metadata.json").read_text())
    assert fail_meta["checkpoint_step"] == success_meta["checkpoint_step"] == step
    assert fail_meta["norm_stats"] == success_meta["norm_stats"]
    fail = np.load(fail_dir / "z_rl.npy", allow_pickle=False)
    success = np.load(success_dir / "z_rl.npy", allow_pickle=False)
    fail, fail_ids, fail_frames = select_frames(
        fail, fail_episode, fail_frame, FAIL_EPISODE_INDICES, "fail"
    )
    success, success_ids, success_frames = select_frames(
        success, success_episode, success_frame, SUCCESS_EPISODE_INDICES, "success"
    )
    vectors = np.concatenate([fail, success])
    episode_ids = np.concatenate([fail_ids, success_ids])
    frame_ids = np.concatenate([fail_frames, success_frames])
    groups = np.concatenate([
        np.zeros(len(fail), dtype=np.uint8),
        np.ones(len(success), dtype=np.uint8),
    ])
    if len(vectors) < 3:
        raise ValueError("At least three sampled frames are needed for t-SNE")
    n_components = min(PYTHON_PARAMS["pca_components"], len(vectors) - 1, vectors.shape[1])
    reduced = PCA(n_components=n_components, svd_solver="randomized",
                  random_state=42).fit_transform(vectors)
    perplexity = min(PYTHON_PARAMS["perplexity"], len(vectors) - 1)
    print(f"step {step}: PCA({n_components}) completed; fitting t-SNE on {len(vectors)} frames", flush=True)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", max_iter=1000, method="barnes_hut",
                random_state=42, n_jobs=8, verbose=1)
    coords = tsne.fit_transform(reduced)
    step_dir = out_dir / f"step{step}"
    step_dir.mkdir(exist_ok=True)
    np.save(step_dir / "tsne_coordinates.npy", coords.astype(np.float32))
    np.save(step_dir / "episode_ids.npy", episode_ids.astype(np.int32))
    np.save(step_dir / "frame_ids.npy", frame_ids.astype(np.int32))
    np.save(step_dir / "outcome_ids.npy", groups)
    metadata = {
        "checkpoint_step": step,
        "fail_frames": len(fail),
        "success_frames": len(success),
        "fail_episodes": FAIL_EPISODE_INDICES,
        "success_episodes": SUCCESS_EPISODE_INDICES,
        "frame_stride": FRAME_STRIDE,
        "source_fps": 30,
        "sampled_fps": 30 / FRAME_STRIDE,
        "outcome_id_mapping": {"0": "fail", "1": "success"},
        "params": {**PYTHON_PARAMS, "pca_components": n_components, "perplexity": perplexity},
        "final_kl_divergence": float(tsne.kl_divergence_),
    }
    (step_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(12, 8))
    episode_handles = []
    for color_index, (outcome, episode) in enumerate(selected_keys):
        outcome_id = 0 if outcome == "fail" else 1
        positions = np.flatnonzero((groups == outcome_id) & (episode_ids == episode))
        if not len(positions):
            continue
        positions = positions[np.argsort(frame_ids[positions])]
        rgb = episode_color(color_index)
        rgba = np.tile((*rgb, 1.0), (len(positions), 1))
        rgba[:, 3] = np.linspace(0.18, 0.95, len(positions))
        marker = "^" if outcome == "fail" else "o"
        ax.scatter(coords[positions, 0], coords[positions, 1], c=rgba,
                   marker=marker, s=17 if outcome == "fail" else 11,
                   linewidths=0, rasterized=True)
        episode_handles.append(Line2D([], [], marker="s", linestyle="", color=rgb,
                                      markersize=7, label=f"{outcome[0].upper()}{episode:02d}"))
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"RLT Stage 1 z · selected episodes · step {step} · 5 Hz")
    ax.grid(alpha=0.15)
    shape_handles = [
        Line2D([], [], marker="^", linestyle="", color="black", markersize=8, label="fail"),
        Line2D([], [], marker="o", linestyle="", color="black", markersize=8, label="success"),
    ]
    shape_legend = ax.legend(handles=shape_handles, title="Outcome / marker",
                             loc="upper left", frameon=True)
    ax.add_artist(shape_legend)
    fig.legend(handles=episode_handles, title="Episode / color",
               loc="center right", bbox_to_anchor=(0.99, 0.5), ncol=2,
               fontsize=8, title_fontsize=9, columnspacing=0.5, handletextpad=0.2)
    fig.subplots_adjust(right=0.75)
    fig.savefig(step_dir / "tsne_scatter.svg", dpi=200)
    fig.savefig(step_dir / "tsne_scatter.png", dpi=150)
    plt.close(fig)
    print(f"step {step}: t-SNE saved; KL={tsne.kl_divergence_:.4f}", flush=True)

cell_w, cell_h = 1200, 820
gallery = Image.new("RGB", (cell_w, cell_h * len(STEPS)), "white")
draw = ImageDraw.Draw(gallery)
for i, step in enumerate(STEPS):
    with Image.open(out_dir / f"step{step}/tsne_scatter.png") as item:
        item.thumbnail((cell_w - 10, cell_h - 25), Image.Resampling.LANCZOS)
        gallery.paste(item, ((cell_w - item.width) // 2, i * cell_h + 22))
    draw.text((12, i * cell_h + 5), f"step {step}", fill="black")
gallery.save(out_dir / "tsne_overview.png", optimize=True)
print("gallery saved", flush=True)
