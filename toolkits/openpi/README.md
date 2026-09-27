# OpenPI and RLT tools

These scripts are offline tools. Training entrypoints and Hydra configs live under `examples/sft/`; standalone environment evaluations live under `evaluations/` or `toolkits/standalone_eval_scripts/`.

## Export Stage 1 RLT vectors

Use the Docker OpenPI environment. If its OpenPI package does not provide the UR10e transforms, run `bash docker/setup_local_openpi.sh` once to install the local source into that same environment.

```bash
source switch_env openpi
python toolkits/openpi/export_rlt_z.py \
  --config-name ur10e_rlt_stage1_sft_openpi_pi05 \
  --dataset-root /app/data/lerobot_data/plug_v2_merge_crop_vid \
  --checkpoint /app/checkpoints/ur10e_rlt_stage1_sft_openpi_pi05/ur10e_plug_rlt_stage1_20260922_181649/global_step_21000 \
  --norm-stats /app/data/openpi_rlinf_checkpoints/pi05_ur10e_plug_lora_ki/plug_20260920_223703/20000/assets/plug_v2_merge_crop_vid/norm_stats.json \
  --output-dir /app/results/ur10e_rlt_stage1_z_step21000 \
  --batch-size 1
```

For a small check, add `--limit 2` and use a different output directory. The command writes `z_rl.npy` (`float32`, `[N, 2048]`), `indices.npy` (`int64`, dataset order), and `metadata.json`. It refuses to overwrite nonempty output and publishes metadata only after validation. Use `--device cpu` when CUDA is unavailable.

The checkpoint must contain trained Stage 1 `full_weights.pt`, and `--norm-stats` must resolve to the same `norm_stats.json` referenced by the original training YAML. The exporter preserves the YAML model settings and records its runtime overrides: repository `root_dir`, checkpoint `model_path`, explicit statistics path, `openpi.task: eval`, and `openpi.rlt_freeze_vla: false`. The YAML stores `fp32` weights and sets FSDP compute precision to `bf16`; export uses `bf16` autocast and records both precisions. Wait until checkpoint transfer has completed before exporting.

For the separate Stage 1 freeze check, see [RLT_STAGE1_FREEZE_VALIDATION.md](RLT_STAGE1_FREEZE_VALIDATION.md).

## Plot exported z vectors with PCA

Run PCA after `metadata.json` confirms that an export finished. The plot script uses NumPy and matplotlib; scikit-learn is not required.

```bash
python toolkits/openpi/plot_rlt_z_pca.py \
  --input-dir results/ur10e_rlt_stage1_z_batch8_smoke_20260924
```

The default output directory is `<input-dir>/pca/`. It contains `pca_scatter.png`, `pca_coordinates.npy`, `pca_components.npy`, `pca_mean.npy`, and `pca_metadata.json`. Points are colored by original dataset index, and the axes show explained variance. At least three vectors are needed. A small smoke sample tests the pipeline but does not describe the full dataset distribution. Use `--output-dir` to choose a different, empty directory; `--chunk-size` controls memory use for a full export.

To redraw the joint fail/success PCA for complete episodes with episode colors and outcome markers, use:

```bash
/root/autodl-tmp/.venvs/rlinf-openpi/bin/python toolkits/openpi/plot_rlt_z_pca_episodes.py \
  --steps 15000 18000 21000
```

This reads the existing `results/rlt_z_fail_success_pca_step<step>/` coordinates and the dataset episode indices. It writes SVG files to `results/rlt_z_pca_full_episodes/`; no model inference or PCA refit is performed. The colors match the full-episode t-SNE plots: each episode has one color, triangles indicate fail, and circles indicate success.
