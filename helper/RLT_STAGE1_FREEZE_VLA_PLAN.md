# RLT Stage 1：仅冻结 VLA 参数的实施方案

给后续 Codex 的指令：在 `/app` 仓库按本文直接实施代码、配置、测试和英中文档修改；运行可执行的验证，最后报告结果与未验证项。保留工作区已有的其他改动。

## 任务

在 `dev/rlt` 上实现第一层优化：当 OpenPI RLinf 的 SFT Stage 1 明确配置为只训练 RLT encoder/decoder 时，冻结完整 Pi0/Pi0.5 VLA 核心的参数。保留当前完整前向、损失和指标计算。本文件是供后续 Codex 直接实施的任务说明；编写本文件时尚未实现代码修改。

### 验收目标

- `rlt_freeze_vla` 默认 `false`，现有配置的训练行为不变。
- 当 `task: sft`、`use_rlt: true`、`rlt_alpha: 0.0`、`rlt_freeze_vla: true` 时，`wrapper.model` 的全部参数 `requires_grad=False`，`wrapper.rlt_module` 的全部参数 `requires_grad=True`。
- 冻结发生在 FSDP 包装及优化器创建之前；包装后的 AdamW 只包含 RLT 参数。反向后 VLA 参数的 `grad is None`，RLT 有有限梯度；优化器更新后 VLA 权重完全不变。
- 更新前，同一模型状态、批次和随机采样下，开关前后的 `vla_loss`、`rlt_loss`、总 `loss` 在合理浮点误差内一致。
- 原有 prefix + suffix 的 VLA 前向和 `vla_loss` 诊断指标保持不变。

## 已核实的项目现状

1. `examples/sft/config/ur10e_rlt_stage1_sft_openpi_pi05.yaml` 目前使用 `task: sft`、`use_rlt: true`、`rlt_alpha: 0.0`、FSDP `no_shard` 和 `use_orig_params: true`；没有设置 `runner.resume_dir`。当前 `model_path` 指向 `/app/data/openpi_rlinf_checkpoints/pi05_ur10e_plug_lora_ki/plug_20260920_223703/20000`，编写本文件时该目录存在 `model.safetensors`。不要恢复先前写错的路径。
2. `rlinf/models/embodiment/openpi_rlinf/__init__.py:get_model()` 先构建 Pi0，加载 `model.safetensors` 到 Pi0，构建 task wrapper；若使用 `full_weights.pt`，则在 wrapper 构建后加载完整权重，最后返回 wrapper。`rlt_utils.py:build_rlt_config()` 被 SFT 和 eval builder 调用；RL builder 当前不传 RLT 配置。
3. `rlinf/workers/sft/fsdp_sft_worker.py:model_provider_func()` 调用 `get_model()`。随后 `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py:setup_model_and_optimizer()` 依次记录可训练参数名、FSDP 包装、按 `requires_grad` 创建 AdamW，并立即预建优化器状态。因此冻结必须在 `get_model()` 返回前完成。
4. `rlinf/models/embodiment/openpi_rlinf/sft_action_model.py:sft_forward()` 在 RLT 模式仍计算 `vla_loss`，返回 `loss = rlt_loss + rlt_alpha * vla_loss` 和两项独立指标。`_sft_forward_with_rlt_prefix()` 在把 `prefix_out` 交给 RLT 前调用 `detach()`；RLT decoder 也 detach 重建目标。`rlt_alpha: 0.0` 并不自动把 VLA 排除出优化器：未冻结时，`0.0 * vla_loss` 仍可产生零梯度张量，AdamW 仍为 VLA 建状态。
5. 当前 `use_orig_params: true` 支持 FSDP 中混合冻结与可训练参数；保留现有 `no_shard` 策略。混合参数的实际显存收益必须测量，不能仅凭参数标志推断。

## 实施步骤

### 1. 配置

在 `rlinf/models/embodiment/openpi_rlinf/utils/rlt_utils.py`：

- 给 `OpenPiPytorchRLTConfig` 增加 `rlt_freeze_vla: bool = False`。
- 在 `build_rlt_config(model_cfg)` 中从 `actor.model.openpi.rlt_freeze_vla` 读取该字段，默认 `False`。

在 UR10e Stage 1 YAML 的 `rlt_alpha: 0.0` 旁增加 `rlt_freeze_vla: true`。不要修改该 YAML 的 `model_path`、FSDP 策略、精度、数据路径或训练步数。

### 2. 统一校验和冻结位置

在 `rlinf/models/embodiment/openpi_rlinf/__init__.py:get_model()` 按**请求的 `model_cfg`**校验开关：仅允许 `task == "sft"`、`use_rlt is True` 且 `rlt_alpha == 0.0`。非法组合抛出包含字段名和要求的 `ValueError`；使用精确的 `0.0` 比较，不用阈值。校验须在 task 分派前或分派时完成，不要只检查 wrapper 的 `rlt_cfg`：RL builder 忽略该配置，检查 wrapper 会漏报 `task: rl` 的非法请求。

完成 wrapper 构建和可选 `full_weights.pt` 加载后，在 `return wrapper` 前仅当开关为真时执行：

```python
wrapper.model.requires_grad_(False)
wrapper.rlt_module.requires_grad_(True)
```

可以复用一次解析得到的 `rlt_cfg`，也可以在 `get_model()` 中读取并校验请求配置，同时沿用 builder 现有的解析逻辑。选择改动更小且避免两个来源不一致的写法。不要冻结整个 wrapper；不要逐个枚举 VLA 子模块。checkpoint 的 `state_dict` 不负责保存 `requires_grad`，因此以加载完成后的最终参数标志为准。

### 3. 保持计算路径

不要修改 `sft_action_model.py` 的前向与损失公式，不添加 `torch.no_grad()`，不跳过 action suffix、Action Expert、flow noise/time 采样、velocity head 或 `vla_loss`。本次不做 prefix-only、KV cache、数据集去 action 等后续优化。

### 4. 文档

在现有 RLT 示例文档的英文和中文页面（`docs/source-en/rst_source/examples/embodied/rlt.rst`、`docs/source-zh/rst_source/examples/embodied/rlt.rst`）简要说明新开关只适用于上述 SFT + 零 `rlt_alpha` 组合，默认关闭，且不减少 VLA 前向计算。保持两种语言一致；不要把通用的 `rlt_alpha: 1.0` 示例改为冻结模式。

## 验证

### 自动测试

在 `tests/unit_tests/models/` 增加针对 OpenPI RLinf RLT 冻结的测试，沿用已有测试风格。至少覆盖：

1. 默认配置解析为 `False`；合法组合通过；`task != sft`、`use_rlt == false`、`rlt_alpha != 0.0` 各自报错。尤其检查 `task: rl` 的非法配置。
2. 使用轻量模型或适当 mock 验证加载后的 wrapper 中：所有 `model.*` 参数冻结，所有 `rlt_module.*` 参数可训练；未开启时不改变既有标志。避免在普通 CI 中实例化 3B 级 Pi0.5。
3. 对实际 `use_orig_params: true`、`no_shard` 路径做单卡集成检查，按参数对象身份比较 FSDP 包装后的 `named_parameters()` 与优化器 `param_groups`：优化器恰好包含 RLT 参数，不包含 VLA 参数。FSDP 可添加 `_fsdp_wrapped_module.` 等名称前缀，不要用未包装模型的名字做脆弱断言。

运行相关 Ruff 检查和新增测试，并报告结果。若真实模型或 GPU 资源使集成检查无法在 CI 执行，应提供可单独运行的验证脚本或明确记录手工验证命令及结果；不得把未运行的检查写成已通过。

### 真实模型单步验收

使用当前 UR10e 配置和同一份起始 `model.safetensors`，在训练进程中记录：总、可训练、冻结参数元素数；可训练参数名前缀；优化器参数对象数和元素数。运行真实 batch 的一次 forward/backward/step：

- 反向后检查每个 VLA 参数 `grad is None`；RLT 梯度存在且有限，报告缺失梯度的参数名及梯度范数。
- 在**实际学习率非零**时更新，比较更新前后抽样 VLA 权重的逐元素相等，并确认至少一个 RLT 权重变化。
- 用同一初始模型及 RLT 状态、同一 batch 和受控 RNG，比较冻结前后的三项 loss；在优化器更新前比较，使用合理浮点容差。
- 可选比较初始化后的 `torch.cuda.memory_allocated()` 以及完整步骤的峰值。优化器在创建时就预建状态，因此仅在训练步骤前重置峰值会遗漏这部分内存。

## Checkpoint 边界

当前 YAML 从 `model.safetensors` 启动，且无 `runner.resume_dir`，不涉及旧优化器状态兼容问题。`runner.resume_dir` 会在新优化器建成后恢复模型、优化器、学习率调度器和 RNG；若旧训练的优化器包含 VLA + RLT，不能假定它能直接恢复到新建的仅含 RLT 参数的优化器。首次切换冻结模式时，从 `model_path` 加载模型权重并新建优化器；之后可用冻结模式保存的 checkpoint 正常续训。如果以后用 `full_weights.pt` 作起始权重，确认其包含完整 `rlt_module.*`；当前 loader 在 `use_rlt: true` 时要求这些键存在。

## 完成条件

提交应包含配置字段、严格校验、加载后且 FSDP 前的冻结、UR10e YAML、自动测试及英中文档。报告实际运行了哪些测试与真实模型验收项、哪些因资源限制未运行。保持本任务以外的工作区改动不变。
