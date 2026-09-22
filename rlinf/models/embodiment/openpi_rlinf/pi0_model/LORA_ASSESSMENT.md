# OpenPI_RLinf LoRA Implementation Assessment

This note records the implementation status observed on 2026-09-08. It is a
design assessment only; no model behavior is changed by this document.

## Two LoRA implementations in RLinf

RLinf currently has two distinct LoRA paths for OpenPI models.

1. The generic PEFT path in `rlinf/models/__init__.py`, enabled by
   `actor.model.is_lora: true`. For `model_type: openpi`, this path wraps the
   PaliGemma VLM with PEFT. It does not reproduce OpenPI's two different LoRA
   ranks for PaliGemma and the action expert.
2. The native-style implementation in this directory. It mirrors the model
   structure of upstream OpenPI and is selected with model variants such as
   `gemma_2b_lora` and `gemma_300m_lora`. This path does not use PEFT.

These paths must not be enabled together. A configuration using the internal
LoRA variants should keep the top-level `actor.model.is_lora` set to `false`.

## Intended native-style configuration

The internal implementation can express the model configuration used by
upstream OpenPI:

```yaml
model_type: openpi_rlinf
is_lora: false
pi05: true

openpi:
  task: sft
  paligemma_variant: gemma_2b_lora
  action_expert_variant: gemma_300m_lora
```

The variant definitions in `gemma.py` specify:

- `gemma_2b_lora`: rank-16 attention and FFN LoRA.
- `gemma_300m_lora`: rank-32 attention and FFN LoRA.

## Current implementation status

### FFN LoRA is implemented

`lora.py` implements the low-rank parameters and forward computation for the
gating and output weights of the feed-forward network. `gemma.Block` selects
this implementation when an expert has an `ffn` LoRA configuration.

### Attention LoRA is not implemented

`gemma.Attention.__init__` currently contains:

```python
lora_cfg = config.lora_configs.get("attn")
# TODO: add lora attn support
if lora_cfg is not None:
    raise NotImplementedError
```

Both `gemma_2b_lora` and `gemma_300m_lora` include an attention LoRA
configuration. Consequently, either variant currently raises
`NotImplementedError` while the model is being constructed. The `Einsum` LoRA
primitive exists in `lora.py`, but it has not been connected to the Q, K, V,
and output projections in `gemma.Attention`.

This is the first blocker for using the internal implementation in training.

### SFT does not apply the upstream freeze filter

The base and LoRA tensors in `lora.py` are all created as ordinary
`nn.Parameter` objects and therefore default to `requires_grad=True`.
`_build_sft_model` in `../utils/model_builders.py` does not apply a freeze
policy.

The only explicit OpenPI_RLinf freeze implementation is `freeze_vlm` in
`../rl_action_model.py`. It is used by the RL/PPO builder when
`train_expert_only` is enabled. Its behavior is not equivalent to the
`Pi0Config.get_freeze_filter()` used by upstream LoRA SFT.

After attention LoRA is implemented, an explicit PyTorch equivalent of the
upstream freeze filter is still needed. The intended behavior must be checked
against the exact upstream OpenPI version. For the currently installed version,
the filter freezes matching base LLM parameters while excluding LoRA
parameters from the frozen set.

### A non-LoRA base checkpoint cannot currently initialize a LoRA model

`load_base_safetensors` in `../utils/rlt_utils.py` calls
`model.load_state_dict(..., strict=True)`. A normal Pi0/Pi0.5 base checkpoint
does not contain newly introduced LoRA A/B tensors, so strict loading into a
model constructed with LoRA variants will report missing keys.

The loader should eventually permit only known missing LoRA tensors while
continuing to reject missing base tensors and unexpected checkpoint tensors.
Using unchecked `strict=False` would hide genuine conversion or checkpoint
errors and is not recommended.

### Numerical parity with upstream OpenPI is not established

The PyTorch port follows the upstream structure, but some details differ from
the currently installed JAX implementation:

- The PyTorch implementation initializes LoRA A randomly and LoRA B to zero.
  The inspected upstream JAX implementation initializes both through its LoRA
  initializer.
- The PyTorch FFN path explicitly applies `scaling_value`; the inspected
  upstream FFN implementation does not visibly apply the same scaling in its
  `_dot` helper, while its attention LoRA does.

These differences may be intentional or may reflect differences between
OpenPI revisions. Before claiming parity, compare against the exact OpenPI
commit used to create the target training configuration and add deterministic
cross-framework numerical tests.

### Checkpoint export has not been validated for internal LoRA tensors

OpenPI_RLinf can save and reload its own full FSDP wrapper checkpoint, but the
conversion path back to the legacy OpenPI PyTorch layout has not been shown to
preserve or merge the internal LoRA tensors. Deployment can initially remain on
`model_type: openpi_rlinf`. Legacy deployment requires either explicit adapter
conversion or merging the LoRA updates into the base weights before export.

### No dedicated internal-LoRA tests were found

No unit or end-to-end test currently demonstrates all of the following:

- construction with `gemma_2b_lora` and `gemma_300m_lora`;
- rank-16 PaliGemma and rank-32 action-expert adapters;
- successful initialization from a non-LoRA Pi0.5 base checkpoint;
- upstream-equivalent freezing;
- forward and backward passes with gradients only on intended parameters;
- checkpoint save and resume with the internal adapters.

## Does `pi0_model` support only Pi0?

No. The directory name denotes the Pi0 model family. The same `Pi0` class also
implements Pi0.5, selected by `Pi0Config.pi05`.

The Pi0.5 branches include:

- `pi05=True` stored by `Pi0`;
- adaptive RMSNorm conditioning for the action expert;
- Pi0.5-specific `time_mlp_in` and `time_mlp_out` layers;
- Pi0/Pi0.5-specific prefix and suffix construction;
- Pi0.5 defaults for prompt length and discrete state input.

The implementation therefore covers Pi0 and Pi0.5. No Pi0-FAST implementation
was identified in this directory. Pi0.5 internal dual LoRA is represented in
the configuration, but is not currently usable because attention LoRA is
unfinished.

## Recommended implementation order

If the internal OpenPI-style LoRA path is completed later, use this order:

1. Implement attention LoRA for every expert's Q, K, V, and output projection.
2. Add structural and numerical tests for attention and FFN LoRA against the
   selected upstream OpenPI revision.
3. Allow a base checkpoint to omit only newly initialized LoRA tensors.
4. Implement and test the PyTorch equivalent of the upstream SFT freeze filter.
5. Reject accidental use of generic PEFT together with internal LoRA variants.
6. Add the required environment-specific SFT dataloader routing, such as UR.
7. Test FSDP forward, backward, checkpoint save, and checkpoint resume.
8. Decide whether deployment stays on OpenPI_RLinf or requires merged/converted
   legacy OpenPI weights, then test that path explicitly.

Until these items are completed, the internal implementation should be treated
as an incomplete native-style LoRA port rather than a production-ready
replacement for the generic PEFT path.
