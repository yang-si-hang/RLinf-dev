UR10e RLT: Stage 1 and Online TD3
==================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/RLT.png
   :align: center
   :width: 80%

   RL Token training overview (image: RLinf).

Train RLT features from UR10e demonstrations, initialize a twin-Q critic from
offline replay, and then collect online episodes with a TD3 actor. This recipe
uses a separate `VLA-Tele-ur <https://github.com/yang-si-hang/VLA-Tele-ur>`_
device server on the robot computer. Use the server implementation that
includes ``scripts/inference/ur10e_rlinf_device_server.py`` and the matching
``ur_control.py`` trajectory-cancellation change.

Overview
--------

Follow the stages in order, or provide the validated artifacts from an earlier
run. See :doc:`RL Token <rlt>` for the method and other task recipes.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Algorithm
      :text-align: center

      RLT Stage 1 + actor-only TD3

   .. grid-item-card:: Models
      :text-align: center

      Frozen π₀.₅ / RLT + TD3 MLP

   .. grid-item-card:: Environments / Data
      :text-align: center

      UR10e / LeRobot demonstrations

   .. grid-item-card:: Training
      :text-align: center

      Offline twin-Q import → online episodes

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Task
     - Config
     - Result
   * - Plug an Ethernet cable into its port
     - ``ur10e_rlt_stage1_sft_openpi_pi05`` and ``ur10e_rlt_stage2_td3_mlp``
     - A Stage 1 feature checkpoint and online TD3 actor checkpoints.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Contract
   * - Observation
     - Two RGB cameras at 224×224 and a 10D state; Stage 1 produces 2048D ``z_rl``.
   * - Action
     - Actor chunk ``[15,10]``; VLA BC reference ``[30,10]``. Training uses the same OpenPI q01/q99 normalized relative action space as the offline critic.
   * - Reward
     - ``-1`` per started step, ``-750`` on the terminal failure step; a 450-step episode limit is a failure.
   * - Prompt
     - ``Plug the Ethernet cable into the Ethernet port.`` comes from the device observation.

Installation
------------

Use the OpenPI Python environment for RLinf and the installed UR, camera, and
gripper dependencies in the robot checkout. Keep the robot computer terminal
interactive for ``c`` (success), ``b`` (failure), and ``x`` (discard). See the
:doc:`real-world robot guide <../../guides/realworld_robot>` for device setup.

Download the Model
------------------

Set ``PROJECT_ROOT`` to your RLinf checkout. The following paths are defaults
in the checked-in YAML; edit the YAML if your data or weights live elsewhere.
Keep the same ``norm_stats.json`` in both stages and in the offline critic.

.. list-table::
   :header-rows: 1

   * - Input
     - Config key and default path under ``/app``
   * - Stage 1 demonstrations
     - ``data.train_data_paths`` → ``data/lerobot_data/plug_v2_merge_crop_vid``
   * - Merged π₀.₅ starting checkpoint
     - ``actor.model.model_path`` in the Stage 1 YAML → ``data/openpi_rlinf_checkpoints/pi05_ur10e_plug_lora_ki/plug_20260920_223703/20000``
   * - OpenPI action statistics
     - ``actor.model.openpi_data.norm_stats_path`` → that checkpoint's ``assets/plug_v2_merge_crop_vid/norm_stats.json``
   * - Offline success/failure demonstrations
     - ``data.success_dataset_path`` and ``data.failure_dataset_path`` in ``ur10e_rlt_stage2_critic_only.yaml``

Run It
------

Stage 1: Train the Feature Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``examples/sft/config/ur10e_rlt_stage1_sft_openpi_pi05.yaml`` so
``data.train_data_paths``, ``actor.model.model_path``, and
``actor.model.openpi_data.norm_stats_path`` resolve on your machine. The
configuration uses ``openpi.task: sft``, ``use_rlt: true``,
``rlt_alpha: 0.0``, and ``rlt_freeze_vla: true``: it trains the RLT module
while freezing the VLA. Run:

.. code-block:: bash

   cd /app
   export PROJECT_ROOT=/app
   bash examples/sft/run_vla_sft.sh ur10e_rlt_stage1_sft_openpi_pi05

The command writes checkpoints under
``logs/<run>/.../checkpoints/global_step_<step>/actor``. For the supplied
Stage 2 configuration, use the ``global_step_18000/actor`` checkpoint and
confirm it contains ``model_state_dict/full_weights.pt``. Set
``rollout.rlt_feature_model.model_path`` in the online YAML to that actor
directory. Do not point it at the merged π₀.₅ starting checkpoint.

Stage 2: Prepare the Offline Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Online TD3 imports twin-Q weights from an offline critic checkpoint. It does
not mix the offline replay into online replay. Edit
``examples/embodiment/config/ur10e_rlt_stage2_critic_only.yaml`` so the two
demonstration paths, two feature export paths, Stage 1 checkpoint step, action
statistics path, and ``trainer.output_dir`` agree with your artifacts. Export
features from the *same* Stage 1 checkpoint for the success and failure
datasets; replace the checkpoint path below if your Stage 1 run wrote it
elsewhere. The exporter requires an empty output directory. If you already
have exports and a matching critic checkpoint, reuse their metadata and
manifest instead of overwriting them:

.. code-block:: bash

   export STAGE1_ACTOR=/app/data/rlt_checkpoints/ur10e_rlt_stage1_sft_openpi_pi05/ur10e_plug_rlt_stage1_20260922_181649/global_step_18000/actor
   export NORM_STATS=/app/data/openpi_rlinf_checkpoints/pi05_ur10e_plug_lora_ki/plug_20260920_223703/20000/assets/plug_v2_merge_crop_vid/norm_stats.json
   python toolkits/openpi/export_rlt_z.py --config-name ur10e_rlt_stage1_sft_openpi_pi05 --dataset-root /app/data/lerobot_data/plug_v1_deploy_success_merge_vid --checkpoint "$STAGE1_ACTOR" --norm-stats "$NORM_STATS" --output-dir /app/results/rlt_z_success_step18000 --batch-size 1
   python toolkits/openpi/export_rlt_z.py --config-name ur10e_rlt_stage1_sft_openpi_pi05 --dataset-root /app/data/lerobot_data/plug_v1_deploy_fail_merge_vid --checkpoint "$STAGE1_ACTOR" --norm-stats "$NORM_STATS" --output-dir /app/results/rlt_z_fail_step18000 --batch-size 1
   python toolkits/openpi/prepare_rlt_stage2_replay.py
   python toolkits/openpi/rlt_stage2_critic_only.py

Each export writes ``z_rl.npy``, ``indices.npy``, and ``metadata.json``.
Replay preparation writes ``replay_manifest.json`` and the offline replay
under ``trainer.output_dir``. Critic training writes
``critic_step_<step>.pt`` there. The checked-in online YAML imports
``critic_step_18000.pt``; update its
``algorithm.ur10e_offline_critic_checkpoint_path`` if you select another
checkpoint. Keep ``env.train.override_cfg.replay_manifest_path``, the two
metadata paths, and ``action_norm_stats_path`` aligned with that checkpoint.
The online preflight checks their SHA256 provenance and exact twin-Q mapping.
Regenerating a manifest can invalidate a checkpoint tied to its previous hash.

Stage 2: Start Online Actor Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``examples/embodiment/config/ur10e_rlt_stage2_td3_mlp.yaml`` to point
``rollout.rlt_feature_model.model_path`` at Stage 1 and to set the critic,
manifest, metadata, and statistics paths above. On the robot computer, start
the device server in an interactive terminal:

.. code-block:: bash

   cd /path/to/VLA-Tele-ur
   python scripts/inference/ur10e_rlinf_device_server.py --robot-ip <UR10e-IP> --endpoint tcp://0.0.0.0:5555

On the RLinf computer, set the reachable service address and run the preflight
before training. With Docker host networking and a server on the same host,
``127.0.0.1`` works; otherwise use the robot computer's reachable address.

.. code-block:: bash

   cd /app
   export PROJECT_ROOT=/app
   export UR10E_DEVICE_ENDPOINT=tcp://127.0.0.1:5555
   python toolkits/standalone_eval_scripts/openpi/check_ur10e_online_td3.py
   bash examples/embodiment/run_embodiment.sh ur10e_rlt_stage2_td3_mlp

Preflight checks local inputs, the OpenPI transforms, critic import, device
``HEALTH``, and an initial dual-camera observation. It sends no action. Before
each episode, physically reset the robot and type ``start`` in the RLinf
terminal. The actor controls every chunk from the first one. A valid complete
episode enters replay; ``INVALID_ABORT`` discards the entire current episode.

.. warning::

   A timed-out ``EXECUTE_CHUNK`` is never retried. Check the robot state before
   another physical reset. The 450-step limit counts started 30 Hz policy
   targets, not wall-clock time. Run preflight while the robot is stationary;
   its initial-observation request resets device-side episode state.

Visualization and Results
-------------------------

Watch the preflight's ``PREFLIGHT PASSED`` before training. Stage 1 produces
``global_step_18000/actor/model_state_dict/full_weights.pt`` when that save
step completes. Stage 2 logs episode results such as ``env/success_once`` and
``ur10e/valid_episode``. Its online actor checkpoint is saved under
``logs/<run>/ur10e_rlt_stage2_td3_mlp/checkpoints/global_step_<step>/actor``
at ``runner.save_interval``. See :doc:`training metrics <../../reference/metrics>`
for metric definitions. The current software checks do not establish that a
UR10e hardware training run has completed successfully.
