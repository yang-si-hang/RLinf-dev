# Offline UR10e Stage 2 critic

Prepare replay first, then run the critic trainer. These commands use cached Stage 1 z vectors and recorded UR10e data; neither command starts Ray or connects to a robot.

    cd /root/autodl-tmp/workspace/RLinf-dev
    source switch_env openpi
    export PROJECT_ROOT=/root/autodl-tmp/workspace/RLinf-dev
    python toolkits/openpi/prepare_rlt_stage2_replay.py
    python toolkits/openpi/rlt_stage2_critic_only.py

Both scripts use the same Hydra config. Preparation converts LeRobot rows into 15-step replay transitions starting at the first frame and drops incomplete action chunks at each episode end, creates an episode-level train/validation split, saves PT replay groups, validation transitions, action statistics, and a manifest under trainer.output_dir. The trainer loads that replay; it does not import LeRobot rows or rebuild the replay.

Run preprocessing again only when creating a fresh replay. If a manifest already exists, the preparer checks that the inputs match and reports the existing replay. For a different Stage 1 checkpoint, export success and failure z arrays separately, update both paths and data.stage1_checkpoint_step, and use a fresh trainer.output_dir.

Hydra overrides are supported by both scripts. For example, set trainer.output_dir=/path/to/run in both invocations. The trainer accepts trainer.resume_checkpoint=/path/to/critic_step_N.pt; the checkpoint manifest is checked against the prepared replay inputs and reward configuration.

The default optimization settings follow the RLT TD3 MLP critic optimizer: learning rate 1e-4, gradient clip 10, and a 256-wide two-layer MLP. The validation fraction is applied per outcome label at the episode level. Replay sampling uses 60% ordinary transitions, 20% success terminals, and 20% failure terminals.

The critic uses RLinf's `MetricLogger` to log training and validation loss, Q values, TD error, and gradient norm every `trainer.log_interval` steps. Configure `runner.logger` in the YAML to select W&B or TensorBoard and set the project and experiment names. W&B defaults to offline logging on this server; sync the resulting run with `wandb sync`, or set `WANDB_MODE=online` when network access is reliable. Set `runner.logger.logger_backends=[]` to disable metric backends. The script also prints the same metrics as JSON.

`trainer.log_interval` controls JSON printing and W&B logging. `trainer.save_interval` saves checkpoints, while `trainer.keep_period` permanently retains checkpoints at that step interval. Every save also retains the newest checkpoint and removes older checkpoints outside the keep period. `keep_period` must be a multiple of `save_interval`; the final training step is saved even when it is off schedule.
