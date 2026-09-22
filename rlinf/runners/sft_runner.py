# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import EarlyStopController, check_progress

if TYPE_CHECKING:
    from rlinf.workers.reward.reward_worker import FSDPRewardWorker
    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

logger = logging.getLogger(__name__)

_STEP_CHECKPOINT_PATTERN = re.compile(r"global_step_(\d+)")


def _format_console_metrics(metrics: dict) -> str:
    """Format SFT metrics on one console line."""
    return ", ".join(
        f"{key}={value:.4g}" if isinstance(value, (int, float)) else f"{key}={value}"
        for prefix in ("time/", "train/", "eval/")
        for key, value in metrics.items()
        if key.startswith(prefix)
    )


def _prune_sft_checkpoints(
    checkpoints_dir: Path, current_step: int, keep_period: int | None
) -> list[Path]:
    """Delete superseded SFT checkpoints and return the removed paths.

    The current checkpoint is always retained. Periodic milestone checkpoints
    are retained when ``keep_period`` is set; all other step checkpoints are
    removed. Non-step entries, symlinks, and files are left untouched.

    Args:
        checkpoints_dir: Directory containing ``global_step_<N>`` directories.
        current_step: Step of the checkpoint that has just finished saving.
        keep_period: Positive milestone interval, or ``None`` to keep only the
            current checkpoint.

    Returns:
        Paths of checkpoint directories that were removed.

    Raises:
        ValueError: If ``keep_period`` is not a positive integer or ``None``.
    """
    if keep_period is not None and (
        isinstance(keep_period, bool)
        or not isinstance(keep_period, int)
        or keep_period <= 0
    ):
        raise ValueError(
            f"runner.keep_period must be a positive integer or null, got {keep_period!r}."
        )

    checkpoints_dir = checkpoints_dir.resolve()
    if not checkpoints_dir.is_dir():
        return []

    removed = []
    for checkpoint_path in checkpoints_dir.iterdir():
        match = _STEP_CHECKPOINT_PATTERN.fullmatch(checkpoint_path.name)
        if (
            match is None
            or checkpoint_path.is_symlink()
            or not checkpoint_path.is_dir()
        ):
            continue
        step = int(match.group(1))
        if step == current_step or (
            keep_period is not None and step % keep_period == 0
        ):
            continue
        # The child-name check above and resolved-parent check keep deletion
        # strictly scoped to checkpoint directories managed by this runner.
        if checkpoint_path.resolve().parent != checkpoints_dir:
            continue
        shutil.rmtree(checkpoint_path)
        removed.append(checkpoint_path)

    return removed


class SFTRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union["FSDPSftWorker", "FSDPRewardWorker"],
        run_timer: Optional[ScopedTimer] = None,
    ) -> None:
        self.cfg = cfg
        self.actor = actor

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0
        early_stop_cfg = cfg.runner.get("early_stop", None)
        self.early_stop = (
            EarlyStopController(early_stop_cfg) if early_stop_cfg is not None else None
        )
        self._checkpoint_retention_enabled = "keep_period" in cfg.runner
        self._checkpoint_keep_period = cfg.runner.get("keep_period", None)
        if self._checkpoint_retention_enabled:
            # Validate before training rather than after the first checkpoint.
            if self._checkpoint_keep_period is not None and (
                isinstance(self._checkpoint_keep_period, bool)
                or not isinstance(self._checkpoint_keep_period, int)
                or self._checkpoint_keep_period <= 0
            ):
                raise ValueError(
                    "runner.keep_period must be a positive integer or null, "
                    f"got {self._checkpoint_keep_period!r}."
                )

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)

    def init_workers(self) -> None:
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.global_step = parse_global_step_from_checkpoint_path(resume_dir)
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()

    def run(self) -> None:
        start_step = self.global_step
        console_log_interval = self.cfg.runner.get(
            "console_log_interval", self.cfg.runner.get("log_interval", 1)
        )
        if (
            isinstance(console_log_interval, bool)
            or not isinstance(console_log_interval, int)
            or console_log_interval < 1
        ):
            raise ValueError("runner.console_log_interval must be a positive integer")
        progress_stream = sys.stderr
        terminal_stream = None
        if not progress_stream.isatty():
            try:
                terminal_stream = open("/dev/tty", "w")
                progress_stream = terminal_stream
            except OSError:
                pass
        interactive = progress_stream.isatty()
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            dynamic_ncols=True,
            mininterval=0,
            disable=not interactive,
            file=progress_stream,
        )
        for _step in range(start_step, self.max_steps):
            if hasattr(self.actor, "set_global_step"):
                # set global step
                self.actor.set_global_step(self.global_step)

            with self.timer("step"):
                actor_handle: Handle = self.actor.run_training()
                actor_metrics = actor_handle.wait()

                self.global_step += 1

                eval_model, save_model, _ = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

                should_stop = False
                if eval_model:
                    eval_handle: Handle = self.actor.run_eval()
                    eval_metrics = eval_handle.wait()

                    if self.early_stop is not None:
                        should_stop, best_val_acc_improved = self.early_stop.update(
                            eval_metrics[0]
                        )
                        if best_val_acc_improved:
                            self._save_checkpoint(is_best=True)

            time_metrics = self.timer.consume_durations()
            time_metrics["training"] = actor_handle.consume_duration()
            if eval_model:
                time_metrics["evaluate"] = eval_handle.consume_duration()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            merged_actor_metrics = {}
            # get the merged actor metrics from all ranks
            for metrics in actor_metrics:
                for k, v in metrics.items():
                    if k not in merged_actor_metrics:
                        merged_actor_metrics[k] = v
            training_metrics = {
                f"train/{k}": v for k, v in merged_actor_metrics.items()
            }
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(training_metrics)

            if eval_model:
                evaluate_metrics = {f"eval/{k}": v for k, v in eval_metrics[0].items()}
                logging_metrics.update(evaluate_metrics)
                self.metric_logger.log(evaluate_metrics, _step)

            global_pbar.update(1)
            if self.global_step % console_log_interval == 0:
                message = _format_console_metrics(logging_metrics)
                if interactive and terminal_stream is not None:
                    global_pbar.clear()
                if interactive and terminal_stream is None:
                    tqdm.write(message, file=sys.stderr)
                else:
                    print(message, file=sys.stderr, flush=True)
                if interactive and terminal_stream is not None:
                    global_pbar.refresh()
            if should_stop:
                break

        global_pbar.close()
        if terminal_stream is not None:
            terminal_stream.close()

        if self.early_stop is not None and self.early_stop.best_val_acc > 0:
            logger.info(
                f"Early stopping triggered! Best val_acc: {self.early_stop.best_val_acc:.4f}"
            )
        self.metric_logger.finish()

    def run_eval(self) -> None:
        with self.timer("evaluate"):
            eval_handle: Handle = self.actor.run_eval()
            eval_metrics = eval_handle.wait()

        time_metrics = self.timer.consume_durations()
        time_metrics["evaluate"] = eval_handle.consume_duration()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

        raw_eval = (
            eval_metrics[0]
            if isinstance(eval_metrics, (list, tuple)) and len(eval_metrics) > 0
            else {}
        )
        evaluate_metrics = {f"eval/{k}": v for k, v in raw_eval.items()}

        logging_metrics = {}
        logging_metrics.update(time_metrics)
        logging_metrics.update(evaluate_metrics)

        logger.info(f"Eval metrics: {evaluate_metrics}")
        self.metric_logger.finish()

    def _save_checkpoint(self, is_best: bool = False) -> None:
        checkpoint_root = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
        )
        if is_best:
            base_output_dir = os.path.join(checkpoint_root, "checkpoints/best_model")
        else:
            base_output_dir = os.path.join(
                checkpoint_root,
                f"checkpoints/global_step_{self.global_step}",
            )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
        if not is_best and self._checkpoint_retention_enabled:
            removed = _prune_sft_checkpoints(
                Path(checkpoint_root) / "checkpoints",
                self.global_step,
                self._checkpoint_keep_period,
            )
            for checkpoint_path in removed:
                logger.info("Removed superseded checkpoint %s", checkpoint_path)
        if is_best and self.early_stop is not None:
            logger.info(
                f"Saved best model (val_acc={self.early_stop.best_val_acc:.4f}) to {base_output_dir}"
            )

    def set_max_steps(self) -> None:
        self.num_steps_per_epoch = self.actor.get_max_steps_per_epoch().wait()[0]
        max_epochs = self.cfg.runner.get("max_epochs", -1)
        max_steps = self.cfg.runner.get("max_steps", -1)

        step_limits = []
        if max_epochs > 0:
            step_limits.append(self.num_steps_per_epoch * max_epochs)
        if max_steps >= 0:
            step_limits.append(max_steps)

        # If both limits are configured, stop at whichever one is reached first.
        # If neither is configured, keep the historical default of one epoch.
        self.max_steps = min(step_limits) if step_limits else self.num_steps_per_epoch

    @property
    def epoch(self) -> int:
        return self.global_step // self.num_steps_per_epoch
