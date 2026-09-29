UR10e RLT：Stage 1 与在线 TD3
========================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/RLT.png
   :align: center
   :width: 80%

   RL Token 训练概览（图片：RLinf）。

用 UR10e 示范数据训练 RLT 特征，导入离线 replay 训练的 twin-Q critic，
再由 TD3 actor 在线采集完整 episode。机器人计算机运行独立
`VLA-Tele-ur <https://github.com/yang-si-hang/VLA-Tele-ur>`_ 仓库中的设备服务。
该仓库需包含 ``scripts/inference/ur10e_rlinf_device_server.py`` 及对应的
``ur_control.py`` 轨迹取消改动。

概览
----

按顺序执行各阶段，或提供先前运行中已验证的产物。方法说明和其他任务入口见
:doc:`RL Token <rlt>`。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 算法
      :text-align: center

      RLT Stage 1 + actor-only TD3

   .. grid-item-card:: 模型
      :text-align: center

      冻结的 π₀.₅ / RLT + TD3 MLP

   .. grid-item-card:: 环境 / 数据
      :text-align: center

      UR10e / LeRobot 示范数据

   .. grid-item-card:: 训练
      :text-align: center

      导入离线 twin-Q → 在线 episode

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 任务
     - 配置
     - 产物
   * - 将网线插入网口
     - ``ur10e_rlt_stage1_sft_openpi_pi05`` 和 ``ur10e_rlt_stage2_td3_mlp``
     - Stage 1 特征 checkpoint 和在线 TD3 actor checkpoint。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 字段
     - 契约
   * - 观测
     - 两路 224×224 RGB 图像和 10 维状态；Stage 1 产生 2048 维 ``z_rl``。
   * - 动作
     - Actor 块为 ``[15,10]``，VLA 的 BC 参考为 ``[30,10]``。训练与离线 critic 均使用 OpenPI q01/q99 归一化的相对动作空间。
   * - 奖励
     - 已开始的每步为 ``-1``，失败终止步为 ``-750``；达到 450 步上限记为失败。
   * - 任务文本
     - 设备观测提供 ``Plug the Ethernet cable into the Ethernet port.``。

安装
----

RLinf 使用 OpenPI Python 环境；机器人仓库需安装现有 UR、相机和夹爪依赖。
机器人端终端保持交互，以便按 ``c`` 标记成功、按 ``b`` 标记失败、按 ``x``
丢弃当前 episode。设备设置参见 :doc:`真机指南 <../../guides/realworld_robot>`。

下载模型
--------

将 ``PROJECT_ROOT`` 指向 RLinf checkout。下表是现有 YAML 的默认路径；
如果数据或权重在其他位置，修改对应配置。两阶段及离线 critic 必须使用同一份
``norm_stats.json``。

.. list-table::
   :header-rows: 1

   * - 输入
     - 配置键及 ``/app`` 下的默认路径
   * - Stage 1 示范数据
     - ``data.train_data_paths`` → ``data/lerobot_data/plug_v2_merge_crop_vid``
   * - 合并后的 π₀.₅ 初始 checkpoint
     - Stage 1 YAML 的 ``actor.model.model_path`` → ``data/openpi_rlinf_checkpoints/pi05_ur10e_plug_lora_ki/plug_20260920_223703/20000``
   * - OpenPI 动作统计
     - ``actor.model.openpi_data.norm_stats_path`` → 上述 checkpoint 的 ``assets/plug_v2_merge_crop_vid/norm_stats.json``
   * - 离线成功 / 失败示范数据
     - ``ur10e_rlt_stage2_critic_only.yaml`` 中的 ``data.success_dataset_path`` 与 ``data.failure_dataset_path``

运行
----

Stage 1：训练特征模型
~~~~~~~~~~~~~~~~~~~~~

先修改 ``examples/sft/config/ur10e_rlt_stage1_sft_openpi_pi05.yaml``，确保
``data.train_data_paths``、``actor.model.model_path`` 和
``actor.model.openpi_data.norm_stats_path`` 均可访问。配置使用
``openpi.task: sft``、``use_rlt: true``、``rlt_alpha: 0.0``、
``rlt_freeze_vla: true``，冻结 VLA 并训练 RLT 模块。执行：

.. code-block:: bash

   cd /app
   export PROJECT_ROOT=/app
   bash examples/sft/run_vla_sft.sh ur10e_rlt_stage1_sft_openpi_pi05

checkpoint 写入 ``logs/<run>/.../checkpoints/global_step_<step>/actor``。
现有 Stage 2 配置使用 ``global_step_18000/actor``；确认该目录包含
``model_state_dict/full_weights.pt``。将这个 ``actor`` 目录填入在线 YAML 的
``rollout.rlt_feature_model.model_path``，不能填入合并后的 π₀.₅ 初始 checkpoint。

Stage 2：准备离线 Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在线 TD3 只导入离线 critic 的 twin-Q 权重，不将离线 replay 混入在线 replay。
修改 ``examples/embodiment/config/ur10e_rlt_stage2_critic_only.yaml`` 中的两份
示范数据路径、两份特征导出路径、Stage 1 checkpoint 步数、动作统计路径与
``trainer.output_dir``。成功和失败数据必须使用同一个 Stage 1 checkpoint
导出特征；如果 checkpoint 位于其他位置，替换下面的路径。导出目录必须为空。
若已有导出和匹配的 critic checkpoint，应复用其 metadata 和 manifest，
不要覆盖：

.. code-block:: bash

   export STAGE1_ACTOR=/app/data/rlt_checkpoints/ur10e_rlt_stage1_sft_openpi_pi05/ur10e_plug_rlt_stage1_20260922_181649/global_step_18000/actor
   export NORM_STATS=/app/data/openpi_rlinf_checkpoints/pi05_ur10e_plug_lora_ki/plug_20260920_223703/20000/assets/plug_v2_merge_crop_vid/norm_stats.json
   python toolkits/openpi/export_rlt_z.py --config-name ur10e_rlt_stage1_sft_openpi_pi05 --dataset-root /app/data/lerobot_data/plug_v1_deploy_success_merge_vid --checkpoint "$STAGE1_ACTOR" --norm-stats "$NORM_STATS" --output-dir /app/results/rlt_z_success_step18000 --batch-size 1
   python toolkits/openpi/export_rlt_z.py --config-name ur10e_rlt_stage1_sft_openpi_pi05 --dataset-root /app/data/lerobot_data/plug_v1_deploy_fail_merge_vid --checkpoint "$STAGE1_ACTOR" --norm-stats "$NORM_STATS" --output-dir /app/results/rlt_z_fail_step18000 --batch-size 1
   python toolkits/openpi/prepare_rlt_stage2_replay.py
   python toolkits/openpi/rlt_stage2_critic_only.py

每次导出产生 ``z_rl.npy``、``indices.npy`` 和 ``metadata.json``。
replay 准备脚本在 ``trainer.output_dir`` 下写入 ``replay_manifest.json``
及离线 replay；critic 训练写入 ``critic_step_<step>.pt``。现有在线 YAML
导入 ``critic_step_18000.pt``；若使用其他 checkpoint，修改
``algorithm.ur10e_offline_critic_checkpoint_path``。确保
``env.train.override_cfg.replay_manifest_path``、两份 metadata 路径与
``action_norm_stats_path`` 对应同一份 critic。在线预检会校验其 SHA256 来源
及 twin-Q 权重映射。重新生成 manifest 可能使绑定旧哈希的 checkpoint 失效。

Stage 2：启动在线 Actor 训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 ``examples/embodiment/config/ur10e_rlt_stage2_td3_mlp.yaml``，使
``rollout.rlt_feature_model.model_path`` 指向 Stage 1，并设置上面的 critic、
manifest、metadata 和统计路径。在机器人计算机的交互终端启动设备服务：

.. code-block:: bash

   cd /path/to/VLA-Tele-ur
   python scripts/inference/ur10e_rlinf_device_server.py --robot-ip <UR10e-IP> --endpoint tcp://0.0.0.0:5555

在 RLinf 计算机上设置可访问的设备地址，先预检再训练。若 Docker 使用 host
网络，且设备服务在同一宿主机，可使用 ``127.0.0.1``；否则填写机器人计算机的
可访问地址。

.. code-block:: bash

   cd /app
   export PROJECT_ROOT=/app
   export UR10E_DEVICE_ENDPOINT=tcp://127.0.0.1:5555
   python toolkits/standalone_eval_scripts/openpi/check_ur10e_online_td3.py
   bash examples/embodiment/run_embodiment.sh ur10e_rlt_stage2_td3_mlp

预检核对本地输入、OpenPI 变换、critic 导入、设备 ``HEALTH`` 和双相机初始
观测，不发送动作。每个 episode 前先物理复位机器人，再在 RLinf 主终端输入
``start``。第一块起即由 actor 控制。完整且有效的 episode 才进入 replay；
``INVALID_ABORT`` 会丢弃当前整轮。

.. warning::

   ``EXECUTE_CHUNK`` 超时后不会自动重试。再次物理复位前先确认机器人状态。
   450 步上限统计已开始的 30 Hz 策略目标，不按墙钟计时。机器人静止时再运行
   预检；它的初始观测请求会重置设备端 episode 状态。

可视化与结果
------------

训练前确认预检输出 ``PREFLIGHT PASSED``。Stage 1 完成对应保存步后产生
``global_step_18000/actor/model_state_dict/full_weights.pt``。Stage 2
可观察 ``env/success_once``、``ur10e/valid_episode`` 等 episode 结果；
在线 actor checkpoint 按 ``runner.save_interval`` 保存在
``logs/<run>/ur10e_rlt_stage2_td3_mlp/checkpoints/global_step_<step>/actor``。
指标定义见 :doc:`训练指标 <../../reference/metrics>`。现有软件验证尚不代表
UR10e 实机在线训练已经成功完成。
