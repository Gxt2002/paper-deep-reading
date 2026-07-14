# MindDrive 架构深度拆解

> MindDrive: A Vision-Language-Action Model for Autonomous Driving Utilizing Language as Action in Online Reinforcement Learning  
> ECCV 2026 · 面向源码理解、强化学习与算法面试

**在线交互版：** https://gxt2002.github.io/paper-deep-reading/papers/MindDrive/

## 1. 一句话理解

MindDrive 在 ORION 的视觉语言轨迹规划骨架上，引入“语言即动作”的在线强化学习：

```text
六相机视觉
  → EVA-ViT + 双 PETR 时序 Head
  → Qwen2 VLM
  ├─ Decision Expert：选择 7 类离散速度语言动作
  └─ Action Expert：把语言决策映射为连续轨迹
       ├─ speed trajectory：7 modes × 6 points
       └─ path trajectory：6 modes × 20 points
  → PID
  → CARLA
  → 路线级奖励
  → PPO 更新 Decision Expert
```

核心洞察是：直接在连续轨迹空间做在线 RL，动作空间巨大、探索效率低。MindDrive 把 RL 动作空间压缩成 7 个离散速度 meta-action token，再由已通过模仿学习训练好的 Action Expert 将语言动作翻译为可执行轨迹。

## 2. MindDrive 解决了什么问题

纯模仿学习存在两个经典问题：

1. **Distribution shift**：训练只看到专家状态，闭环一旦偏离专家轨迹，后续输入分布改变，误差持续累积。
2. **Causal confusion**：模型可能学到与专家动作相关但不是真正因果的视觉特征。

在线 RL 允许车辆在闭环中试错，但直接探索 steering、throttle 或完整连续轨迹非常低效。MindDrive 的处理方式是：

```text
连续动作 RL
    ↓ 降维
有限语言决策 RL
    ↓ 由 Action Expert 翻译
连续可执行轨迹
```

这使 RL 主要负责“当前应该加速、减速还是停车”，而不是重新学习完整的几何规划和低层控制。

## 3. 整体架构

### 3.1 视觉与几何骨架

MindDrive 复用 ORION 的主要感知结构：

- 六路 RGB 相机；
- EVA-ViT 图像编码器；
- 基于相机内外参和深度 bin 的 3D positional embedding；
- `MinddriveHead`：目标检测、交通灯、周围 agent 运动；
- `MinddriveHeadM`：车道线、中心线、交通灯和 stop sign；
- StreamPETR 风格时序 query memory。

典型张量流：

| 阶段 | 张量形状 |
| --- | --- |
| 输入图像 | `(B, 6, 3, 640, 640)` |
| EVA-ViT 输出 | `(B, 6, 1024, 40, 40)` |
| 展平图像 token | `(B, 9600, 1024)` |
| PETR 投影与位置编码 | `(B, 9600, 256)` |
| Det VLM token | 约 `(B, 257, H_llm)` |
| Map VLM token | `(B, 256, H_llm)` |
| 合并视觉前缀 | 约 `(B, 513, H_llm)` |

其中：

- Qwen2-0.5B：`H_llm = 896`
- Qwen2.5-3B：`H_llm = 2048`

核心代码：

- `mmcv/models/detectors/minddrive.py`
- `mmcv/models/dense_heads/minddrive_head.py`
- `mmcv/models/dense_heads/minddrive_head_map.py`
- `mmcv/models/utils/petr_transformers.py`

### 3.2 513 个结构化视觉 token

视觉 token 不是直接把 9600 个 patch 全部塞入 LLM，而是由两个 PETR head 的 query 压缩得到：

| 来源 | 数量 | 含义 |
| --- | ---: | --- |
| Detection extra query | 256 | 目标、运动和场景语义 |
| CAN bus / ego token | 约 1 | 速度、姿态、导航等 ego 状态 |
| Map extra query | 256 | 道路拓扑与交通规则元素 |
| 总计 | 约 513 | 替换文本中的 `<image>` |

这样做比单个全局 pooled vector 保留更多对象级和地图级结构，又比把全部 9600 个视觉 patch 输入 LLM 更节省上下文和计算。

## 4. 双 LoRA 专家

MindDrive 的 Qwen2 主干上挂载两套 LoRA：

| 专家 | LoRA 名称 | 作用 |
| --- | --- | --- |
| Decision Expert | `decision_expert` | 场景推理并选择离散速度 meta-action |
| Action Expert | `action_expert` | 提取 waypoint token hidden state，生成连续轨迹 |

二者共享同一个基础 LLM，但使用不同 LoRA 参数：

```python
load_model(
    ...,
    adapter_names=["action_expert", "decision_expert"],
)
```

这种设计的好处：

- 比复制两个完整 LLM 节省大量参数；
- 共享语言和视觉表示；
- RL 只更新 Decision Expert，避免破坏已经学好的轨迹生成能力；
- 运行时可按 QA round 切换 adapter。

代价：

- 推理时 adapter 切换增加代码复杂度；
- 如果激活了错误 adapter，meta-action 或 trajectory 会直接失效；
- 两个专家的能力耦合依赖 checkpoint 初始化与 token 接口。

核心代码：

- `mmcv/utils/misc.py:414-465`
- `mmcv/models/detectors/minddrive.py:245`
- `mmcv/utils/llava_qwen.py`

## 5. 语言动作空间

### 5.1 七类速度动作

PPO 实际优化的动作空间只有 7 类速度 token：

```text
<maintain_moderate_speed>
<stop>
<maintain_slow_speed>
<speed_up>
<slow_down>
<maintain_fast_speed>
<slow_down_rapidly>
```

Decision Expert 从对应位置的 LLM logits 中抽取这 7 个 token 的概率分布：

```text
π(a | s) ∈ R⁷
```

Rollout 时从分布采样，闭环评测时通常取 argmax。

### 5.2 六类路径动作

路径 meta-action 包括：

```text
<lanefollow>
<straight>
<turn_left>
<change_lane_left>
<turn_right>
<change_lane_right>
```

但一个容易被忽略的重要事实是：

> 当前 RL rollout 并没有对路径动作做 PPO 探索。运行时 path mode 来自 CARLA RoutePlanner 的 `ego_fut_cmd`，PPO 只学习速度决策。

这是一种有意识的动作空间降维：

- 优点：避免速度 × 路径的组合爆炸，显著降低在线探索难度；
- 缺点：RL 无法学习更优的横向决策或变道时机，路径仍依赖规则导航命令。

## 6. 速度与路径解耦

Stage 2/3 打开 `is_decoupling=True`，使用两个 trajectory token：

- `<waypoint_ego>`：速度/短期运动轨迹；
- `<path_waypoint_ego>`：更长的几何路径。

Action Expert 在两个 token 位置提取 hidden state：

```text
ego_feature: (B, 2, H_llm)
├─ speed feature
└─ path feature
```

之后进入两套 VAE/GRU decoder：

| 分支 | 模态数 | 时间/空间点数 | 输出 |
| --- | ---: | ---: | --- |
| Speed trajectory | 7 | 6 | `(B, 7, 6, 2)` |
| Path trajectory | 6 | 20 | `(B, 6, 20, 2)` |

Speed trajectory 表达 3 秒内、每 0.5 秒一个点的运动；path trajectory 以约 1 米间距描述更长的道路几何。

两者训练输出都是位移增量，推理选择 mode 后通过 `cumsum` 转成 ego frame 下的绝对 waypoint。

## 7. VAE 轨迹生成器

MindDrive 延续 ORION 的条件 VAE 思路：

```text
训练 posterior：
q(z | ego_feature, GT trajectory)

推理 prior：
p(z | ego_feature)
```

VAE latent dimension 为 32。训练时使用包含 GT future 的分布采样，推理时只能从当前条件分布采样，再由 GRU 风格的 `PredictModel` 展开未来状态。

```text
ego_feature
  → DistributionModule
  → latent z
  → PredictModel / SpatialGRU
  → per-step hidden state
  → trajectory decoder
```

VAE 的作用不是替代语言决策，而是处理同一决策下轨迹细节的一对多不确定性。

## 8. 三阶段模仿学习

MindDrive 继承 ORION 的三阶段 IL 框架，但加入 meta-action 和速度/路径解耦。

| 阶段 | 主要目标 | 关键配置 |
| --- | --- | --- |
| Stage 1 | 感知、VQA 与 meta-action 语言预训练 | `pretrain=True`, `use_gen_token=False` |
| Stage 2 | Planning-only，建立语言到轨迹的一对一映射 | `use_gen_token=True`, `is_decoupling=True` |
| Stage 3 | 通用 QA 与 planning 联合微调 | `mix_qa_training=True`, `planning_qa_ratio=0.8` |

### 8.1 Stage 1

- 使用 Chat-B2D-plus；
- 数据中增加每帧 meta-action VQA；
- 训练检测、地图和 VLM；
- 不启用 waypoint trajectory branch。

### 8.2 Stage 2

- 只保留 meta-action 与 planning QA；
- 加入 `<waypoint_ego>` 和 `<path_waypoint_ego>`；
- 开启 speed/path 双轨迹 VAE；
- 开启轨迹、地图边界和碰撞监督。

### 8.3 Stage 3

- 80% 样本以 planning 为主；
- 20% 样本混入场景描述与普通 VQA；
- 在保持语言推理能力的同时稳定轨迹规划。

README 明确说明，IL 阶段主要只训练 Action Expert 的 LoRA，以降低显存和训练成本。

## 9. IL 损失

### 感知与地图

- 3D detection focal loss；
- 3D box L1；
- DETR denoising loss；
- 交通灯状态 loss；
- 地图分类与 11 控制点回归。

### Agent motion

- 周围 agent 多模态轨迹回归；
- agent trajectory mode 分类。

### Ego speed/path planning

- `loss_plan_reg`：选中 mode 的轨迹 L1，权重 3.0；
- `loss_plan_bound`：地图边界约束，权重 3.0；
- `loss_plan_col`：与 agent future 的碰撞约束，权重 1.0；
- `loss_vae_gen`：prior/posterior 分布约束，权重 3.0；
- path 分支有对应的 `pw_loss_plan_*`。

感知、地图和规划不是彼此独立的：

- collision loss 使用 detection/motion head 的 agent future；
- bound loss 使用 map head 的 lane prediction；
- trajectory loss 通过 waypoint token 回传到 Action Expert。

## 10. 从 IL 到 RL 的 Checkpoint 桥接

Stage 3 完成后执行：

```bash
python rl_projects/convert_checkpoint.py
```

它完成两项关键复制：

```text
action_expert LoRA
  → decision_expert LoRA

lm_head
  → value_net
```

为什么需要复制：

- IL 主要训练 Action Expert；
- Decision Expert 在开始 PPO 前需要具备合理的语言/动作表示；
- critic 也需要从已训练的多模态表示初始化；
- 否则从随机 policy/value 开始在线探索几乎不可行。

## 11. 在线 Rollout

调用链：

```text
minddrive_run_collection_multi.sh
  → rollout.py
  → RLIterBasedRunner.collect_rollouts_scenario()
  → CarlaScenarioEnv
```

每个 CARLA step：

```text
CARLA observation
  → inference pipeline
  → EVA-ViT + PETR heads
  → Decision Expert
      ├─ 7-way speed logits
      ├─ sample speed action
      └─ value_net predicts V(s)
  → Action Expert
      ├─ speed trajectory
      └─ path trajectory
  → PID
  → CARLA step
  → reward / done
  → RolloutBuffer
```

Buffer 每步保存：

- `actions`：速度动作 index；
- `rewards`；
- `values`；
- `ref_log_probs`：7 维旧策略 log probability；
- `advantages` 与 `returns`；
- `meta_action_info`：缓存的 `inputs_embeds` 和 `new_input_ids`。

缓存 `inputs_embeds` 的目的，是 PPO 时不再重新运行高成本的视觉 backbone。代价是 RL 梯度不会回到视觉编码器，且 rollout 数据会占用较大磁盘。

## 12. Reward 设计

奖励非常稀疏：

| 事件 | Reward | 是否结束 |
| --- | ---: | --- |
| 路线完成 100% | `+1` | 是 |
| 指定交通违规或碰撞 | `-1` | 是 |
| 异常 / crash | `-1` | 是 |
| 普通中间 step | `0` | 否 |

默认惩罚包括：

- 与行人、车辆或静态物体碰撞；
- 闯红灯或 stop sign；
- 偏离路线；
- 驶出车道。

稀疏奖励与 Bench2Drive 路线成功语义一致，但会带来高方差和低 credit assignment 效率。

## 13. PPO 实现

### 13.1 Clipped policy objective

对于 rollout 中动作 \(a_t\)：

```text
r_t(θ) = exp(log π_θ(a_t|s_t) - log π_old(a_t|s_t))

L_clip = -min(
  r_t A_t,
  clip(r_t, 1-ε, 1+ε) A_t
)
```

代码中 `ε = 0.2`。

### 13.2 Value loss

```text
L_value = 0.5 × MSE(V_θ(s_t), return_t)
```

### 13.3 Advantage

使用 GAE：

```text
δ_t = r_t + γV(s_{t+1}) - V(s_t)
A_t = δ_t + γλA_{t+1}
```

配置中：

- `gamma = 0.99`
- `gae_lambda = 1.0`

当 `λ=1` 时接近 Monte Carlo return；在 ±1 稀疏终止奖励下无偏性更强，但方差较高。

### 13.4 RL 实际更新哪些参数

PPO 阶段主要训练：

- `decision_expert` LoRA；
- `value_net_pro` 标量 value projection。

被冻结的部分包括：

- EVA-ViT；
- PETR perception heads；
- Action Expert；
- 连续 trajectory decoder；
- value_net 主体的大部分参数。

因此，RL 优化的是“选哪个速度语言动作”，而不是重新端到端训练视觉到轨迹的全部网络。

## 14. PPO 的实现细节与风险

### 14.1 PPO 是否真正 on-policy

rollout 先被保存为 pickle/NPZ，再进行多个 epoch 的训练。它更接近使用固定 rollout 数据的 PPO update，而不是每次 policy 更新后立刻重新采样。

随着训练 epoch 增加，当前 policy 与采样 policy 的差异会变大，clipping 和 KL 约束承担稳定作用。

### 14.2 配置与实际代码可能不一致

配置中的 `no_use_kl_and_entro=True` 并不能简单理解为 PPO 不使用 KL。默认 `EpochBasedRunner.ppo_train` 路径仍会计算 KL；相关配置 wiring 有注释掉的代码。

面试或复现实验时应以实际 runner 调用链为准，而不是只读 config 变量。

### 14.3 只优化速度

PPO action 是速度 index，path 来自 RoutePlanner。结果是：

- longitudinal decision 可通过在线 reward 改善；
- lateral/path decision 不直接接受 PPO credit；
- 系统能力仍受 route command 和 Action Expert 限制。

## 15. 数据管线

### 15.1 离线数据

MindDrive 使用：

- Bench2Drive Base 六相机数据；
- Bench2Drive map；
- `b2d_infos_train.pkl / val.pkl / map_infos.pkl`；
- Chat-B2D-plus meta-action VQA。

典型字段：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `img` | `(6,3,640,640)` | 六相机 |
| `ego_fut_trajs` | `(6,2)` | 0.5 秒间隔的未来增量轨迹 |
| `path_points_future` | `(20,2)` | 约 1 米间隔的路径增量 |
| `ego_fut_cmd` | `(6,)` | route command one-hot |
| `cmd_speed` | `(7,)` | 速度 meta-action one-hot |
| `cmd_path` | `(6,)` | 路径 meta-action one-hot |
| `can_bus` | `(18,)` | ego 运动状态 |

### 15.2 时间基准

- 离线 Bench2Drive annotation：10 Hz；
- future trajectory 每 5 帧采样一次，即 0.5 秒；
- 6 个 future point 对应 3 秒；
- CARLA rollout/eval：20 Hz。

训练和仿真时间基准不同，依赖 timestamp 与 temporal memory 正确处理。

### 15.3 坐标系

轨迹统一转换到当前 ego/LiDAR frame：

- x forward；
- y left；
- GT 以位移增量保存；
- 推理后 `cumsum`。

CARLA runtime 还包含 Y 轴符号、compass 到 yaw、GPS 到局部坐标等手工转换，是最容易出现静默错误的部分。

## 16. 闭环控制

MindDrive 的网络输出两类 waypoint：

| 输入 PID 的轨迹 | 形状 | 用途 |
| --- | --- | --- |
| Path waypoint | `(20,2)` | 横向 steering aim point |
| Speed waypoint | `(6,2)` | 估计 desired speed |

调用：

```python
PIDController.control_pid(
    path_waypoint,
    speed_waypoint,
    current_speed,
    route_target,
)
```

PID 和 agent 仍包含启发式规则：

- brake 很小时置零；
- throttle 与 brake 互斥；
- throttle 上限 0.75；
- `speed > 5 m/s` 时强制 throttle 为 0；
- 约 3.5 米前方 path aim point 用于 steering。

因此 MindDrive 是 VLA trajectory planning + classical control，而不是纯 sensor-to-control。

## 17. MindDrive 与 ORION 的关系

| 维度 | ORION | MindDrive |
| --- | --- | --- |
| 视觉骨架 | EVA-ViT + 双 PETR | 基本继承 |
| LLM | LLaMA 路线，约 7B/4096 hidden | Qwen2-0.5B 或 Qwen2.5-3B |
| LoRA | 单一 adapter | Decision + Action 双 adapter |
| Planning token | 单个 waypoint token | speed/path 两个 token |
| 轨迹模式 | 6 个 route-command mode | 7 speed + 6 path mode |
| Meta-action | 无 | 7 speed + 6 path token |
| 在线 RL | 无 | CARLA rollout + PPO |
| RL action | — | 只优化 7 类速度语言动作 |
| 闭环控制 | waypoint + PID | decoupled waypoint + PID |

MindDrive 并不是推倒 ORION 重做，而是在 ORION 的 reasoning-to-action 接口上增加一层 decision/action decomposition，并把在线 reward 主要反馈给 Decision Expert。

## 18. 结果

README 结果表：

| 模型 | L2 @ 2s | Driving Score | Success Rate |
| --- | ---: | ---: | ---: |
| ORION-7B | 0.68 m | 77.74 | 54.62% |
| MindDrive-0.5B | 0.69 m | 78.04 | 55.09% |
| MindDrive-3B | 0.66 m | 80.59 | 58.26% |

重要观察：

- 0.5B MindDrive 的开环 L2 略差于 ORION，但闭环 DS/SR 略高；
- 说明在线决策优化未必显著改善 imitation trajectory matching，却可能改善交互式闭环行为；
- 3B 同时获得更好的开环和闭环结果；
- README News 中 3B SR 写成 55.09，而结果表写 58.26，存在文档不一致；引用结果时应明确采用表格或官方 eval JSON。

## 19. 关键源码阅读顺序

1. `README.md:16-118`：论文目标、训练流程和结果。
2. `mmcv/models/detectors/minddrive.py:545-851`：IL forward 与损失。
3. `mmcv/models/detectors/minddrive.py:872-1261`：Decision/Action Expert 推理。
4. `mmcv/utils/llava_qwen.py:45-560`：Qwen2、meta-action logits 和 RL forward。
5. `mmcv/datasets/pipelines/transforms_3d.py:2193-2408`：meta-action VQA。
6. `mmcv/models/detectors/minddrive.py:1965-2169`：VAE 与 planning loss。
7. `rl_projects/convert_checkpoint.py:5-70`：IL 到 RL 的权重桥接。
8. `mmcv/runner/iter_based_runner.py:339-405`：CARLA rollout。
9. `mmcv/runner/buffers.py:237-339`：GAE 与 rollout buffer。
10. `mmcv/runner/epoch_based_runner.py:74-139`：PPO loss。
11. `team_code/carla_env/carla_env_scenario.py:365-473`：reward 与环境 step。
12. `team_code/minddrive_b2d_agent.py:202-437`：闭环评测。

## 20. 高频面试问题

### Q1：为什么不直接对轨迹做 RL？

连续高维轨迹空间探索效率低，CARLA rollout 成本高。把动作压缩成 7 类速度语言 token，可以显著缩小动作空间，再复用 IL 训练的 Action Expert 生成连续轨迹。

### Q2：Decision Expert 和 Action Expert 为什么用双 LoRA？

二者共享视觉语言知识，但目标不同。Decision Expert 接受稀疏在线 reward，Action Expert 保持稳定的语言到轨迹映射。双 LoRA 能隔离 RL 更新，避免 catastrophic forgetting，同时远小于两套完整 LLM。

### Q3：奖励如何回传到语言空间？

CARLA rollout 保存速度动作、旧策略 log probability、advantage 和缓存的 LLM inputs embedding。PPO 对 7 个 meta-action token 的概率执行 clipped policy update，梯度只更新 Decision Expert LoRA。

### Q4：为什么 path 不参与 RL？

这是为了降低探索维度，并利用 RoutePlanner 的强先验。代价是 RL 只优化纵向决策，无法直接改善横向路径选择和变道时机。

### Q5：为什么需要 convert checkpoint？

IL 主要训练 Action Expert。PPO 前将其表示复制给 Decision Expert 和 value net，使 policy/critic 从有意义的视觉语言表示开始，而不是随机探索。

### Q6：缓存 inputs_embeds 有什么好处和代价？

避免 PPO 重跑 EVA-ViT 和 PETR，显著节省训练计算；但视觉 backbone 不接收 RL 梯度，且缓存 rollout 会占用磁盘和内存。

### Q7：`gae_lambda=1` 在稀疏奖励下意味着什么？

advantage 接近 Monte Carlo return，偏差较低但方差很高。可以尝试更小 λ、reward shaping 或更密集的 progress reward。

### Q8：MindDrive 是 on-policy 吗？

算法目标是 PPO，但实现先保存 rollout，再用固定数据训练多个 epoch。随着 policy 更新，数据会逐渐变旧，因此 clipping/KL 和定期重新 rollout 很关键。

### Q9：0.5B 为什么可能超过 ORION-7B？

参数量不是闭环性能的唯一决定因素。ORION 只做 IL，而 MindDrive 用在线交互 reward 修正决策边界；同时有限 meta-action 让小模型也能高效探索。

### Q10：最大的工程风险是什么？

- 双 adapter 切换错误；
- RoutePlanner command 与 path mode 错位；
- 稀疏奖励和高方差；
- 10 Hz 数据与 20 Hz 仿真的时间差；
- 坐标系与固定外参；
- checkpoint adapter 名称不一致；
- PID 启发式掩盖或破坏网络收益。

## 21. 三句总结

1. MindDrive 的创新不是“用 PPO 微调整条轨迹”，而是把在线 RL 限制在 7 类速度语言动作上。
2. Decision Expert 负责“做什么”，Action Expert 负责“如何变成轨迹”，双 LoRA 隔离稀疏 reward 与稳定执行能力。
3. 它改善的是闭环交互决策，不一定直接改善开环轨迹拟合；这解释了 0.5B 模型 L2 略差但 DS/SR 略高的现象。
