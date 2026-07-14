# ORION 架构深度拆解

> ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation  
> ICCV 2025 · 面向代码理解与算法面试

**在线交互版：** https://gxt2002.github.io/paper-deep-reading/papers/ORION/

## 1. 一句话理解

ORION 将六相机视觉、结构化目标/地图查询、语言模型推理和生成式轨迹规划联合起来：

```text
6 路相机
  → EVA-ViT
  → 检测 PETR Head + 地图 PETR Head
  → 513 个结构化视觉 token
  → LLaVA-Llama
  → <waypoint_ego> hidden state
  → VAE + GRU
  → 6 个命令模态 × 6 步二维轨迹
  → PID（闭环）
```

关键创新不只是“使用 LLM”，而是利用特殊 token `<waypoint_ego>` 的 hidden state，建立语言推理空间到连续动作空间的可微接口。

## 2. 模型组件

### 2.1 EVA-ViT 图像主干

输入是六路环视图像：

```text
(B, 6, 3, 640, 640)
```

每张图经过 patch size 为 16 的 EVA-ViT：

```text
(B, 6, 1024, 40, 40)
```

六个相机共得到 `6 × 40 × 40 = 9600` 个图像 token。

核心代码：

- `mmcv/models/detectors/orion.py:331-365`
- `mmcv/models/backbones/eva_vit.py`

### 2.2 3D 位置编码

ORION 没有先构建显式 BEV 栅格。它对特征图上的像素中心采样 64 个 LID 深度 bin，通过 `lidar2img` 的逆矩阵反投影到 LiDAR 坐标系，再用 MLP 编码为 256 维位置特征。

```text
pixel (u, v) + depth d
  → homogeneous camera point
  → lidar2img⁻¹
  → normalized 3D point
  → inverse sigmoid + MLP
  → position embedding (B, 9600, 256)
```

核心代码：`mmcv/models/detectors/orion.py:367-418`。

### 2.3 双 PETR Head

ORION 使用两个独立的时序 PETR head：

1. `OrionHead`：3D 检测、交通灯状态、周围车辆运动预测。
2. `OrionHeadM`：车道线、中心线、交通灯和 stop sign 等地图元素。

分头设计保留了不同任务的归纳偏置：

- 目标检测输出离散对象及其位置、尺寸和速度；
- 地图 head 输出由固定控制点表示的折线；
- 两者最终投影到同一个 LLM hidden space。

核心代码：

- `mmcv/models/dense_heads/orion_head.py`
- `mmcv/models/dense_heads/orion_head_map.py`

## 3. 513 个视觉 token

两个 head 都保留 256 个 `num_extra` query，专门生成供 VLM 使用的结构化视觉表示。

| 来源 | 数量 | 维度 |
| --- | ---: | ---: |
| 检测 head VLM queries | 256 | 4096 |
| CAN bus / ego token | 1 | 4096 |
| 地图 head VLM queries | 256 | 4096 |
| 合计 | 513 | 4096 |

```python
vision_embeded = torch.cat(
    [vision_embeded_obj, vision_embeded_map],
    dim=1,
)  # (B, 513, 4096)
```

这些 token 会替换 LLaVA 文本序列中的 `<image>` 占位符。

重要纠偏：代码中没有独立的 BLIP-2 式 Q-Former 类。`pretrain_qformer` 是权重目录名称；实际的视觉聚合和桥接由 PETR extra queries、时序 scene memory 与 `256 → 4096` 投影完成。

## 4. 时序建模

### 4.1 时序不是多帧图像堆叠

配置中 `queue_length=1`，backbone 每次只处理当前帧。真实的时序能力来自：

1. 连续序列 sampler：让同一个 batch slot 持续读取同一路线；
2. PETR query memory：保存历史 query、参考点、时间戳和 ego pose；
3. ego-motion compensation：把历史参考点变换到当前坐标系。

### 4.2 Memory 更新

检测 head 维护约 600 个 memory slot。每帧从当前结果中选择 top-300 proposal，插入 memory 前部。

场景变化或时间间隔过大时，memory 会被刷新。否则：

```text
历史 reference points
  → ego_pose_inv
  → 当前坐标系
  → 与当前 query 一起进入 temporal attention
```

核心代码：`mmcv/models/dense_heads/orion_head.py:448-576`。

这种设计相比堆叠多帧图像显著节省显存，但依赖正确的帧顺序、坐标变换和 scene reset。

## 5. LLM 与规划接口

### 5.1 为什么不让 LLM 直接输出坐标

让语言模型输出类似 `(1.23, 2.45)` 的文本存在三个问题：

- 数值精度不稳定；
- 格式可能无法解析；
- 文本 token 空间与连续轨迹空间存在表示鸿沟。

ORION 在 prompt 中加入特殊 token：

```text
Here is the planning trajectory <waypoint_ego>
```

模型找到该 token 在序列中的位置，提取对应 4096 维 hidden state：

```text
ego_feature: (B, 4096)
```

规划 loss 可以通过它反向传播到 LLM LoRA 和视觉投影，实现 reasoning-to-action 对齐。

核心代码：

- `mmcv/utils/llava_arch.py`
- `mmcv/utils/llava_llama.py:242-311`
- `mmcv/models/detectors/orion.py:563-687`

### 5.2 VAE + GRU Planner

驾驶未来具有一对多特性。ORION 使用条件 VAE 表达潜在意图：

```text
q(z | ego_feature, GT future)   # 训练 posterior
p(z | ego_feature)              # 推理 prior
```

两者通过 `ProbabilisticLoss` 对齐。采样出的 latent 与 `ego_feature` 一起进入 GRU 式未来状态预测器，再生成：

```text
(B, 6 modes, 6 timesteps, 2)
```

六个 mode 对应：

- 左转
- 右转
- 直行
- 沿车道行驶
- 向左变道
- 向右变道

推理时根据 `ego_fut_cmd` 选择一个 mode。

训练轨迹是逐步位移增量；推理时通过 `cumsum` 得到当前位置坐标系下的绝对 waypoint。

## 6. 三阶段训练

| 阶段 | 主要目标 | VQA 数据 | 规划分支 |
| --- | --- | --- | --- |
| Stage 1 | 感知与 VQA 对齐 | Chat-B2D + 在线目标 QA | 关闭 |
| Stage 2 | Planning 专项训练 | Planning QA only | VAE + GRU + collision |
| Stage 3 | 推理与规划联合微调 | 80% planning，20% 混合 QA | 同 Stage 2 |

### Stage 1

- `use_gen_token=False`
- 不生成 `<waypoint_ego>`；
- 训练检测、地图和 VLM CE；
- 从 `eva02_petr_proj.pth` 初始化。

### Stage 2

- `use_gen_token=True`
- `planning_qa_only=True`
- 打开 ego trajectory 和 agent motion 分支；
- 加入轨迹、边界、碰撞及 VAE loss。

### Stage 3

- `mix_qa_training=True`
- `planning_qa_ratio=0.8`
- 约 80% 样本做 planning，20% 保留完整 Chat-B2D/在线 QA；
- 用混合微调缓解 Stage 2 后语言推理能力退化。

普通 QA 样本没有 `<waypoint_ego>`。代码会使用 dummy ego feature，并把对应 planning mask 置零，因此不会产生伪轨迹监督。

注意：Stage 2/3 配置里的 `load_from=None` 是占位值。实际训练必须手动设置为上一阶段 checkpoint。

## 7. 损失函数

### 感知

- `loss_cls`：3D 目标分类；
- `loss_bbox`：3D 框回归；
- `dn_loss_*`：DETR denoising query 监督；
- `loss_traffic`：交通灯状态；
- 地图分类和控制点回归。

### Agent motion

- `loss_traj`：周围交通参与者未来轨迹；
- `loss_traj_cls`：多模态轨迹分类。

### Ego planning

- `loss_plan_reg`：预测轨迹与 GT 的 L1；
- `loss_plan_bound`：道路边界约束；
- `loss_plan_col`：与周围 agent 未来轨迹的碰撞约束；
- `loss_vae_gen`：prior/posterior 分布对齐。

规划 branch 并不是孤立训练：边界 loss 使用地图 head 输出，碰撞 loss 使用检测/运动 head 输出。因此感知、地图与规划在 loss 层也存在耦合。

## 8. 数据与坐标系

典型模型输入：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `img` | `(B,6,3,640,640)` | 六路相机 |
| `lidar2img` | `(B,6,4,4)` | 投影矩阵 |
| `can_bus` | `(B,18)` | ego 运动状态 |
| `gt_bboxes_3d` | `list[(Nᵢ,9)]` | 目标框 |
| `gt_attr_labels` | `list[(Nᵢ,34)]` | agent 未来运动 |
| map points | `list[(Mᵢ,11,2)]` | 地图折线 |
| `ego_fut_trajs` | `(B,1,6,2)` | 未来位移增量 |
| `ego_fut_cmd` | `(B,1,1,6)` | 导航命令 |
| `input_ids` | `(B,≤2048)` | VQA/规划文本 |

CARLA/UE4 原始坐标是左手系，预处理会转换到右手 LiDAR/nuScenes 风格坐标。框、地图、轨迹和相机外参必须使用一致变换，否则容易出现不报错但轨迹镜像或横向偏移的问题。

核心代码：

- `mmcv/datasets/prepare_B2D.py`
- `mmcv/datasets/b2d_orion_dataset.py`
- `mmcv/datasets/pipelines/transforms_3d.py`

## 9. 闭环控制

闭环运行于 CARLA：

```text
6 RGB + GPS + IMU + speed
  → RoutePlanner 得到 command
  → ORION 预测 6 个 waypoint
  → PIDController
  → steer / throttle / brake
```

PID 的作用：

- 在预测轨迹上选取约 3.5 m 的 aim point，计算横向 steering；
- 根据前两个 waypoint 距离估计目标速度；
- 根据目标速度与当前速度决定 throttle/brake；
- 包含 `speed > 5 m/s → throttle = 0` 等工程启发式。

因此，ORION 是端到端轨迹规划系统，但不是严格的 sensor-to-control 纯端到端系统。

核心代码：

- `team_code/orion_b2d_agent.py:282-436`
- `team_code/pid_controller.py`
- `team_code/planner.py`

## 10. 开环与闭环的差异

开环评测：

- 输入固定的专家数据；
- 当前预测不影响下一帧；
- 主要评估 trajectory L2、collision、mAP/NDS。

闭环评测：

- 输入来自车辆当前实际状态；
- 一次预测或控制错误会改变后续画面；
- 误差经车辆动力学、PID 和环境交互累积；
- 主要评估 Driving Score 与 Success Rate。

因此，较好的开环 L2 不保证较好的闭环驾驶。

## 11. 高频面试问题

### Q1：ORION 的核心创新是什么？

不是简单引入 LLM，而是把检测和地图的结构化时序 token 注入 VLM，再用 `<waypoint_ego>` hidden state 驱动连续生成式 planner，实现语义推理与动作空间的可微对齐。

### Q2：QT-Former 在代码哪里？

代码没有独立的 BLIP-2 式 Q-Former。实际功能由 PETR extra queries、scene memory 和 `256 → 4096` projection 承担。`pretrain_qformer` 只是预训练权重目录名。

### Q3：没有显式 BEV，如何理解 3D？

通过相机内外参和多个深度 bin，把图像像素反投影为 3D positional embedding，再让 PETR query 对多相机图像 token 做 cross-attention。

### Q4：为什么是六个规划 mode？

六个 mode 与六类 route command 对齐，可表达导航条件下的多模态未来，避免直接回归产生均值轨迹。

### Q5：时序信息如何融合？

连续 sampler 保证序列顺序；head 保存 top-k query memory，并通过 ego pose 把历史参考点变换到当前坐标系。

### Q6：Stage 2 和 Stage 3 的区别？

架构不变。Stage 2 是 planning-only；Stage 3 混合 planning 和通用 QA，避免语言推理能力退化。

### Q7：ORION 是真正端到端吗？

视觉到轨迹是联合训练的，但轨迹到执行器由 PID 完成。因此它属于 end-to-end planning，而不是严格 sensor-to-control。

### Q8：最大的工程风险是什么？

- 坐标系和外参一致性；
- 时序 memory 的 reset；
- Stage checkpoint 手工串联；
- 闭环 JPEG 压缩导致的域偏移；
- command 选错导致 mode 选择错误；
- PID 启发式与网络轨迹不匹配。

## 12. 建议源码阅读顺序

1. `mmcv/models/detectors/orion.py:331-687`：完整训练 forward。
2. `mmcv/models/dense_heads/orion_head.py:448-880`：query 与 temporal memory。
3. `mmcv/utils/llava_arch.py:49-184`：视觉 token 注入。
4. `mmcv/utils/llava_llama.py:242-311`：规划 token hidden state。
5. `mmcv/models/detectors/orion.py:197-304,1437-1504`：VAE 和规划 loss。
6. `mmcv/datasets/b2d_orion_dataset.py:164-517`：数据与轨迹 GT。
7. `team_code/orion_b2d_agent.py:282-436`：闭环执行。

## 13. 三句总结

1. ORION 的核心是用特殊 token hidden state 建立语言推理到连续轨迹的可微接口。
2. 它的时序能力来自 query memory 与 ego-motion compensation，而不是多帧图像堆叠。
3. 它是端到端轨迹规划系统，但闭环执行仍依赖 PID，所以开环指标和闭环指标不能等价。
