import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type Tab = "架构" | "训练" | "时序与闭环" | "面试";

const tabs: Tab[] = ["架构", "训练", "时序与闭环", "面试"];

function Architecture() {
  return (
    <Stack gap={16}>
      <Callout tone="info">
        ORION 的核心不是简单加入 LLM，而是用 &lt;waypoint_ego&gt; hidden state
        建立语言推理到连续轨迹的可微接口。
      </Callout>
      <Grid columns={4} gap={12}>
        <Stat value="6" label="环视相机" />
        <Stat value="9600" label="图像 token" />
        <Stat value="513" label="VLM 视觉 token" tone="info" />
        <Stat value="6×6×2" label="多模态轨迹" tone="success" />
      </Grid>
      <H2>前向传播</H2>
      <Table
        headers={["步骤", "张量/模块", "目的"]}
        rows={[
          ["1", "(B,6,3,640,640) → EVA-ViT → (B,6,1024,40,40)", "提取六相机特征"],
          ["2", "64 个 LID depth bins + lidar2img⁻¹", "生成 3D position embedding"],
          ["3", "OrionHead + OrionHeadM", "检测/运动与地图分别建模"],
          ["4", "257 det token + 256 map token", "构成 (B,513,4096) 视觉前缀"],
          ["5", "LLaVA-Llama + LoRA", "多模态语义推理"],
          ["6", "<waypoint_ego> → VAE + GRU", "输出六命令模态、六步轨迹"],
        ]}
        striped
      />
      <Callout tone="warning">
        代码中没有独立 BLIP-2 式 Q-Former；视觉桥接由 PETR extra query、scene memory
        和 256→4096 projection 完成。
      </Callout>
    </Stack>
  );
}

function Training() {
  return (
    <Stack gap={16}>
      <H2>三阶段训练</H2>
      <Table
        headers={["阶段", "目标", "QA", "规划"]}
        rows={[
          ["Stage 1", "感知 + VQA 对齐", "Chat-B2D + 在线 QA", "关闭"],
          ["Stage 2", "规划专项训练", "Planning-only", "VAE + GRU + collision"],
          ["Stage 3", "联合微调", "80% planning / 20% mixed QA", "同 Stage 2"],
        ]}
        rowTone={["info", "warning", "success"]}
        striped
      />
      <H2>损失耦合</H2>
      <Grid columns={2} gap={12}>
        <Card>
          <CardHeader>感知与地图</CardHeader>
          <CardBody>
            <Text>检测 cls/bbox、DN、交通灯状态、地图分类与 11 控制点回归。</Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>运动与规划</CardHeader>
          <CardBody>
            <Text>agent motion、ego L1、地图边界、碰撞约束、VAE 分布对齐。</Text>
          </CardBody>
        </Card>
      </Grid>
      <Callout tone="warning">
        Stage 2/3 配置中的 load_from=None 是占位值；实际训练应手动加载上一阶段 checkpoint。
      </Callout>
    </Stack>
  );
}

function TemporalControl() {
  return (
    <Stack gap={16}>
      <H2>真实时序机制</H2>
      <Text>
        queue_length=1，backbone 只处理当前帧。连续 sampler 保证帧顺序；PETR head 保存
        top-k query memory，并用 ego_pose_inv 将历史参考点补偿到当前坐标系。
      </Text>
      <Table
        headers={["环节", "实现", "风险"]}
        rows={[
          ["数据", "CARLA 左手系转右手 LiDAR 坐标", "变换不一致导致镜像/偏移"],
          ["时序", "top-300 proposal 写入 600-slot memory", "scene reset 或顺序错误"],
          ["规划", "增量 waypoint 经 cumsum 变绝对坐标", "漏做 cumsum"],
          ["闭环", "轨迹经 PID 变 steer/throttle/brake", "PID 启发式与模型不匹配"],
        ]}
        striped
      />
      <Callout tone="info">
        ORION 是 end-to-end planning，但不是严格 sensor-to-control：最终执行仍依赖 PID。
        因此开环 L2 与闭环 Driving Score 不能等价。
      </Callout>
    </Stack>
  );
}

function Interview() {
  return (
    <Stack gap={16}>
      <H2>高频盘问</H2>
      <Table
        headers={["问题", "回答核心"]}
        rows={[
          ["核心创新？", "结构化时序 token → VLM → 特殊 token hidden state → 连续 planner。"],
          ["如何获得 3D？", "像素沿深度 bins 反投影，作为 PETR 位置编码。"],
          ["为什么双 head？", "对象与道路拓扑具有不同 query、匹配和 loss，最终在 LLM 前融合。"],
          ["为什么六个 mode？", "与六类 route command 对齐，避免多模态未来均值化。"],
          ["Stage 2/3 区别？", "架构相同；planning-only 与 mixed QA + planning 的区别。"],
          ["为什么 VAE？", "表达一对多未来；训练 posterior 与推理 prior 通过分布 loss 对齐。"],
          ["最大工程风险？", "坐标系、memory reset、checkpoint 串联、闭环域偏移和 PID。"],
        ]}
        striped
      />
      <Callout tone="success">
        回答结构：Problem → Design → 代码或张量证据 → Trade-off。
      </Callout>
    </Stack>
  );
}

function App() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState<Tab>("orion-tab", "架构");
  return (
    <div style={{ minHeight: "100%", padding: 24, background: theme.bg.editor, color: theme.text.primary }}>
      <Stack gap={18} style={{ maxWidth: 1100, margin: "0 auto" }}>
        <div>
          <H1>ORION 架构深度拆解</H1>
          <Text tone="secondary">代码理解与秋招面试讲义；完整文字版见同目录 README.md。</Text>
        </div>
        <Row gap={8} wrap>
          {tabs.map((item) => (
            <Pill key={item} active={tab === item} onClick={() => setTab(item)}>
              {item}
            </Pill>
          ))}
        </Row>
        <Divider />
        {tab === "架构" && <Architecture />}
        {tab === "训练" && <Training />}
        {tab === "时序与闭环" && <TemporalControl />}
        {tab === "面试" && <Interview />}
      </Stack>
    </div>
  );
}

export default App;
