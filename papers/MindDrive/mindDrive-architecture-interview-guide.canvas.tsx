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

type Tab = "核心" | "架构" | "IL→RL" | "PPO" | "系统" | "面试";
const tabs: Tab[] = ["核心", "架构", "IL→RL", "PPO", "系统", "面试"];

function Core() {
  return (
    <Stack gap={16}>
      <Callout tone="info">
        MindDrive 不直接在连续轨迹空间做 RL，而是让 Decision Expert 在 7 类速度语言动作中探索，
        再由 Action Expert 把语言动作翻译为连续轨迹。
      </Callout>
      <Grid columns={4} gap={12}>
        <Stat value="7" label="PPO 速度动作" tone="info" />
        <Stat value="2" label="LoRA 专家" />
        <Stat value="513" label="结构化视觉 token" />
        <Stat value="80.59" label="3B Driving Score" tone="success" />
      </Grid>
      <H2>核心分解</H2>
      <Grid columns={3} gap={12}>
        <Card><CardHeader>Decision Expert</CardHeader><CardBody>在线 RL 更新，负责选择停车、加速、减速等 7 类语言动作。</CardBody></Card>
        <Card><CardHeader>Action Expert</CardHeader><CardBody>模仿学习训练，把 speed/path waypoint token hidden state 解码为连续轨迹。</CardBody></Card>
        <Card><CardHeader>Classical Control</CardHeader><CardBody>PID 使用 20 点 path 控制转向，使用 6 点 speed trajectory 控制纵向。</CardBody></Card>
      </Grid>
      <Callout tone="warning">
        Path mode 在 rollout 中来自 RoutePlanner，不是 PPO action。MindDrive 的在线 RL 主要优化纵向速度决策。
      </Callout>
    </Stack>
  );
}

function Architecture() {
  return (
    <Stack gap={16}>
      <H2>张量流</H2>
      <Table
        headers={["阶段", "形状", "含义"]}
        rows={[
          ["六相机", "(B,6,3,640,640)", "环视 RGB"],
          ["EVA-ViT", "(B,6,1024,40,40)", "共 9600 个视觉 token"],
          ["双 PETR", "(B,9600,256)", "检测、地图和时序 query"],
          ["VLM prefix", "(B,≈513,Hllm)", "Det + ego + map 结构化 token"],
          ["Waypoint hidden", "(B,2,Hllm)", "Speed feature + path feature"],
          ["Speed VAE", "(B,7,6,2)", "七种速度模态"],
          ["Path VAE", "(B,6,20,2)", "六种路径模态"],
        ]}
        striped
      />
      <H2>双 LoRA 的设计价值</H2>
      <Grid columns={2} gap={12}>
        <Card><CardHeader>共享</CardHeader><CardBody>两个专家共享 Qwen2 基础视觉语言表示，远小于复制两套完整 LLM。</CardBody></Card>
        <Card><CardHeader>隔离</CardHeader><CardBody>稀疏 PPO reward 只更新 Decision LoRA，避免破坏稳定的轨迹执行映射。</CardBody></Card>
      </Grid>
      <Callout tone="neutral">0.5B 的 Qwen2 hidden size 为 896；3B 的 Qwen2.5 hidden size 为 2048。</Callout>
    </Stack>
  );
}

function ILRL() {
  return (
    <Stack gap={16}>
      <H2>训练阶段</H2>
      <Table
        headers={["阶段", "目标", "关键变化"]}
        rows={[
          ["IL Stage 1", "感知 + VQA + meta-action", "不启用 waypoint trajectory"],
          ["IL Stage 2", "Planning-only", "打开 speed/path 双 token 与双 VAE"],
          ["IL Stage 3", "联合微调", "80% planning / 20% mixed QA"],
          ["Checkpoint bridge", "初始化 RL", "Action LoRA→Decision LoRA；LM→value net"],
          ["CARLA rollout", "在线数据采集", "采样 7-way speed action，保存 reward/buffer"],
          ["PPO", "优化闭环决策", "只训练 Decision LoRA + value projection"],
        ]}
        rowTone={["info", "info", "success", "warning", "warning", "success"]}
        striped
      />
      <Callout tone="warning">
        convert_checkpoint.py 不是普通格式转换，而是将 IL 学到的表示复制给 policy 和 critic；
        否则昂贵的 CARLA 探索会从随机策略开始。
      </Callout>
    </Stack>
  );
}

function PPO() {
  return (
    <Stack gap={16}>
      <H2>PPO 实现</H2>
      <Grid columns={3} gap={12}>
        <Card><CardHeader>Policy</CardHeader><CardBody>7-way action ratio，clip ε=0.2。</CardBody></Card>
        <Card><CardHeader>Value</CardHeader><CardBody>0.5 × MSE(V, return)。</CardBody></Card>
        <Card><CardHeader>GAE</CardHeader><CardBody>γ=0.99，λ=1.0，稀疏 reward 下方差较高。</CardBody></Card>
      </Grid>
      <H2>Reward</H2>
      <Table
        headers={["事件", "Reward", "结果"]}
        rows={[
          ["路线完成 100%", "+1", "episode 结束"],
          ["碰撞/闯灯/偏航/出车道", "-1", "episode 结束"],
          ["普通 step", "0", "继续"],
        ]}
        striped
      />
      <Callout tone="danger">
        Rollout 被保存后用固定数据训练多个 epoch，会逐渐偏离严格 on-policy；
        clipping、KL 和定期重新 rollout 是稳定训练的关键。
      </Callout>
    </Stack>
  );
}

function System() {
  return (
    <Stack gap={16}>
      <H2>训练与仿真的关键差异</H2>
      <Table
        headers={["维度", "离线 IL", "Rollout / Eval"]}
        rows={[
          ["数据", "Bench2Drive + Chat-B2D-plus", "Live CARLA"],
          ["时间", "10 Hz annotation，0.5s 轨迹点", "20 Hz simulator"],
          ["速度决策", "JSON teacher forcing", "Rollout 采样 / Eval argmax"],
          ["路径决策", "cmd_path 标签", "RoutePlanner ego_fut_cmd"],
          ["优化", "Action Expert + perception/planning", "Decision Expert + value projection"],
        ]}
        striped
      />
      <H2>闭环控制</H2>
      <Text>20 点 path waypoint 用于 steering；6 点 speed waypoint 用于 desired speed；最后由 PID 输出 CARLA control。</Text>
      <Callout tone="warning">
        系统仍包含 speed&gt;5m/s 时 throttle=0、油门刹车互斥等启发式，因此不是纯 sensor-to-control。
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
          ["为什么不直接对轨迹做 RL？", "连续空间探索低效；7 类语言动作显著降低 CARLA 采样成本。"],
          ["为什么双 LoRA？", "共享基础表示，同时隔离在线决策与稳定轨迹执行。"],
          ["Reward 如何回到语言空间？", "PPO 更新 7 个 meta-action token 的概率，只训练 Decision LoRA。"],
          ["Path 为什么不参与 PPO？", "利用 RoutePlanner 强先验减少组合爆炸；代价是无法优化横向决策。"],
          ["为什么缓存 inputs_embeds？", "避免 PPO 重跑视觉 backbone；代价是视觉无 RL 梯度且数据会 stale。"],
          ["λ=1 有何影响？", "稀疏终止 reward 下接近 Monte Carlo，偏差低、方差高。"],
          ["0.5B 为什么能超过 ORION-7B？", "在线 reward 修正闭环决策边界，参数量并非闭环性能唯一因素。"],
          ["最大工程风险？", "adapter 切换、route mode、稀疏 reward、10/20 Hz、坐标外参、PID。"],
        ]}
        striped
      />
      <Callout tone="success">
        三句总结：RL 只优化 7 类速度语言动作；Decision 负责做什么，Action 负责如何执行；
        0.5B 开环略差但闭环略优，说明收益主要来自决策而非轨迹拟合。
      </Callout>
    </Stack>
  );
}

function App() {
  const theme = useHostTheme();
  const [tab, setTab] = useCanvasState<Tab>("minddrive-tab", "核心");
  return (
    <div style={{ minHeight: "100%", padding: 24, background: theme.bg.editor, color: theme.text.primary }}>
      <Stack gap={18} style={{ maxWidth: 1120, margin: "0 auto" }}>
        <div><H1>MindDrive 架构深度拆解</H1><Text tone="secondary">双 LoRA、语言动作空间、CARLA rollout 与 PPO 源码讲义</Text></div>
        <Row gap={8} wrap>{tabs.map(item => <Pill key={item} active={tab===item} onClick={()=>setTab(item)}>{item}</Pill>)}</Row>
        <Divider />
        {tab==="核心"&&<Core />}{tab==="架构"&&<Architecture />}{tab==="IL→RL"&&<ILRL />}
        {tab==="PPO"&&<PPO />}{tab==="系统"&&<System />}{tab==="面试"&&<Interview />}
      </Stack>
    </div>
  );
}

export default App;
