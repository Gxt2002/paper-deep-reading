# 论文精读 / Paper Deep Reading

面向算法、自动驾驶与多模态方向的论文精读仓库。

每篇论文使用独立目录，内容包括：

- 架构与数据流解析
- 关键代码调用链
- 训练目标与实现细节
- 设计动机、取舍与失败模式
- 面试高频问题与回答框架
- 浏览器可直接访问的交互式 HTML
- Cursor Canvas 交互式讲义（如适用）

## 目录

| 论文 | 主题 | 在线讲义 | Markdown |
| --- | --- | --- | --- |
| ORION (ICCV 2025) | 视觉语言端到端自动驾驶 | [交互式网页](https://gxt2002.github.io/paper-deep-reading/papers/ORION/) | [完整笔记](papers/ORION/README.md) |
| MindDrive (ECCV 2026) | 语言动作与在线强化学习 | [交互式网页](https://gxt2002.github.io/paper-deep-reading/papers/MindDrive/) | [完整笔记](papers/MindDrive/README.md) |

## 目录约定

```text
paper-deep-reading/
├── README.md
└── papers/
    ├── ORION/
    │   ├── README.md
    │   ├── index.html
    │   └── orion-architecture-interview-guide.canvas.tsx
    └── MindDrive/
        ├── README.md
        ├── index.html
        └── mindDrive-architecture-interview-guide.canvas.tsx
```

`index.html` 通过 GitHub Pages 提供跨平台交互阅读；`README.md` 用于检索和归档；`.canvas.tsx` 可在 Cursor Canvas 中打开。
