<div class="index-page">

  <!-- 顶部标题区 -->
  <div class="header-container">
    <div class="header-small">XI'AN JIAOTONG UNIVERSITY · SCHOOL OF MATHEMATICS AND STATISTICS</div>
    <h1 class="header-title">宇树机器人强化学习源码精读笔记</h1>
    <div class="header-subtitle">Research Notes on Unitree RL-Gym</div>
    <div class="header-author">作者：田淞文 | 西安交通大学 数学与统计学院</div>
  </div>

  <!-- 关于本笔记 -->
  <div class="about-card">
    <h3 class="card-title-accent">关于本笔记</h3>
    <p>大家好，我是田淞文。在学习宇树 G1 机器人强化学习的过程中，我整理了这份笔记，记录了从底层环境配置、PPO 算法原理、到 Sim2Sim 部署的完整学习路径。希望这份笔记能帮助到同样在学习这个仓库的你。</p>
    <div class="tags-container">
      <span class="tag">Embodied AI</span>
      <span class="tag">强化学习</span>
      <span class="tag">宇树机器人行走</span>
      <span class="tag">PPO</span>
      <span class="tag">Sim2Sim</span>
    </div>
  </div>

  <!-- 目录卡片网格 -->
  <div class="contents-title">
    <h2>目录 / Contents</h2>
  </div>

  <div class="grid-container">
    
    <!-- 卡片 1：总览 -->
    <a href="./学习路线" class="grid-item">
      <div class="section-num">OVERVIEW</div>
      <h3 class="card-title">总览与基础 / Overview</h3>
      <p class="card-desc">完整的 19 天学习路线表格，以及看源码需要了解的强化学习前置知识。</p>
      <div class="card-keywords">包含 Roadmap、前置知识</div>
    </a>

    <!-- 卡片 2：第一阶段 -->
    <a href="./Day1" class="grid-item">
      <div class="section-num">STAGE I</div>
      <h3 class="card-title">第一阶段：环境与控制系统</h3>
      <p class="card-desc"><strong>Day 1 - Day 7</strong><br>理解训练入口、机器人模型、观测、动作空间、PD 控制、奖励函数与 Episode 生命周期。</p>
      <div class="card-keywords">Observation, Action, PD Controller, Reward</div>
    </a>

    <!-- 卡片 3：第二阶段 -->
    <a href="./Day8" class="grid-item">
      <div class="section-num">STAGE II</div>
      <h3 class="card-title">第二阶段：RSL-RL 与 PPO</h3>
      <p class="card-desc"><strong>Day 8 - Day 11</strong><br>深入 Rollout 机制、GAE、PPO 更新流程、Actor-Critic 网络结构及 LSTM。</p>
      <div class="card-keywords">Rollout, GAE, Actor-Critic, LSTM</div>
    </a>

    <!-- 卡片 4：第三阶段 -->
    <a href="./Day12" class="grid-item">
      <div class="section-num">STAGE III</div>
      <h3 class="card-title">第三阶段：自主修改与实验</h3>
      <p class="card-desc"><strong>Day 12 - Day 13</strong><br>实验设计与 Baseline，进行单因素修改实验（如 Reward 或 Domain Randomization）并对比。</p>
      <div class="card-keywords">Baseline, Domain Randomization</div>
    </a>

    <!-- 卡片 5：第四阶段 -->
    <a href="./Day14" class="grid-item">
      <div class="section-num">STAGE IV</div>
      <h3 class="card-title">第四阶段：Sim2Sim 与部署</h3>
      <p class="card-desc"><strong>Day 14 - Day 16</strong><br>模型加载与推理，MuJoCo Sim2Sim 环境运行，以及 Observation/Action 接口对齐。</p>
      <div class="card-keywords">Play, Sim2Sim, MuJoCo, 接口对齐</div>
    </a>

    <!-- 卡片 6：第五阶段 -->
    <a href="./Day17" class="grid-item">
      <div class="section-num">STAGE V</div>
      <h3 class="card-title">第五阶段：项目整理与复盘</h3>
      <p class="card-desc"><strong>Day 17 - Day 19</strong><br>整体架构复盘，关键代码与实验总结，完成最终项目文档与 GitHub 整理。</p>
      <div class="card-keywords">架构图, 复盘, 项目文档</div>
    </a>

  </div>

  <!-- 作者信息 -->
  <div class="author-card">
    <h3 class="card-title">作者信息 / About the Author</h3>
    <div class="author-grid">
      <div class="author-label">姓名：</div>
      <div class="author-value">田淞文</div>
      <div class="author-label">单位：</div>
      <div class="author-value">西安交通大学 / 数学与统计学院</div>
      <div class="author-label">GitHub：</div>
      <div class="author-value"><a href="https://github.com/SHini-CHi33" target="_blank">SHini-CHi33</a></div>
      <div class="author-label">兴趣领域：</div>
      <div class="author-value">Embodied AI & AI Agent & Reinforcement Learning</div>
    </div>
  </div>

  <!-- 页脚 -->
  <div class="custom-footer">
    © 2026 田淞文 · Unitree 机器人强化学习源码精读与学习平台<br>
    本笔记仅供学术交流使用 | Last Updated: November 2026
  </div>

</div>