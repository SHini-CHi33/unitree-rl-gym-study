<style>
/* 首页专属样式 */
.index-page {
  font-family: "Times New Roman", "SimSun", "Songti SC", serif;
}
.index-page .header-container {
  text-align: center;
  margin-bottom: 3rem;
  padding-top: 2rem;
}
.index-page .header-small {
  font-family: sans-serif;
  letter-spacing: 2px;
  font-size: 0.85rem;
  color: var(--gray);
  text-transform: uppercase;
  margin-bottom: 1rem;
}
.index-page .header-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  color: var(--dark);
}
.index-page .header-subtitle {
  font-size: 1.3rem;
  font-style: italic;
  color: #f97316;
  margin-bottom: 1.5rem;
}
.index-page .header-author {
  font-size: 1rem;
  color: var(--darkgray);
}
.index-page .about-card, .index-page .author-card {
  background: rgba(128, 128, 128, 0.05);
  border: 1px solid var(--lightgray);
  border-radius: 8px;
  padding: 2rem;
  margin-bottom: 3rem;
}
.index-page .about-card {
  border-left: 4px solid #f97316;
}
.index-page .card-title-accent {
  color: #f97316;
  font-family: sans-serif;
  font-weight: 600;
  margin-bottom: 1rem;
}
.index-page .card-title {
  font-family: sans-serif;
  font-weight: 600;
  margin-bottom: 1.5rem;
  color: var(--dark);
}
.index-page .tags-container {
  margin-top: 1.5rem;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.index-page .tag {
  font-family: sans-serif;
  font-size: 0.85rem;
  padding: 4px 12px;
  background: rgba(249, 115, 22, 0.1);
  color: #f97316;
  border-radius: 4px;
  border: 1px solid rgba(249, 115, 22, 0.3);
}
.index-page .contents-title {
  margin-bottom: 1.5rem;
  border-bottom: 2px solid #f97316;
  padding-bottom: 10px;
}
.index-page .contents-title h2 {
  font-size: 1.8rem;
  font-weight: 600;
  margin: 0;
}
.index-page .grid-container {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.5rem;
  margin-bottom: 3rem;
}
@media (max-width: 800px) {
  .index-page .grid-container {
    grid-template-columns: 1fr;
  }
}
.index-page .grid-item {
  display: block;
  background: rgba(128, 128, 128, 0.05);
  border: 1px solid var(--lightgray);
  border-left: 4px solid #f97316;
  border-radius: 4px;
  padding: 1.5rem;
  text-decoration: none;
  color: inherit;
  transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
}
.index-page .grid-item:hover {
  transform: translateY(-3px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  border-color: #f97316;
}
.index-page .section-num {
  font-family: monospace;
  font-size: 0.8rem;
  color: #f97316;
  margin-bottom: 0.5rem;
  font-weight: 600;
}
.index-page .card-desc {
  font-size: 0.95rem;
  color: var(--darkgray);
  line-height: 1.6;
  margin-bottom: 1rem;
}
.index-page .card-keywords {
  font-size: 0.85rem;
  color: var(--gray);
  font-style: italic;
}
.index-page .author-grid {
  display: grid;
  grid-template-columns: 100px 1fr 100px 1fr;
  gap: 1rem;
  font-size: 0.95rem;
}
@media (max-width: 600px) {
  .index-page .author-grid {
    grid-template-columns: 80px 1fr;
  }
}
.index-page .author-label {
  color: #f97316;
  font-weight: 600;
}
.index-page .author-value {
  color: var(--darkgray);
}
.index-page .author-value a {
  color: var(--darkgray);
  text-decoration: underline;
}
.index-page .author-value a:hover {
  color: #f97316;
}
.index-page .custom-footer {
  text-align: center;
  margin-top: 4rem;
  padding-top: 2rem;
  border-top: 1px solid var(--lightgray);
  color: var(--gray);
  font-size: 0.85rem;
  font-family: sans-serif;
  line-height: 1.8;
}
</style>

<div class="index-page">
<div class="header-container">
<div class="header-small">XI'AN JIAOTONG UNIVERSITY · SCHOOL OF MATHEMATICS AND STATISTICS</div>
<h1 class="header-title">Unitree 机器人强化学习源码精读与学习平台</h1>
<div class="header-subtitle">Research Platform on Unitree RL-Gym</div>
<div class="header-author">作者：田淞文 | 西安交通大学 数学与统计学院</div>
</div>
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
<div class="author-card">
<h3 class="card-title">作者信息 / About the Author</h3>
<div class="author-grid">
<div class="author-label">姓名：</div>
<div class="author-value">田淞文</div>
<div class="author-label">单位：</div>
<div class="author-value">西安交通大学 / 数学与统计学院</div>
<div class="author-label">GitHub：</div>
<div class="author-value"><a href="https://github.com/SHini-CHi33" target="_blank" rel="noopener noreferrer">SHini-CHi33</a></div>
<div class="author-label">兴趣领域：</div>
<div class="author-value">Embodied AI & AI Agent & Reinforcement Learning</div>
</div>
</div>
<div class="custom-footer">
© 2026 田淞文 · Unitree 机器人强化学习源码精读与学习平台<br>
本笔记仅供学术交流使用 | Last Updated: November 2026
</div>
</div>