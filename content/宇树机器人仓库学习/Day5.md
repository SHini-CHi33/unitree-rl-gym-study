# 今日阅读任务

今天是整个学习过程中一个非常重要的节点。

前几天我们已经追踪了：

- Day 1：训练入口、环境创建、OnPolicyRunner。
- Day 2：G1 机器人关节、初始姿态、控制参数。
- Day 3：Observation 的构成与 47 维输入。
- Day 4：12 维 Action 如何变成目标关节角度。

今天要把这些内容连接起来。

核心问题：为什么 Policy 只输出一次 Action，机器人却需要执行四次物理仿真？这四次仿真中，PD 控制器究竟在做什么？

这次我特别结合你之前关于 5ms、20ms 和 Reward 归属的讨论来讲。

---

## 一、先确认今天要阅读的代码

官方仓库：

Unitree RL Gym 官方仓库

今天只需要打开三个文件：

| 文件 | 重点 |
|---|---|
| `legged_gym/envs/g1/g1_config.py` | `G1RoughCfg.control` |
| `legged_gym/envs/base/legged_robot_config.py` | `LeggedRobotCfg.sim` |
| `legged_gym/envs/base/legged_robot.py` | `step()`、`_compute_torques()`、`post_physics_step()` |

本次核对的是官方仓库当前 main 分支。G1 配置没有单独覆盖 `sim.dt`，因此继承基础配置中的仿真时间步。你本地如果修改过配置，应以实际加载的 `sim_params` 为准。

---

## 二、第一部分：找到两个最重要的参数

先打开：

`legged_gym/envs/g1/g1_config.py`

找到：

```python
class control(LeggedRobotCfg.control):
    control_type = 'P'
    stiffness = {
        'hip_yaw': 100,
        'hip_roll': 100,
        'hip_pitch': 100,
        'knee': 150,
        'ankle': 40,
    }
    damping = {
        'hip_yaw': 2,
        'hip_roll': 2,
        'hip_pitch': 2,
        'knee': 4,
        'ankle': 2,
    }
    action_scale = 0.25
    decimation = 4
```

今天重点关注：

```python
decimation = 4
```

再打开：

`legged_gym/envs/base/legged_robot_config.py`

找到：

```python
class sim:
    dt = 0.005
    substeps = 1
```

这里：

$$
dt_{\text{sim}}=0.005s=5ms
$$

而 G1 设置：

$$
decimation=4
$$

因此：

$$
dt_{\text{policy}}
=
decimation\times dt_{\text{sim}}
$$

$$
=4\times0.005
$$

$$
\boxed{dt_{\text{policy}}=0.02s=20ms}
$$

这就是为什么：

- 物理仿真每隔 5ms 更新一次。
- Policy 每隔 20ms 才重新计算一次 Action。

官方 G1 配置也明确把 `decimation` 注释为仿真时间步与策略时间步之间的控制更新次数。

### G1 的两种时间尺度

| 物理仿真频率      | 200 Hz | 每 5ms 一次  |
| ----------- | -----: | --------- |
| Policy 决策频率 |  50 Hz | 每 20ms 一次 |

这里的 Hz 是仿真时间中的更新频率，不代表实际计算机的墙钟运行速度。

---

## 第二部分：追踪 step(self, actions)

这是今天最重要的代码。

打开：

`legged_gym/envs/base/legged_robot.py`

找到：

```python
def step(self, actions):
```

官方代码的核心结构如下：

```python
clip_actions = self.cfg.normalization.clip_actions
self.actions = torch.clip(
    actions,
    -clip_actions,
    clip_actions
).to(self.device)

self.render()

for _ in range(self.cfg.control.decimation):
    self.torques = self._compute_torques(
        self.actions
    ).view(self.torques.shape)

    self.gym.set_dof_actuation_force_tensor(
        self.sim,
        gymtorch.unwrap_tensor(self.torques)
    )

    self.gym.simulate(self.sim)

    if self.device == 'cpu':
        self.gym.fetch_results(self.sim, True)

    self.gym.refresh_dof_state_tensor(self.sim)

self.post_physics_step()
```

这里请特别注意循环内部的三行：

```python
self.torques = self._compute_torques(self.actions)
self.gym.simulate(self.sim)
self.gym.refresh_dof_state_tensor(self.sim)
```

它们的顺序非常重要。

我们逐行理解。

### 1. Policy 输出一次 Action

```python
self.actions = torch.clip(actions, ...)
```

这里将 Policy 输出保存到：

```python
self.actions
```

例如：

$$
a_t=
[0.1,0.2,-0.1,\ldots]
$$

接下来进入四次循环。

### 2. 第一次物理仿真

```python
self.torques = self._compute_torques(self.actions)
```

根据：

- 当前 Action；
- 当前关节角度；
- 当前关节速度；

计算 PD 力矩。

然后：

```python
self.gym.simulate(self.sim)
```

物理引擎推进 5ms。

机器人状态从：

$$
(q_t,\dot q_t)
$$

变成：

$$
(q_{t+1/4},\dot q_{t+1/4})
$$

这里的 $$t+1/4$$ 只是表示 20ms 策略周期内的第一个物理子步，并不是 PPO 的新时间步。

然后：

```python
self.gym.refresh_dof_state_tensor(self.sim)
```

刷新关节状态 Tensor。

这样下一次 PD 计算就能使用更新后的关节位置和速度。

### 3. 第二次、第三次、第四次物理仿真

接下来重复同样的过程。

最重要的是：

四次循环中，`self.actions` 没有重新赋值，因此 Policy Action 保持不变。

但是关节状态在不断更新。

因此，PD 控制器每次都可以根据新的关节状态重新计算力矩。

---

## 第三部分：把四次循环真正展开

我们不再只看抽象代码，直接把它展开成时间轴。

假设 Policy 在 $$t=0$$ 时刻输出：

$$
a_0
$$

那么整个过程如下：

| 阶段        | 时间                          | 内容                                                                               |
| --------- | --------------------------- | -------------------------------------------------------------------------------- |
| Policy 决策 | $t=0\text{ ms}$             | Actor 输出 $a_0$，读取当前 Observation，产生一次 12 维 Action                                 |
| 第 1 次     | $0\rightarrow5\text{ ms}$   | 当前关节状态 $q_0$、$\dot q_0$，计算 $\tau_1$，物理仿真 5ms                                     |
| 第 2 次     | $5\rightarrow10\text{ ms}$  | 更新后的 $q_1$、$\dot q_1$，计算 $\tau_2$，物理仿真 5ms                                       |
| 第 3 次     | $10\rightarrow15\text{ ms}$ | 更新后的 $q_2$、$\dot q_2$，计算 $\tau_3$，物理仿真 5ms                                       |
| 第 4 次     | $15\rightarrow20\text{ ms}$ | 更新后的 $q_3$、$\dot q_3$，计算 $\tau_4$，物理仿真 5ms                                       |
| 下一次策略决策   | $t=20\text{ ms}$            | 返回新 Observation，环境完成 `post_physics_step()`，随后 PPO 才能基于新 Observation 计算下一次 Action |

你可以把它理解成：

```text
Action a0 ──────────────────────────────
          |        |        |        |
          v        v        v        v
        PD 1     PD 2     PD 3     PD 4
          |        |        |        |
        Sim 1    Sim 2    Sim 3    Sim 4
          |        |        |        |
          5ms      5ms      5ms      5ms
          └────────┴────────┴────────┘
                       |
                     20ms
                       |
                       v
                  新 Observation
                       |
                       v
                 Policy 输出 a1
```

这里有两个需要你牢牢记住的区别：

- Action 在四个子步中保持不变。
- PD 力矩不是保持不变的，而是每个子步重新计算。

这正是高层策略与底层控制器的分工。

---

## 第四部分：为什么需要每 5ms 重新计算 PD 力矩？

回顾 Day 4 学习的公式：

$$
\tau_t=
K_p(q_{\text{target}}-q_t)-K_d\dot q_t
$$

其中：

$$
q_{\text{target}}
=
q_{\text{default}}+0.25a_t
$$

现在假设：

- Policy 输出的 Action 不变。
- 默认关节角度不变。
- 目标角度不变。

但机器人实际关节状态在不断变化。

例如：

$$
q_{\text{target}}=0.4
$$

假设同一目标角度下的状态变化：

| 时刻 | 实际关节角度 | 角度误差 |
|---:|---:|---:|
| 0ms | 0.30 | 0.10 |
| 5ms | 0.33 | 0.07 |
| 10ms | 0.36 | 0.04 |
| 15ms | 0.38 | 0.02 |

> 示意数据，假设目标角度为 0.4 rad，不代表实际 G1 轨迹。

假设：

$$
K_p=100
$$

那么只看 P 项：

$$
\tau_{P,0}=100(0.4-0.3)=10
$$

$$
\tau_{P,1}=100(0.4-0.33)=7
$$

$$
\tau_{P,2}=100(0.4-0.36)=4
$$

$$
\tau_{P,3}=100(0.4-0.38)=2
$$

你会发现：

虽然 Policy 一直没有改变 Action，但控制器可以根据机器人实际运动状态不断调整力矩。

这就是为什么我们需要把两个频率分开：

- Policy 负责较高层的运动决策。
- PD 控制器负责较高频率的关节跟踪。

如果每次 PD 计算都必须重新运行神经网络，就会增加策略推理的计算负担，而且没有必要。

---

## 第五部分：20ms 结束后，Reward 是怎么计算的？

这一部分和你之前关于 PPO 收敛、Action 与 Reward 对应关系的讨论直接相关。

继续阅读：

`legged_gym/envs/base/legged_robot.py`

找到：

```python
def post_physics_step(self):
```

核心逻辑：

```python
self.gym.refresh_actor_root_state_tensor(self.sim)
self.gym.refresh_net_contact_force_tensor(self.sim)

self.episode_length_buf += 1

self._post_physics_step_callback()

self.check_termination()
self.compute_reward()

env_ids = self.reset_buf.nonzero(
    as_tuple=False).flatten()

self.reset_idx(env_ids)
self.compute_observations()
```

这说明：

四次物理仿真结束后，环境才进入 `post_physics_step()`。

在这里：

1. 刷新机器人基座状态和接触力。
2. 更新速度、姿态等状态量。
3. 检查机器人是否终止。
4. 计算 Reward。
5. 对终止环境进行 Reset。
6. 生成新的 Observation。

然后 `step()` 返回：

```python
return (
    self.obs_buf,
    self.privileged_obs_buf,
    self.rew_buf,
    self.reset_buf,
    self.extras)
```

##### 🏭 1. `post_physics_step()` 到底在干什么？

这就像是下班前的盘点：

- **刷新数据**：`refresh_...`。物理引擎刚跑完，但底层的 C++ 数据还没同步到 Python 里。这两句把最新的基座位置（机器人是不是摔了）和接触力（脚是不是踩到地了）拉取出来。
- **计时**：`episode_length_buf += 1`。这只狗又活过了 20ms。
- **判决**：`check_termination()`。检查机器人是不是摔死了？还是超时了？如果死了，就在 `reset_buf`（结束标志）里打个勾。
- **发钱（核心）**：`compute_reward()`。这就是你苦苦寻找的 Reward 产生的地方！
- **重启**：`reset_idx(env_ids)`。谁死了（`reset_buf` 为 True），就把谁原地复活（重置回起点）。
- **拍照**：`compute_observations()`。把这些狗复活后、或者走了一步后的最新画面，拍成下一帧的 Observation。

##### 🎁 2. 为什么返回这一串东西？（这是给 PPO 的“大礼包”）

`step()` 最后返回了这 5 个东西：

```python
return (obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras)
```

这套组合拳，完美对应了 PPO 需要的：

$$
(s_{t+1}, r_t, done_t)
$$

具体对应关系如下：

- `obs_buf` 和 `privileged_obs_buf`（$$s_{t+1}$$）：这是机器人的最新状态。PPO 拿到它，下一轮就可以继续让 Actor 输出动作了。
	（至于为什么有两个变量储存 $Next\_obs$ ，回到Day3`privileged_obs_buf`（50维）：给 `Critic` ; `obs_buf`（47维）：平民视角（给`Actor`）

- `rew_buf`（$$r_t$$）：这就是奖励！是刚才 `compute_reward()` 算出来的。PPO 会把它存进 Rollout Buffer 里。
- `reset_buf`（$$done_t$$）：这是回合结束标志。PPO 看到它是 True，就知道“这狗摔死了，算 GAE 的时候千万别把下一个回合的奖励算进来”。
- `extras`：这是个杂货铺，里面装着诸如“速度追踪误差是多少”、“脚滑扣了多少分”的统计数据，主要是给 TensorBoard 画图用的。

##### 💰 3. Reward 到底在哪？为什么乘以 `self.dt`？

Reward 就在 `self.compute_reward()` 这个函数里生成的。

在 `_prepare_reward_function()` 里，代码干了这么一件事：

```python
self.reward_scales[key] *= self.dt
```

其中：

$$
self.dt = decimation \times sim\_params.dt = 4 \times 0.005 = 0.02s
$$

为什么要乘 0.02 秒（20ms）？这是极其精妙的物理积分思想！

- **如果按物理步算**：假设每 5ms 算一次奖励，那 1 秒钟要算 200 次奖励。如果每步奖励是 1，1 秒就是 200。这会导致 PPO 的 Q 值爆炸。
- **折算成速率**：乘以 `self.dt`，相当于把“点”累加变成了“线积分”。它告诉网络：“你在这个动作上保持的时间越长（20ms），你拿到的总奖励（面积）才越大。” 如果把reward理解为保持某种状态，一秒平均可以获得 $reward = 1分$，这样，我们就可以把reward拆解到时间层面，20ms的控制周期中也可以计算 reward，不管你把 `decimation` 设成 4 还是 10，奖励的累积尺度永远是按“策略控制周期（Policy Step）”来算的。这让超参数的调节变得极其稳定。

对于一次 `env.step(actions)`：

- Policy 产生一次 Action。
- 环境执行四次物理仿真。
- 环境在控制周期结束时计算一次 Reward。
- 返回一次新的 Observation。

因此，一次环境交互对应 PPO Rollout 中的一条 Transition。

---

# 今日问题

### 问题 1

在 `step(actions)` 的四次循环中：

```python
self.actions
```

始终不变，但：

```python
self.torques
```

为什么可以发生变化？

请结合 `_compute_torques()` 的公式解释。

答：`self.torques = self._compute_torques(self.actions)`, `self.torques`是根据动作和关节角度计算的PD力矩，由于要让$Policy$ $20ms$输出一次策略，而$5ms$计算一次物理仿真，所以在 $4$ 步 $5ms$ 的物理仿真推导中，不改变$Action$，但是由于每一步计算PD力矩后关节会持续朝目标角度转动，所以每一次需要根据关节位置与目标位置的差距计算新的PD力矩，从而实现高频控制

#### 💡GPT的修正与补充

这一题你已经真正理解了。

这里再补充一个严谨性问题：

你说关节会持续朝目标角度转动，但实际上并不保证每一步都单调接近目标角度。重力、惯性、地面接触等因素都可能影响运动方向。

更准确的表述是：

**机器人状态在物理仿真中不断变化，所以 PD 控制器需要根据实时状态重新计算反馈力矩。**

### 问题 2

为什么不让 Policy 每隔 5ms 就重新计算一次 Action，而是每隔 20ms 才计算一次？

从计算开销、底层控制和运动连续性三个角度考虑。

答：5ms计算一次action的话，首先，计算开销是原来的四倍，其次，从底层控制和运动连续性角度来看，20ms计算一次action虽然在运动控制连续性上不如5ms，但是对于机器人行走这类任务来说绰绰有余

你的核心方向正确，但有两个地方值得修正。

#### 💡GPT的修正与补充

**1. 计算开销：正确，但要明确是策略推理开销。**

但注意，不意味着整个仿真程序的总计算量一定增加四倍，因为物理仿真本来就已经是 200Hz。

**2. 底层控制：你需要理解一个关键区别。**

当前系统并不是 20ms 才控制一次机器人。

它实际上是：

- Policy：50Hz，产生目标控制指令。
    
- PD：200Hz，根据实时关节状态计算力矩。
    
- Physics：200Hz，推进动力学状态。
    

因此，即使 Policy 20ms 才更新一次，PD 仍然会每 5ms 进行反馈控制。

**3. 运动连续性：这里需要修正你的理解。**

不能简单地说：

> 20ms 的运动连续性不如 5ms。

因为高频更新 Policy 并不一定让机器人运动更平滑。

反而可能出现：

- 策略输出频繁变化；
    
- 目标关节角不断变化；
    
- 关节控制指令出现高频波动。
    

当前 20ms 的设计，相当于在一个策略周期内保持目标指令，再由 200Hz 的 PD 控制器持续跟踪。

这种分层控制方式可以在计算开销与控制响应之间取得平衡。

最后，你说 20ms 对行走任务来说绰绰有余，这在当前配置下是一个合理的工程判断，但不能推广到所有机器人任务。更高动态、快速接触变化的任务可能需要不同的策略频率。

**本题建议记忆：Policy 负责高层决策，PD 负责高频反馈控制。**

### 问题 3

四次物理仿真结束后，环境才调用：

```python
self.compute_reward()
```

为什么一次 `env.step(actions)` 只会产生一条 PPO Transition，而不是四条？

答 ：我们设定策略20ms计算一次action，当然输入的是输出动作后20ms时的观测以及对应的reward，所以四次物理仿真结束后，环境才输出一条transition$(s_{t+1}, r_t, done_t)$

#### 💡GPT的修正与补充

还有一个细节：

**Unitree 这里使用的是 Vectorized Environment。**

假设同时有 4096 个环境。

一次 `env.step(actions)` 实际上会返回一批数据：

```
4096 个环境
     |
     | 每个环境执行 4 次物理仿真
     v
4096 组新 Observation
4096 个 Reward
4096 个 Done
     |
     v
4096 条 Transition
```

因此，更严谨的表述是：

> 一次 `env.step(actions)` 对每个并行环境产生一条 Transition，而不是整批环境只产生一条 Transition。四次物理子步都封装在一次环境交互内部，没有分别交给 PPO 作为独立的四条 Transition。

你这题差的主要是对 Transition 数据结构的完整理解。
### 问题 4

如果将：

```python
decimation = 4
```

改为：

```python
decimation = 8
```

同时保持：

```python
sim.dt = 0.005
```

那么：

- Policy 的决策频率会发生什么变化？
- PD 控制器的计算频率会发生什么变化？
- 每条 PPO Transition 对应的仿真时间会发生什么变化？

先推导，不要实际修改训练参数。

答：Policy 决策时间 $8 \times 5 = 40ms$,决策频率 $\frac{1}{40 \times 10^{-3}} = 25$,决策频率降低，PD控制频率不变，仍是一次5ms，PPO Transition对应仿真时间由20ms变到40ms

---

# 今天的实践练习

今天不需要重新训练机器人。

只需要完成一个小练习：自己打印并验证 Policy 时间步和物理仿真时间步。

在环境创建完成后，可以查看：

```python
print("sim_dt:", env.sim_params.dt)
print("decimation:", env.cfg.control.decimation)
print("policy_dt:", env.dt)
print("policy_frequency:", 1.0 / env.dt)
```

默认配置预期：

```text
sim_dt: 0.005
decimation: 4
policy_dt: 0.02
policy_frequency: 50.0
```

然后自己写一个简单计算：

```python
sim_dt = 0.005
decimation = 4
policy_dt = sim_dt * decimation

print("Physics steps:", decimation)
print("Physics dt:", sim_dt)
print("Policy dt:", policy_dt)
```

再把 `decimation` 临时改成 8，只做手算，不要真正启动训练：

$$
dt_{\text{policy}}=8\times0.005=0.04s
$$

得到：

$$
f_{\text{policy}}=25Hz
$$

这就完成了今天的实践。