# 今日阅读任务

今天我们重点回答一个问题：

PPO Actor 输出一个 12 维向量，这个向量是关节角度、关节角速度，还是关节力矩？

## 一、今天需要阅读的文件和函数

今天主要看两个文件。

| 文件                                     | 函数 / 配置                           | 作用             |
| -------------------------------------- | --------------------------------- | -------------- |
| `legged_gym/envs/g1/g1_config.py`      | `G1RoughCfg.control`              | Action 控制参数    |
| `legged_gym/envs/g1/g1_config.py`      | `G1RoughCfg.env`                  | Action 维度      |
| `legged_gym/envs/base/legged_robot.py` | `step(self, actions)`             | 接收策略输出         |
| `legged_gym/envs/base/legged_robot.py` | `_compute_torques(self, actions)` | Action 转换为关节力矩 |
| `legged_gym/envs/base/legged_robot.py` | `_init_buffers()`                 | 默认关节角和关节顺序     |

官方源码：

- `g1_config.py`
- `legged_robot.py`

本次核对的是官方仓库 main 分支。

版本提醒：官方安装文档推荐 PyTorch 2.3.1、Isaac Gym Preview 4 和 RSL-RL v1.0.2。由于 RSL-RL 是独立依赖，策略网络部分应以实际安装版本为准。

## 二、第一部分：先找到 Action 的维度

打开 `g1_config.py`。

找到：

```python
class env(LeggedRobotCfg.env):
    num_observations = 47
    num_privileged_obs = 50
    num_actions = 12
```

这里：

$$
a_t\in\mathbb{R}^{12}
$$

代表每次策略决策输出 12 个连续数值。

为什么是 12？

因为当前 G1 使用的是 12 个可控关节：

- 左腿 6 个关节。
- 右腿 6 个关节。

所以每一个 Action 分量，都需要对应一个可控关节。

但需要注意：

12 维 Action 并不意味着策略直接输出 12 个关节的实际角度。

它输出的是一个控制指令，后面还需要经过转换。

另外，当前 G1 配置使用：

```python
class runner(LeggedRobotCfgPPO.runner):
    policy_class_name = "ActorCriticRecurrent"
```

并且配置了 LSTM。

因此，当前策略实际上是带有循环状态的 Actor-Critic，而不只是一个简单的前馈 MLP。不过这不改变今天研究的重点：最终送给环境的仍然是 12 维 Action。

## 三、第二部分：Action 到底代表什么？

打开 `g1_config.py`，找到：

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

今天最重要的是这两个参数：

```python
control_type = 'P'
action_scale = 0.25
```

先看注释：

```python
# action scale: target angle = actionScale * action + defaultAngle
```

它已经告诉我们 Action 的真正含义了。

当前 G1 使用的是：

基于默认关节角度的目标角度偏移量。

数学表达式：

$$
\boxed{
q_{\text{target}}
=
q_{\text{default}}+0.25a_t
}
$$

其中：

- $$q_{\text{target}}$$：目标关节角度。
- $$q_{\text{default}}$$：默认关节角度。
- $$a_t$$：Policy 输出的 Action。
- $$0.25$$：Action 缩放系数。

注意，关节角度使用 rad（弧度），因此这里的 0.25 可以理解为每单位 Action 对应 0.25 rad 的目标角度偏移。

### 举个例子

假设左膝关节：

$$
q_{\text{default}}=0.3
$$

Policy 输出：

$$
a_t=0.4
$$

那么：

$$
q_{\text{target}}
=0.3+0.25\times0.4
$$

得到：

$$
\boxed{q_{\text{target}}=0.4\text{ rad}}
$$

如果 Policy 输出：

$$
a_t=0
$$

则：

$$
q_{\text{target}}=0.3\text{ rad}
$$

如果：

$$
a_t=-0.4
$$

则：

$$
q_{\text{target}}=0.2\text{ rad}
$$

所以你可以这样理解：

Policy 并不是告诉机器人：

> 你的膝盖现在应该处于 0.4 rad。

而是在告诉机器人：

> 以默认关节姿态为参考，将目标角度向某个方向偏移一定幅度。

这也是为什么默认关节角度如此重要。

## 四、第三部分：找到 Action 转换的真正代码

现在进入：

`legged_gym/envs/base/legged_robot.py`

找到：

```python
def step(self, actions):
```

重点看：

```python
clip_actions = self.cfg.normalization.clip_actions
self.actions = torch.clip(
    actions,
    -clip_actions,
    clip_actions
).to(self.device)
```

这里完成了 Action 裁剪。

注意一个容易产生误解的地方：

不要默认认为环境会把 Policy 输出裁剪到 `[-1, 1]`。

当前官方基础配置中：

```python
clip_actions = 100.
```

所以这里的裁剪阈值是 ±100，而不是 ±1。

这里的动作的 `clip` 和 `PPO Loss` 中动作 `clip` 不同

虽然它们都叫 `clip`（截断/裁剪），但在整个 PPO 闭环里，它们**发生的位置、保护的对象、本质的原理完全不同**。

我们把这两个 `clip` 放在一起对比，你瞬间就能看透：

###### 🛡️  Action 裁剪（环境层的物理保护罩）

就是你截图中看到的：

python

```python

clip_actions = 100.
self.actions = torch.clip(actions, -clip_actions, clip_actions)

```

- **发生在哪**：在物理仿真器**即将执行动作之前**。
    
- **防什么**：防止神经网络突然“抽风”，输出了一个极其离谱的数值（比如 $9999$ 或 $NaN$）。如果不裁剪，底层 PD 控制器会计算出一个毁天灭地的巨大扭矩，Isaac Gym 的物理引擎瞬间就被算崩（产生 NaN）或者直接把机器人的腿甩飞。
    
- **为什么是 100**：这其实是个“兜底安全阀”。正常训练良好的网络，输出动作通常在 `[-3, 3]` 左右。设成 ±100 就像给汽车方向盘装了个限位器，平时根本碰不到，但遇到极值能保命。

---

接下来：

```python
for _ in range(self.cfg.control.decimation):
    self.torques = self._compute_torques(
        self.actions
    ).view(self.torques.shape)

    self.gym.set_dof_actuation_force_tensor(
        self.sim,
        gymtorch.unwrap_tensor(self.torques)
    )

    self.gym.simulate(self.sim)
```

这里发生了：

1. 接收 Action。
2. 裁剪 Action。
3. 将 Action 送入 `_compute_torques()`。
4. 得到关节力矩。
5. 将力矩交给物理引擎。

注意：这里的 `_compute_torques()` 每次物理仿真都会重新执行。

也就是说，Action 在一个策略周期内保持不变，但关节状态不断变化，控制器可以根据最新状态重新计算力矩。

具体的 PD 控制与时间关系，我们放到 Day 5。

## 五、第四部分：最重要的函数 _compute_torques()

这是今天的核心。

找到：

```python
def _compute_torques(self, actions):
```

首先：

```python
actions_scaled = actions * self.cfg.control.action_scale
control_type = self.cfg.control.control_type
```

然后：

```python
if control_type == "P":
    torques = (
        self.p_gains
        * (
            actions_scaled
            + self.default_dof_pos
            - self.dof_pos
        )
        - self.d_gains * self.dof_vel
    )
```

最后：

```python
return torch.clip(
    torques,
    -self.torque_limits,
    self.torque_limits)
```

来源：官方 `legged_robot.py` 的 `_compute_torques()`。

我们把公式拆开。

第一步：

$$
a_{\text{scaled}}=0.25a_t
$$

第二步：

$$
q_{\text{target}}
=
a_{\text{scaled}}+q_{\text{default}}
$$

第三步：

$$
\boxed{
\tau
=
K_p(q_{\text{target}}-q)
-
K_d\dot q
}
$$

其中：

- $$q$$：机器人当前真实关节角度。
- $$\dot q$$：机器人当前关节角速度。
- $$K_p$$：比例增益。
- $$K_d$$：微分增益。
- $$\tau$$：最终计算出的关节力矩。

这就是位置目标型 PD 控制。

注意配置中虽然写的是：

```python
control_type = 'P'
```

但代码实际上包含 P 和 D 两项，因此不能理解成纯 P 控制。

### 这里有一个非常重要的区别

我们现在已经有三个不同的量：

| 变量         | 含义             |
| ---------- | -------------- |
| `actions`  | Policy 输出的控制动作 |
| `q_target` | 期望关节角度         |
| `dof_pos`  | 机器人实际关节角度      |

三者不能混淆。

尤其是：

$$
q_{\text{target}}\neq q
$$

Policy 给出目标角度，并不意味着机器人能够立即到达目标角度。

机器人实际如何运动，还要受到：

- 关节动力学；
- PD 增益；
- 力矩限制；
- 重力；
- 地面接触；
- 机器人整体运动状态；

等因素影响。

**问：之前在Day2中我们好像没有D项，这里到底用的是哪一种？

你之前的印象是 `τ = Kp * (q_target - q)`，这叫**纯 P 控制（比例控制）**。  

但真实的代码里，多了一项 `- Kd * \dot{q}`，这叫**D 控制（微分控制）**。


**1. 前半部分：`Kp(q_target - q)`（P 项，比例控制 = 弹簧）**

- `q_target` 是目标角度，`q` 是当前角度。两者相减是误差。
    
- **意义**：这就好比绑了一根弹簧。差得越远，拉力（力矩）越大。**但只有弹簧的话，机器人会怎样？会疯狂震荡！** 因为它冲到目标角度时，因为惯性刹不住车，会冲过头，然后弹簧又把它往回拉，像帕金森一样疯狂抽搐。

我们在Day2主要是讨论了stiffness即$K_p$，类比弹簧的劲度系数

```python
stiffness = {
    'hip_yaw': 100,
    'hip_roll': 100,
    'hip_pitch': 100,
    'knee': 150,
    'ankle': 40,
}
```

**2. 后半部分：`- Kd\dot{q}`（D 项，微分控制 = 阻尼/刹车）**

- `\dot{q}` 是关节当前的**速度（角速度）**。
    
- **意义**：这是一个**阻尼器（刹车）**。只要机器人动得越快，这个项产生的反向力矩就越大。它强行阻止机器人“冲过头”。
    

###### 💡 总结：最后用的是哪一种？

**最后用的是完整的 PD 控制（图里的第三步）。**

- 如果你只用 `Kp`（纯比例）：机器人冲到目标角度会刹不住，疯狂震荡。
    
- 如果你只用 `Kd`（纯微分）：机器人根本到不了目标，因为它一动就被“刹车”按死了。
    
- **P + D 结合**：`Kp` 负责把它拉过去，`Kd` 负责在靠近目标时踩刹车，让它平滑、稳定地停在目标角度

###### 六、第五部分：Action 的顺序如何对应关节？

这里又回到了 Day 2 的讨论

在 `legged_robot.py` 的 `_create_envs()` 中：

```python
self.dof_names = self.gym.get_asset_dof_names(
    robot_asset)
self.num_dofs = len(self.dof_names)
```

而在 `_init_buffers()` 中：

```python
for i in range(self.num_dofs):
    name = self.dof_names[i]
    angle = self.cfg.init_state.default_joint_angles[name]
    self.default_dof_pos[i] = angle
```

这意味着：

Action 的第 i 维、当前关节角度的第 i 维、默认关节角度的第 i 维，都是按照同一个 DOF 顺序对应的。

例如：

```text
Action[0]  → DOF[0]
Action[1]  → DOF[1]
Action[2]  → DOF[2]
...
Action[11] → DOF[11]
```

不能想当然地认为 Action 的排列顺序就是 `default_joint_angles` 字典的书写顺序。

应该检查实际加载 URDF 后得到的：

```python
self.dof_names
```

这才是最可靠的对应依据。

## 七、今天的完整数据流

现在把整个过程串起来。

```text
             PPO Actor
                 |
                 | 12维连续动作
                 v
             actions
                 |
                 v
          step(actions)
                 |
                 v
             Action Clip
                 |
                 v
          actions * 0.25
                 |
                 v
       default_dof_pos + scaled_action
                 |
                 v
             q_target
                 |
                 v
       ┌─────────────────────┐
       │    PD Controller    │
       │                     │
       │  Kp(q_target - q)   │
       │      - Kd * q_dot   │
       └─────────────────────┘
                 |
                 v
               Torque
                 |
                 v
          Torque Limitation
                 |
                 v
           Isaac Gym
                 |
                 v
          Robot Movement
```

今天你应该真正建立这个认识：

Policy 输出的是目标关节角度的偏移指令，而不是直接输出机器人关节的实际角度，也不是直接输出关节力矩。

这是当前 G1 配置下的行为，因为它设置了 `control_type='P'`。

如果将来改成：

```python
control_type = 'T'
```

那么 `_compute_torques()` 会直接将缩放后的 Action 作为力矩输入。

因此，Action 的物理含义并不是 PPO 算法决定的，而是由策略输出维度和环境控制接口共同决定的。

# 什么是“环境接口（Environment Interface）”？

在你的代码里，`legged_gym` 这个环境继承自 OpenAI Gym 的标准接口。它主要包含两个函数：

1. `obs, reward, done, info = env.reset()` （重置环境）
    
2. **`obs, reward, done, info = env.step(action)`** （执行动作）
    

这里的 **`env.step(action)` 就是环境接口**。

它是连接“纯数学的神经网络”和“真实的物理引擎（Isaac Gym）”的中间桥梁。任何塞进 `step()` 里面的数字，都会经过这个接口的一顿“内部加工”（比如缩放、裁剪、PD 转换），最后变成物理引擎能懂的力
# 今日问题


### 问题 1：

为什么 Policy 输出的 Action 不是直接等于关节目标角度？

为什么需要：

$$
q_{\text{target}}
=
q_{\text{default}}+0.25a_t
$$

答：不是，由于神经网络对数字的敏感性，我们设定系数为0.25让网络学习，$4(q_{\text{target}}-q_{\text{default}}) = a_t$

#### 💡GPT的修正与补充

首先，你写的数学关系没有问题。

我们从：

$$
q_{\text{target}}=q_{\text{default}}+0.25a_t
$$

移项得到：

$$
a_t=4(q_{\text{target}}-q_{\text{default}})
$$

但是，你对为什么设置 `0.25` 的解释不够准确。

##### 真正的目的是什么？为什么不直接学习`q_target` 

**理论上完全可以，但在实际工程中，直接输出 `q_target` 会让神经网络“学废”，训练难度呈指数级上升。**

我们用“画蒙娜丽莎”**来打个比方，你就彻底明白了：

##### 🎨 1. 直接学 `q_target` = 让一个新手在白纸上从头画蒙娜丽莎

如果你让网络直接输出绝对角度（比如膝盖 1.2 弧度，髋骨 -0.5 弧度）。

- **初始化灾难**：网络刚初始化时，输出的数字是随机的（可能在 [-5, 5] 乱跳）。此时 `q_target` 完全偏离了人类正常的站立姿态，机器人在物理引擎里一开局就直接“原地劈叉骨折”，根本活不到拿奖励的那一刻。
    
- **探索空间太大**：每个关节的绝对角度范围都不一样（有的只有 ±0.5，有的有 ±2.5）。网络要在这个巨大的高维空间里，精准地找出一套能让机器人走路的完美绝对角度序列，就像大海捞针。
    

##### 🖍️ 2. `q_default + 0.25 * a` = 给蒙娜丽莎画好轮廓，让网络拿细笔去微调

工程师把机器人的“标准站立姿态”写死了，也就是 `q_default`。

- **超级先验（Prior）**：网络什么都不用干，只要输出全是 0，机器人就是完美站立的！这就保证了网络在训练初期，即使是个傻子，也不会马上摔倒。
    
- **局部微调（残差）**：网络不需要去算“膝盖到底是 1.2 度还是 1.3 度”，它只需要算“比默认角度多弯一点（+0.1）还是少弯一点（-0.1）”。
    
- **统一量纲（归一化）**：所有关节的 Action 都被统一到了 `[-1, 1]`（乘以 0.25 后变成 `[-0.25, 0.25]` 的偏移量）。神经网络的输出层极其喜欢这种统一的标准区间，这极大地降低了网络的拟合难度。
    

---

##### 🔍 3. 为什么 `action_scale` 必须是固定的 0.25？（不能被学习？）

如果 scale 太大，**你微小的策略改变，在物理世界就会变成毁灭性的动作突变**。网络根本不敢乱动，一动就死，这就导致它无法探索，最后只能学出一个“原地站桩”的保守策略。
**而 `0.25` 这个数字，是工程师做过无数次实验找出来的“甜点（Sweet Spot）”**。

假设 PPO 网络当前输出的均值 $μ=0.5$，标准差 $σ=0.3$（在训练初期探索幅度很大）。  
每次从分布里采样出来的值，可能落在 `[-0.4, 1.4]` 之间。

- 如果 `action_scale = 2.0`：采样到 `1.4` 时，目标角度偏移 = `2.8 rad`（约 160 度）。物理引擎里的机器人直接原地大腿骨折。
    
- 如果 `action_scale = 0.25`：采样到 `1.4` 时，目标角度偏移 = `0.35 rad`（约 20 度）。物理引擎里的机器人只是步子迈大了一点，虽然会踉跄，但不会瞬间死掉。

走路、跑步，是极其需要**微操**的。

- 如果 scale 很大，网络输出 0.01 对应 0.02 rad 的移动。网络就像在拿一根粗大的毛笔在画清明上河图，根本画不出细节。
    
- 如果 scale 是 0.25，网络需要输出 0.1 才能对应 0.025 rad 的移动。这就逼迫网络在输出的时候，必须精准地在 `0.1`、`0.2` 这种小数级别上做文章，从而实现了高分辨率的微操。

### 问题 2：

如果：

$$
a_t=0
$$

是否意味着：

$$
\tau=0
$$

为什么？

提示：看看 `_compute_torques()` 中还有哪些变量。

答：

并不意味着 $\tau = 0$，而是说明 $q_{\text{target}}=q_{\text{default}}$，而不是$q_{\text{target}}=q_{\text{dof\_pos}}$

代入得到：

$$
\boxed{
\tau=K_p(q_{\text{default}}-q)-K_d\dot q
}
$$

这里应该注意两个东西：

##### 第一：P 项仍然可能产生力矩

如果：

$$
q\neq q_{\text{default}}
$$

那么：

$$
K_p(q_{\text{default}}-q)\neq0
$$

控制器仍然会产生力矩，让关节尝试回到默认角度。

##### 第二：D 项也可能产生力矩

即使：

$$
q=q_{\text{default}}
$$

但如果：

$$
\dot q\neq0
$$

那么：

$$
-K_d\dot q\neq0
$$

控制器仍然会产生阻尼力矩。

### Action = 0 时的三种情况

| 当前状态 | P 项 | D 项 |
|---|---|---|
| 角度偏离，速度为零 | 非零 | 0 |
| 角度正确，速度非零 | 0 | 非零 |
| 角度正确，速度为零 | 0 | 0 |

最后一行是力矩为零的一个充分条件；实际也可能出现 P、D 两项恰好抵消的情况。


### 问题 3：

假设左膝关节：

$$
q_{\text{default}}=0.3
$$

Policy 输出：

$$
a_t=0.8
$$

那么目标关节角度是多少？

注意单位。

答：$$
q_{\text{target}}
=
q_{\text{default}}+0.25a_t
=
0.5\,rad
$$

### 问题 4：

如果把：

```python
control_type = 'P'
```

改成：

```python
control_type = 'T'
```

那么 Action 的物理含义发生了什么变化？

为什么不能继续把它解释为关节角度偏移？

答：网络输出（Action）：直接就是力矩（比如：输出 `10.5` 牛·米），由于环境接口以及reward的设置，导致策略网络不断改变参数矩阵，逐渐理解action输出应该是什么

#### 💡GPT的修正与补充

你已经抓住最重要的变化：

> P 模式下，Action 控制目标角度；T 模式下，Action 控制力矩。

但还需要深入一步。

我重新核对了官方仓库当前 main 分支的 `legged_robot.py`。

找到：

```python
def _compute_torques(self, actions):
    actions_scaled = (
        actions * self.cfg.control.action_scale
    )
    control_type = self.cfg.control.control_type

    if control_type == "P":
        torques = (
            self.p_gains
            * (
                actions_scaled
                + self.default_dof_pos
                - self.dof_pos
            )
            - self.d_gains * self.dof_vel
        )
    elif control_type == "V":
        ...
    elif control_type == "T":
        torques = actions_scaled

    return torch.clip(
        torques,
        -self.torque_limits,
        self.torque_limits
    )
```

来源：官方 `legged_robot.py`。

这里有一个非常关键的细节：

> T 模式仍然会乘以 `action_scale`(这里还是默认的0.25)，并且仍然会进行力矩限幅。

我们逐步看。

## P 模式：位置控制

```text
Policy Action
    |
    v
乘以 0.25，得到角度偏移
    |
    v
加上默认关节角度，得到目标角度
    |
    v
PD 控制器计算力矩
```

## T 模式：力矩控制

```text
Policy Action
    |
    v
乘以 action_scale
    |
    v
直接作为目标力矩
    |
    v
Torque Limits 限幅
```

### 举个例子

假设：

$$
a_t=10.5
$$

$$
action\_scale=0.25
$$

那么：

$$
\tau=10.5\times0.25
$$

$$
\boxed{\tau=2.625\text{ Nm}}
$$

这里假设没有触发力矩限幅。

注意：当前 G1 的 `0.25` 是按照 P 模式的目标角度偏移来设计的。如果直接切换成 T 模式而不调整 Scale，那么 `0.25` 会变成力矩缩放系数，物理意义完全不同。通常也需要重新训练策略，不能直接假设原来训练好的位置控制策略可以直接用于力矩控制。

## 两种模式对比

| 模式  | Action 经 `action_scale` 后的含义 | 后续处理                                                               | 最终输出         |
| --- | ---------------------------- | ------------------------------------------------------------------ | ------------ |
| `P` | 目标角度偏移量                      | 加上默认关节角，再由 PD 控制器计算（$\tau=K_p(q_{\mathrm{default}}-q)-K_d\dot{q}$） | 关节力矩（$\tau$） |
| `T` | 目标力矩缩放值                      | 直接作为力矩，再限幅                                                         | 关节力矩（$\tau$） |

###### 🚗 1. `control_type = 'P'`（Position 位置控制模式）

- **网络输出（Action）**：目标角度（比如：膝盖弯曲 30 度）。
    
- **底层代码执行**：引擎在内部自动开启**完整的 PD 控制器（包含我们刚才讲的 Kp 和 Kd）**，计算出力矩交给电机。
    
- **人话翻译**：你对司机说：“**请开到 60 迈**”。至于司机踩多少油门、踩多少刹车，由司机（底层 PD 控制器）自己决定。
    
- **行业现状**：这是宇树、波士顿动力等绝大多数足式机器人的**绝对主流**选择。因为底层有 Kp 和 Kd 兜底，即使网络输出有点瑕疵，电机也不会瞬间烧毁。
    

###### 🚀 2. `control_type = 'T'`（Torque 力矩控制模式）

- **网络输出（Action）**：直接就是力矩（比如：输出 `10.5` 牛·米）。
    
- **底层代码执行**：不经过任何 PD 计算，直接把 `10.5` 塞进物理引擎的发动机里。
    
- **人话翻译**：你对司机说：“**给我把油门踩到底 10.5 毫米**”。完全不顾车会不会撞墙。
    
- **行业现状**：极少有人直接在行走任务里用纯力矩控制。因为神经网络很难直接输出完美的力矩序列，太容易瞬间把电机烧了或者让机器人原地疯狂抽搐。
    

---

###### 💡 为什么说“Action 的物理含义不是 PPO 决定的”？

这句话是极其核心的工程哲学。

PPO 算法是个瞎子，它根本不知道机器人有腿。它只知道：“老板让我输出 12 个数字，只要这 12 个数字能拿到高分，我就拼命调参数。”

- 如果配置成 `P` 模式，这 12 个数字就是**目标角度**。
    
- 如果配置成 `T` 模式，这 12 个数字就是**力矩**。
    
- 如果配置成 `V` 模式（速度控制），这 12 个数字就是**目标角速度**。
    

**PPO 算法没有任何改变，它的网络结构依然是 47 进，12 出。但是底层环境（Env）会把这 12 个数字解释成不同的物理意义。**


虽然PPO此时并不知道这串数字的含义，但是，由于人类设定的reward以及环境接口对输出数字的处理方式，经过几百万次的试错，PPO 逐渐”理解“这段数字的含义。  
它虽然依然不知道“0.1”到底是 0.1 弧度，还是 0.1 牛·米，但它的大脑（神经网络参数）已经完全记住了：

> **“只要观测到这种状态，我输出 `[0.1, -0.2...]` 这个特定组合，物理引擎里的那个‘翻译官’就会帮我把腿摆好，我就能拿到高分！”**

只不过，计算机不是像人一样用语言去理解，而是用**参数矩阵的权重**去理解。

## **一个补充思考题**

在进入 Day 6 之前，我希望你再思考一个问题：

假设我们使用 T 模式：

```
control_type = 'T'
```

现在 PPO 输出：

$a_t=0$

那么：

$\tau=0$

既然没有力矩输入，机器人是不是就会完全静止？

注意考虑：

- 机器人本身的重力；
    
- 当前关节是否在运动；
    
- 地面接触力；
    
- 机器人本身的惯性。
    
答：显然，就像速度与加速度无关，机器人禁止与否和当前产生的力矩无关，还有机器人本身的重力；当前关节是否在运动； 地面接触力；机器人本身的惯性等因素

# 今天的动手实践

今天不需要训练，也不需要修改任何配置。

只做一个小实验：

自己找出左膝关节在 12 维 Action 中的位置，并计算一个目标角度。

如果已经能够创建环境，在环境创建完成后执行：

```python
# 找到左膝关节的索引
i = env.dof_names.index("left_knee_joint")

# 获取左膝默认角度
q_default = env.default_dof_pos[0, i].item()

# 假设 Policy 输出 0.4
action = 0.4

# 获取缩放系数
scale = env.cfg.control.action_scale

# 计算目标关节角度
q_target = q_default + scale * action

print("Joint index:", i)
print("Default angle:", q_default)
print("Action:", action)
print("Action scale:", scale)
print("Target angle:", q_target)
```

预期：

```text
Default angle: 0.3
Action: 0.4
Action scale: 0.25
Target angle: 0.4
```

如果暂时不想启动 Isaac Gym，也可以直接用公式手算。

第二个小任务：

自己画出以下三个变量的关系：

```text
actions
   |
   v
q_target
   |
   v
dof_pos
```

并在旁边写出它们分别代表什么。

## 十、今天的完成标准

完成以下四件事即可结束今天的学习：

- [ ] 能解释为什么 G1 的 Action 是 12 维。
- [ ] 能解释 `action_scale=0.25` 的实际物理意义。
- [ ] 能从 `_compute_torques()` 推导出目标关节角度。
- [ ] 能区分 Policy Action、目标关节角度和实际关节角度。

建议今天留下这条核心公式：

$$
\boxed{
q_{\text{target},t}
=
q_{\text{default}}+0.25a_t
}
$$

以及：

$$
\boxed{
\tau_t=
K_p(q_{\text{target},t}-q_t)
-K_d\dot q_t
}
$$