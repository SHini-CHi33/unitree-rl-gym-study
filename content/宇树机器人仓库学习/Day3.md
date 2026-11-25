# 欧拉角与四元数组

**欧拉角（Euler Angles），就是用三个独立的旋转角度，来描述一个物体在三维空间里的姿态。**

这是人类最容易理解、最直观的姿态表示方法。为了让你秒懂，我们想象一架飞机（或者你的宇树 H1 机器人），在它身上绑一个三维坐标系：

- **X 轴**：指向机头（正前方）
    
- **Y 轴**：指向机翼（正左方）
    
- **Z 轴**：指向机背（正上方）
    

所谓的欧拉角，就是把机器人的所有复杂旋转，拆解成围绕这三个轴的三个独立动作：

1. **Roll（横滚角 / 翻滚）**：绕 **X 轴**旋转。
    
    - **动作**：飞机向左或向右“歪翅膀”。
        
    - **机器人**：人形机器人往左边侧弯腰，或者四足狗往左翻滚摔倒。
        
2. **Pitch（俯仰角 / 点头）**：绕 **Y 轴**旋转。
    
    - **动作**：飞机机头向上抬起（爬升）或向下俯冲。
        
    - **机器人**：人形机器人“点头”、“鞠躬”，或者往前栽倒。
        
3. **Yaw（偏航角 / 摇头）**：绕 **Z 轴**旋转。
    
    - **动作**：飞机在平地上向左转或向右转。
        
    - **机器人**：你让狗“原地转圈”，或者人形机器人左右“摇头”。
        

---

### 🤔 既然这么直观，为什么机器人不用它，非要用四元数？

因为欧拉角有一个极其致命的物理学 Bug，叫 **“万向节死锁（Gimbal Lock）”**。具体可以看b站视频BV1Nr4y1j7kn来理解

简单的说就是，因为旋转就是本体乘一个矩阵变换，$R_x \times R_y \times R_z$ 这个顺序已经设定好了，是一次性的变换，我们先乘$R_x$，绕X轴旋转，再乘$R_y$，绕Y轴旋转，是有顺序的，而这个顺序导致了一件事，如果把旋转想象成一个连续的过程，先绕X轴转10°，再绕Y轴转90度，转过来后，Y轴不会带着X轴转动，而带着Z轴转动了，因为后面还要乘$R_z$，如果我们后面要再绕X转，需要再改Rx矩阵，而此时和绕Z轴旋转效果一样，导致丢失了自由度，本质上是由于欧拉旋转变换是单次变换且矩阵乘法不可交换，如果我们再乘一个$R_{x_2}$矩阵，将他变成本地坐标系，就可以解决这个问题，但是程序中不可能反复乘矩阵去调整，于是有了四元数等一些改进方法。


程序中，当 Pitch 接近 ±90 度时，计算 Roll 和 Yaw 的公式分母会变成 0，导致程序直接崩溃（或者疯狂输出无穷大），机器人在仿真里就会瞬间像发了疯一样疯狂抽搐。

---

### 💡 欧拉角在具身智能里的真实地位

既然有四元数，欧拉角是不是彻底没用了？  
**不是的。**

1. **给人类看**：你在看 TensorBoard 曲线，或者在查看机器人的遥测数据时，如果给你看四元数 `(0.707, 0.0, 0.707, 0.0)`，你根本不知道机器人现在是什么姿势。但如果告诉你 `Roll=0, Pitch=45, Yaw=90`，你脑子里立刻就有画面了。
    
2. **给简单任务用**：如果你只是控制一个在平地上慢慢走路的机器人，Pitch 永远不会超过 90 度，用欧拉角算起来不仅没问题，而且计算量极小。
    
3. **作为惩罚项**：在你的 H1 代码里，有一个奖励叫 `rew_ang_vel_xy`（惩罚机器人躯干前后左右剧烈摇晃），它其实针对的就是欧拉角里的 Roll 和 Pitch 的速度。
    

**总结一句话：**

- **欧拉角**：是给人看的“说明书”，直观，但有死角（死锁）。
    
- **四元数**：是给计算机算的“黑盒”，抽象，但天下无敌（无死角，算得快）。
    

# 今日阅读任务

今日主题：Observation 的构成与维度——PPO 的输入

今日目标：从 G1 环境代码中找到 Observation 的真实来源，逐项理解 47 维输入的物理含义，并弄清楚为什么还存在一个 50 维的 Privileged Observation。

前两天我们已经研究了：

- Day 1：训练入口、配置加载、环境创建、Runner 和 learn()。
- Day 2：G1 配置、机器人关节、初始状态、控制参数。

今天把这些内容和你之前学习的 PPO 联系起来。

你之前学过 CleanRL PPO，知道 Actor 根据状态输入生成动作：

$$
a_t\sim\pi_\theta(a_t|s_t)
$$

但现在有一个更具体的问题：

对于 G1 人形机器人，程序中的 $$s_t$$ 到底是什么？它从哪里来？

今天我们就追踪这个问题。

###### 一、今天需要阅读的三个位置

今天重点阅读以下文件：

| 文件 | 需要关注的内容 |
|---|---|
| `legged_gym/envs/g1/g1_env.py` | `compute_observations()` |
| `legged_gym/envs/g1/g1_env.py` | `_post_physics_step_callback()` |
| `legged_gym/envs/base/legged_robot.py` | `post_physics_step()`、`_init_buffers()` |

另外回看：

`legged_gym/envs/g1/g1_config.py`

重点找到：

```python
class env(LeggedRobotCfg.env):
    num_observations = 47
    num_privileged_obs = 50
    num_actions = 12
```

这些是今天要验证的三个数字。

官方源码：

- G1 环境 `g1_env.py`
- G1 配置 `g1_config.py`
- 基础环境 `legged_robot.py`

本次核对的是官方仓库 main 分支截至 2026-11-10 的代码。你的本地版本如果不同，应以本地实际代码为准。

###### 二、第一部分：Observation 是什么时候生成的？

先打开：

`legged_gym/envs/base/legged_robot.py`

找到：

```python
def step(self, actions):
```

重点看：

```python
self.actions = torch.clip(
    actions,
    -clip_actions,
    clip_actions
).to(self.device)

for _ in range(self.cfg.control.decimation):
    self.torques = self._compute_torques(
        self.actions
    ).view(self.torques.shape)

    self.gym.set_dof_actuation_force_tensor(
        self.sim,
        gymtorch.unwrap_tensor(self.torques)
    )

    self.gym.simulate(self.sim)

self.post_physics_step()
```

这里我们可以看到：

1. 接收 PPO 输出的 Action。
2. 对 Action 进行裁剪。
3. 使用控制器计算力矩。
4. 执行物理仿真。
5. 调用 `post_physics_step()`。

继续看：

```python
def post_physics_step(self):
```

里面有：

```python
self.gym.refresh_actor_root_state_tensor(self.sim)
self.gym.refresh_net_contact_force_tensor(self.sim)

self.episode_length_buf += 1

self.base_pos[:] = self.root_states[:, 0:3]
self.base_quat[:] = self.root_states[:, 3:7]

self.base_lin_vel[:] = quat_rotate_inverse(
    self.base_quat,
    self.root_states[:, 7:10]
)

self.base_ang_vel[:] = quat_rotate_inverse(
    self.base_quat,
    self.root_states[:, 10:13]
)

self.projected_gravity[:] = quat_rotate_inverse(
    self.base_quat,
    self.gravity_vec
)
```

然后：

```python
self._post_physics_step_callback()
self.check_termination()
self.compute_reward()

env_ids = self.reset_buf.nonzero(
    as_tuple=False).flatten()

self.reset_idx(env_ids)
self.compute_observations()
```

这里有一个很重要的认识：

Observation 不是 PPO 自己计算出来的，而是环境根据机器人当前的仿真状态整理出来的。

整个流程：

```text
PPO Actor
    |
    | 输出 Action
    v
LeggedRobot.step(actions)
    |
    v
执行 PD 控制
    |
    v
Isaac Gym 物理仿真
    |
    v
post_physics_step()
    |
    ├── 更新机器人状态
    |
    ├── 更新速度、姿态
    |
    ├── 更新 Gait Phase
    |
    ├── 检查终止条件
    |
    ├── 计算 Reward
    |
    v
G1Robot.compute_observations()
    |
    v
生成新的 Observation
    |
    v
返回 PPO Actor
```

注意：如果某些环境在这一步发生 Reset，返回的 Observation 也可能已经是重置后的状态，而不是终止前的状态。

源码参考：`LeggedRobot.step()` 和 `post_physics_step()`。

###### 三、第二部分：G1 的 Observation 到底包含什么？

今天最重要的函数：

`legged_gym/envs/g1/g1_env.py`

找到：

```python
def compute_observations(self):
```

你会看到：

```python
sin_phase = torch.sin(
    2 * np.pi * self.phase).unsqueeze(1)

cos_phase = torch.cos(
    2 * np.pi * self.phase).unsqueeze(1)

self.obs_buf = torch.cat((
    self.base_ang_vel * self.obs_scales.ang_vel,
    self.projected_gravity,
    self.commands[:, :3] * self.commands_scale,
    (self.dof_pos - self.default_dof_pos)
        * self.obs_scales.dof_pos,
    self.dof_vel * self.obs_scales.dof_vel,
    self.actions,
    sin_phase,
    cos_phase), dim=-1)
```

这就是 G1 的 Actor Observation 构造过程。

我们逐项拆解。

###### 1. base_ang_vel：机身角速度

```python
self.base_ang_vel * self.obs_scales.ang_vel
```

维度：$3$

代表机器人机身绕三个轴的角速度：

- X 轴：Roll 方向角速度。
- Y 轴：Pitch 方向角速度。
- Z 轴：Yaw 方向角速度。

注意：这里的 `base_ang_vel` 已经通过四元数转换到了机器人机身坐标系，而不是直接使用世界坐标系下的角速度。

它来自：

```python
self.base_ang_vel[:] = quat_rotate_inverse(
    self.base_quat,
    self.root_states[:, 10:13]
)
```

######  为什么后面要乘 `obs_scales.ang_vel`（缩放因子）？

这是深度强化学习工程落地里**最极其重要的“炼丹基本操作”——数据归一化**。

- **物理现实**：机器人在翻滚或者快速旋转时，真实的角速度可能达到 `5.0`、`10.0` 甚至更大（单位：rad/s）。
    
- **神经网络的本能**：神经网络的神经元最喜欢输入的数值在 `[-1, 1]` 左右。如果直接把 `10.0` 这种大数值喂进去，经过几层矩阵乘法，数值会爆炸，梯度直接变成 `NaN`，训练瞬间崩溃。
    
- **缩放**：所以框架会在 `config` 里设一个缩放因子（比如 `obs_scales.ang_vel = 0.25`）。  
    `10.0 * 0.25 = 2.5`。  
    这一下就把狂野的物理数据，温柔地压缩到了神经网络舒适的区间，训练不仅稳，收敛速度还快。
###### 2. projected_gravity：投影重力

```python
self.projected_gravity
```

维度：

$$
3
$$

表示重力向量在机器人机身坐标系下的投影。

为什么需要这个信息？

因为机器人需要知道自己当前的身体姿态。

例如：

- 正常直立时，重力主要投影在机身的竖直轴方向。
- 向前倾倒时，重力在机身坐标系中的分量会发生变化。
- 向侧面倾倒时，也会出现不同的重力投影。

它实际上为策略提供了一种姿态信息。

注意：这里不是直接输入 Roll、Pitch、Yaw 三个欧拉角，而是使用投影重力。

注意：后面我们输入的 $(q−q_{default})$ 12维姿态情况，只是腿部的姿态描述，机器人有上半身，我们需要靠重力投影看看机器人是不是前倾的快倒了

1. **本体运动状态（Proprioception）**：包括机身角速度（经四元数逆旋转至机身坐标系）、12个腿部关节偏差 $(q−q_{default})$及关节速度。这部分告诉网络‘我当前的肉体和关节处于什么运动趋势’。特别是关节输入减去默认姿态，完成了数据的中心化，加速了网络收敛。
    
2. **绝对空间姿态（Spatial Orientation）**：输入 `projected_gravity`（重力投影）。因为机器人是浮动基座（Floating Base）系统，仅靠关节角度无法得知机身是否倾斜。投影重

###### 3. commands：运动指令

```python
self.commands[:, :3] * self.commands_scale
```

维度：

$$
3
$$

这里需要回看 `legged_robot.py` 中 `_init_buffers()`：

```python
self.commands = torch.zeros(
    self.num_envs,
    self.cfg.commands.num_commands,
    dtype=torch.float,
    device=self.device,
    requires_grad=False
) # x vel, y vel, yaw vel, heading
```

因此前三维分别代表：

| 维度 | 含义 |
|---|---|
| 1 | X 方向目标线速度 |
| 2 | Y 方向目标线速度 |
| 3 | 目标偏航角速度 |

也就是说，机器人不仅需要知道自己现在的运动状态，还需要知道应该往什么方向、以什么速度运动。

举个例子：

```python
commands = [0.5, 0.0, 0.2]
```

可以理解为：

- 希望机器人向前移动。
- 不要求侧向移动。
- 希望机器人以一定角速度转向。

这里的数值只是示意，实际数值需要结合命令范围和缩放系数理解。

这里为什么是这三个速度呢，实际上是有六个目标速度：

如果是**无人机**，或者做**极限跑酷（Parkour）**、**空翻**的机器人，就绝对需要了。

- **无人机**：指令直接就是 `[Vx, Vy, Vz, Roll_rate, Pitch_rate, Yaw_rate]`。因为它在三维空间飞，必须能翻滚和俯仰。因为是走路，所以只需要绕着Y轴转
    
- **跑酷/特技机器人**：如果要跨越很高的障碍，或者后空翻，指令里就会加上“期望的俯仰角/角速度”，让机器人腾空时做出转体动作。

**注：在这个项目中，我们是固定速度\[0.5,0,0],然后在reward中设置只要达到这个目标速度就能拿到reward，从而让他学会听从人类指令的速度执行，这个command有利于我们遥控机器人，在机器人运动会中，机器人遥控就是这么做到的，虽然是强化学习，但是只要把达成人类遥控指令能获得很多reward，那么他就会往人类遥控的方向走

**为什么在前面的机身角速度，我们不止给一个指令Yaw_rate呢？

因为command是我们期望机器人达到的目标，在平面运动任务中，通常仅包含 2 维线速度和 1 维偏航角速度。  

但机身角速度是物理引擎反馈的真实客观状态。由于多足机器人高度动态的行走特性，机身不可避免地会产生横滚（Roll）和俯仰（Pitch）的角速度分量。如果观测输入中不包含完整的 3 维机身角速度，策略网络将无法感知自身的高频摇晃趋势，也就无法通过动作调整来满足奖励函数中 `ang_vel_xy` 的惩罚约束，极易导致摔倒。  
因此，完整保留 3 维角速度是维持动态平衡的必要条件
###### 4. dof_pos：关节位置误差

```python
(self.dof_pos - self.default_dof_pos)
    * self.obs_scales.dof_pos
```

维度：

$$
12
$$

这里有一个非常重要的细节。

输入的不是绝对关节角度：

$$
q_t
$$

而是相对于默认关节角度的偏差：

$$
q_t-q_{\text{default}}
$$

这和我们昨天学习的：

`default_joint_angles`

联系起来了。

例如：

假设某个膝关节：

$$
q_{\text{default}}=0.3
$$

当前实际角度：

$$
q_t=0.5
$$

那么输入网络的原始位置误差就是：

$$
0.5-0.3=0.2
$$

这样策略更容易理解机器人相对于默认姿态偏离了多少。

**注：这里是（现在的关节角度-初始的关节角度）用于让机器人理解现在关节距离关节初始状态，而不是PD控制时，计算需要多少力的$\Delta p$=(现在的关节角度-目标关节角度)
一个是$q_{default}$，一个是$q_{target}$

###### 5. dof_vel：关节速度

```python
self.dof_vel * self.obs_scales.dof_vel
```

维度：

$$
12
$$

分别对应 G1 的 12 个可控关节的角速度。

为什么需要它？

仅仅知道关节角度是不够的。

例如，同样处于：

$$
q=0.5
$$

但一种情况下关节正在向前运动，另一种情况下关节正在向后运动。

两者虽然角度相同，但运动状态不同。比如向上跳高，同样1m处的位置，是正在向上跳和正在下落不同

所以需要同时输入：

- 关节位置偏差。
- 关节速度。

**理解：为什么每个关节只有一个角速度，而不是三个欧拉角速度

🛩️ 1. 机身（Base）：需要三个轴（三维旋转）

- 你的身体躯干。你可以向前弯腰（Pitch），向侧面扭转（Roll），也可以原地转圈（Yaw）。
    
- 所以在代码里，机身的角速度 `base_ang_vel` 是一个 **3 维向量**（X轴、Y轴、Z轴），或者用四元数表示。
    

 🦵 2. 关节（Joint）：只需要一个轴（一维旋转）

- 现在看看你的**膝盖**。你的膝盖只能在“伸直”和“弯曲”这一个平面内转动。你绝对没法把膝盖向左右侧掰（除非你骨折了）。
    
- 再比如你的**手肘**，大部分时候也只能在一个轴向上开合。
    
- 在机器人学里，这种叫“铰链关节（Hinge Joint）”**。它被物理机械结构死死限制住了，只能绕着一个固定的轴旋转**。
    

所以，对于机器人的一个膝盖关节，描述它的状态，只需要一个数字：**它当前弯曲的角速度（rad/s）**。它没有 Roll 和 Pitch，因为它的机械结构不允许。

###### 6. actions：上一时刻的动作

```python
self.actions
```

维度：

$$
12
$$

这里有一个容易混淆的地方。

在 `step(actions)` 中：

```python
self.actions = torch.clip(actions, ...)
```

之后才执行仿真，最后生成新的 Observation。

所以，当新 Observation 返回给 PPO 时，`self.actions` 保存的是刚刚执行过的动作。

从下一次策略决策的角度看，它就是上一时刻的 Action。

为什么要输入它？

因为机器人运动具有连续性。

相同的当前关节状态，如果上一时刻采取的动作不同，可能对应不同的运动趋势。

此外，输入上一时刻的动作也有助于策略学习动作的连续性，减少不必要的剧烈变化。

**注：具身中也常常利用这个trick，将上一次输入的动作做为一个显示的输入输入策略网络，这样可以保证输出动作的连续性，那你问，为什么神经网络记不住上一轮自己的输出呢？
这是一个非常核心的 MDP（马尔可夫决策过程）设计问题。普通的 Feedforward 神经网络（MLP）不具备时序记忆能力，它的每次前向传播都是独立的。  
在机器人控制中，由于真实的加速度和电机受力情况无法仅从当前时刻的观测中完美推断（不满足严格的马尔可夫性），如果只输入当前状态，网络就会出现‘失忆’。  
为了解决这个问题，我们在观测中显式地引入了上一时刻的动作 at−1at−1​，人为地构造了一个包含历史信息的近似马尔可夫状态。这不仅补全了动力学中的趋势信息，也有效约束了动作输出的连续性。当然，如果采用 RNN/LSTM 架构，网络就可以通过内部的 Hidden State 来实现时序记忆。

###### 7. sin_phase 和 cos_phase：步态相位

```python
sin_phase = torch.sin(
    2 * np.pi * self.phase).unsqueeze(1)

cos_phase = torch.cos(
    2 * np.pi * self.phase).unsqueeze(1)
```

维度：

$$
1+1=2
$$

这两个量代表机器人步态周期中的相位信息。

继续看：

```python
def _post_physics_step_callback(self):
    self.update_feet_state()

    period = 0.8
    offset = 0.5

    self.phase = (
        self.episode_length_buf * self.dt
    ) % period / period

    self.phase_left = self.phase

    self.phase_right = (
        self.phase + offset
    ) % 1

    self.leg_phase = torch.cat((
        self.phase_left.unsqueeze(1),
        self.phase_right.unsqueeze(1)
    ), dim=-1)
```

这里：

- `period = 0.8`：步态周期为 0.8 秒。
- `phase`：根据 Episode 时间计算当前周期相位。
- `phase_left`：左腿相位。
- `phase_right`：右腿相位，相对左腿偏移 0.5。

为什么不用一个 `phase` 直接输入？

因为相位具有周期性。

例如：

$$
phase=0
$$

和：

$$
phase=1
$$

实际上代表同一个周期位置。

直接输入 `phase` 时，数值会从接近 1 突然跳回 0。

而使用：

$$
\sin(2\pi phase)
$$

和：

$$
\cos(2\pi phase)
$$

可以更加连续地表达周期位置。

注意：当前 G1 的 Observation 只拼接了 `sin_phase` 和 `cos_phase`，没有直接拼接左右腿的两个 `phase`。左右腿相位还会用于其他逻辑，例如接触奖励。

**理解：这就是为机器人走路特意设计的<运动节奏>，用三角函数的周期性模拟左右腿轮流迈出，即步态周期：

###### 📻 什么是步态周期（Phase）？

想象你在做广播体操，教练喊：“一二三四，二二三四，三二三四……”

- **周期（period）**：就是喊完“一二三四”这 4 拍的时间（比如 0.8 秒）。
    
- **相位（phase）**：就是教练现在喊到了“第几拍”。
    
    - 第一拍时，相位是 0.0（左腿刚刚抬起）。
        
    - 第二拍时，相位是 0.25。
        
    - 第三拍时，相位是 0.5（左腿落地，右腿抬起）。
        
    - 第四拍时，相位是 0.75。
        

**所以，这个“进度条”指的是整个步态循环（从左脚起步，到右脚落下，再回到左脚起步）的时间进度。** 它就像一个节拍器，在数字 0 到 1 之间无限循环。

---

###### 🚶 为什么要用相位，而不是让网络自己学？

如果没有这个相位（进度条）输入：  

网络就像一个没有节奏感的新手，它完全不知道现在该抬左腿还是右腿。它只能靠“瞎蒙”：看到身体快往左倒了，赶紧迈左腿；看到快往右倒了，赶紧迈右腿。这种纯粹的“应激反应”不仅走起来极其难看（像帕金森患者），而且极其耗电、极易摔倒。

有了这个进度条（相位）输入：  

相当于你直接给网络塞了一张“广播体操时间表”。网络看了时间表：

- “哦！现在相位是 0.1，时间表上写着这时候左腿应该在最高点，那我赶紧输出左腿的抬腿指令！”

在端到端（End-to-End）的强化学习理论中，理论上只要有足够多的数据和算力，网络可以通过隐式的历史特征推算出时间节奏。但在实际的机器人工程中，由于普通 MLP 缺乏时序记忆能力，且纯粹依靠试错去探索复杂的连续周期步态，会导致样本效率极低并极易陷入局部最优。  

引入 `sin_phase` 和 `cos_phase` 本质上是**人工注入的物理先验（Prior Knowledge）**。它类似于 Transformer 架构中的位置编码（Positional Encoding），为网络提供了一个无歧义的时间基准。它剥离了网络自己去‘盲猜’节奏的负担，让网络能将算力集中在学习关节的动力学配合上，这是工程落地中提升收敛速度和策略稳定性的核心 Trick。

###### 四、第三部分：为什么刚好是 47 维？

现在把刚才的内容整理成表格。

| Observation 内容                  |     维度 |   累计维度 |
| ------------------------------- | -----: | -----: |
| base_ang_vel（机身角速度）             |  $$3$$ |  $$3$$ |
| projected_gravity（上半身姿态）        |  $$3$$ |  $$6$$ |
| commands（速度指令）                  |  $$3$$ |  $$9$$ |
| dof_pos - default_dof_pos（腿部姿态） | $$12$$ | $$21$$ |
| dof_vel（关节速度）                   | $$12$$ | $$33$$ |
| actions（上一轮动作）                  | $$12$$ | $$45$$ |
| sin_phase（动作节奏）                 |  $$1$$ | $$46$$ |
| cos_phase（动作节奏）                 |  $$1$$ | $$47$$ |

所以：

$$
N_{\text{obs}}
=
3+3+3+12+12+12+1+1
$$

$$
\boxed{N_{\text{obs}}=47}
$$

这就对应了配置文件：

```python
num_observations = 47
```

这里要记住：

配置文件声明 Observation 的维度，而 `compute_observations()` 真正决定 Observation 的内容和排列顺序。

两者必须保持一致。

###### 五、第四部分：为什么还有 50 维的 Privileged Observation？

继续看 `g1_env.py` 中的：

```python
self.privileged_obs_buf = torch.cat((
    self.base_lin_vel * self.obs_scales.lin_vel,
    self.base_ang_vel * self.obs_scales.ang_vel,
    self.projected_gravity,
    self.commands[:, :3] * self.commands_scale,
    (self.dof_pos - self.default_dof_pos)
        * self.obs_scales.dof_pos,
    self.dof_vel * self.obs_scales.dof_vel,
    self.actions,
    sin_phase,
    cos_phase), dim=-1)
```

你会发现它与普通 Observation 非常相似。

唯一多出来的是：

```python
self.base_lin_vel * self.obs_scales.lin_vel
```

维度：

$$
3
$$

所以：

$$
N_{\text{privileged}}
=
47+3
$$

$$
\boxed{N_{\text{privileged}}=50}
$$

为什么需要这个额外信息？

这里涉及强化学习中的一个概念：

Asymmetric Actor-Critic（非对称 Actor-Critic）。

它的基本思想是：

- Actor 使用相对有限的 Observation 生成动作。
- Critic 可以使用额外的 Privileged Observation，帮助估计状态价值。

对于当前 G1：

| 网络 | 输入 | 维度 |
|---|---|---:|
| Actor | obs_buf | $$47$$ |
| Critic | privileged_obs_buf | $$50$$ |

其中，额外的三维是机器人机身线速度：

$$
v_{\text{base}}=[v_x,v_y,v_z]
$$

为什么不直接给 Actor？

可以理解为一种训练设计：让 Critic 在训练时获得更多状态信息，以帮助价值估计，而 Actor 使用较少的观测学习控制策略。

这也是一种典型的非对称训练方式。

为什么要这么训练呢

###### 🌀原因一： Sim2Real的迁移

我们用一个**“新手开车”**的比喻，你瞬间秒懂：

###### 🚗 1. 什么是特权信息（Privileged Info）？

多出来的那 3 维，是 `base_lin_vel`（机身真实的线速度 $v_x,v_y,v_z$​）。

在**仿真器（Isaac Gym）里，物理引擎拥有绝对的“上帝视角”。它知道你每一毫秒的真实加速度和绝对速度，所以可以直接把 `base_lin_vel` 提取出来。

但在**真实的机器人**上，这玩意儿极难测量！

- 你可以用轮速计，但足式机器人没有轮子。
    
- 你可以用视觉里程计（VIO），但会有延迟和漂移。
    
- 如果你给机器人强行装个 GPS，一旦进了室内，线速度直接归零。
    

所以，这 3 维线速度，在仿真里是**特权信息（Privileged Info）**，在真机上是**不可靠/不存在**的。

###### 👑 2. 为什么只给 Critic 看？（非对称）

- **Actor（行动者）**：它是要部署到**真实机器人**上的。所以它只能吃“真机上能拿到的最普通的传感器数据”（47维）。如果你让 Actor 吃了这 3 维线速度，它在仿真里会走得非常完美，一上真机，发现没有这个输入了，瞬间变成脑瘫。
    
- **Critic（评论家）**：它只在**训练阶段**干活。它的任务是给 Actor 打分（估算 $V(s)$）。既然它在仿真里能开“上帝视角”，我们就直接把真实线速度喂给它。
    

###### 🎮 3. 为什么开了上帝视角，训练就变快了？

想象你在教一个新手倒车入库（Actor）。

- 如果你（Critic）看不见车到底挪了多少距离（线速度），你只能凭感觉瞎评价，新手学得很慢。
    
- 但如果你有一台极其精确的测距仪（Privileged Info），你能精确算出他刚才那一脚油门踩得怎么样，你的打分就会极其精准。
    

**打分的评判官（Critic）越精准，Actor 拿到的 Advantage（优势函数）就越准，梯度更新的方向就越正确，收敛速度直接起飞！**

###### 🛡️ 4. 还有哪些特权信息？（拓展知识）

除了线速度，有些前沿的代码里，Critic 还会偷偷看：

- 真实的摩擦力系数
    
- 真实的地面高度图
    
- 真实的电机延迟时间
    
- 甚至直接看对方的动作（足球对抗里）
    
只要 Actor 在真机上拿不到的信息，统统都可以塞给 Critic！
###### 🌀原因二： 为什么角速度能算，线速度不能算？

这不是算法的问题，这是**物理硬件**的锅。

1. **角速度为什么极好算？**  
    因为真机上装了一个神器叫 **IMU（惯性测量单元）**，里面有一个**陀螺仪（Gyroscope）**。陀螺仪可以直接测出机器人绕三个轴的旋转角速度（Roll, Pitch, Yaw rate）。它的精度极高，几乎不受外部干扰。
    
    - 无论你是在平地上走，还是从天上掉下来，只要你在转，陀螺仪立刻就能测出来。**它是一个直接测量的绝对物理量。**
        
2. **线速度为什么极难测？（这是世界级难题）**  
    足式机器人没有车轮（没有轮速计）。如果你想测机身线速度，只有以下几种极其坑爹的方法：
    
    - **积分法（加速度计）**：加速度对时间积分就是速度。听起来很完美？错！任何微小的加速度噪声，一积分就会疯狂放大（这叫**积分漂移**）。走个 10 分钟，机器人可能以为自己速度有 100m/s。
        
    - **视觉里程计（摄像头）**：极其吃算力，有几十毫秒的延迟，一旦遇到白墙（没有纹理）或者跑步时画面模糊，直接抓瞎。
        
    - **腿部运动学（正运动学推算）**：通过脚踩在地上的位置反推。这需要依赖“脚不打滑”。一旦踩到冰面（打滑），推算的速度瞬间爆炸。
        

**所以，在真机上，线速度只能是一个“充满噪声和延迟的估计值”，而在仿真里，引擎直接给你100%完美的真实线速度。这就是为什么它成了特权信息。**

###### 六、第五部分：Observation 有没有加入噪声？

最后再看一个小细节。

在 `G1Robot.compute_observations()` 最后：

```python
if self.add_noise:
    self.obs_buf += (
        2 * torch.rand_like(self.obs_buf) - 1
    ) * self.noise_scale_vec
```

这说明在启用噪声时，普通 Observation 会加入随机噪声。

而当前这段代码没有对 `privileged_obs_buf` 执行同样的噪声操作。

再看 `_get_noise_scale_vec()`：

```python
noise_vec = torch.zeros_like(self.obs_buf[0])

noise_vec[:3] = (
    noise_scales.ang_vel
    * noise_level
    * self.obs_scales.ang_vel
)

noise_vec[3:6] = (
    noise_scales.gravity
    * noise_level
)

noise_vec[6:9] = 0. # commands

noise_vec[9:9+self.num_actions] = ...

noise_vec[9+self.num_actions:9+2*self.num_actions] = ...

noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.

noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0.
```

你可以发现：

- 角速度：可以加入噪声。
- 投影重力：可以加入噪声。
- commands：不加入噪声。
- 关节位置：可以加入噪声。
- 关节速度：可以加入噪声。
- actions：不加入噪声。
- phase：不加入噪声。

为什么这样设计？

因为训练时可以模拟传感器测量误差，让策略不要过度依赖完全精确的状态。

今天只需要知道噪声是在 Observation 构造完成之后加入的，不需要研究具体噪声参数。
这就是为什么所有高精度的物理引擎（Isaac Gym, MuJoCo）底层解算全用四元数，而在输入给神经网络的 Observation 时，也会把四元数转成重力向量或旋转矩阵，而不是直接喂三个欧拉角。

深入探讨：
##### 🛠️ 1. 代码是怎么加噪声的？

你看这一行核心代码：


```python

self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

```

- `torch.rand_like(...)`：生成一堆 0 到 1 之间的随机数。
    
- `(2 * rand - 1)`：把它变成 **-1 到 1 之间的均匀分布**。
    
- `* self.noise_scale_vec`：这是一个极其精妙的“遮罩（Mask）”。它决定了每个维度的噪声幅度。
    
    - 如果某一位是 `0`，噪声就是 0（完全不加）。
        
    - 如果某一位是 `0.05`，噪声就是 `[-0.05, 0.05]` 里的随机数。


***注：为什么不加高斯噪声？在 $TD3、DDPG$ 等众多算法中，对 $Action$ 加噪声全是加高斯噪声 

###### 🧱 一、 致命原因：高斯噪声没有“天花板”（无界性）

高斯分布（正态分布）的理论取值范围是**负无穷到正无穷**。虽然大部分数据集中在均值附近，但理论上，它依然有极小的概率抽到一个极其离谱的异常值。

- **如果给动作加高斯噪声**：就算动作偶尔偏大，我们通常会用 `np.clip` 把它硬性裁剪到 `[-1, 1]` 的安全范围内，保护电机。
    
- **如果给观测加高斯噪声**：假设真实的角速度是 `0.1`，你用高斯噪声，极小概率抽到一个 `100.0` 的疯狗值。网络看到这个假数据，会以为机器人正在以光速旋转，瞬间输出一个极其离谱的救命动作，直接导致机器人真机炸机。
    

**而均匀分布（Uniform）** `2 * rand - 1`，它的生成区间被**死死锁在了 `[-1, 1]`**，再乘上你的 `noise_scale`，它绝对不可能超出这个范围。**安全感拉满。**

###### 🎛️ 二、 物理现实：很多真实的传感器误差就是均匀的

我们在仿真里加噪声，是为了模拟真机的传感器误差。真实的误差到底是什么样的？

- **量化误差（ADC误差）**：比如一个 12 位的编码器，把物理量转成数字时，最后一位的跳动，本质上就是一种**均匀分布**。
    
- **截断误差**：数值计算中因为位数限制被截断，也是均匀的。
    
- 虽然真机也有热噪声（高斯），但工程师在仿真里加噪声时，为了**确保训练的安全边界**，往往更倾向于用均匀分布来粗糙但安全地覆盖掉不确定的误差范围。
    

###### ⚡ 三、 计算效率（GPU 性能）

在 Isaac Gym 里，你的 4096 个环境，每秒要生成几百万个随机数。

- 生成**高斯分布（正态分布）**，底层用的是 Box-Muller 算法，需要算对数（`log`）、算根号（`sqrt`）、算三角函数（`sin`, `cos`），极其耗费 GPU 算力。
    
- 生成**均匀分布**，底层就是一个极其简单的线性映射（伪随机数生成器），速度极快。  
    对于追求极致吞吐量的强化学习来说，能省一点算力是一点。
    

###### 💡 四、 拓展：为什么“动作（Action）”探索偏偏要用高斯噪声？

既然均匀分布这么好，为什么我们在连续控制的探索阶段（比如 $TD3$ 的 `expl_noise`），全都用高斯噪声？

因为**探索的逻辑不同**。

- 我们希望机器人的探索**集中在当前认为最优的动作附近**。高斯分布是钟形曲线，均值附近概率最高（代表微调），两侧极低。这完美契合了“大步探索，但重心依然在好动作附近”的直觉。
    
- 而在观测里加噪声，只是“模拟传感器坏了”，均匀分布这种“范围固定、随便乱跳”的特性刚好合适。

---

##### 🤔 2. 为什么有的能加，有的死活不能加？

这就是在**模拟真实的物理世界**。

**✅ 必须加噪声的（传感器能测的）：**

- **角速度、重力投影**：真机上的 IMU 传感器极其便宜，电磁干扰、震动、温漂都会导致数据乱跳。
    
- **关节位置、关节速度**：真机上的电机编码器是有量化误差的，而且算出来的速度会有极高频的噪声。
    
- _目的_：如果在仿真里太干净，网络就会变成“温室里的花朵”。到了真机上，传感器一抖，网络就疯了。加噪声，是逼迫网络学会“抗干扰”，这叫**鲁棒性（Robustness）**。
    

**❌ 绝对不能加的（上帝给的或自己算的）：**

- **commands（目标速度）**：你是用遥控器给的指令，遥控器的数字是极其精准的，不存在“本意是 0.5，遥控器抽风发出 0.52”。
    
- **actions（上一时刻动作）**：这是神经网络自己上一帧算出来的绝对精确的存储值。
    
- **phase（步态相位）**：这是代码里的数学公式用时间算出来的节拍器，不经过任何物理传感器，极其精确。
    
- _目的_：给这些加噪声，等于给网络塞假情报，只会让训练直接崩溃。
    

---

##### 🧩 3. 这跟四元数和欧拉角有什么关系？（极其关键）

这就是为什么我们在 Observation 里用 `projected_gravity`，而**不敢直接用四元数或欧拉角**。（上半身姿态）

**如果直接对四元数加噪声，会出大问题：**  
四元数的物理约束是 $w^2+x^2+y^2+z^2=1$。如果你给这四个数随便加个噪声，这个四元数就不归一化了（长度不是 1 了）。如果不重新归一化直接扔给网络，网络会算出非常诡异的姿态；如果重新归一化，计算又很麻烦。

**如果直接对欧拉角加噪声，会出大问题：**  
假设真实角度是 179°（Roll 轴），你加了 2° 的噪声，变成了 181° 。但在欧拉角的数学表达里，它可能会跳变成 −179°。网络突然看到数值从 179 变成了 -179，会以为机器人原地转了 360 度，网络的训练崩坏。

**所以，最完美的方案是：把姿态转成重力投影向量（3维）**  
重力投影本来就是 **[-1, 1]** 之间的三个普通数字，没有死锁，也没有长度约束。你直接给它加上噪声，网络只会觉得“嗯，我稍微有点倾斜”，极其自然。

**不同于机身角速度，那个不是角度，用三维对应欧拉角对应速度即可

---

# 今日问题

### 问题 1

为什么 G1 的 Actor Observation 中没有：

```python
self.base_lin_vel
```

但 Privileged Observation 中却有？

这和 Asymmetric Actor-Critic 有什么关系？

答：这是一种Actor-Critic常见非对称训练方式，常常用于Sim2Real，由于机器人线速度在真机中不好测量或测量误差较大，真机阶段Actor无法获取正确的线速度，而在仿真中物理仿真器可以直接算出机器人线速度，所以在阶段，我们会给Critic输入真实线速度，让Actor获得更加准确的指导（具体体现在 $\mathcal{L}^{\mathrm{Actor}}=-\mathbb{E}\left[\min\left(\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}A_t^{(\mathrm{GAE})},\operatorname{clip}\left(\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)},1-\epsilon,1+\epsilon\right)A_t^{(\mathrm{GAE})}\right)\right]$中$A_t^{(\mathrm{GAE})}$中$V(S)$的$S$为50维的），这样在真机中会有更好的效果。
### 问题 2

为什么关节位置输入使用：

```python
self.dof_pos - self.default_dof_pos
```

而不是直接输入：

```python
self.dof_pos
```

答：输入self.dof_pos，网络只能得到关节位置，而不知道这个关节弯曲了多少，现在处于什么状态，而如果我们输入self.dof_pos - self.default_dof_pos，网络就会知道我们的关节距离初始角度的差距，如果弯曲了，下一步就要伸直，加速网络的收敛。

#### 💡GPT的修正与补充

**更准确的答案：**

> 使用关节角度相对于默认角度的偏差，可以使 Observation 以默认姿态为参考中心，让网络更直接地理解关节偏离正常姿态的程度，降低网络自己学习参考偏置的负担，通常有利于策略学习。

### 问题 3

为什么 Observation 中还要加入：

```python
self.actions
```

如果不输入上一时刻的 Action，可能有什么影响？

答：机器人的运动是连续控制，如果不输入上一时刻的动作，由于神经网络仅仅是一个映射，没有过去动作的历史，下一步输出的动作可能和上一步动作无法连续衔接，导致机器人运动卡顿或者不协调
#### 💡GPT的修正与补充

###### 修正：G1 当前 Actor 不是普通 MLP，而是 LSTM Recurrent Actor-Critic

此前“神经网络仅仅是一个映射，没有过去动作的历史”这一说法，对当前 G1 并不完全成立。

###### 1. G1 当前策略网络是 LSTM Recurrent Actor-Critic

在 `g1_config.py` 中：

```python
class runner(LeggedRobotCfgPPO.runner):
    policy_class_name = "ActorCriticRecurrent"
```

并且：

```python
class policy:
    rnn_type = 'lstm'
    rnn_hidden_size = 64
    rnn_num_layers = 1
```

这说明当前 G1 使用的不是普通前馈 Actor，而是带 LSTM 记忆的 Recurrent Actor-Critic。

在 RSL-RL v1.0.2 的 `actor_critic_recurrent.py` 中：

```python
self.memory_a = Memory(
    num_actor_obs,
    type=rnn_type,
    num_layers=rnn_num_layers,
    hidden_size=rnn_hidden_size
)
```

Actor 的 `act()` 会先经过 LSTM：

```python
def act(self, observations, masks=None, hidden_states=None):
    input_a = self.memory_a(
        observations,
        masks,
        hidden_states
    )
    return super().act(input_a.squeeze(0))
```

因此，当前 Actor 已经能够利用之前时间步的信息，LSTM 本身具有记忆能力。

###### 2. 为什么 LSTM 有记忆，还要把上一时刻 Action 加入 Observation？

实际上，**不仅不是多余，反而是一个极其精妙的“双保险”设计。** 我们用“打乒乓球”来比喻，你就秒懂了：

###### 🏓  LSTM 是“长期肌肉记忆”，上一帧动作是“瞬时手感”

- **LSTM（长短期记忆）**：就像你打乒乓球的**宏观战术记忆**。它记得过去几秒钟你一直在后退，或者一直在拉弧圈球。它是用“隐藏状态（Hidden State）”来压缩过去的复杂历史。
    
- **上一帧动作 at−1at−1​**：就像你上一拍**击球时的精确角度和力度**。
    

LSTM 在传递信息时，为了让记忆能够流传下去，它会对历史信息进行**“压缩和模糊化”**（就像你把 100 张照片压缩成一个 10MB 的压缩包，画质肯定受损）。LSTM 可能会忘记你 0.02 秒前到底给膝盖施加了 0.5 还是 0.51 的扭矩，它只记得“我刚刚在往前走”。

###### 🎯  算“平滑惩罚”的绝对刚需

还记得你代码里的 `action_rate = -0.01` 吗？这个奖励项是在惩罚神经网络：**$Δa=a_t−a_{t−1}$​**（动作变化太快就扣分）。

```python
def _reward_action_rate(self):
    return torch.sum(
        torch.square(
            self.last_actions - self.actions
        ),
        dim=1
    )
```

如果网络只依赖 LSTM，它要自己从模糊的记忆里“反推”出上一帧到底输出了什么，才能计算这个差值，这太难了。  

**直接把 $a_{t−1}​$ 塞到它手边**，网络只要看一眼第一层矩阵，瞬间就能算出让电机平滑的 $a_t$，避免了机械臂帕金森式的高频抽动。

也就是同时使用上一时刻 Action 作为 Observation，并通过 Reward 对相邻动作变化进行惩罚。

这两种设计相互配合，但作用不同

###### 3. 所以更准确的答案是

输入上一时刻 Action，可以给策略提供显式的控制历史信息，帮助策略了解前一步采取了什么控制指令，从而更好地决定下一步动作。当前 G1 虽然还使用 LSTM 来记忆历史，但上一时刻 Action 仍然可以作为直接的状态输入。

###### 5. 关键提醒

以后判断此类问题时，一定要结合实际网络结构，不能把所有 Actor 都当成普通 MLP。G1 当前使用的是 `ActorCriticRecurrent` + LSTM，因此“Actor 没有历史记忆”的说法不成立。
### 问题 4

为什么 G1 要把：

```python
self.phase
```

转换成：

```python
sin_phase
cos_phase
```

而不是直接输入 `phase`？

答：如果直接用phase，如果运动节奏从0回到1，证明下一个循环开始，而网络会理解为数值从0突变到1，导致网络训练失败，如果用周期函数，这样phase从0回到1时，周期函数值会连续变化，保证了机器人运动的流畅性

#### 💡GPT的修正与补充

这题理解基本正确！

不过你把相位变化的方向说反了。

当前代码：

```python
self.phase = (
    self.episode_length_buf * self.dt
) % period / period
```

Phase 是：

$$
0\rightarrow 1\rightarrow 0
$$

也就是说，从接近 1 跳回 0，而不是从 0 跳到 1。

例如：

| 时间位置 | phase |
|---|---:|
| 周期开始 | 0.00 |
| 周期中间 | 0.50 |
| 周期快结束 | 0.99 |
| 下一周期开始 | 0.00 |

如果直接把 `phase` 输入网络，那么：

$$
0.99\rightarrow0
$$

数值会突然发生很大的变化。

但是使用：

$$
(\sin(2\pi\phi),\cos(2\pi\phi))
$$

可以得到：

当：

$$
\phi=0
$$

有：

$$
(0,1)
$$

当：

$$
\phi=1
$$

同样有：

$$
(0,1)
$$

所以周期开始和结束在二维表示中是同一个点。

这里还要修正一句：

> 保证了机器人运动的流畅性。

更严谨地说：

**它消除了周期边界在输入特征上的人为数值跳变，但不能单靠这个编码保证机器人运动一定平滑。**

机器人最终是否平滑，还受到策略、PD 控制、动力学和 Reward 等因素影响。

### 问题 5

为什么配置文件中的：

```python
num_observations = 47
```

必须和 `compute_observations()` 中 `torch.cat()` 拼接出来的维度保持一致？

如果不一致，可能会在哪个环节出现问题？

答：如果不一致，在输入网络时可能出现问题，如果配置文件中维数更小，那可能导致输入的信息被截断，而使得网络收敛速度减慢或者训练失败，如果反之，则不一定会出现问题，比如配置文件中action维度为13，而实际只有十二个action关节

#### 💡GPT的修正与补充

这是这次最需要纠正的一题。

你的回答：

> 如果配置文件中维数更小，那可能导致输入的信息被截断，而使得网络收敛速度减慢或者训练失败，如果反之，则不一定会出现问题，比如配置文件中 action 维度为13，而实际只有十二个 action 关节。

这里有两个错误。

###### 错误一：维度不匹配，通常不是简单的信息截断

你需要先理解：

```python
num_observations = 47
```

这个参数不仅是一个说明信息的数字。

它会被用于初始化神经网络的输入维度。

我核对了当前项目的 `BaseTask`：

```python
self.num_obs = cfg.env.num_observations
```

而 RSL-RL v1.0.2 的 `OnPolicyRunner` 使用：

```python
actor_critic_class(
    self.env.num_obs,
    num_critic_obs,
    self.env.num_actions,
    **self.policy_cfg)
```

也就是说，网络在创建时就已经确定了输入维度。

**你的直觉：** 如果配置是 47，实际是 46，网络可能会“自动截断”或者“忽略”，只是效果变差。  
**计算机的现实：** **直接崩溃（Crash）！** 根本跑不起来。

**为什么？**  
因为神经网络的第一层是一个**矩阵乘法**（`nn.Linear`）。  
在 PyTorch 里，配置里写了 `num_observations = 47`，它就会在内存里建一个长得像 `[47, 256]` 的权重矩阵。  
如果你这时候硬塞给它一个 46 维的数据。矩阵乘法规则是 `(1, 46) × (47, 256)`。**数学上根本乘不了！**  
PyTorch 会直接甩你一个红字报错：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x46 and 47x256)`。

###### 错误二：你把 num_observations 和 num_actions 混在一起了

这两个参数完全不同：

| 配置参数 | 含义 | 当前 G1 |
|---|---|---:|
| `num_observations` | Actor 输入维度 | $$47$$ |
| `num_privileged_obs` | Critic 输入维度 | $$50$$ |
| `num_actions` | Actor 输出维度 | $$12$$ |

你举的例子中，配置中虽然写了13个关节，但是：

```python
num_actions = 12
```

输入网络的动作维数是严格的十二维，直接忽略了第十三个关节，所以程序可以正常运行

而且需要纠正的一点是：

假设：

```
num_actions = 13
```

Actor 可以正常输出 13 维 Action，但 G1 实际只有 12 个可控关节。

进入：

```
def _compute_torques(self, actions):
```

时：

```
actions_scaled + self.default_dof_pos - self.dof_pos
```

可能发生：

```
actions_scaled:      [4096, 13]
default_dof_pos:     [4096, 12]
dof_pos:             [4096, 12]
```

这时张量运算就会报维度不匹配错误。

**注意：这不是网络输入维度错误，而是网络输出与机器人执行接口不匹配。**


# 今天的动手实践

今天不需要重新训练，也不需要修改任何 PPO 参数。

只做一件事：

自己验证 47 维 Observation 的维度分布。

先根据 `compute_observations()` 写出：

```python
obs_dimensions = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "commands": 3,
    "dof_pos_error": 12,
    "dof_vel": 12,
    "actions": 12,
    "sin_phase": 1,
    "cos_phase": 1,
}

print(sum(obs_dimensions.values()))
```

预期：

```text
47
```

再计算 Privileged Observation：

```python
privileged_obs_dim = (
    obs_dimensions["base_ang_vel"]
    + obs_dimensions["projected_gravity"]
    + obs_dimensions["commands"]
    + obs_dimensions["dof_pos_error"]
    + obs_dimensions["dof_vel"]
    + obs_dimensions["actions"]
    + obs_dimensions["sin_phase"]
    + obs_dimensions["cos_phase"]
    + 3  # base_lin_vel
)

print(privileged_obs_dim)
```

预期：

```text
50
```

如果你本地已经能够创建 G1 环境，还可以在 `compute_observations()` 拼接完成之后临时打印：

```python
print("obs_buf:", self.obs_buf.shape)
print(
    "privileged_obs_buf:",
    self.privileged_obs_buf.shape
)
```

如果并行环境数量为 4096，预期：

```text
obs_buf: torch.Size([4096, 47])

privileged_obs_buf: torch.Size([4096, 50])
```

不需要执行完整训练。

