# 今日阅读任务



### 1. `stiffness` (刚度 / 比例增益 Kp) —— 像“弹簧的硬度”

- **数值**：髋关节(hip) 100，膝关节(knee) 150，踝关节(ankle) 40。
    
- **作用**：告诉电机：“你现在的实际角度离目标角度差了多少，你要用多大的力气去拉回来。”
    
- **公式**：$P=Kp×(目标角度−当前角度)$  可以用 $F = -kx$ 理解
    
- **直觉理解**：
    
    - 假设目标角度是 0度，当前角度是 -10度。
        
    - 如果 `stiffness = 100`，电机就会输出一个力，这个力等于 $100×10=1000$（这里单位是抽象比例，实际单位是 $N·m/rad$）。
        
    - **为什么不同关节数值不同？** 膝关节（150）承受全身重量，且需要快速伸直，所以“弹簧”非常硬。踝关节（40）主要用来微调平衡，不需要那么大的蛮力，所以比较软。
        

### 2. `damping` (阻尼 / 微分增益 Kd) —— 像“避震器的阻力”

- **数值**：髋关节 2，膝关节 4，踝关节 2。
    
- **作用**：告诉电机：“你要动可以，但必须踩刹车，不能冲过头。”
    
- **公式**：D=Kd×(目标速度−当前速度)D=Kd​×(目标速度−当前速度)。因为目标速度通常设为 0，所以简化为 −Kd×当前速度−Kd​×当前速度。
    
- **直觉理解**：
    
    - 如果只有 P（只有弹簧），你拉一下它，它会在目标位置上下疯狂震荡（就像一辆没有避震器的车）。
        
    - 加上 D（避震器）后，它就会变得平稳。当关节移动速度越快，D 产生的“反向刹车力”就越大，防止关节“哐”地一下砸到目标位置。
        

> **💡 结合上一节的 `control_type = 'P'`**：  
> 
> 你可能会问，既然有 P 也有 D，为什么叫 'P'？  
> 
> 因为这里的控制目标是**“位置”**（Position）。虽然底层用的是 PD 控制，但网络输入的是位置目标，不是速度目标（速度控制通常叫 'V' 或 'VMC'）。

### 3. `action_scale` (动作缩放系数) —— 神经网络的“变速箱”

- **数值**：0.25
    
- **作用**：防止神经网络输出的动作把机器人“甩飞”。
    
- **直觉理解**：
    
    - 强化学习的神经网络通常输出在 `[-1, 1]` 之间（或者接近这个范围）。
        
    - 如果 Policy 输出 `action = 1`，而 `action_scale = 0.25`，那么实际目标角度只增加了 1×0.25=0.251×0.25=0.25 弧度（大约 14 度）。
        
    - 如果没有这个缩放，网络输出 1 就直接转 1 弧度（57 度），机器人瞬间动作太大，直接当场摔死。**这个系数就是给机器人的运动限个速。**
        

---

### 把它们连起来看：一次完整的动作是怎么发生的？

1. **Policy（神经网络）**：算出一个动作 `action = 0.8`。
    
2. **Target Position（目标角度）**：
    

q_{\text{target}} = q_{\text{default}} + (\text{action} \times \text{action_scale})  
]  
假设髋关节默认是 0，那么 qtarget=0+0.8×0.25=0.2qtarget​=0+0.8×0.25=0.2 弧度。

3. **PD 控制器（物理仿真底层每 5ms 算一次）**：
    
    - 假设此时关节实际角度是 0.1 弧度，速度是 0.5 rad/s。
        
    - **P 的力量**：$100×(0.2−0.1)=10$
        
    - **D 的阻力**：$−2×0.5=−1$
        
    - **总力矩**：$10−1=9 N·m$
        
4. **物理引擎**：给这个关节施加 $9 N·m$ 的力矩，机器人关节开始转动。
    
5. **`decimation = 4`**：这个力矩算好后，物理引擎推进 5ms。然后再算一次 PD，再推进 5ms。重复 4 次（凑够 20ms），再把新的 Observation 喂给 Policy。
    

### 🎯 为什么这些参数这么重要？

这些参数本质上定义了**机器人的“肌肉反应速度”**。

- 如果 `stiffness` 设置得太大，机器人走路会像钢板一样僵硬，容易把物理引擎算崩。
    
- 如果 `stiffness` 设置得太小，机器人走路会像软脚虾，承不住重量。
    
- 如果你改了这个配置，Policy 原来的训练结果可能就作废了，因为机器人变成了“另一个人”。
    

所以明天你学 Observation 和 Action 的时候，只要记住：**Policy 只是给出了一串抽象的数字，而正是这些 PD 增益和 Scale，把这些数字变成了物理世界里真实的推拉扭转。**
# 今日问题

## **问题 1**

为什么 `G1RoughCfg` 中的：

```
pos = [0.0, 0.0, 0.8]
```

和：

```
default_joint_angles = {...}
```

需要分开设置？

它们分别描述机器人什么状态？

**答：pos是机器人重心在仿真物理世界的位置坐标，而 `default_joint_angles` 是机器人关节的初始角度，两者没有关系，所以分开设置

#### 💡GPT的修正与补充

前半部分正确，但有一个术语需要修正：

`pos` 不是机器人重心（Center of Mass，CoM）的位置，而是机器人基座（Root/Base）的初始位置。

---

什么是基座：

简单来说，**基座（Base / Root）是机器人骨骼树（Kinematic Tree）的“根节点”**。对于人形机器人（比如 G1），基座通常就是它的**骨盆（Pelvis）**或者**躯干（Torso）**。

为了让你彻底明白，我们分三点来解释：

###### 1. 什么是“基座”？（从机器人结构说起）

机器人 URDF 文件是一个树状结构。想象一棵倒挂的树：

*   **根节点（Root/Base）**：G1 的 `pelvis`（骨盆）。
*   **树枝（Links）**：大腿、小腿、脚、手臂、头。
*   **树杈（Joints）**：髋关节、膝关节、踝关节。

因为 G1 是人形机器人，它没有被固定在地面上（不像工厂里的机械臂螺丝打在地上）。所以它的基座是一个**“浮动基座（Floating Base）”**。它在物理世界里有 6 个自由度：可以上下左右前后移动（3个平移），也可以前后翻转、左右倾斜（3个旋转）。

当你在配置文件里写：
```python
pos = [0.0, 0.0, 0.8]
```
它的意思是：**“把 G1 的骨盆（Pelvis）放在仿真世界的坐标原点上，高度为 0.8 米。”** 只要骨盆放好了，引擎就会根据关节角度，自动把两条腿和手臂“挂”在这个骨盆下面。

###### 2. 为什么它不是“重心”？

*   **重心（CoM, Center of Mass）**：是机器人所有质量平均的一个点。当你抬起左腿时，因为左腿的质量移动了，整个机器人的重心也会跟着偏移。重心在机器人行走时是**不断变化**的。
*   **基座（Base）**：是一个物理上的连杆（零件）。只要你不改变骨盆的位置，G1 的基座坐标就永远在那里。

在仿真引擎（Isaac Gym）里，你无法直接设置重心的位置，因为重心是物理引擎根据 URDF 里的质量和惯性自动算出来的。你能直接“捏”住的，只有基座（根节点）。你把基座放在 0.8 米高，机器人就站在那里。

###### 3. 为什么代码里要这么写？（回看截图代码）
官方 `legged_robot.py`代码：
```python
base_init_state_list = (
    self.cfg.init_state.pos   # 基座的 x, y, z 坐标
    + self.cfg.init_state.rot # 基座的翻滚、俯仰、偏航角
    ...
)
```

在 Isaac Gym 的底层接口里，创建一个机器人（`create_actor`）时，你必须告诉引擎：**“请把你的根节点（Base）放在这个位置，摆成这个姿势。”** 接着，引擎再把 `default_joint_angles`（关节初始角度）应用上去。

*   `pos` 决定了机器人**整体**被丢在世界的哪个角落。
*   `default_joint_angles` 决定了机器人的**四肢**是什么姿势（是站着还是半蹲）。

这就好比：你买了一个手办（机器人），`pos` 是你要把手办放在桌子上的哪个位置；而 `default_joint_angles` 是你把它的胳膊和腿掰成什么造型。这两者确实没有直接关系，所以需要分开设置。

---

另外，你说的“两者没有关系”也不够严谨。

它们属于两个不同维度的初始状态，但共同决定机器人初始构型：

<div style="border: 1px solid #d1d5db; border-radius: 10px; padding: 20px; max-width: 800px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; box-sizing: border-box;">

  <!-- 蓝色块 -->
  <div style="background-color: #eaf2fd; border-radius: 8px; padding: 16px 20px;">
    <div style="font-weight: bold; font-size: 16px; color: #111827; margin-bottom: 12px;">pos: 基座在世界坐标系中的位置</div>
    <div style="font-size: 22px; font-weight: bold; color: #3b82f6; margin-bottom: 8px;">[0, 0, 0.8]</div>
    <div style="font-size: 13px; color: #6b7280;">机器人整体放置在哪里</div>
  </div>

  <!-- 加号 -->
  <div style="text-align: center; font-size: 20px; color: #6b7280; margin: 12px 0; font-weight: bold; line-height: 1;">+</div>

  <!-- 绿色块 -->
  <div style="background-color: #eaf7f0; border-radius: 8px; padding: 16px 20px;">
    <div style="font-weight: bold; font-size: 16px; color: #111827; margin-bottom: 12px;">default_joint_angles: 关节初始角度</div>
    <div style="font-size: 22px; font-weight: bold; color: #10b981; margin-bottom: 8px;">12 个活动关节</div>
    <div style="font-size: 13px; color: #6b7280;">机器人以什么姿态开始运动</div>
  </div>

  <!-- 箭头 -->
  <div style="text-align: center; font-size: 20px; color: #6b7280; margin: 12px 0; font-weight: bold; line-height: 1;">↓</div>

  <!-- 紫色块 -->
  <div style="background-color: #f5effa; border-radius: 8px; padding: 16px 20px;">
    <div style="font-weight: bold; font-size: 16px; color: #111827; margin-bottom: 8px;">完整的初始机器人构型</div>
    <div style="font-size: 13px; color: #6b7280;">基座位置与内部关节姿态共同决定各个刚体在世界中的位置。</div>
  </div>

</div>



例如，即使基座位置完全相同，改变膝关节角度，也会改变腿部姿态，进而改变机器人整体的重心位置。


## **问题 2**

为什么 G1 的：

```
num_actions = 12
```

而 `default_joint_angles` 中却有 13 个键？

为什么不能直接使用字典长度作为 Action 维度？

**答：`default_joint_angles`中多了一个`torso_joint`但是其为固定参数，不加入机器人控制

#### 💡GPT的修正与补充

你抓住了核心：多出来的配置键没有成为第 13 个 Action。

不过我需要把这里说得更精确。

**不要去配置文件里数 Action 的数量，要去 URDF 模型里数。配置文件里多写的键，只要 URDF 里没有对应的活动关节，程序就直接无视了。** 所以，第 13 个 Action 从头到尾都不存在

我重新核对了官方仓库的 G1 URDF：

```
<joint name="waist_yaw_joint" type="fixed">
```

当前 URDF 中确实存在一个固定的腰部连接关节，但它的名字是 `waist_yaw_joint`，不是配置字典里的 `torso_joint`。


而且，在官方 `legged_robot.py` 中：

```python
for i in range(self.num_dofs):

    name = self.dof_names[i]

    angle = self.cfg.init_state.default_joint_angles[name]

    self.default_dof_pos[i] = angle
```

注意这里的逻辑：

程序遍历的是实际加载模型得到的 DOF 名称，而不是配置字典中的所有键。

这是最核心的。程序不是数一数配置文件里写了几个键，就生成几个 Action。程序是先去问物理引擎：“你加载的模型到底有几个能动的关节（DOF）？”引擎回答：“12 个。” 然后程序就建了一个长度为 12 的数组，**只去配置文件里找这 12 个关节的名字**。多出来的那一个，程序连看都不看一眼

因此：

1. URDF 的活动关节决定实际 DOF。
    
2. 当前 G1 12DOF 模型有 12 个活动关节。
    
3. `torso_joint` 是配置字典中的额外键，在当前这段初始化代码中不会被访问。
    
4. 固定关节本身不提供独立的运动自由度。
    

所以更准确的答案是：

> `torso_joint` 虽然存在于 `default_joint_angles` 字典中，但它不是当前 URDF 中的活动 DOF 名称。程序只遍历 Isaac Gym 返回的实际 DOF 列表，因此这个额外配置项不会产生第 13 维 Action。

这里有一个很值得记住的工程知识：

配置字典中存在一个参数，不代表程序实际使用了它。必须追踪它是否进入了执行路径

## **问题 3**

为什么 `legged_robot.py` 要使用：

```
self.dof_names[i]
```

再去查找：

```
self.cfg.init_state.default_joint_angles[name]
```

而不是直接按照配置字典的顺序生成 Tensor？

答：由于配置文件`G1RoughCfg`中的关节位置与URDF中的关节位置不一定相同，如：

- **URDF 告诉代码**：我的关节顺序是 `[左髋, 左膝, 右髋, 右膝]`。
    
- **配置文件告诉代码**：默认角度是 `{"左膝": 0.5, "左髋": 0.1, "右膝": 0.6, "右髋": -0.1}`

若按照字典顺序生成，将配置文件中的 左膝=0.5 配置给左髋，那么机器人控制可能完全乱套

#### 💡GPT的修正与补充

这一题你理解得非常好。

你举的例子也很恰当。

例如配置字典：

```
default_joint_angles = {    "left_knee_joint": 0.5,    "left_hip_joint": 0.1,    "right_knee_joint": 0.6,    "right_hip_joint": -0.1}
```

而程序实际得到的关节顺序是：

```
self.dof_names = [    "left_hip_joint",    "left_knee_joint",    "right_hip_joint",    "right_knee_joint"]
```

如果按照字典顺序直接生成 Tensor，就可能造成关节角度错配。

正确的方式：

```python
for i in range(self.num_dofs):

    name = self.dof_names[i]

    angle = self.cfg.init_state.default_joint_angles[name]

    self.default_dof_pos[i] = angle
```

这样无论字典中的键如何排列，都会按实际 DOF 名称匹配。

这里我再帮你提高一点严谨性：

`self.dof_names` 是通过：

```
self.gym.get_asset_dof_names(robot_asset)
```

从 Isaac Gym 已经加载的模型中获得的。

什么意思呢？
###### 1. 代码是怎么拿到关节名字的？

在 `_create_envs()` 里：

```python

##### 第一步：把 URDF 文件交给 Isaac Gym
robot_asset = self.gym.load_asset(...) 
##### 第二步：问 Isaac Gym 要数据
self.num_dof = self.gym.get_asset_dof_count(robot_asset) # 问：有几个能动的关节？答：12
self.dof_names = self.gym.get_asset_dof_names(robot_asset) # 问：它们都叫啥？答：['left_hip...', 'right_knee...']

```

这个 `self.dof_names` 列表，就是**物理引擎解析完 URDF 文件后，吐出来的真实名单**。它代表了现实世界（仿真世界）里的物理真相。

### 2. 这段话在“严谨”什么？

在之前的讨论中，你可能会误以为：“配置字典里写了几个键，代码就遍历几个键”。  
但这个 AI 提醒你：**程序的执行路径，是严格以 `self.dof_names`（引擎返回的名单）为准的。**

回到初始化默认角度的代码：


```python

for i in range(self.num_dofs): # 这里的 num_dofs 是 12
    name = self.dof_names[i]   # 严格按照物理引擎给的名单来
    angle = self.cfg.init_state.default_joint_angles[name] 

```

- 循环跑了 12 次。
    
- 每次按名字去字典里找值。
    
- 如果字典里多了一个 `torso_joint`，但这 12 次循环里根本没点到它的名字，它就直接被**忽略**了。

配置是可以随便写的，甚至可以写错、写多；但 `self.dof_names` 是 Isaac Gym 读出来的真实结果，程序后面的所有逻辑，都是围绕这个真实的列表转的。

因此，我们真正应该信任的是仿真器返回的 DOF 顺序，而不是直接假设它一定等于 URDF 文件中 XML 标签的文本排列顺序。

**这一题最值得记住的编程思想：

按名称查找  $Name-based Mapping$

使用关节名称建立映射关系，而不是依赖两个容器碰巧拥有相同的排列顺序。

这一点在以后阅读机器人控制代码、部署策略和 Sim2Sim 时都非常重要。

## **问题 4**

为什么髋关节、膝关节和踝关节的 stiffness 不一样？

如果膝关节的 stiffness 从 150 改成 50，机器人可能出现什么变化？

今天只需要提出合理假设，不需要实际修改参数。

答：stiffness可以看作弹簧的劲度系数，不同关节的硬度等因素不同，则在旋转时需要配置不同的力，将膝关节的 stiffness 从 150 改成 50，可能导致膝盖在运动时施加的力不够而不转动，最终导

致机器人因为左膝不协调而左脚一跨步就摔倒在地

#### 💡GPT的修正与补充

前半部分非常好。

你已经理解了 PD 控制中的 P 项，本质上类似一个旋转弹簧 $$F = -kx$$
不过对于机器人关节，我们讨论的不是普通弹簧的线性力，而是关节角度误差产生的恢复力矩。

G1 配置中实际上写明了：
```python
stiffness = {
    'hip_yaw': 100,
    'hip_roll': 100,
    'hip_pitch': 100,
    'knee': 150,
    'ankle': 40,
}

# [N*m/rad]
```


我们用一个具体例子理解。

假设膝关节目标角度与实际角度相差：

$$ e=0.1\text{ rad} $$

当 stiffness 为 150 时：

$$tau_P=150\times0.1=15\text{ Nm} $$

当 stiffness 降低到 50 时：

$$tau_P=50\times0.1=5\text{ Nm} $$



因此，你说的“施加的力不够”方向正确，但应该改成：

***相同角度误差下，P 项产生的关节力矩变小了。

##### 然后有三个需要修正的地方：

**第一，不一定会导致关节不转动。

因为 PD 控制还有 D 项，而且机器人仍然受到重力、惯性和其他动力学因素影响。

降低 stiffness 只是降低了角度误差对应的比例力矩，不等于关节失去运动能力。

**第二，可能导致姿态跟踪变差。

例如膝盖需要支撑身体重量，控制器的恢复力矩不足时，可能出现：

- 膝关节弯曲过多；
    
- 目标角度跟踪误差增大；
    
- 机器人下蹲或支撑不稳定；
    
- 最终影响行走甚至摔倒。
    

这是一种合理的风险，但不一定一跨步就摔倒。

**第三，你说左膝不协调，但实际会同时改变左右膝盖。

因为代码中有：

```python
for dof_name in self.cfg.control.stiffness.keys():

    if dof_name in name:

        self.p_gains[i] = (
        
            self.cfg.control.stiffness[dof_name]
            
        )    
```

当 `dof_name = 'knee'` 时，左右两个膝关节名称都会匹配。

所以修改：

```
'knee': 150
```

为：

```
'knee': 50
```

会同时改变左右膝关节的 P 增益。

这题的正确结论：

**降低膝关节 stiffness，会降低相同位置误差下 P 项产生的恢复力矩，使膝关节的位置跟踪能力减弱，可能影响机器人支撑和步态稳定性，但不意味着膝关节一定不转动或机器人必然摔倒。

#### 注：什么是恢复力矩

问：这个恢复力矩到底是什么阶段做的，是我们输出动作0要把关节转到某个角度，然后开始计算 $K_p(q_{\text{target}} - q_t)$还是?

**答：在每一小步物理仿真（5ms）开始前，都会重新算一次 $(K_p \times \Delta p)$**。

这就是我们在最开始聊的 `decimation = 4` 的核心意义。

让我用最直观的时间轴，把你刚才这句话补充完整：

###### 完整的控制闭环（时间轴拆解）

假设现在是 $(t=0$) 毫秒，强化学习策略刚刚算出了一个新的动作 `action`，比如它输出了 `[0.1, -0.2, ...]`。

**第一阶段：Policy 决策（每 20ms 一次）**
1. 拿到网络输出的 `action`。
2. 算出**目标角度（Target Position）**：
   $$
 q_{\text{target}} = q_{\text{default}} + \text{action\_scale} \times \text{action} 
$$
   *（这就是你说的“要把关节转到某个角度”）*

**第二阶段：底层 PD 循环与物理仿真（每 5ms 一次，重复 4 次）**
接下来，代码进入 `for _ in range(4):` 循环（这就是 `decimation=4` 在做的事）。

*   **第 1 次循环（\(t=0\) 到 \(5\) ms）：**
    1. **算恢复力矩**：底层控制器拿刚才算出的 $q_{\text{target}}$，减去**当前真实角度** $q_t$，算出 $K_p(q_{\text{target}} - q_t)$，再加上阻尼项。
    2. **物理推进**：把这个力矩交给 Isaac Gym，物理引擎推进 5ms。此时关节转动了一点点，角度变成了 $q_{t+1}$。

*   **第 2 次循环（\(t=5\) 到 \(10\) ms）：**
    1. **重新算恢复力矩**：注意！**Action 没有变，目标角度 \(q_{\text{target}}\) 没变。** 但是！因为物理引擎刚才把关节转动了，**当前真实角度变成了 \(q_{t+1}\)**。
    2. 控制器根据最新的误差 \(q_{\text{target}} - q_{t+1}\)，**重新计算一次 \(K_p \Delta p\)**。
    3. 物理推进 5ms。

*   **第 3 次、第 4 次循环（\(t=10\) 到 \(20\) ms）：**
    同理，每次循环开始时，都用**最新的实际关节位置**，去和目标角度做差，重新算一遍 PD 力矩，然后再推进物理仿真。

**第三阶段：返回新状态（\(t=20\) ms）**
4 次循环结束，把这 20ms 内机器人发生的变化打包成新的 Observation，还给 PPO，准备下一次决策。

---

###### 纠正你的一个概念：“输出动作 0”

你说“输出动作 0 要把关节转到某个角度”，这句话容易引起歧义。

在 Unitree 的代码设定里，`action = 0` 并不代表“关节转到 0 度”，而是代表**“保持默认的站立姿态（`default_joint_angles`）”**。

*   如果 Policy 输出全 0，那 $(q_{\text{target}} = q_{\text{default}} + 0 \times 0.25 = q_{\text{default}})$。
*   这时候，PD 控制器算的就是：$K_p(q_{\text{default}} - \text{当前角度})$。它会把机器人拼命拉向初始设定的站立姿势。

---

# 今日的动手实践

今天做一个非常小的实验：

**尝试自己读取 URDF，统计机器人实际具有多少个可运动关节。**

在项目根目录运行：

```python
import xml.etree.ElementTree as ET

urdf_path = "resources/robots/g1_description/g1_12dof.urdf"

tree = ET.parse(urdf_path)
root = tree.getroot()

dof_names = []

for joint in root.findall("joint"):
    if joint.attrib["type"] != "fixed":
        dof_names.append(joint.attrib["name"])

print("DOF 数量:", len(dof_names))

for i, name in enumerate(dof_names):
    print(i, name)
```

你应该能够看到：

```
DOF 数量: 12
```

然后完成第二步：

在 `legged_robot.py` 中找到：

```
self.dof_names = self.gym.get_asset_dof_names(robot_asset)
```

思考：

**URDF 中的关节名称列表，与 Isaac Gym 加载之后得到的 `self.dof_names` 有什么关系？**

进一步思考：如果你想知道 PPO 输出的 `action[0]` 对应哪个关节，应该检查什么变量？

提示：不要只看 `default_joint_angles` 的排列顺序，应该追踪实际的 `dof_names`。



---
###### 解释这段代码：

这段代码非常棒！它实际上是在**脱离强大的 Isaac Gym 引擎，用最基础的 Python 原生代码，去窥探 URDF 文件的底层秘密**。

这段话的作用，就是让你亲手验证我们上一节讨论的结论：**“真实的可动关节到底有几个？它们叫什么名字？”**

我为你逐行拆解这段代码，并说明它与 Isaac Gym 底层逻辑的完美对应：

###### 1. 逐行代码分析

```python
import xml.etree.ElementTree as ET
```
**解释**：URDF（Unified Robot Description Format）本质上就是一个 XML 文件（你可以用记事本打开它，里面全是尖括号 `<xxx>`）。`xml.etree.ElementTree` 是 Python 自带的 XML 解析库，专门用来读取这种树状结构的文件。

```python
urdf_path = "resources/robots/g1_description/g1_12dof.urdf"
tree = ET.parse(urdf_path)
root = tree.getroot()
```
**解释**：
*   `ET.parse` 把 URDF 文件读进内存，变成一棵“树”。
*   `getroot()` 拿到这棵树的总根节点（在 URDF 里，最外层通常是 `<robot name="g1">`）。

```python
dof_names = []
```
**解释**：准备一个空购物车，用来装真正能动的关节名字。

```python
for joint in root.findall("joint"):
```
**解释**：在整棵 XML 树里，把所有名叫 `<joint>` 的标签全找出来。URDF 里每个关节都是一个 `<joint>` 标签。

```python
    if joint.attrib["type"] != "fixed":
        dof_names.append(joint.attrib["name"])
```
**解释**：这是整个代码的**核心灵魂**！

*   `joint.attrib["type"]` 是读取这个关节的属性。URDF 里关节类型有 `revolute`（旋转）、`continuous`（连续旋转）、`prismatic`（滑动）等等，还有一种最特殊的叫 **`fixed`（固定死，不能动）**。

*   这行 `if` 逻辑就是：**“把所有不是 `fixed` 的关节名字，装进购物车里！”**

```python
print("DOF 数量:", len(dof_names))
for i, name in enumerate(dof_names):
    print(i, name)
```
**解释**：打印出能动的关节总数（12），并按顺序打印出它们的名字和序号。

---

###### 2. 这段代码和 Isaac Gym 的联动关系

这段代码完美解释了 Isaac Gym 底层到底是怎么得到 `self.dof_names` 的。

*   在这个纯 Python 实验里：代码通过 `if type != "fixed"` 过滤出了 12 个关节。
*   在 Isaac Gym 的 C++ 引擎里：当我们调用 `self.gym.load_asset()` 时，引擎内部跑的逻辑和上面**一模一样**。它把 `fixed` 的关节（比如你配置里多出来的那个腰部）焊死，把剩下 12 个能动的丢出来。所以当我们调用 `get_asset_dof_count()` 时，它坚决返回 `12`。

###### 3. 为什么这个实验非常有意义？

这个实验戳破了两个新手常犯的误区：

1.  **误区一：“URDF 里写了 13 个 `<joint>`，我就有 13 个电机。”**
    
    *   **真相**：如果里面有 1 个是 `fixed` 的（比如把头部死死固定在了躯干上），那它就只是个铁疙瘩，物理引擎根本不会给它分配电机和自由度。所以实际上只有 12 DOF。
    
2.  **误区二：“程序员写代码是数配置文件的键。”**
    
    *   **真相**：看这段代码，它只从 URDF 里数 `type != "fixed"` 的关节，根本不看配置文件。
    

###### 💡 拓展知识：URDF 里还有哪些 Joint 类型？

既然有 `fixed`，那真正能动的是什么？

*   `revolute`：像门铰链一样的旋转关节（限制最大最小角度，G1的膝盖就是这种）。
*   `continuous`：像车轮一样无限制旋转的关节（可以无限转圈）。
*   `prismatic`：滑动关节（像抽屉一样），人形机器人通常用不到。

**建议实验**：

你可以按图里的代码在自己电脑上跑一下，或者直接去 VSCode 里打开 `g1_12dof.urdf` 文件，`Ctrl+F` 搜一下 `type="fixed"`，看看它到底长啥样，名字是不是叫 `waist_yaw_joint`。亲眼看到那

个被焊死的关节，你对代码的理解就彻底落地了。

---

###### 如何使用这段代码

###### 🚨 错误示范：直接粘贴到 Bash 终端
如果你在终端（提示符为 `root@...:~/unitree_rl_gym#`）直接粘贴 Python 代码，会报错：
```text
Command 'import' not found...
bash: syntax error near unexpected token `('
```
**原因**：Bash 不认识 Python 语法，它把 `import` 当成了系统命令去查找，自然找不到。

---

###### ✅ 正确操作步骤

###### 第一步：进入 Python 交互模式
在终端输入 `python` 或 `python3` 并回车。看到提示符变成 `>>>` 时，说明你进入了 Python 环境：

```bash
(rl) root@LAPTOP-TG1QBTTV:~/unitree_rl_gym# python
Python 3.8.20 (default, Oct  3 2024, 15:24:27) 
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

###### 第二步：逐行粘贴代码
将以下代码粘贴进去（注意每一行回车确认）：

```python
import xml.etree.ElementTree as ET

urdf_path = "resources/robots/g1_description/g1_12dof.urdf"
tree = ET.parse(urdf_path)
root = tree.getroot()

dof_names = []

for joint in root.findall("joint"):
    if joint.attrib["type"] != "fixed":
        dof_names.append(joint.attrib["name"])

print("DOF 数量:", len(dof_names))
```

此时你会看到输出：
```text
DOF 数量: 12
```

###### 第三步：打印具体的关节名字
接着粘贴最后一段循环打印代码：

**在 Python 交互模式（出现 `>>>` 和 `...`）里，当你写完一个 `for` 循环的代码块：

```python
>>> for i, name in enumerate(dof_names):
...     print(i, name)
...
```

 **注意**：在 Python 交互模式下，写完 `for` 循环的代码块后，你需要**再按一次回车（输入一个空行）**，告诉 Python “代码写完了，开始执行”。

**最后输出**

```
0 left_hip_pitch_joint
1 left_hip_roll_joint
2 left_hip_yaw_joint
3 left_knee_joint
4 left_ankle_pitch_joint
5 left_ankle_roll_joint
6 right_hip_pitch_joint
7 right_hip_roll_joint
8 right_hip_yaw_joint
9 right_knee_joint
10 right_ankle_pitch_joint
11 right_ankle_roll_joint
```

---

###### 🧠 核心知识点回顾

通过这段代码，我们解答了之前的疑问：

1. **为什么只有 12 个 Action？**
   代码里的 `if joint.attrib["type"] != "fixed":` 是关键。URDF 里有一个关节被设为了 `fixed`（比如 `waist_yaw_joint`），它被这行代码无情过滤掉了。

2. **配置文件里的 `torso_joint` 去哪了？**
   在打印出来的 12 个名字里，**根本没有 `torso_joint`**。这说明即使你在 `g1_config.py` 里写了 13 个键，因为 URDF 里没有这个名字的可动关节，程序根本不会去读它。它就是个纯粹的“废键”（可能是复制粘贴遗留的）。

3. **物理引擎的真实逻辑**：
   Isaac Gym 在底层调用 `get_asset_dof_names()` 时，干的事情和这段 Python 代码一模一样。只有 URDF 决定“谁能动”，配置文件无权干涉。**URDF 才是唯一的真理。**