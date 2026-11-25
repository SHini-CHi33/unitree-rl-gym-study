# 今日代码阅读

今日主题：Reward 函数——机器人究竟怎样判断自己走得好不好？

---

前五天，我们已经把 G1 的控制链路逐渐串起来了：

```text
Observation
    ↓
Actor / Policy
    ↓
12维 Action
    ↓
目标关节角度
    ↓
PD Controller
    ↓
4次物理仿真
    ↓
机器人产生新的运动状态
```

今天开始研究一个新的问题：

机器人产生运动之后，强化学习算法如何判断这个动作是好是坏？

这就是 Reward。

今天我不希望你直接背诵 G1 有哪些奖励项，而是希望你理解三个层次：

1. Reward 函数从哪里获取数据？
2. Reward 函数如何把机器人状态转换为奖励数值？
3. 多个 Reward 如何组合成最终交给 PPO 的奖励？

今天还会遇到一个很重要的 Python 设计：为什么配置文件中写一个 Reward 名称，程序就能自动找到对应的函数？

---

## 一、今天需要打开的三个文件

我已经核对了官方仓库当前 main 分支的代码结构。今天涉及的文件如下：

### 官方代码文件

**1. `legged_gym/envs/g1/g1_config.py`**

打开 `g1_config.py`，重点：

```python
G1RoughCfg.rewards.scales
```

**2. `legged_gym/envs/base/legged_robot.py`**

打开 `legged_robot.py`，重点：

- `_parse_cfg()`
- `_prepare_reward_function()`
- `compute_reward()`
- `_reward_tracking_lin_vel()`
- `_reward_action_rate()`

**3. `legged_gym/envs/g1/g1_env.py`**

打开 `g1_env.py`，重点：

- `_reward_contact()`
- `_reward_feet_swing_height()`
- `_reward_contact_no_vel()`

如果你的本地代码和官方 main 有差异，优先使用本地实际运行的版本。今天先不修改任何参数。

---

## 二、第一部分：先看看 G1 到底设计了哪些 Reward

打开 `g1_config.py`，找到：

```python
class rewards(LeggedRobotCfg.rewards):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.78

class scales(LeggedRobotCfg.rewards.scales):
    tracking_lin_vel = 1.0
    tracking_ang_vel = 0.5
    lin_vel_z = -2.0
    ang_vel_xy = -0.05
    orientation = -1.0
    base_height = -10.0
    dof_acc = -2.5e-7
    dof_vel = -1e-3
    feet_air_time = 0.0
    collision = 0.0
    action_rate = -0.01
    dof_pos_limits = -5.0
    alive = 0.15
    hip_pos = -1.0
    contact_no_vel = -0.2
    feet_swing_height = -20.0
    contact = 0.18
```

这里有一个非常重要的区别。

Reward 的名字对应一个评价目标，而数值是它的权重（Scale）。

例如：

| Reward | Scale | 目的 |
|---|---:|---|
| `tracking_lin_vel` | +1.0 | 跟踪目标线速度 |
| `tracking_ang_vel` | +0.5 | 跟踪目标角速度 |
| `orientation` | -1.0 | 惩罚身体倾斜 |
| `base_height` | -10.0 | 惩罚基座高度偏离目标 |
| `action_rate` | -0.01 | 惩罚 Action 变化过快 |
| `contact` | +0.18 | 鼓励脚部接触与步态相位匹配 |
| `feet_swing_height` | -20.0 | 惩罚摆动脚高度偏离目标 |

这里正负号很重要：

- 正数：这个指标越大，通常越有利。
- 负数：这个指标越大，通常惩罚越重。

但要注意，Scale 的正负不能单独说明一个 Reward 的好坏，还必须结合 Reward 函数本身的计算方式。

### 一个小观察

你会发现：

```python
feet_air_time = 0.0
collision = 0.0
```

这两个奖励虽然写在配置中，但它们的权重是零。

那么问题来了：

> 代码中有对应的 Reward 函数，是否就意味着它一定会被执行？

先记住这个问题，第三部分会揭晓。

---

## 三、第二部分：Reward 是怎样被自动注册的？

这是今天最值得研究的代码设计。

打开 `legged_robot.py`，找到：

```python
def _parse_cfg(self, cfg):
    self.dt = (
        self.cfg.control.decimation
        * self.sim_params.dt
    )
    self.reward_scales = class_to_dict(
        self.cfg.rewards.scales
    )
```

这里把配置中的 Reward Scale 转换为字典。

接下来找到：

```python
def _prepare_reward_function(self):
```

其关键逻辑是：

```python
# remove zero scales + multiply non-zero ones by dt
for key in list(self.reward_scales.keys()):
    scale = self.reward_scales[key]
    if scale == 0:
        self.reward_scales.pop(key)
    else:
        self.reward_scales[key] *= self.dt
```

接着：

```python
self.reward_functions = []
self.reward_names = []

for name, scale in self.reward_scales.items():
    if name == "termination":
        continue

    self.reward_names.append(name)
    name = '_reward_' + name

    self.reward_functions.append(
        getattr(self, name)
    )
```

##### 📖 1. `items()` 是什么？

在 Python 字典（Dictionary）里，`items()` 是一个内置方法。

- 如果你只写 `self.reward_scales`，你只能拿到菜名（`tracking_lin_vel`）或者价格（`1.0`）。
    
- 但加上 `.items()`，它就会把**菜名**和**价格**打包成一对，同时返回给你。

---
##### 2. 权重为 0 的 Reward 会被删除

例如：

```python
feet_air_time = 0.0
```

会在准备阶段被移除，因此不会进入普通 Reward 函数列表。

所以并不是代码中有 `_reward_feet_air_time()`，它就一定执行。

##### 🍽️ 为什么在这里要用 `pop`？（工程意义）

在 `_prepare_reward_function()` 里，代码的逻辑是：

> “遍历字典里所有的 reward 名字，如果发现它的权重是 0，就把它从字典里踢出去。”

**为什么要踢出去？**

1. **节省算力（最重要）**：如果 `feet_air_time = 0.0`，说明这次训练你根本不想要这个奖励。如果不删掉它，在后续 `compute_reward()` 的循环里，程序还是会**傻乎乎地去算一遍这个函数**，再乘以 0。算一遍涉及大量的张量（Tensor）运算，在几千个并行环境里，这会白白浪费巨大的 GPU 算力。
    
2. **清理列表**：删掉它之后，后面的代码 `for name, scale in self.reward_scales.items():` 就只会遍历真正有用的奖励。这就叫“净化数据”。
    

##### 🤔  为什么不用 `del` 删除？

在 Python 里，删除字典里的元素，还有一种写法是：


```python

del self.reward_scales[key]

```
它和 `pop` 的区别是：

- `del` 只是纯粹地删掉。
    
- `pop` 是删掉的同时，**把值当作返回值扔给你**。比如 `val = my_dict.pop(key)`，你不仅删了它，还拿到了它。
    

在咱们的代码里，作者选择用 `pop` 是一种常见的 Python 编程习惯。

##### ⚠️ 一个极其重要的细节：为什么有个 `list()`？

你仔细看那一行的代码：

```python

for key in list(self.reward_scales.keys()):

```

**为什么要用 `list()` 包一层？**  
因为如果你直接遍历字典 `for key in self.reward_scales.keys():`，然后在循环内部去 `pop` 这个字典。Python 会直接报错（RuntimeError: dictionary changed size during iteration）。  
这就好比你正在数一箱苹果，你数一个我拿走一个，你肯定会数错。用 `list()` 相当于**拍了一张快照**，你对着照片数，我拿走真实的苹果，互不干扰。

---
##### 3. 程序使用 getattr() 自动找到函数

假设配置中有：

```python
tracking_lin_vel = 1.0
```

程序会构造字符串：

```python
name = '_reward_' + name
```

最终得到：

```python
'_reward_tracking_lin_vel'
```

再通过：

```python
getattr(self, name)
```

获取对应的 Python 方法。

也就是说：

```python
self._reward_tracking_lin_vel
```

会被自动加入：

```python
self.reward_functions
```

这就是为什么配置文件只需要填写 Reward 名称，而不需要手动把每个函数注册到一个列表里。


##### ❌ 场景 A：不用 `getattr`（硬编码 / 强耦合）

假设你现在要给 H1 机器人**新加一个奖励**，叫 `foot_slip`（打滑惩罚），你在配置文件里写了 `foot_slip = -0.1`。

这时候，底层的主计算函数 `compute_reward()` 必须长这样：


```python

def compute_reward(self):
    total_reward = 0
    
    # 必须手动写死每一个调用
    if self.cfg.rewards.scales.tracking_lin_vel != 0:
        total_reward += self._reward_tracking_lin_vel() * self.cfg.rewards.scales.tracking_lin_vel
        
    if self.cfg.rewards.scales.alive != 0:
        total_reward += self._reward_alive() * self.cfg.rewards.scales.alive
        
    # 因为你新加了 foot_slip，你必须打开底层代码，在这里再写一遍！
    if self.cfg.rewards.scales.foot_slip != 0:
        total_reward += self._reward_foot_slip() * self.cfg.rewards.scales.foot_slip
        
    return total_reward

```
**这有什么致命缺点？**

1. **极其啰嗦**：如果有 20 个奖励，你得手写 20 个 `if`。
    
2. **容易出错**：如果手滑把 `_reward_foot_slip` 拼错了，程序直接崩溃。
    
3. **不敢改底层**：底层代码是整个框架的心脏。如果你每次加新奖励都要打开心脏改一刀，迟早有一天会把代码搞出 Bug。
    

##### 🪄 场景 B：用了 `getattr`（动态绑定 / 完全解耦）

在 `legged_gym` 里，不管你加多少个奖励，底层的 `compute_reward()` 永远只需要写这**三行**：


```python

def compute_reward(self):
    total_reward = 0
    # 遍历自动注册好的函数列表
    for i in range(len(self.reward_functions)):
        total_reward += self.reward_functions[i]() * self.reward_scales[self.reward_names[i]]
    return total_reward

```
**当你新加 `foot_slip = -0.1` 时，发生了什么？**  
你**完全不需要**去动底层的主循环代码。你只需要做两件事：

1. 去配置文件里加上 `foot_slip = -0.1`。
    
2. 在 `h1_env.py` 里写个 `def _reward_foot_slip(self): return ...`。
    

底层的自动注册机制（利用 `getattr`）会在启动时自动扫描：

- 它发现配置里有个 `"foot_slip"`。
    
- 它拼接出 `"_reward_foot_slip"`。
    
- 它用 `getattr(self, "_reward_foot_slip")` 成功在 `h1_env.py` 里找到了你写的函数。
    
- 它自动把这个函数加入到 `self.reward_functions` 列表里。
    

接下来，底层那个只有三行的主循环，就会自动把你新加的奖励算进去。

##### 🔍 `getattr` 是去哪里找这个 `name`？

`getattr(self, name)` 里的 `self`，就是当前实例化的那个机器人环境对象（比如 `H1Robot` 的实例）。

当 Python 执行 `getattr(self, '_reward_alive')` 时，它的查找顺序是：

1. **先找子类（H1Robot）**：去 `h1_env.py` 里找有没有 `def _reward_alive(self)`。
    
2. **再找父类（LeggedRobot）**：如果子类里没有，它会顺着继承链往上找，去 `legged_robot.py` 里找。
    

所以，不管是通用奖励还是 H1 专属奖励，只要在这个实例的“族谱（继承链）”里，`getattr` 都能精准把它揪出来。

##### 🤔 `self.reward_names` 好像没用到？

你在 `_prepare_reward_function` 这个函数里看，它确实只是被 `append`（添加）进去了，没看到后续使用。

**但是，如果你切到 `compute_reward()` 这个真正算分的函数里，你会发现它是绝对的核心！**

你看这段代码：


```python

for i in range(len(self.reward_functions)):
    name = self.reward_names[i]  # <--- 看这里！用到了！
    rew = self.reward_functions[i]() * self.reward_scales[name]
    self.rew_buf += rew
    self.episode_sums[name] += rew # <--- 极其重要！

```
##### 🧠 为什么非要存一个 `self.reward_names`？（两个核心原因）

###### 为了去字典里查“权重”（`self.reward_scales[name]`）

`self.reward_functions` 里面存的是**函数对象**（内存地址），它没有名字。  
当你算出 `_reward_alive()` 的值后，你需要去乘以它的权重 0.15。怎么查？只能靠 `self.reward_names[i]` 拿到字符串 `'alive'`，然后去字典里查 `self.reward_scales['alive']`。

###### 为了给 TensorBoard 记录单项数据（最关键的工程意义！）

你看最后那行 `self.episode_sums[name] += rew`。  
如果你只用 Total Reward，你在 TensorBoard 里只能看到一条总曲线。  
但有了 `name`，框架就可以把每一个子奖励单独记录。你打开 TensorBoard 就能看到：`Episode/rew_alive` 涨了，但 `Episode/rew_contact` 跌了。**这是你作为算法工程师调参的“仪表盘”！**

###### 列表索引的“对号入座”

`self.reward_functions` 是一个列表，`self.reward_names` 也是一个列表。它们俩是**严格平行对应的**：

- `self.reward_functions[0]` 算出来的值，肯定属于 `self.reward_names[0]`。
    
- 这样用 `for i in range(...)` 遍历时，就能完美匹配。

##### 🪄 `getattr` 到底返回了什么？

假设类里有这么个方法：


```python

def _reward_alive(self):
    return 1.0

```

当你执行 `func = getattr(self, "_reward_alive")` 时，Python 并不是把函数里面的代码给返回了，而是返回了一个“绑定了 `self` 的方法对象”**。  
你可以把它理解为一个**“带了遥控器的机器人”**：

- 这个对象自带 `self` 环境。
    
- 你现在不需要手动传 `self` 进去，只要在它后面加个括号 `func()`，它就会自动执行 `self._reward_alive()` 并返回结果。
    

##### 📦 为什么函数能放进列表？

因为在 Python 里，**函数是一等公民（First-class citizen）**。

- 在 C 语言等老派语言里，函数是指令的集合，你没法把它存在数组里。
    
- 但在 Python 里，**函数就是一块内存地址**。你把它 `append` 进列表，本质上就是把这个内存地址存进了列表。
    
- **你可以像遍历数字一样遍历函数**：


```python
    
    my_list = [print, max, min] # 存函数
    for func in my_list:
        func([1, 2, 3]) # 挨个执行！
        
```    

##### 🍽️ 在咱们代码里，这种设计的意义是什么？

这叫做“延迟执行”**或者**“回调（Callback）”**。

在 `_prepare_reward_function` 里，代码只是把函数“装进盒子”：


```python

self.reward_functions.append(getattr(self, name))

```
注意，这里**没有加括号**！它只是把“做菜的方法”放进了盒子里，**并没有真正开始做菜**。

等到环境运行起来，需要算奖励时，底层的 `compute_reward()` 才会从盒子里把函数拿出来，挨个加上括号执行 `self.reward_functions[i]()`。这就是为什么框架能做到动态加载、灵活插拔的核心！

---

##### 🚧 4. `if name=="termination": continue` 是什么意思？

`continue` 是 Python 的专属语法，意思是：**“这次循环到此为止，后面的代码不执行了，直接跳到字典里的下一个项去。”**

这就好比你在过安检，只要机器报警（`name=="termination"`），保安直接把你拦下（`continue`），不让你进下一个流程。

**那么termination的奖励被放在哪里呢

虽然 `termination` 被 `continue` 跳过了，没有加入自动注册的列表。但在 `compute_reward()` 这个函数的**最末尾**，作者留了一手专门对付它：


```python

def compute_reward(self):
    self.rew_buf[:] = 0.
    
    # 1. 跑所有自动注册的常规奖励
    for i in range(len(self.reward_functions)):
        name = self.reward_names[i]
        rew = self.reward_functions[i]() * self.reward_scales[name]
        self.rew_buf += rew
        self.episode_sums[name] += rew
        
    # 2. 只有这里！专门给 termination 开小灶！
    if "termination" in self.reward_scales:
        rew = self._reward_termination() * self.reward_scales["termination"]
        self.rew_buf += rew
        self.episode_sums["termination"] += rew

```
##### 🧠 为什么非要把它单拎出来开小灶？

因为**它的计算公式和触发机制，跟普通的奖励完全不一样！**

###### 1. 普通奖励是“连续积分”（每一帧都在算）

比如 `tracking_lin_vel`，机器人只要在走，每一帧（20ms）都会产生一个奖励值。这就是我们前面讲的，要乘以 `dt`。

###### 2. `termination` 是“瞬间爆炸”（事件驱动）

你看上面代码里的 `self._reward_termination()`。如果点开这个函数，你会发现它长这样：


```python

def _reward_termination(self):
    # Terminal reward / penalty
    return self.reset_buf * ~self.time_out_buf

```
- 如果机器人没死，`reset_buf` 是 0，这个函数返回 0。
    
- 如果机器人因为**摔断腿（非超时）死了**，`reset_buf` 变成 1，这个函数瞬间返回 1。
    

**为什么不能放进循环里？**  
如果放进循环里，它也会被自动乘上 `dt`！  
原本设定死亡惩罚是 `-50.0`，如果乘了 `dt`（0.02），就变成了 `-1.0`。  
那机器人心里会想：“我摔死才扣 1 分，但我每走一步都能拿 0.5 分，那我干脆每走两步就摔死一次，疯狂重置刷分好了！”  
**死亡惩罚是瞬间的、离散的，绝对不能乘时间积分。** 所以必须把它单独提出来，不乘 `dt`，死一次就直接扣 50 分。

---

### 总结Reward 自动注册机制

```text
G1 配置文件
    |
    | tracking_lin_vel = 1.0
    v
字符串拼接
    |
    | "_reward_" + "tracking_lin_vel"
    v
getattr(self, name)
    |
    | 找到实际 Python 方法
    v
self.reward_functions
    |
    | 加入待执行函数列表
    v
```

还有一个进阶细节：

`G1Robot` 继承了 `LeggedRobot`。

因此，`getattr(self, name)` 不仅可以找到 `legged_robot.py` 中定义的父类方法，也可以找到 `g1_env.py` 中定义的 G1 子类方法。

例如：

```python
_reward_tracking_lin_vel
```

来自父类，而：

```python
_reward_contact
```

来自 G1 子类。

这意味着 G1 可以在通用机器人环境基础上增加自己的步态奖励。

这一点很方便，以为无论是机器人还是机器狗都需要摔倒、线速度惩罚，这些可以放在父类，而在子类只需要写机器人或者机器狗独特的reward就行，如接触地面reward

---

## 四、第三部分：研究三个真正重要的 Reward 函数

今天不要把所有奖励项都读完。只需要理解三个。

### 1. `_reward_tracking_lin_vel()`：机器人有没有按照目标速度走？

打开 `legged_robot.py`：

```python
def _reward_tracking_lin_vel(self):
    # Tracking of linear velocity commands (xy axes)
    lin_vel_error = torch.sum(
        torch.square(
            self.commands[:, :2]
            - self.base_lin_vel[:, :2]
        ),
        dim=1
    )
    return torch.exp(
        -lin_vel_error / self.cfg.rewards.tracking_sigma
    )
```

先理解输入：

- `self.commands[:, :2]`：机器人被要求达到的 X、Y 方向速度。（在 Day5 我们讲过，command作为状态输入，后面还有一个$yaw_{vel}$角速度，所以我们只取$[0,1]$两个速度，注意$[2]$取不到）
- `self.base_lin_vel[:, :2]`：机器人实际的 X、Y 方向速度。

这里的实际速度是在机器人基座坐标系中表示的。

计算过程：

$$
e_v=
(v_x^{cmd}-v_x)^2+
(v_y^{cmd}-v_y)^2
$$

然后：

$$
r_{\text{tracking}}
=
\exp\left(-\frac{e_v}{\sigma}\right)
$$

其中 $\sigma$ 是配置中的 `tracking_sigma`。

这个函数有一个很有意思的性质：

##### 目标速度跟踪误差与原始 Reward 的关系

| 速度误差平方和 | 原始 Reward |
|---:|---:|
| 0 | 1.000 |
| 0.05 | 0.819 |
| 0.25 | 0.368 |
| 0.50 | 0.135 |
| 1.00 | 0.018 |

> 以下是数学示意，假设 $\sigma = 0.25$
> 误差越大，奖励越接近 0；误差为 0 时奖励为 1。

因此，这个 Reward 实际在告诉 Policy：

> 你不仅要让机器人移动，还要让机器人按照我指定的速度移动。


**思考：为什么不直接使用** $-lin\_vel\_error$ **，而要使用指数函数？**

答：用指数函数添加符号，可以将奖励严格控制在$[0,1]$之间，至于$\sigma$，则是根据机器人速度大小和具体调参设计出的缩放比例

### 2. `_reward_action_rate()`：机器人是否频繁改变动作？

继续看 `legged_robot.py`：

```python
def _reward_action_rate(self):
    # Penalize changes in actions
    return torch.sum(
        torch.square(
            self.last_actions - self.actions
        ),
        dim=1
    )
```

这个函数的逻辑很简单：

$$
r_{\text{action-rate}}
=
\sum_i(a_{t-1,i}-a_{t,i})^2
$$

注意，这个函数返回的是一个非负数，但 G1 的 Scale 是：

$$
w_{\text{action-rate}}=-0.01
$$

所以最终它是一个惩罚项。

为什么需要它？

假设 Policy 每隔 20ms 输出一次 Action：

**情况 A：动作变化平缓**

相邻 Action 差值较小，惩罚较小。

**情况 B：动作频繁跳变**

相邻 Action 差值较大，惩罚较大。

这里要特别区分：

> `action_rate` 直接约束的是 Policy 输出的 Action 变化，而不是直接计算关节角速度或力矩变化。

因为 Action 会经过 `action_scale` 转换为目标关节角度，所以它会间接影响机器人运动的平滑程度。

**为什么追踪线速度用exp，而动作连续却不用
###### 🎁 1. 收益型（如追踪速度）：为什么必须用 `exp()`？

假设你不用指数函数，直接用 −lin_vel_error

- 如果机器人在仿真里刚开始，它还不会走，误差极大（比如误差平方和是 100）。
    
- 那么这一帧的奖励就是 `-100`。
    
- **灾难来了**：这一帧的 -100 会直接主导整个 PPO 的梯度方向，网络为了拯救这个巨大的负分，会疯狂更新参数，导致之前学到的所有东西瞬间被抹除，这叫**梯度爆炸/支配**。
    

**用 `exp(-error / sigma)` 的三大好处：**

1. **完美封顶**：不管你做得多完美，哪怕误差是 0，奖励最高就是 `exp(0) = 1.0`。它不会因为你表现好，就给你一个 10000 分去破坏整个系统的奖励平衡。
    
2. **下限平滑**：如果你表现极差（误差极大），奖励会趋近于 0，但**永远不会变成巨大的负数**。它只是失去奖金，不会把你直接打入死牢。
    
3. **平滑的梯度**：指数函数曲线极其平滑，越接近 0 误差（完美状态），梯度越小，网络越稳定。而在误差很大时，它能提供一个温和的拉扯力，让网络慢慢变好，而不是瞬间崩盘。
    

---

###### ⚡ 2. 惩罚型（如动作变化率）：为什么用平方和 `square()`？

`action_rate` 的目的是：**“你不许高频抽搐，动得越猛，我罚得越狠。”**

假设我们也给它加个 `exp()`，变成 `exp(-(a_t - a_{t-1})^2)`。

- 如果动作变化极大，这个值最高是 1；变化极小，值也是接近 1（因为 -0 还是 0，exp(0)=1）。这不仅没有惩罚效果，还变成了**奖励**网络去抽搐！方向反了。
    

如果变成 `-exp(error)`（越乱动负得越多）：

- 假设动作剧烈变化（差值平方和为 100），惩罚为 `-exp(100)`，这个数字大到宇宙爆炸，网络直接瘫痪。
    

**用平方和 `square()` 的巧妙之处：**

1. **无上限惩罚**：平方和没有上限！如果你只是微微抖动，平方和是 0.001，乘以 `-0.01` 的权重，扣分微乎其微；但如果你疯狂抽搐，平方和变成 100，扣分立刻放大到 `-1.0`，甚至更大。**它逼着网络必须保持动作极度平滑。**
    
2. **物理上的能量约束**：在物理世界里，能量的消耗、电机的发热、磨损，通常与速度/加速度的**平方**成正比。用平方和惩罚，完美对齐了真实的物理痛点。
    
3. **无需封顶**：惩罚项的 Scale（如 -0.01）是人定的，工程师可以通过调节这个 Scale，确保它在总奖励中占据合理的权重（比如扣分最多也就扣个 -0.5），不需要用 `exp()` 来限制它。

### 3. `_reward_contact()`：脚的接触状态是否符合步态？

现在进入 G1 特有的 Reward。

打开 `g1_env.py`：

```python
def _reward_contact(self):
    res = torch.zeros(
        self.num_envs,
        dtype=torch.float,
        device=self.device
    )

    for i in range(self.feet_num):
        is_stance = self.leg_phase[:, i] < 0.55

        contact = (
            self.contact_forces[
                :, self.feet_indices[i], 2
            ] > 1
        )

        res += ~(contact ^ is_stance)

    return res
```

这一段是今天最值得你认真读的地方。

先理解两个变量：

`is_stance`

根据步态相位，判断当前脚是否应该处于支撑阶段。

```python
is_stance = self.leg_phase[:, i] < 0.55
```

***解读代码：

**一、为什么是0.55

###### 🚶 仿生学原因：人类走路时，脚不是 50/50 分配的

如果设置成 `0.5`，意味着左脚踩地 50%，右脚踩地 50%，严丝合缝，没有一丝重叠。  
但在真实的生物力学中，人类走路时，**两只脚是有一段时间同时踩在地上的**，这叫**“双支撑相（Double Support Phase）”**。

- 左脚相位是 `0 ~ 0.55`（踩地）。
    
- 右脚相位是 `0.5 ~ 1.05`（因为代码里有一个 `+0.5` 的偏移，取模后就是 `0.5 ~ 0`）。
    
- 你看，在 `0.5 ~ 0.55` 这 **5% 的时间**里，**左右两只脚同时都处于踩地状态**。
    

###### 🧗 控制学原因：为了“重心转移”留出容错时间

人形机器人在走路时，最危险的动作就是“换脚”。当它把重心从左脚移到右脚时，如果严格执行 50/50 交替，换脚的瞬间会极其突兀，机器狗很容易因为重心没稳住而摔倒。

留出这 5% 的双脚同时踩地时间，相当于给了机器人一个**“缓冲带”**，让它能从容地把重心从一只脚挪到另一只脚，这就是走路稳的秘诀。

###### 💡 行业里的 “Duty Factor（占空比）”

在机器人界，这个支撑相占整个周期的比例，叫做**占空比（Duty Factor）**。

- **步行（Walking）**：占空比 > 0.5（比如 0.6）。总有脚在地上，绝不腾空。
    
- **小跑（Trotting）**：占空比 = 0.5。对角线上的两只脚同时落地。
    
- **奔跑（Running）**：占空比 < 0.5。有双脚同时离地的“腾空期（Flight Phase）”。
    


**二、$i$ 取什么

###### 🦶  物理上：`i` 就是 0 和 1

G1 机器人有两只脚。  
在代码里：

- 当 `i = 0` 时，代表**左脚**。
    
- 当 `i = 1` 时，代表**右脚**。
    


###### 🧮  张量切片：`[:, i]` 在干嘛？

`self.leg_phase` 是一个矩阵（张量），它的形状是 `[num_envs, 2]`。

- `num_envs` 是并行环境的数量（比如 4096 个机器狗同时在跑）。
    
- `2` 代表两只脚。
    

`self.leg_phase[:, i]` 的语法是：

- `:`（冒号）代表**“所有行”**（也就是把 4096 个环境的都要了）。
    
- `i` 代表**“第 i 列”**。
    

**所以：**

- `self.leg_phase[:, 0]` 取出来的是一个包含 4096 个数字的数组，代表这 4096 个环境里，**每一只左脚**当前的步态相位。
    
- `self.leg_phase[:, 1]` 取出来的是 4096 个环境里，**每一只右脚**当前的步态相位。
    

`contact`

根据实际接触力，判断脚是否真的接触地面。

```python
contact = self.contact_forces[
    :, self.feet_indices[i], 2
] > 1
```

***解读代码：

###### 🌍 1. 力的来源：物理引擎

首先，`self.contact_forces` 这个东西**不是代码算出来的，是物理引擎（Isaac Gym）算出来的**。  
在环境的初始化阶段，代码调用了 Isaac Gym 的 API（类似 `acquire_net_contact_force_tensor`），拿出了一个巨大的三维张量。

- **它的形状是**：`[num_envs, num_bodies, 3]`。
    
    - `num_envs`：有多少个并行环境（比如 4096）。
        
    - `num_bodies`：机器人身上有多少个刚体（躯干、大腿、小腿、脚掌等，比如有 20 个）。
        
    - `3`：代表 X、Y、Z 三个方向的受力（牛顿）。
        

所以，这个张量里装的是**整个机器人全身每一块骨头受到的所有碰撞力**。

###### 🔪 2. 切片：精准找到那只脚

代码里的 `[:, self.feet_indices[i], 2]` 干了三件事：

1. “`:`” 表示取所有并行环境（4096 个机器人的数据我全要）。
    
2. `self.feet_indices[i]` ：这是一个极其关键的索引。它根据脚在机器人刚体列表里的位置，精准定位到那只脚的刚体 ID。假设左脚的编号是 10，右脚的编号是 11。
    
3. `2` ：取 Z 轴的力。因为在平地上踩踏，主要靠的是垂直方向的支撑力（X 和 Y 通常是地面摩擦力的滑动，Z 才是真正的压力）。
    

###### ⚖️ 3. 阈值判断：为什么是 `> 1`？

如果你光拿到 Z 轴的力，它可能是一个很小的浮点数（比如 `0.05` 牛顿），因为脚在空气中轻微晃动，或者传感器自带微小的数值噪声。  
如果直接用 `> 0`，那机器人随便抖一下，或者膝盖碰到空气，都会被判定为“踩在地上”，这就乱套了。

所以，设置了一个阈值 **`> 1`（牛顿）**。

- 力大于 1 牛顿，说明脚结结实实地踩在了地面上。
    
- 力小于 1 牛顿，说明脚还在空中。
    

**最关键的是，`> 1` 这个判断，返回的不再是力的大小，而是一个布尔值（True / False）！**  
这就是为什么前面的代码里，`contact` 可以直接和 `is_stance` 做异或运算。

---

接下来：

```python
~(contact ^ is_stance)
```

这里：

- `^` 是 XOR，即异或。
- `~` 是布尔取反。

我们把它展开：

1. **预期踩地 (True)，实际也踩地 (True)**：异或结果 = `False`。取反 = **`True`（奖励 1 分）**。完美！
    
2. **预期踩地 (True)，实际没踩 (False)**：异或结果 = `True`（因为不一样）。取反 = **`False`（0 分）**。没踩稳，不奖励。
    
3. **预期抬腿 (False)，实际抬腿 (False)**：异或结果 = `False`。取反 = **`True`（奖励 1 分）**。完美！
    
4. **预期抬腿 (False)，实际踩了 (True)**：异或结果 = `True`。取反 = **`False`（0 分）**。绊到脚了，不奖励。
    
> **一句话总结：如果你在“该踩地的时候踩地”，或者在“该抬腿的时候抬腿”，我就给你加 1 分。如果反着来，就不给分。**
##### ❌ 为什么不能用 `OR` 或者 `AND`？

你看代码想要达成的目标：

1. 预期踩地 (True) 且 实际踩地 (True) 👉 **得分**。
    
2. 预期抬腿 (False) 且 实际抬腿 (False) 👉 **得分**。
    
3. 预期踩地，实际没踩 👉 **不得分**。
    
4. 预期抬腿，实际踩了 👉 **不得分**。
    

如果你用 **`OR`**（`is_stance or contact`）：

- 预期踩地(True) OR 实际踩地(True) = True（得分，正确）。
    
- 预期抬腿(False) OR 实际抬腿(False) = False（**不得分，错误！** 这种情况应该得分啊！）
    
- 预期踩地(True) OR 实际没踩(False) = True（**得分，错误！** 这里应该扣分啊！）
    

如果你用 **`AND`**（`is_stance and contact`）：

- 预期踩地(True) AND 实际踩地(True) = True（得分，正确）。
    
- 预期抬腿(False) AND 实际抬腿(False) = False（**不得分，错误！** 应该得分啊！）
    
- 预期踩地(True) AND 实际踩地(False) = False（不得分，正确）。
    

**你看，无论是 `OR` 还是 `AND`，都无法完美表达“预期和实际完全一致（哪怕是同为 False）”这个物理逻辑。**

左右两只脚分别计算，因此原始 Reward 可以是 0、1 或 2。

这实际上是一个步态先验：

> Policy 不仅要学会让机器人向前走，还需要让脚部接触状态与预设的步态相位相匹配。

注意，这里不是直接规定左右腿的关节角度，而是通过接触状态对步态进行引导。

---

## 五、第四部分：所有 Reward 怎样合并？

现在回到 `legged_robot.py`，找到：

```python
def compute_reward(self):
    self.rew_buf[:] = 0.

    for i in range(len(self.reward_functions)):
        name = self.reward_names[i]
        rew = (
            self.reward_functions[i]()
            * self.reward_scales[name]
        )
        self.rew_buf += rew
        self.episode_sums[name] += rew
```

你会发现，所有 Reward 最终都累加到：

```python
self.rew_buf
```

而且之前 `_prepare_reward_function()`（就是`scale==0`在字典中删去，不为零就乘$dt$的那个部分） 已经做过：

```python
self.reward_scales[key] *= self.dt
```

因此，对于普通 Reward 项，可以写成：

$$
\boxed{
r_t=
\sum_i w_i\cdot dt\cdot f_i(s_t,a_t)
}
$$

这里：

- $f_i$：某一个 Reward 函数的原始输出。
- $w_i$：配置文件中的 Scale。
- $dt$：Policy 时间步。
- $r_t$：所有普通 Reward 项的加权和。

在当前 G1 配置中：

$$
dt=4\times0.005=0.02s
$$

例如，假设某一时刻：

$$
f_{\text{tracking}}=0.8
$$

那么它对最终 Reward 的贡献为：

$$
1.0\times0.02\times0.8=0.016
$$

再假设：

$$
f_{\text{action-rate}}=2
$$

那么：

$$
-0.01\times0.02\times2=-0.0004
$$

两个项目合计：

$$
r=0.016-0.0004=0.0156
$$

> 以上是用于理解计算流程的假设数据，不是实际训练日志。

最后：

```python
self.rew_buf += rew
```

就把不同奖励项累加起来。

另外，`termination` 是单独处理的：普通 Reward 循环之后，如果配置中有这个项目，会另外计算终止惩罚。因此上面的公式主要描述普通 Reward 项。

# 今日问题

这次希望你不要只解释代码做了什么，还要解释为什么这样设计。

### 问题 1：Reward 注册机制

假设 G1 配置中有：

```python
action_rate = -0.01
collision = 0.0
```

为什么 `_reward_action_rate()` 会进入 `reward_functions`，而 `_reward_collision()` 不会？

请结合 `_prepare_reward_function()` 解释。

答：`_prepare_reward_function()` 中对`reward_scale`进行了处理，`scale`为零的用pop移除，`scale`不为零的乘$self.dt$，进入`reward_functions`


### 问题 2：速度跟踪

假设：

$$
v^{cmd}=[1.0,0]
$$

情况 A：

$$
v=[0.9,0]
$$

情况 B：

$$
v=[0.2,0]
$$

哪一种情况会得到更高的 `_reward_tracking_lin_vel()`？

为什么要对速度误差平方求和，再使用指数函数？

答：情况A的`reward`更高； `_reward_tracking_lin_vel()`主要是控制机器人线速度，让它尽可能接近我们输入的$command$，比这个数很多或低很多都给较小的$reward$，因为对于奖励型而非惩罚型`reward`，我们需要将输出控制在$[0,1]$，所以用指数函数，并且在平方前乘一个负号，这样既保证了`reward`在$[0,1]$之间又保证了误差为0时奖励为1，误差越大奖励越接近0

#### 💡DeepSeek的修正与补充

- **补充漏点（为什么需要“平方”）**：你解答了指数函数的作用，但忽略了一个细节——**为什么要在指数里用 `torch.square()`（平方）？**
    
    - **修正/补充**：用平方（L2 范数）有两个原因：一是消除正负误差的符号影响（如果不用平方，X轴误差+1和Y轴误差-1加起来可能抵消为0，但实际偏离极大）；二是**平方会给予大误差更严厉的惩罚**（误差 2 平方是 4，误差 1 平方是 1）。配合外面的指数负号，能让速度偏离越严重，奖励断崖式下跌。


### 问题 3：Action Rate

如果连续两个 Policy 时间步输出完全相同的 Action：

$$
a_t=a_{t-1}
$$

那么：

```python
_reward_action_rate()
```

返回多少？

如果 Action 变化很大，结合 G1 的负权重，最终对总 Reward 有什么影响？

答：返回$0$，如果变化很大，平方和后乘以$-0.01$，将会给很大的惩罚，其他奖励在$[0,1]$之间，会被巨大惩罚盖过，机器人会优先并且不断牢记应该让$Action$变化较小

#### 💡DeepSeek的修正与补充

- **面试拔高（防挖坑）**：面试官可能会顺着你的话问：“那如果这个惩罚太大，把正向奖励都盖过了，会发生什么？”
    
    - 你要回答：“如果惩罚过大，机器人会陷入**‘什么都不做最安全’的局部最优**。它宁愿原地不动（动作差值为 0，不扣分），也不愿迈步（迈步必然产生动作差值）。所以 `reward_scale` 的配比必须由工程师精心调教，防止惩罚项喧宾夺主。”

### 问题 4：G1 特有的 Contact Reward

为什么 `_reward_contact()` 同时需要：

```python
self.leg_phase
```

和：

```python
self.contact_forces
```

如果只使用 `contact_forces`，不使用 `leg_phase`，还能够判断机器人是否符合预设步态吗？

答：不行，因为`self.leg_phase`判断做机器人步态相位判断，`self.contact_forces`做机器人接触判断，两者用异或取反来判断机器人有没有符合预设步态，假设只用`contact_forces`，此时发现机器人`contact_forces` 是`true`，但不知道此刻是否处于抬脚相位，如果应该抬脚却接触，则应该返回0，如果应该接触，则返回1，但是没有`leg_phase`就无法判断

### 问题 5：Reward 的时间缩放

为什么 `_prepare_reward_function()` 要将：

```python
self.reward_scales[key]
```

乘以：

```python
self.dt
```

如果将 Policy 时间步从 20ms 改成 40ms，但其他 Reward Scale 不变，单次 Transition 的普通 Reward 会受到什么影响？

答：由公式$r_t= \sum_i w_i\cdot dt\cdot f_i(s_t,a_t)$ ，单次`reward`会增大，对应的就是动作持续40ms的平均`reward`

#### 💡DeepSeek的修正与补充

- **纠正细节（“平均” vs “积分”）**：
    
    - 你说的“动作持续 40ms 的平均 reward”这个词不太准确，应该是“累积”**。
        
    - **严谨解释**：配置里的 $w_i$ 代表的是“每秒的奖励速率”**。如果 `dt` 从 0.02s 变成 0.04s，单次 transition 的时间变长了。乘以 `dt` 后，单次的 `reward` 确实增大了（翻倍了）。但从总物理时间（比如 1 秒）来看，policy 的步数减半了（50步变25步），单步奖励翻倍，总回报（Return）其实是保持不变的**。这样才能保证 PPO 的 $γ$ 折扣因子在相同的物理时间尺度上工作。

---

# 今天的实践练习：手动计算一次 Reward

无需训练 · 只需 10 分钟

今天不需要启动 Isaac Gym。

你只需要自己写一个小 Python 程序，计算三个 Reward。

```python
import math

# 1. Tracking velocity
command = [1.0, 0.0]
actual = [0.8, 0.0]
sigma = 0.25

error = sum(
    (command[i] - actual[i]) ** 2
    for i in range(2)
)
tracking_reward = math.exp(-error / sigma)

# 2. Action rate
last_action = [0.1, 0.2, 0.3]
action = [0.2, 0.1, 0.3]

action_rate = sum(
    (last_action[i] - action[i]) ** 2
    for i in range(3)
)

# 3. Contact reward
# 假设左右脚都处于预期支撑阶段
is_stance = [True, True]
contact = [True, False]

contact_reward = sum(
    not (contact[i] ^ is_stance[i])
    for i in range(2)
)

print("tracking:", tracking_reward)
print("action_rate:", action_rate)
print("contact:", contact_reward)
```

你应该能够得到：

```text
tracking: 0.8521...
action_rate: 0.02
contact: 1
```

然后再自己添加：

```python
dt = 0.02

total_reward = (
    tracking_reward * 1.0 * dt
    + action_rate * (-0.01) * dt
    + contact_reward * 0.18 * dt
)

print("total_reward:", total_reward)
```

这一步的意义是：你不再只是知道 G1 有哪些 Reward，而是能独立复现它们的计算过程。

---

## 八、今天应该留下的学习笔记

完成后，建议你留下下面这张表：

| 层次 | 需要理解的内容 |
|---|---|
| 配置 | `G1RoughCfg.rewards.scales` |
| 注册 | `_prepare_reward_function()` |
| 计算 | `_reward_xxx()` |
| 加权 | `Reward × Scale × dt` |
| 汇总 | `self.rew_buf` |
| 记录 | `self.episode_sums` |
| 交给算法 | `env.step()` 返回的 Reward |

今天尤其要记住：

$$
\boxed{
\text{配置决定启用哪些 Reward}
}
$$

$$
\boxed{
\text{Reward 函数决定如何评价机器人}
}
$$

$$
\boxed{
\text{Scale 决定每种评价对最终目标的影响}
}
$$