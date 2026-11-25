# 今日阅读任务

**今日主题：Reset、Termination、Command 采样与 Domain Randomization

今天开始进入环境管理机制。前六天我们已经研究了机器人如何感知、决策、执行动作，以及如何计算奖励。

现在要回答一个非常重要的问题：

> 如果 4096 个并行环境中，有一个 G1 摔倒了，训练程序会怎么办？

是整个训练停止？还是只重置摔倒的机器人？

另外，机器人每次重新站起来时，为什么初始状态不完全一样？为什么训练过程中还会受到随机推力？

这些都属于今天的内容。

---
## 一、今天需要打开的文件

我核对了 Unitree 官方仓库当前 main 分支。

| 文件                                            | 具体函数 / 配置                       |
| --------------------------------------------- | ------------------------------- |
| `legged_gym/envs/g1/g1_config.py`             | `domain_rand`、`asset`           |
| `legged_gym/envs/base/legged_robot.py`        | `post_physics_step()`           |
| `legged_gym/envs/base/legged_robot.py`        | `check_termination()`           |
| `legged_gym/envs/base/legged_robot.py`        | `reset_idx()`                   |
| `legged_gym/envs/base/legged_robot.py`        | `_reset_dofs()`                 |
| `legged_gym/envs/base/legged_robot.py`        | `_reset_root_states()`          |
| `legged_gym/envs/base/legged_robot.py`        | `_post_physics_step_callback()` |
| `legged_gym/envs/base/legged_robot.py`        | `_resample_commands()`          |
| `legged_gym/envs/base/legged_robot.py`        | `_process_rigid_shape_props()`  |
| `legged_gym/envs/base/legged_robot.py`        | `_process_rigid_body_props()`   |
| `legged_gym/envs/base/legged_robot.py`        | `_push_robots()`                |
| `legged_gym/envs/base/legged_robot_config.py` | `commands`、`env`、`domain_rand`  |

源码链接：

- 官方 G1 配置
- 官方基础机器人环境
- 官方基础配置

有一个版本细节：G1 的 `commands` 配置没有在 `G1RoughCfg` 中重新定义，而是继承基础配置。因此今天要同时阅读 G1 和基础配置，不能只看 G1 文件。

## 二、第一部分：Termination——机器人什么时候算摔倒？

打开：

`legged_gym/envs/base/legged_robot.py`

找到：

```python
def check_termination(self):
```

官方当前代码的核心逻辑：

```python
self.reset_buf = torch.any(
    torch.norm(
        self.contact_forces[
            :, self.termination_contact_indices, :
        ],
        dim=-1
    ) > 1.,
    dim=1)

self.reset_buf |= torch.logical_or(
    torch.abs(self.rpy[:, 1]) > 1.0,
    torch.abs(self.rpy[:, 0]) > 0.8)

self.time_out_buf = (
    self.episode_length_buf > self.max_episode_length)

self.reset_buf |= self.time_out_buf
```

我们拆成三个条件。

### 1. 关键部位发生碰撞

G1 配置：

```python
class asset(LeggedRobotCfg.asset):
    penalize_contacts_on = ["hip", "knee"]
    terminate_after_contacts_on = ["pelvis"]
```


- **物理意义**：如果你（机器人）的**骨盆（pelvis）** 或者**脑袋**重重地砸在了地上，你就“死”了。
    
- **代码解析**：
    
    - `self.termination_contact_indices`：这里提取了你在配置里写的 `terminate_after_contacts_on = ["pelvis"]`（骨盆的 ID）。
        
    - `torch.norm(..., dim=-1)`：对 X、Y、Z 三个方向的接触力求模长（总受力大小）。
        
    - `> 1.`：如果这个总受力大于 1 牛顿，说明真撞上了。
        
    - `torch.any(..., dim=1)`：只要这些关键部位里，**有任何一个**撞到了，就返回 True，把当前环境标记为需要重置（Reset）。
        

 **区分两个极易搞混的概念：**
 
> 
> - `penalize_contacts_on`（惩罚）：比如 G1 写了 `["hip", "knee"]`。如果膝盖磕了一下，**不判死**，只扣除奖励。
>     
> - `terminate_contacts_on`（终止）：比如写了 `["pelvis"]`。骨盆着地，直接判死，立刻重开。
>

而 `check_termination()` 通过：

```python
self.contact_forces[
    :, self.termination_contact_indices, :
]
```

检查这些部位的接触力。

当接触力模长超过 1 N，就会触发对应环境的 Reset 标记。

**问题：为什么受力大于1就是碰到了，难道电机给他的力不可能有1N吗

##### 🚗 1. 概念区分：电机力矩 vs 接触力

- **电机发出的力（Torque）**：这是机器人的内部驱动力。比如你想让膝盖弯曲，电机需要输出 50 牛·米的力矩。这个力非常大。
    
- **接触力（Contact Force）**：这是**外界环境对机器人的反作用力**。根据牛顿第三定律，只有当机器人的某个部位**撞到了外物（地面、墙壁等）**，这个力才会产生。
    

**打个比方**：你在空中挥拳，你可以用很大的力气（电机力），但你的拳头打不到人，就没有接触力。只有当你的拳头打在沙袋上，你才会感受到沙袋给你的反作用力（接触力）。

`self.contact_forces` 提取的，就是这个环境对机器人的**反作用力**。

##### 🧮 2. 为什么阈值是 1 N？（1N 到底有多大？）

1 牛顿的力，大约是**100 克物体的重力**（相当于你手里拿着一个鸡蛋感受到的重量）。

在真实的物理世界和仿真引擎里：

- 如果机器人的骨盆（pelvis）悬在空中，物理引擎计算出的接触力是**绝对的 0**（或者由于浮点精度偶尔会飘出个 `0.00001` 的极微小噪声）。
    
- 只有当骨盆真的磕到了地面，哪怕只是轻轻擦了一下，接触力都会瞬间飙升到**几十、几百甚至上千牛顿**（因为机器人本身可能有二三十公斤重）。
    

所以，把阈值设为 `> 1 N`，根本不是因为电机力会超过 1N，而是为了**过滤掉空气中或者浮点计算带来的极小数值噪声**。只要数值大于 1N，说明骨盆绝对实实在在地接触到了地面。

##### 🚨 3. 为什么非要盯死骨盆（Pelvis）？

在人形机器人的设定里，**骨盆是“绝对禁区”**。

- 正常的走路、跑步，哪怕摔得很惨（膝盖磕破了、脚踝扭了），骨盆通常都不会着地。
    
- 一旦骨盆的接触力大于 1N，说明机器人已经**彻底失去平衡，一屁股坐在地上了**。这时候这局游戏已经没有抢救的必要了，立刻判定摔倒（Terminate），重开下一局。
    

如果阈值设得很大（比如 > 500N），那会发生什么？  
机器人的屁股已经在地上拖行了 1 秒钟，压得很实了，系统才判定它“死”了。这会给神经网络注入大量“在地上蹭也能活”的错误经验，极其浪费算力。

---
### 2. 机器人姿态倾斜超过阈值

代码：

```python
torch.abs(self.rpy[:, 1]) > 1.0
```

以及：

```python
torch.abs(self.rpy[:, 0]) > 0.8
```

分别表示：

- Pitch 绝对值超过 1.0 rad。
- Roll 绝对值超过 0.8 rad。

注意，这里的阈值并不是机器人真正摔倒的唯一物理定义，而是工程上设定的终止条件。

- **代码解析**：

为什么设定 `Pitch > 1.0 rad`（约 57 度）和 `Roll > 0.8 rad`（约 46 度）就判定死亡？为什么说它是“工程设定”而不是“纯物理定义”？我们把它彻底扒开：

##### 🧭 1. 先看懂 `self.rpy[:, 0]` 和 `[:, 1]`

`self.rpy` 是我们之前聊过的欧拉角（Roll, Pitch, Yaw）。

- `self.rpy[:, 1]` 是 **Pitch（俯仰角）**：代表机器人**向前栽倒或向后仰倒**的角度。
    
- `self.rpy[:, 0]` 是 **Roll（横滚角）**：代表机器人**向左或向右歪斜**的角度。
    
- `torch.abs(...)` 取绝对值：因为无论是向前倒 60 度，还是向后倒 60 度（或者向左倒、向右倒），都是摔倒，我们只看倾斜的**幅度**。
    

##### ⚖️ 2. 为什么 Pitch (1.0) 和 Roll (0.8) 的阈值不一样大？

这是一个极其符合人体生物力学和机器人控制特性的工程设定：

- **Pitch 容错率高（1.0 rad ≈ 57度）**：  
    人类在走路、甚至快跑的时候，身体为了保持重心，本来就会发生前后倾斜。而且，如果机器人向前倾得厉害了，它只要猛地往前迈出一大步（跨步），就能把重心拉回来。所以对前后的倾斜，我们可以给到 57 度的容忍度。
    
- **Roll 容错率极低（0.8 rad ≈ 46度）**：  
    人形机器人侧向的支撑面（两只脚并排的宽度）是非常窄的。如果你往左或往右歪了 46 度，由于腿部的物理结构和干涉，你**很难通过“横着迈一大步”把重心救回来**（极大概率会两腿交叉绊倒）。一旦侧向倾斜超过 46 度，在动力学上基本宣判死刑了，所以阈值设得更严。

![[宇树H1机器人姿态图解.png|779]]

##### 🛠️ 3. 为什么是“工程设定”，而不是“物理定义”？

在纯物理世界里，根本没有“死（Done）”这个概念。哪怕机器人躺在地上四脚朝天，物理引擎依然在老老实实地模拟它在地面摩擦的受力，它没死。

**但这是强化学习，我们必须人为给它制造一个“物理定义”。**

- **防止无意义的摆烂（作弊）**：假设我们不设这个阈值，只靠骨盆接触地面来判定死亡。网络会发现一个捷径：“我只要一屁股坐在地上（骨盆没碰地，是用大腿根碰的），我虽然走不了路，但我也不扣分，我还能一直白嫖 `alive`（存活）奖励！” 这会毁掉整个训练。
    
- **避免计算资源浪费**：如果一个环境里的机器人已经歪到了 70 度，其实你已经肉眼可见它没救了，它接下来 1 秒钟一定会摔倒。如果我们等它 1 秒钟后头砸在地上再判定，这 1 秒钟内产生的几十个 Transition 全是垃圾数据，白白浪费 GPU 的算力。**在它倾斜超阈值的那一刻立刻 `Kill`（重置），才是最高效的**。
    
- **强行纠正姿态**：通过这个硬性终止条件，就是在逼着网络学会：“不管你怎么走，你绝对不能让自己倾斜超过这个红线！”
    

_(补充硬核细节：在不同的任务里，这个阈值是完全不同的。比如宇树四足狗如果做“侧空翻”特技，Roll 会翻转 360 度，那时候这个阈值就必须被代码动态禁用，否则狗一翻身就被判定死亡了。)_

---

### 3. Episode 超时

```python
self.time_out_buf = (
    self.episode_length_buf > self.max_episode_length)
```

这个条件和机器人摔倒不同。

假设一个 Episode 规定最长运行 20 秒：

$$
T_{\max}=20s
$$

当前 Policy 时间步为：

$$
dt=0.02s
$$

那么最大步数约为：

$$
N_{\max}=\frac{20}{0.02}=1000
$$

达到时间限制后，即使机器人没有摔倒，也需要结束这一轮 Episode。

---


<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; color: #333; line-height: 1.6; max-width: 800px;">

    
    <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 0 20px; background-color: #fff;">
        <div style="padding: 20px 0 15px 0; font-size: 18px; font-weight: bold; color: #222;">
            三种终止情形
        </div>
        
        <!-- 异常接触 -->
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 15px 0; border-bottom: 1px dashed #eaeaea;">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 24px; color: #d93025; margin-right: 15px;">⚠️</div>
                <div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-bottom: 4px;">异常接触</div>
                    <div style="font-size: 14px; color: #666;">Pelvis 接触力超过阈值</div>
                </div>
            </div>
            <div style="background-color: #fce8e6; color: #d93025; padding: 4px 12px; border-radius: 16px; font-size: 14px; font-weight: 500;">Reset</div>
        </div>

        <!-- 姿态超限 -->
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 15px 0; border-bottom: 1px dashed #eaeaea;">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 24px; color: #d93025; margin-right: 15px;">☑️</div>
                <div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-bottom: 4px;">姿态超限</div>
                    <div style="font-size: 14px; color: #666;">Roll 或 Pitch 超过阈值</div>
                </div>
            </div>
            <div style="background-color: #fce8e6; color: #c5221f; padding: 4px 12px; border-radius: 16px; font-size: 14px; font-weight: 500;">Reset</div>
        </div>

        <!-- 时间到期 -->
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 15px 0;">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 24px; color: #1a73e8; margin-right: 15px;">ℹ️</div>
                <div>
                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-bottom: 4px;">时间到期</div>
                    <div style="font-size: 14px; color: #666;">达到 Episode 最大步数</div>
                </div>
            </div>
            <div style="background-color: #e8f0fe; color: #1a73e8; padding: 4px 12px; border-radius: 16px; font-size: 14px; font-weight: 500;">Timeout</div>
        </div>
    </div>

    
</div>

***三个条件如何用OR串起来

##### 🪜 第一步：初始化 `reset_buf`


```python
self.reset_buf = torch.any(...骨盆碰撞...)
```

此时，`reset_buf` 里面装的是：“哪些机器人因为骨盆着地死掉了”。（比如 `[False, False, True, False]`）

##### 🪜 第二步：追加“姿态倾斜”的死亡条件


```python
self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8)
```

- 等号右边：计算出“哪些机器人因为歪得太厉害死掉了”（比如 `[False, True, False, False]`）。
    
- `|=` 的作用：把右边的结果和左边原来的结果**合并（取并集）**。
    
- 结果：现在 `reset_buf` 变成了 `[False, True, True, False]`。这说明 1 号因为歪倒死掉了，2 号因为撞地死掉了。
    

### 🪜 第三步：追加“超时”的死亡条件


```python
self.reset_buf |= self.time_out_buf
```

- 等号右边：计算出“哪些机器人活得太久，时间到了被强制结束”。
    
- 再次使用 `|=` 合并。
    
- 结果：不管是撞死、歪死，还是超时死，只要占了其中一样，最终的 `reset_buf` 对应位置就是 `True`。


这里先记住一个重要区别：

> Fall 和 Timeout 都会结束当前 Episode，但对 PPO 的价值估计而言，两者不一定应该采用完全相同的处理方式。

我们会在 Day 8 学习 Time-limit Bootstrapping 时进一步研究。

## 三、第二部分：Reset——究竟重置谁？

继续看：

```python
def post_physics_step(self):
```

关键代码：

```python
self._post_physics_step_callback()
self.check_termination()
self.compute_reward()

env_ids = self.reset_buf.nonzero(
    as_tuple=False).flatten()

self.reset_idx(env_ids)

if self.cfg.domain_rand.push_robots:
    self._push_robots()

self.compute_observations()
```

这里有一个非常重要的执行顺序：

1. 更新仿真状态。
2. 检查哪些环境需要终止。
3. 计算 Reward。
4. 找到需要 Reset 的环境 ID。
5. 只重置这些环境。
6. 生成新的 Observation。

### 🕵️ 怎么找到哪几个环境结束了？

重点看：

```python
env_ids = self.reset_buf.nonzero(
    as_tuple=False).flatten()
```

核心就在这两行代码：

1. **`self.reset_buf` 是什么？**  
    它是一个**布尔数组（Boolean Mask）**，长度等于环境总数（比如 4096）。在前面 `check_termination()` 中，只要某个机器人摔死了或者超时了，它对应的位置就会变成 `True`。  
    假设 0号、1号没死，2号摔死了，3号没死，4号摔死了。那么 `reset_buf` 长这样：  
    `[False, False, True, False, True, ...]`
    
2. **`nonzero()` 是什么意思？**  
    它是 PyTorch 的函数，作用是**把所有非零（也就是 `True`）的元素的索引（ID）给揪出来！**  
    经过 `nonzero()` 之后，结果就变成了：`tensor([[2], [4]])`。
    
3. **`.flatten()` 是什么？**  
    把刚才那个二维的列向量，拉平成一维的数组：**`[2, 4]`**。
    

**结论**：通过这一行极其高效的张量操作，代码瞬间就拿到了“第 2 号和第 4 号机器人死亡，需要重置”的情报。这就是 `env_ids`。

现在假设有 4096 个环境：

```python
env_ids = [3, 25, 100, 512]
```

这意味着：

> 只有第 3、25、100、512 号环境需要重置。（这个325是GPT生成的，这也言周）

其他 4092 个环境继续运行。


<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; color: #333; line-height: 1.6; max-width: 800px;">

    <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; background-color: #fafafa;">
        <div style="font-size: 18px; font-weight: bold; color: #222; margin-bottom: 15px;">
            并行环境 Reset 示意
        </div>
        
        <!-- 网格容器 -->
        <div style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px;">
            <!-- 第1行 -->
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">0</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">1</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">2</div>
            <div style="background-color: #fce8e6; color: #d93025; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">3</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">4</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">5</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">6</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">7</div>
            <!-- 第2行 -->
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">8</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">9</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">10</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">11</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">12</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">13</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">14</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">15</div>
            <!-- 第3行 -->
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">16</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">17</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">18</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">19</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">20</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">21</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">22</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">23</div>
            <!-- 第4行 -->
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">24</div>
            <div style="background-color: #fce8e6; color: #d93025; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">25</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">26</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">27</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">28</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">29</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">30</div>
            <div style="background-color: #e6f4ea; color: #137333; padding: 12px 0; text-align: center; border-radius: 4px; font-size: 14px;">31</div>
        </div>
        
        <!-- 图例 -->
        <div style="display: flex; justify-content: space-between; margin-top: 15px; font-size: 14px;">
            <div style="color: #137333;">绿色：继续运行</div>
            <div style="color: #d93025;">红色：本轮示意中的 Reset</div>
        </div>
    </div>


</div>

这就是 Vectorized Environment 的一个重要优势。

某个机器人摔倒，不需要让整个训练中的其他机器人一起重新开始。

### Reset 具体做了什么？

找到：

```python
def reset_idx(self, env_ids):
```

核心代码：

```python
if len(env_ids) == 0:
    return

self._reset_dofs(env_ids)
self._reset_root_states(env_ids)

self._resample_commands(env_ids)

self.actions[env_ids] = 0.
self.last_actions[env_ids] = 0.
self.last_dof_vel[env_ids] = 0.
self.feet_air_time[env_ids] = 0.
self.episode_length_buf[env_ids] = 0

self.reset_buf[env_ids] = 1
```

可以看到，它不只是重置机器人位置，还需要清理与当前 Episode 有关的历史数据。

##### 🛑  `return` 

在 Python 函数里，`return` 意味着“立即终止当前函数的执行，并且把后面的代码全部跳过”**。在这里，它返回的是空值（`None`）。

我们分别研究两个函数。

##### 1. `_reset_dofs(env_ids)`

```python
self.dof_pos[env_ids] = (
    self.default_dof_pos
    * torch_rand_float(
        0.5,
        1.5,
        (len(env_ids), self.num_dof),
        device=self.device
    ))

self.dof_vel[env_ids] = 0.
```

这里有两个重要点：

- 关节位置围绕默认关节姿态进行随机化。
- 关节速度设置为零。

注意，这里是乘法随机化：

$$
q_{\text{reset}}
=
q_{\text{default}}\times u
$$

其中：

$$
u\sim U(0.5,1.5)
$$

并不是简单地给关节角度加一个随机偏差。

**为什么是“乘法”而不是“加法”？

```python
self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, ...)
```

- **如果是加法**：假设默认膝盖弯曲 0.5 弧度，你加一个随机数，可能变成 `0.5 + 0.8 = 1.3`（腿向后折断了）。
    
- **乘法（0.5 ~ 1.5 倍）**：它保留了关节的正负号（物理方向）。它只是在默认姿态的基础上，让腿稍微伸直一点（0.5倍）或者弯得深一点（1.5倍）。这就叫**形态随机化**。配合速度清零，逼迫神经网络学会从各种“奇形怪状”的姿势中站起来。

**代码中的维数是如何对应的：
###### 📦 1. 先看原数据有多大（全局）

- `self.dof_pos`：这是一个包含了所有环境的关节角度大矩阵。假设你有 4096 个环境，每个机器人 12 个关节。它的形状是 **`[4096, 12]`**。
    
- `self.default_dof_pos`：这是机器人的标准站立姿态。因为所有机器人都长一样，它的形状是 **`[12]`**（或者 `[1, 12]`）。
    
###### 🔪 2. 左边：`self.dof_pos[env_ids]` 变成了多大？

假设这一轮死了 4 个机器人，那么 `env_ids = [3, 25, 100, 512]`，长度 `len(env_ids) = 4`。  
当你写下 `self.dof_pos[env_ids]` 时，PyTorch 会去那个 4096 行的大矩阵里，精准地把第 3、25、100、512 行**抽出来**。  
所以，**左边的形状瞬间变成了 `[4, 12]`**（只包含这 4 个死掉的机器人的数据）。

###### ✨ 3. 右边：那坨复杂的运算变成了多大？

右端是：`self.default_dof_pos * torch_rand_float(..., (len(env_ids), self.num_dof), ...)`

1. `torch_rand_float(0.5, 1.5, (4, 12))`：直接生成了一个形状为 **`[4, 12]`** 的随机数矩阵。
    
2. `self.default_dof_pos` 是 **`[12]`**。
    
3. **广播机制（Broadcasting）登场**：PyTorch 发现你要把一个 `[12]` 的向量，和一个 `[4, 12]` 的矩阵相乘。它会自动把那个 `[12]` 的向量**复制 4 份**，变成 `[4, 12]`。
    
4. 然后 `[4, 12]` 和 `[4, 12]` 逐元素相乘，结果还是 **`[4, 12]`**。
    
注：
###### 🛠️ 若要取出具体关节该怎么改？

假设 `self.dof_pos` 的形状是 `[4096, 12]`（4096 个环境，12 个关节）。

**如果你要取所有的第 3 列和第 25 列（取特定列）：**


```python
selected_cols = self.dof_pos[:, [3, 25]]
```

- **解释**：前面的 `:` 表示取所有行（4096 个环境），后面的 `[3, 25]` 表示取第 3 和 25 列。
    
- **结果形状**：`[4096, 2]`。
     

##### 2. `_reset_root_states(env_ids)`

这个函数负责重置机器人基座状态。

核心代码：

```python
self.root_states[env_ids] = self.base_init_state
self.root_states[env_ids, :3] += (
    self.env_origins[env_ids])

self.root_states[env_ids, 7:13] = (
    torch_rand_float(
        -0.5,
        0.5,
        (len(env_ids), 6),
        device=self.device
    ))
```

 解析这三行代码：
 
```python
# 1. 恢复到初始出生状态
self.root_states[env_ids] = self.base_init_state
# 2. 加上环境的物理偏移量
self.root_states[env_ids, :3] += self.env_origins[env_ids]
# 3. 给速度注入灵魂（随机打乱）
self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6))
```

- **第一句**：把死掉的机器人的位置、姿态、速度全部重置为出厂默认值（`base_init_state`）。
    
- **第二句**：为什么要加 `env_origins`？因为有 4096 个机器人同时在一个大地图上训练。如果把他们都放在坐标 `(0,0,0)`，他们一出生就会因为互相碰撞挤成一团废铁。所以每个环境都有自己的“格子”，这一步是把机器人平移到属于它的那个格子里。
    
- **第三句**：提取 `[7:13]`，即三轴线速度 + 三轴角速度，并随机替换为 `-0.5 到 0.5` 之间的数。

注：`root_states`与`base_init_state`是什么？有什么区别？

用一句话概括：**`base_init_state` 是“出厂设置（静态图纸）”，而 `root_states` 是“实时监控（动态数据）”。**

| 索引范围    | 维度含义                            | 物理意义                    |
| ------- | ------------------------------- | ----------------------- |
| `0:3`   | `Position (x, y, z)`            | 机器人在世界地图上的三维坐标。         |
| `3:7`   | `Orientation (qw, qx, qy, qz)`  | **四元数**，描述机器人的绝对旋转姿态。   |
| `7:10`  | `Linear Velocity (vx, vy, vz)`  | **线速度**，机器人在三个方向上的平移速度。 |
| `10:13` | `Angular Velocity (wx, wy, wz)` | **角速度**，机器人绕三个轴旋转的速度。   |

##### 🏗️ 1. `base_init_state`：出厂设置（静态蓝图）

- **身份**：它是根据配置文件（`config.py` 里的 `pos` 和 `rot` 等）拼出来的**默认初始状态**。
    
- **维度**：形状是 **`[13]`**（或者 `[1, 13]`），仅仅代表**一台**机器人的出生位置、姿态和默认速度（通常是0）。
    
- **生命周期**：在环境初始化的时候，它被创建一次。在后续的几万步训练中，它的数值**绝对不变**。
    
- **物理意义**：它告诉环境：“所有机器人第一次出生时，以及以后每次摔倒复活时，都应该以这个姿势、在这个坐标出生。”

在Day 2的问题一中我们其实介绍过：

```python
base_init_state_list = (
    self.cfg.init_state.pos   # 基座的 x, y, z 坐标
    + self.cfg.init_state.rot # 基座的翻滚、俯仰、偏航角
    ...
)
```
这里还有细节：

 **🪄 Python 中列表的 `+` 魔法

在 Python 里，两个列表相加 `[1, 2] + [3, 4]`，结果不是 `[4, 6]`，而是**把后面的列表拼到前面的列表后面**，变成 `[1, 2, 3, 4]`。

所以：

```python
base_init_state_list = (
    [0.0, 0.0, 0.42]      # pos
    + [0.0, 0.0, 0.0, 1.0] # rot
    + [0.0, 0.0, 0.0]      # 线速度
    + [0.0, 0.0, 0.0]      # 角速度
)
```

拼接后的结果就是一个**长度正好为 13 的一维列表**：  
`[0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

**❌ 为什么不能用逗号 `,`？

如果你用了逗号：


```python
# 错误写法
base_init_state_list = (
    self.cfg.init_state.pos,
    self.cfg.init_state.rot
)
```
这样会生成一个**嵌套元组**：`([0,0,0.42], [0,0,0,1])`。这在赋给 PyTorch 张量或者 Isaac Gym 引擎时，会因为维度不对（变成了 2 维结构，而非平铺的 1 维）直接报错。

##### 📊 2. `root_states`：实时监控（动态数据池）

- **身份**：它是 Isaac Gym 物理引擎中，**所有并行环境**中机器人基座（Root）的**实时状态**张量。
    
- **维度**：形状是 **`[num_envs, 13]`**（比如 `[4096, 13]`）。注意这里多了一维，因为它包含了**4096 个机器人**各自的数据。
    
- **生命周期**：**每经过一个物理子步（比如 5ms），物理引擎就会刷新一次这个张量**。机器人在往前走，或者摔倒了，这里面的数字就会疯狂变化。
    
- **物理意义**：它记录着：“1号机器人现在走到哪里了？2号机器人现在是不是头朝下摔了？”

---

### 两个容易忽略的细节

1、 `post_physics_step()` 中：

```python
self.compute_reward()
self.reset_idx(env_ids)
```

顺序是先计算 Reward，再 Reset。

为什么？

- **如果反过来**：假设机器人摔死了，如果你先把它重置回完美的站立姿态，再去算 Reward，系统会以为它现在站得很好，不仅不给它死亡惩罚（-50分），甚至还可能给它发一个“姿态优秀奖”。
    
- **正确的逻辑**：必须用它**摔死前那一帧的惨状**来算奖励。重罚之后，再让它投胎。这样网络才会深刻记住：“刚才那个动作会导致摔死”。

这个机制我们在 Day 8 会和 PPO Storage 联系起来。

2、`_reset_dofs(env_ids)`中将关节位置随机化但将关节速度重制为0，为什么这里的基座速度却不归零？


这背后藏着极其严谨的**物理引擎约束**与**训练策略考量**。

###### ❌ 为什么 `dof_vel`（关节速度）必须归零？

你在 `_reset_dofs` 里看到代码写了 `self.dof_vel[env_ids] = 0.`。

1. **保护物理引擎不崩溃**：物理引擎计算刚体运动是通过时间步积分算出来的。如果你给某个膝关节随机赋一个 `10 rad/s` 的初始速度，由于时间步的离散性，下一秒膝关节可能会因为位移过大，直接和另一根骨头“穿模”或者产生上百万牛顿的碰撞力，导致整个仿真环境瞬间蓝屏/NaN报错。
    
2. **电机无法瞬间爆发**：真实机器人的电机是有极限的。你让机器人刚通电的瞬间，关节就自带每秒转三圈的速度，这在物理上根本做不到。
    
3. **姿态已经随机了**：上面 `dof_pos` 已经乘了 `0.5~1.5` 的随机数。如果你同时随机化关节位置和速度，机器人的初始状态会极其混乱，网络根本学不会。
    

###### ✅ 为什么基座速度（Root Velocity）必须随机化？

1. **打破温室效应**：如果基座速度永远是 0，网络会学到一个“依赖平稳地面起步”的死策略。一旦真机上底盘有晃动，它就摔。
    
2. **模拟真实环境**：在现实中，机器人可能是从卡车上掉下来的，可能是被侧面来的风正在吹着，甚至是在移动的履带平台上启动的。基座给一个 `[-0.5, 0.5]` 的随机初速度（大约相当于被猛推一把），就是在逼迫策略：“开局自带一个盲盒推力，你得立刻调整肌肉把它稳住！”
    
3. **基座速度的容错率高**：基座不是电机直连的，它是空气动力学和整体重心的体现。给基座一个乱速，物理引擎能平稳计算，不容易穿模。


## 四、第三部分：Command 采样——机器人到底听谁的指令？

前几天我们研究 Observation 时，已经知道：

```python
self.commands[:, :3]
```

会进入 Actor 的 Observation。

但今天需要进一步追踪：

> 这些 commands 是谁生成的？多久生成一次？

先打开：

`legged_gym/envs/base/legged_robot_config.py`

找到：

```python
class commands:
    curriculum = False
    max_curriculum = 1.
    num_commands = 4
    resampling_time = 10.
    heading_command = True

    class ranges:
        lin_vel_x = [-1.0, 1.0]
        lin_vel_y = [-1.0, 1.0]
        ang_vel_yaw = [-1, 1]
        heading = [-3.14, 3.14]
```

这里 G1 没有重新定义 `commands`，所以使用继承的基础配置。

这四个量分别是：

| Command | 含义 |
|---|---|
| `commands[:,0]` | X 方向目标线速度 |
| `commands[:,1]` | Y 方向目标线速度 |
| `commands[:,2]` | 目标偏航角速度 |
| `commands[:,3]` | 目标 Heading |

但 G1 启用了：

```python
heading_command = True
```

所以第三维角速度并不是直接随机采样，而是根据目标 Heading 和当前姿态计算出来的。

##### 🔄 1. 传统模式 vs Heading 模式

- **传统模式（直接给转速）**：如果代码直接采样 `ωz = 0.5 rad/s`，并且持续 10 秒钟。那机器人在这 10 秒内，就会像陀螺一样原地打转，转 5 弧度（大概 286 度），根本停不下来。**这也是为什么代码里设置了 `heading_command = True`，彻底抛弃了这种反人类的设定。**
    
- **Heading 模式（给目标角度）**：代码采样的其实是 `commands[:, 3]`，也就是**目标朝向（Heading）**。比如目标是：**“面朝正北”**。
    

##### 🧮 2. 动态计算：如果目标转速一直不为零，它是怎么自动停下来的？

你看这段代码：


```python
self.commands[:, 2] = torch.clip(
    0.5 * wrap_to_pi(self.commands[:, 3] - heading),
    -1., 1.
)
```

这里发生了一个极其经典的**闭环控制**：

1. **算误差**：`self.commands[:, 3] - heading`。目标朝向 减去 当前朝向。如果机器人现在面朝东，目标是面朝北，误差就是 90 度（约 1.57 弧度）。
    
2. **算转速**：`0.5 * 1.57`，大概算出来当前应该转 `0.78 rad/s`。
    
3. **实时更新**：机器人开始往北转。到了下一帧（20ms后），它离北更近了，误差变小了，`commands[:, 2]` 的数值就会立刻**自动变小**。
    
4. **精准停下**：等机器人完全对准正北时，误差变成了 `0`。此时 `0.5 * 0 = 0`。**目标转速变成了 0，机器人完美停下！**
    

**所以，这 10 秒内发给 Actor 的 `commands[:, 2]`（目标角速度）根本不是固定的，而是每一帧都在随着机器人的转身而衰减的！** 它就像一个老司机在微调方向盘，快对准了就不转了。正因为这样，我们在输入`obs`时，`command`也只需要输入前三维$[:,:3]$而不用输入第四维$[:,:4]$

---

继续看：

```python
def _post_physics_step_callback(self):
    env_ids = (
        self.episode_length_buf
        % int(
            self.cfg.commands.resampling_time
            / self.dt
        ) == 0
    ).nonzero(as_tuple=False).flatten()

    self._resample_commands(env_ids)

    if self.cfg.commands.heading_command:
        forward = quat_apply(
            self.base_quat,
            self.forward_vec
        )

        heading = torch.atan2(
            forward[:, 1],
            forward[:, 0]
        )

        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(
                self.commands[:, 3] - heading
            ),
            -1.,
            1.
        )
```

这段代码非常重要。

首先：

$$
T_{\text{command}}=10s
$$

而：

$$
dt=0.02s
$$

所以每隔：

$$
\frac{10}{0.02}=500
$$

个策略时间步，重新采样一次运动指令。


**为什么不每次步进都重新随机生成指令？**

- **防止精神分裂**：如果每 20 毫秒指令就变一次（刚让你往前，下一秒让你往左，再下一秒让你后退），机器人根本无法建立稳定的动力学步态，它会像个无头苍蝇一样原地抽搐。
    
- **逼迫长时间追踪**：给你 10 秒钟的时间，你就得在这个时间段内，死死咬住当前的目标速度。这也是工业界轮式机器人/足式机器人测试时的常用操作。
    

_(注：除了定时采样，机器人如果在 10 秒内摔死了，`reset_idx` 里会调用 `_resample_commands` 给它重新抽一组指令。)_

**这段代码具体如何实现构想：

⏱️ 第一部分：定时重置（给机器人重新发任务）

```python
env_ids = (
    self.episode_length_buf
    % int(
        self.cfg.commands.resampling_time
        / self.dt
    ) == 0
).nonzero(as_tuple=False).flatten()
self._resample_commands(env_ids)
```

- **`self.episode_length_buf % int(500) == 0`**：  
    你之前知道，`resampling_time = 10s`，而 `dt = 0.02s`，所以每 500 步换一次任务。这行代码利用取模运算（`%`），精准抓出当前步数是 0、500、1000... 的那些环境。
    
    - _注意：因为每个机器人开局时间不同，所以每个机器人步数 500 的节点也是错开的，不会出现 4096 个机器人同时换任务卡顿的情况。_
        
- **`nonzero().flatten()`**：把满足条件的**环境编号（ID）**揪出来，比如 `[5, 102, 3096]`。
    
- **`self._resample_commands(env_ids)`**：只给这几个被选中的机器人重新随机抽样速度（vx, vy）和绝对目标朝向（Heading）。
    

🧭 第二部分：把 Heading（目标角度）变成转速（角速度）


```python
if self.cfg.commands.heading_command:
    forward = quat_apply(self.base_quat, self.forward_vec)
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    self.commands[:, 2] = torch.clip(
        0.5 * wrap_to_pi(self.commands[:, 3] - heading),
        -1., 1.
    )
```

这一段就是那个“包工头算误差”的现场：

1. **`quat_apply(self.base_quat, self.forward_vec)`**
    
    - **动作**：提取机器人的当前四元数姿态（`base_quat`），把代表机身正前方的向量（通常是 `[1, 0, 0]`），通过旋转变换，**投影到世界坐标系**中。
        
    - **人话**：算一下机器人现在车头到底指着世界地图的哪个东南西北的方向。
        
2. **`heading = torch.atan2(forward[:, 1], forward[:, 0])`**
    
    - **动作**：用 `atan2(y, x)` 反正切函数，把这个向量变成具体的弧度角。
        
    - **人话**：查一下罗盘，算出机器人当前在地图上的“绝对方位角”，范围是 [−π,π][−π,π]。
        
3. **`wrap_to_pi(self.commands[:, 3] - heading)`**
    
    - **动作**：用目标朝向（第 4 维）减去当前朝向，求误差。并且用 `wrap_to_pi` 做角度环绕。
        
    - **人话**：目标在 179 度，你现在在 -179 度，直接减是 358 度。但机器人知道其实只要转 2 度就行了。`wrap_to_pi` 就是为了防止机器人绕远路转大圈。
        
4. **`0.5 * ...`**
    
    - **动作**：乘以 0.5。这叫 **P 控制（比例控制）**。
        
    - **人话**：如果误差很大（比如差了 90 度），就转快点（算出来 0.78）；如果误差很小了，就转慢点。
        
5. **`torch.clip(..., -1., 1.)`**
    
    - **动作**：把计算出的角速度限制在 `-1.0` 到 `1.0 rad/s` 之间。
        
    - **人话**：给机器人踩一脚刹车。哪怕误差极大，也不许超过 1.0 rad/s，否则机器人会像陀螺一样失控。
        
6. **`self.commands[:, 2] = ...`**
    
    - **动作**：把算出来的这组完美的动态角速度，填入第 3 维（索引 2）。
    
---

接着 `_resample_commands()`：

```python
self.commands[env_ids, 0] = torch_rand_float(
    self.command_ranges["lin_vel_x"][0],
    self.command_ranges["lin_vel_x"][1],
    ...
)

self.commands[env_ids, 1] = torch_rand_float(
    self.command_ranges["lin_vel_y"][0],
    self.command_ranges["lin_vel_y"][1],
    ...
)

if self.cfg.commands.heading_command:
    self.commands[env_ids, 3] = torch_rand_float(
        self.command_ranges["heading"][0],
        self.command_ranges["heading"][1],
        ...
    )
```

所以当前 G1 的命令流程是：

```text
随机采样目标前进速度
随机采样目标侧向速度
随机采样目标 Heading
          |
          v
   当前机器人姿态
          |
          v
   计算 Heading Error
          |
          v
   计算目标偏航角速度
          |
          v
      commands
          |
          v
   输入 Actor Observation
```

这里的设计有一个很重要的意义：

> 策略不只是学习从当前状态走路，还要学会根据不同的目标运动指令调整自己的行为。

## 五、第四部分：Domain Randomization——为什么要故意把仿真变得不一样？

今天最后一个知识点：域随机化

域随机化在具身中应用甚多，是Sim2Real中让机器人获得强力泛化性的一大法宝

打开 `g1_config.py`：

```python
class domain_rand(LeggedRobotCfg.domain_rand):
    randomize_friction = True
    friction_range = [0.1, 1.25]
    randomize_base_mass = True
    added_mass_range = [-1., 3.]
    push_robots = True
    push_interval_s = 5
    max_push_vel_xy = 1.5
```

今天重点研究三种随机化。

### 1. 摩擦系数随机化

找到：

```python
def _process_rigid_shape_props(self, props, env_id):
```

关键代码：

```python
if self.cfg.domain_rand.randomize_friction:
    if env_id == 0:
        friction_range = (
            self.cfg.domain_rand.friction_range
        )

        num_buckets = 64

        bucket_ids = torch.randint(
            0,
            num_buckets,
            (self.num_envs, 1)
        )

        friction_buckets = torch_rand_float(
            friction_range[0],
            friction_range[1],
            (num_buckets, 1),
            device='cpu'
        )

        self.friction_coeffs = (
            friction_buckets[bucket_ids]
        )

    for s in range(len(props)):
        props[s].friction = (
            self.friction_coeffs[env_id]
        )
```

这里使用了 64 个摩擦系数 Bucket。

当前 G1 的摩擦系数在：

$$
\mu\sim U(0.1,1.25)
$$

范围内随机采样。

这意味着不同并行环境中的机器人可能处在不同的地面摩擦条件下。

例如：

- Environment 1：摩擦系数比较低，脚容易打滑。
- Environment 2：摩擦系数比较高，脚更容易获得地面牵引力。

这样 Policy 不会只适应某一个固定的地面摩擦系数。

注意：从代码执行位置看，摩擦系数是在创建环境、配置刚体形状属性时随机化的，并不是每个 Policy Step 都重新随机。

现在我们来看每一行到底做了什么。

##### **第一步：判断是否开启摩擦随机化**

```
if self.cfg.domain_rand.randomize_friction:
```

这个不用复杂理解。

如果 G1 配置是：

```
randomize_friction = True
```

就执行后面的随机化逻辑。

如果是 `False`，就直接返回原始属性，不做这项随机化。

##### **第二步：为什么要判断** `env_id == 0`**？**


```python
if env_id == 0:
    friction_range = self.cfg.domain_rand.friction_range  # 例如 [0.1, 1.25]
    num_buckets = 64
```

还记得刚才的循环吗？

```
for i in range(self.num_envs):    ...
```

每创建一个 Environment，就会调用一次：

```
_process_rigid_shape_props(props, env_id)
```

所以：

- 第一次调用：`env_id = 0`
    
- 第二次调用：`env_id = 1`
    
- 第三次调用：`env_id = 2`
    
- ...
    
- 最后一次调用：`env_id = 4095`
    

那么，为什么只在 `env_id == 0` 时生成 Bucket？

因为我们只需要生成一次候选摩擦系数表，以及所有环境对应的随机索引。

后面其他环境直接使用这张表即可。

否则每创建一个环境，就重新生成一次 64 个摩擦系数，之前生成的结果就没有意义了。


##### 第三步：生成 64 个“摩擦系数桶”（Buckets）


```python
friction_buckets = torch_rand_float(
    friction_range[0], friction_range[1], (num_buckets, 1), device='cpu'
)
```
在$[μ_{min}⁡,μ_{max}⁡]$ 区间内均匀采样 **64 个**具体的摩擦系数值。这 64 个值就是整个训练过程中所有环境可能用到的**全部摩擦系数候选**。

例如：

```python
friction_buckets = [    [0.22],    [0.73],    [1.12],    ...]
```

##### 第四步：为每个环境随机分配一个桶的 ID


```python
bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
self.friction_coeffs = friction_buckets[bucket_ids]
```

这一步为**全部** `num_envs`（例如 4096）个环境各随机抽取一个整数 ID（0–63），然后通过索引得到每个环境对应的摩擦系数。`self.friction_coeffs[env_id]` 就是第 `env_id` 个环境在整个训练过程中使用的**固定摩擦系数**。

举例：

```
bucket_ids = [    [2],    [5],    [2],    [0],    [63],    ...]
```

它代表：

```
Environment 0 选择 Bucket 2

Environment 1 选择 Bucket 5

Environment 2 选择 Bucket 2

Environment 3 选择 Bucket 0

Environment 4 选择 Bucket 63
```

注意这里允许重复抽样。

比如 Environment 0 和 Environment 2 都选择 Bucket 2。

所以不同环境完全可能具有相同的摩擦系数。


##### **第五步：每个环境使用自己对应的摩擦系数**

```python
for s in range(len(props)):    

props[s].friction = (self.friction_coeffs[env_id])
```

这里要再区分两个索引：

- `env_id`：当前正在创建第几个环境。
    
- `s`：当前机器人 Asset 的第几个刚体形状。
    

`props` 是机器人 Asset 的刚体形状属性列表。

例如，一个机器人有多个碰撞形状，那么：

```python
for s in range(len(props)):
```

就逐个处理这些形状。

而：

```python
props[s].friction = self.friction_coeffs[env_id]
```

表示：

**当前环境中，写入它所有碰撞几何体（通常是脚部接触形状）的摩擦属性中**

最终这个函数把修改后的 `props` 返回，随后代码会将属性设置到 Asset，再创建该环境里的机器人 Actor。

**这里还有一个更深层的细节：随机化的到底是地面还是机器人？

我需要特别纠正一个容易产生的误解。

之前我们把它简单描述成不同机器人处在不同的地面摩擦条件下，但严格看这段代码，实际操作是：

```python
props[s].friction = self.friction_coeffs[env_id]
```

这里修改的是 **G1 机器人 Asset 的刚体形状摩擦属性**，不是直接修改地面属性。

因为在前面的代码中：

```python
rigid_shape_props_asset = (self.gym.get_asset_rigid_shape_properties(robot_asset))
```

获得的是机器人 Asset 的形状属性。

而地面是通过另一个函数创建的：

```python
def _create_ground_plane(self):   
 plane_params = gymapi.PlaneParams()    
 plane_params.static_friction = (self.cfg.terrain.static_friction)    
 plane_params.dynamic_friction = (self.cfg.terrain.dynamic_friction)    
 self.gym.add_ground(self.sim, plane_params)
```

这两段代码修改的对象不同。

所以更严谨的说法应该是：

- 地面 Plane 的基础摩擦参数由 `terrain.static_friction` 和 `terrain.dynamic_friction` 设置。
    
- G1 各刚体碰撞形状的 `friction` 属性由 `_process_rigid_shape_props()` 随机化。
    
- 两个接触物体的摩擦属性共同参与接触摩擦计算，最终的组合方式由物理引擎的接触参数规则决定。
    
因此，不能直接说这段代码为每个机器人创建了一块独立的、摩擦系数不同的地板。

### 2. Base Mass 随机化

找到：

```python
def _process_rigid_body_props(self, props, env_id):
```

核心：

```python
if self.cfg.domain_rand.randomize_base_mass:
    rng = self.cfg.domain_rand.added_mass_range

    props[0].mass += np.random.uniform(
        rng[0],
        rng[1]
    )
```

这里是给基座刚体增加或减少质量：

$$
m_{\text{new}}=m_{\text{original}}+\Delta m
$$

其中：

$$
\Delta m\sim U(-1,3)\text{ kg}
$$

为什么这么做？

因为真实机器人与仿真模型可能存在：

- 质量估计误差；
- 外部负载；
- 不同配置造成的重量变化。

训练时让机器人面对不同的质量条件，可以提高策略对模型变化的适应能力。

注意，当前代码修改的是 `props[0].mass`，不是随机修改所有关节连杆的质量。

### 3. Push Robots：训练期间随机推机器人

这项随机化和前两项非常不同。

前两项是在创建环境时设置物理属性。

而 `_push_robots()` 是运行过程中执行的扰动。

打开：

```python
def _push_robots(self):
```

关键代码：

```python
push_env_ids = env_ids[
    self.episode_length_buf[env_ids]
    % int(self.cfg.domain_rand.push_interval)
    == 0
]

max_vel = self.cfg.domain_rand.max_push_vel_xy

self.root_states[:, 7:9] = torch_rand_float(
    -max_vel,
    max_vel,
    (self.num_envs, 2),
    device=self.device
)
```

然后通过：

```python
self.gym.set_actor_root_state_tensor_indexed(...)
```

把选中的机器人的状态写回仿真器。

这里有一个特别重要的细节：

> 当前官方实现并不是直接施加一个随机外力，而是通过修改机器人基座的水平线速度，模拟受到外部冲击。

注意：回顾表格$[7:9]=[v_x,v_y]$对应水平线速度

| 索引范围    | 维度含义                            | 物理意义                    |
| ------- | ------------------------------- | ----------------------- |
| `0:3`   | `Position (x, y, z)`            | 机器人在世界地图上的三维坐标。         |
| `3:7`   | `Orientation (qw, qx, qy, qz)`  | **四元数**，描述机器人的绝对旋转姿态。   |
| `7:10`  | `Linear Velocity (vx, vy, vz)`  | **线速度**，机器人在三个方向上的平移速度。 |
| `10:13` | `Angular Velocity (wx, wy, wz)` | **角速度**，机器人绕三个轴旋转的速度。   |


当前 G1：

$$
T_{\text{push}}=5s
$$

最大水平速度扰动范围：

$$
v_x,v_y\in[-1.5,1.5]
$$

为什么这么做？

注意：前面我们讲过

例如机器人正在向前走：

```text
正常行走
   |
   v
突然受到侧向扰动
   |
   v
机器人身体发生偏移
   |
   v
Actor 观察新的状态
   |
   v
Policy 尝试恢复平衡
```

通过这种方式，策略不仅学习正常条件下的行走，也会学习如何面对突发扰动。

这对 Sim2Real 非常重要。

几个问题：
#### **一、为什么没有用** `push_env_ids` **修改速度？而是用`num_envs`**

原代码：

```python
push_env_ids = env_ids[
    self.episode_length_buf[env_ids]
    % int(self.cfg.domain_rand.push_interval) == 0
]

max_vel = self.cfg.domain_rand.max_push_vel_xy

self.root_states[:, 7:9] = torch_rand_float(
    -max_vel,
    max_vel,
    (self.num_envs, 2),
    device=self.device
)

env_ids_int32 = push_env_ids.to(dtype=torch.int32)

self.gym.set_actor_root_state_tensor_indexed(
    self.sim,
    gymtorch.unwrap_tensor(self.root_states),
    gymtorch.unwrap_tensor(env_ids_int32),
    len(env_ids_int32)
)
```

这里其实有两个不同的操作：

1. 修改 PyTorch Tensor 中的数据。
    
2. 把修改后的数据写入 Isaac Gym 仿真器。
    

**关键在最后这个 API：**

```python
self.gym.set_actor_root_state_tensor_indexed(
    self.sim,
    gymtorch.unwrap_tensor(self.root_states),
    gymtorch.unwrap_tensor(env_ids_int32),
    len(env_ids_int32)
)
```

根据 Isaac Gym 的 Tensor API 文档，它接收的是完整的 Actor Root State Tensor，但只会将指定索引的 Actor 状态写入仿真器。

因此：

```python
self.root_states[:, 7:9] = torch_rand_float(
    -max_vel,
    max_vel,
    (self.num_envs, 2),
    device=self.device
)

```

虽然修改了全部环境在 Tensor 中的 XY 速度，但：

```python
push_env_ids
```

决定了最终哪些机器人的状态真正被更新到仿真器中。
#### 二、为什么不直接用`push_env_ids`

这样有两个好处：

- 不会生成实际上用不到的随机数。
    
- 不会无意修改未被 Push 的环境在 Tensor 中的状态缓存。
    

所以，官方当前这段实现从结果上可以实现选择性 Push，但前面那句对所有环境赋值并不必要。它是一种可以改进的实现。

#### **三、究竟是增加速度，还是直接覆盖速度？为什么这样可以模拟外部推力**

你理解得完全正确。

看：

```
self.root_states[:, 7:9] = torch_rand_float(...)
```

这里使用的是：

```
=
```

而不是：

```
+=
```

因此它是直接覆盖。

Isaac Gym 中 Root State 的 13 个维度分别是：

|索引|含义|
|---|---|
|0:3|Position|
|3:7|Quaternion|
|7:10|Linear Velocity|
|10:13|Angular Velocity|

因此：

```
self.root_states[:, 7:9]
```

就是：

\[ [v_x,v_y] \]

而：

```
torch_rand_float(    -max_vel,    max_vel,    ...)
```

在 G1 配置中：

```
max_push_vel_xy = 1.5
```

所以最终：

$v_x^{new}\sim U(-1.5,1.5)$

$v_y^{new}\sim U(-1.5,1.5)$

注意，它们是两个独立采样的分量，不是保证 XY 平面合速度的模长小于 1.5。

#### 那它为什么可以模拟外部推力？

我们从物理角度理解。

真实世界中，手推机器人相当于施加一个短时间外力：

$$ F_{\text{external}}(t) $$

根据牛顿第二定律：

$$ F=ma $$

外力产生加速度：

$$ a=\frac{F}{m} $$

在一段时间内：

$$ \Delta v=\int a(t)\,dt $$

因此，真实外力作用之后，机器人的速度发生改变。

但 Unitree 这里没有模拟外力持续作用的过程，而是直接将速度设置为随机值：

$$ v^{new}\sim U(-1.5,1.5) $$

代码注释也明确说明：

```
""" Random pushes the robots.Emulates an impulse by setting a randomized base velocity."""
```

所以它实际上是用速度突变来近似模拟一次冲击。


# 今日问题

### 问题 1：Termination 与 Reset

在 4096 个环境中，只有 3 个机器人摔倒。

为什么程序只重置这 3 个环境，而不重置全部 4096 个？

请结合：

```python
env_ids = self.reset_buf.nonzero(
    as_tuple=False).flatten()
```

解释。

答：`env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()`通过三个终止判定条件，锁定终止机器人的索引，重制它们的环境，如果每有一个人摔倒就重制4096个机器人的环境，那么机器人永远也学不会走路，强化学习本质是不断试错，如果机器人一准备走起来，因为其它机器人摔倒就被重制，即使走的很好机器人也无法走出几步，更学不会如何走路

#### 💡GPT的修正与补充

**1.** `nonzero()` **本身并不负责判断机器人是否摔倒。**

真正的终止条件在：

```python
def check_termination(self):

    self.reset_buf = torch.any(
        torch.norm(
            self.contact_forces[
                :, self.termination_contact_indices, :
            ],
            dim=-1
        ) > 1.,
        dim=1
    )

    self.reset_buf |= torch.logical_or(
        torch.abs(self.rpy[:, 1]) > 1.0,
        torch.abs(self.rpy[:, 0]) > 0.8
    )

    self.time_out_buf = (
        self.episode_length_buf > self.max_episode_length
    )

    self.reset_buf |= self.time_out_buf
```

这里将接触异常、姿态超限和 Timeout 的结果合并到 `reset_buf`。

然后：

```python
env_ids = self.reset_buf.nonzero(    as_tuple=False).flatten()
```

这句话只是找出 `reset_buf=True` 的环境索引。

例如：

```python
reset_buf = torch.tensor([    False, False, True, False, True])
```

结果就是：

```python
env_ids = tensor([2, 4])
```

**2. 重置全部环境并不意味着永远学不会走路。**

这是你的推论略有夸大的地方。

更准确地说，如果其中一个机器人摔倒就重置全部 4096 个环境：

- 其他正常运行的 Episode 会被无故中断。
    
- 已经积累的有效轨迹会被浪费。
    
- 平均 Episode 长度会受到影响。
    
- 训练数据采集效率下降。
    

因此，选择性 Reset 的核心是保留其他环境正在进行的独立经验。

**建议记忆：**

> `reset_buf` 记录各环境是否需要终止，`nonzero()` 提取所有需要重置的环境索引，`reset_idx(env_ids)` 只重置这些环境，其他环境继续运行。

### 问题 2：Reward 与 Reset 顺序

为什么：

```python
self.compute_reward()
self.reset_idx(env_ids)
```

而不是先 Reset，再计算 Reward？

如果颠倒顺序，可能会出现什么问题？

答：如果颠倒顺序，机器人摔倒后，应该给予惩罚，但环境重置后恢复站立，此时策略可能会因为它站立的很好反而给他很高的$Reward$，机器人反而会人为策略在奖励这种摔倒的行为

#### 💡GPT的修正与补充

这道题回答得非常好。

你已经理解了一个非常重要的强化学习原则：

**Reward 必须与执行 Action 后产生的实际状态对应。**

看实际代码：

```python
self.check_termination()

self.compute_reward()

env_ids = self.reset_buf.nonzero(
    as_tuple=False
).flatten()

self.reset_idx(env_ids)
```

这里的顺序非常有讲究。

正确过程：

```
机器人执行 Action
       ↓
物理仿真
       ↓
机器人摔倒
       ↓
检查 Termination
       ↓
根据摔倒前的物理状态计算 Reward
       ↓
     Reset
       ↓
生成下一轮 Observation
```

如果反过来：

```
机器人摔倒
    ↓
Reset，恢复站立
    ↓
计算 Reward
```

那么 Reward 使用的就可能是重置后的关节状态、基座状态和新 Command，而不是刚才执行 Action 后产生的状态。

这会造成 Transition 中的 Action 和 Reward 错误对应。

这里再补充一个代码级细节：

当前 G1 继承的基础配置里：

```python
termination = -0.0
```

而 G1 没有将这个权重改为非零。零权重会在 `_prepare_reward_function()` 中被移除。因此，不能简单地说当前 G1 一定启用了独立的摔倒终止惩罚；具体是否有这项惩罚，要看合并后的配置。

但这不影响你的核心判断：Reward 必须在 Reset 前，根据终止前状态计算。

**这一题可以给满分附近。**

### 问题 3：Fall 和 Timeout

为什么机器人摔倒与 Episode 正常运行到时间上限，都需要 Reset，但又分别使用：

```python
self.reset_buf
self.time_out_buf
```

为什么官方代码还会将：

```python
self.extras["time_outs"]
```

传给 PPO？

这题可以结合你之前学过的 GAE 思考。

答：time out的情况下机器人后面的trajectory 还可以正确产生reward，而如果因为摔倒终止需要直接截断后面的reward，所以虽然同样要重制环境，但是在PPO中GAE的处理方式不同

#### 💡GPT的修正与补充

你的直觉正确，但理论关键点需要修正

你抓住了一个重要事实：

**Fall 和 Timeout 虽然都会触发 Reset，但它们在 PPO 中的处理并不完全相同。**

不过，你把 Timeout 的处理理解成“后面的 trajectory 继续产生 Reward”，这不准确。

我们重新梳理。

##### **1. Fall：真正的 Terminal State**

假设机器人走路时摔倒：

```
s_t
 ↓
a_t
 ↓
机器人摔倒
 ↓
r_t
 ↓
Episode 结束
```

摔倒意味着当前任务轨迹真正结束。

因此 GAE 不应该继续把下一轮站起来后的经验算入当前 Episode。

一般来说，Fall 的 Bootstrap Value 为零：

$$ V_{\text{bootstrap}}=0 $$

##### **2. Timeout：Time-limit Truncation**

再假设机器人没有摔倒，只是达到规定的 20 秒 Episode 时间上限。

这并不代表机器人在物理任务中已经失败。

它可能仍然可以继续行走。

因此，如果把 Timeout 当作一个真正的 Terminal State，强行让后续价值为零，就可能低估当前状态的价值。

我们希望通过 Critic 对时间截断处的价值进行估计：

$$ R_t^{\text{timeout}} = r_t+\gamma V_{\text{bootstrap}} $$

这就是 **Timeout Bootstrapping**。

注意：

这不是把下一轮 Episode 的实际 Reward 接回来，而是用 Critic 估计截断处尚未观测到的未来回报。

<div style="max-width: 750px; margin: 20px auto; background-color: #ffffff; border-radius: 12px; border: 1px solid #e5e7eb; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.05); box-sizing: border-box;">
  
  <div style="font-weight: bold; font-size: 18px; color: #111827; margin-bottom: 20px;">Fall 和 Timeout 的区别</div>
  
  <div style="display: flex; gap: 16px; flex-wrap: wrap;">
    
    <!-- 左侧：Fall 卡片 -->
    <div style="flex: 1; min-width: 250px; background-color: #fff5f5; border-radius: 8px; padding: 20px; box-sizing: border-box;">
      <div style="font-size: 20px; font-weight: bold; color: #111827; margin-bottom: 12px;">Fall</div>
      
      <div style="margin-bottom: 16px;">
        <span style="display: inline-block; background-color: #fee2e2; color: #ef4444; border-radius: 20px; padding: 4px 12px; font-size: 13px; font-weight: 500;">真正终止</span>
      </div>
      
      <div style="font-size: 15px; color: #111827; margin-bottom: 8px;">机器人摔倒</div>
      
      <div style="font-size: 24px; font-weight: bold; color: #ef4444; margin-bottom: 12px;">Bootstrap = 0</div>
      
      <div style="font-size: 13px; color: #6b7280; line-height: 1.5;">不再估计当前 Episode 后续价值</div>
    </div>

    <!-- 右侧：Timeout 卡片 -->
    <div style="flex: 1; min-width: 250px; background-color: #eff6ff; border-radius: 8px; padding: 20px; box-sizing: border-box;">
      <div style="font-size: 20px; font-weight: bold; color: #111827; margin-bottom: 12px;">Timeout</div>
      
      <div style="margin-bottom: 16px;">
        <span style="display: inline-block; background-color: #dbeafe; color: #3b82f6; border-radius: 20px; padding: 4px 12px; font-size: 13px; font-weight: 500;">时间截断</span>
      </div>
      
      <div style="font-size: 15px; color: #111827; margin-bottom: 8px;">时间达到上限</div>
      
      <div style="font-size: 24px; font-weight: bold; color: #3b82f6; margin-bottom: 12px;">Bootstrap V</div>
      
      <div style="font-size: 13px; color: #6b7280; line-height: 1.5;">利用 Critic 估计截断处价值</div>
    </div>

  </div>
</div>

#### **3. 结合实际 RSL-RL 代码**

这里我专门重新核对了 Unitree 推荐的 RSL-RL v1.0.2。

在 Unitree 环境中：

```python
self.time_out_buf = (
    self.episode_length_buf > self.max_episode_length
)

self.reset_buf |= self.time_out_buf
```

所以 Fall 和 Timeout 都会被放进 `reset_buf`，都触发环境 Reset。

但是在 `reset_idx()` 中：

```python
if self.cfg.env.send_timeouts:

    self.extras["time_outs"] = self.time_out_buf
```

单独把 Timeout 信息传给 PPO。


接着在 RSL-RL v1.0.2 的 `ppo.py`：

```python
def process_env_step(self, rewards, dones, infos):

    self.transition.rewards = rewards.clone()

    self.transition.dones = dones

    if 'time_outs' in infos:

        self.transition.rewards += (
            self.gamma
            * torch.squeeze(
                self.transition.values
                * infos['time_outs'].unsqueeze(1).to(self.device),
                1
            )
        )

    self.storage.add_transitions(self.transition)

    self.transition.clear()

    self.actor_critic.reset(dones)
```

这里有一个特别值得你注意的细节。

在 `PPO.act()` 中：

```python
self.transition.values = (
    self.actor_critic.evaluate(critic_obs).detach()
)
```

这是环境执行 Action 之前计算并保存的 Value。

因此，在这个指定版本的实际实现中，Timeout 修正使用的是：

$$ \boxed{ r_t\leftarrow r_t+\gamma V(s_t) } $$

这里使用的是存储的 `transition.values`，也就是前一个决策时刻计算的价值，不是重新计算出来的终止后最终 Observation 的价值。

这一点需要特别区分：

- 从强化学习理论角度，通常讨论的是在截断状态进行价值 Bootstrap。
    
- 从具体的 RSL-RL v1.0.2 代码角度，它使用 `transition.values` 实现这一修正，是该版本的具体实现方式。
    

#### **4. 那 GAE 如何阻止下一轮 Episode 的数据串进来？**

打开 `rollout_storage.py`：

```python
next_is_not_terminal = (
    1.0 - self.dones[step].float()
)

delta = (
    self.rewards[step]
    + next_is_not_terminal * gamma * next_values
    - self.values[step]
)

advantage = (
    delta
    + next_is_not_terminal * gamma * lam * advantage
)
```

这里非常重要：

**无论 Fall 还是 Timeout，**`dones=True` **都会切断 GAE 向下一轮 Episode 的递推。**

区别在于：

|项目|Fall|Timeout|
|---|---|---|
|`done`|True|True|
|`time_outs`|False|True|
|Reset|是|是|
|Reward Bootstrap 修正|无|有|
|GAE 跨 Episode 传播|不允许|不允许|

所以你原来的回答应该改成：

> Fall 和 Timeout 都会终止当前 Episode，GAE 都需要通过 Done Mask 阻止后续 Episode 的数据传播进来。但 Fall 是真正的终止状态，不需要 Bootstrap；Timeout 是人为设置的时间截断，机器人实际上可能仍然能够继续完成任务，因此 PPO 会通过 Timeout 标记进行价值修正，避免把时间截断错误地当成价值为零的终止状态。

这是 Day 7 最重要的一道题。


### 问题 4：Command 采样

当前 G1 的：

```python
resampling_time = 10.
```

如果：

$$
dt=0.02s
$$

为什么大约 500 个策略时间步才更新一次 Command？

为什么不能每个策略时间步都随机生成新命令？

答：如果每个策略时间步都随机生成新命令，命令变化太快，机器人每次根据reward，更新速度一步步接近command，command再每一次输入都改变的话，就像再打一个移动靶子，机器人很难在高频command更新中学习到如何接近command，而且每次因为更新command会导致机器人一直拿到很低的reward，可能导致机器人训练崩坏
#### 💡GPT的修正与补充

你已经理解了 Command 的本质：

**它是环境给 Policy 的目标，而不是 Policy 自己产生的动作。**

当前代码：

```python
env_ids = (
    self.episode_length_buf
    % int(
        self.cfg.commands.resampling_time / self.dt
    ) == 0
).nonzero(as_tuple=False).flatten()

self._resample_commands(env_ids)
```

GitHub

根据：

$$ T_{\text{resampling}}=10s $$

$$ dt=0.02s $$

得到：

$$ N=\frac{10}{0.02}=500 $$

因此，每个环境约每 500 个策略步重新采样一次。

你的解释基本完全正确。

这里再补充一个细节：

在 G1 当前配置：

```python
heading_command = True
```

虽然目标 Heading 不是每个策略时间步都重新采样，但代码会根据当前机器人朝向，持续更新第三维偏航角速度指令：

```python
self.commands[:, 2] = torch.clip(
    0.5 * wrap_to_pi(
        self.commands[:, 3] - heading
    ),
    -1.,
    1.
)
```

所以：

- 目标前进速度、侧向速度和 Heading：按一定周期重新采样。
    
- 当前需要的偏航角速度：根据目标 Heading 与机器人当前朝向持续计算。
    

另外，你说每步改变 Command 会导致训练崩坏，这个结论有点过强。

更准确地说，学习目标会变成高频变化的指令跟踪问题，策略必须同时处理很快变化的目标，学习难度通常会上升，Reward 也更难稳定。

如果专门设计高频指令跟踪任务，也可以采用更快的指令更新，但那已经是另一种任务设计了。
### 问题 5：Domain Randomization

为什么摩擦系数和 Base Mass 在创建环境时随机化，而 Push Robots 在运行过程中执行？

这三种随机化分别在模拟真实机器人中的什么不确定性？

答：因为前两种随机化是在一局游戏中固定的随机化，如果在运行过程中不断变化环境，容易导致机器人完全无法适应，而Push Robots是模拟机器人在行走过程中受到冲击的随机化，所以需要每隔一定时间在运行过程中执行，让机器人学会在冲击中保持平衡然后继续行走

#### 💡GPT的修正与补充

你的核心理解正确。

这里最重要的是，你已经认识到了：

**环境物理参数变化和外部动态扰动是两种不同性质的不确定性。**

我们具体对应代码。

**1. Friction：摩擦系数随机化**

```python
def _process_rigid_shape_props(self, props, env_id):
```

这一函数的注释明确说明，它在环境创建期间调用。每个环境取得摩擦系数后，会将其应用到对应刚体形状属性。

模拟真实世界中的：

- 不同地面材质；
    
- 不同地面摩擦；
    
- 脚底可能发生的打滑。
    

**2. Base Mass：基座质量随机化**

```python
def _process_rigid_body_props(self, props, env_id):

    if self.cfg.domain_rand.randomize_base_mass:

        rng = self.cfg.domain_rand.added_mass_range

        props[0].mass += np.random.uniform(
            rng[0], rng[1]
        )
```

它是在创建每个环境的刚体属性时，修改基座质量。

模拟真实机器人可能出现的：

- 负载变化；
    
- 质量估计误差；
    
- 硬件配置差异。
    

**3. Push Robots：运行期间的动态扰动**

```python
def _push_robots(self):
```

它通过修改基座水平线速度，模拟外部冲击：

```python
self.root_states[:, 7:9] = torch_rand_float(
    -max_vel,
    max_vel,
    (self.num_envs, 2),
    device=self.device
)
```

并通过环境索引将选定机器人的新状态写回仿真器。

模拟：

- 被人推了一下；
    
- 遭遇突然的外部冲击；
    
- 行走中产生意外速度扰动。
    

这里我再纠正一个小细节：

摩擦系数和质量不只是“一局 Episode 里固定”，而是在当前实现中**每个并行环境创建时随机设定，并在该环境后续 Episode Reset 时保留**。

这意味着同一个环境可能连续跑多个 Episode，但其摩擦系数和基座质量保持不变。不同环境之间则具有不同的物理参数。

**三种 Domain Randomization 的本质：**

|方法|随机化对象|主要模拟|
|---|---|---|
|Friction|地面接触参数|地面条件不同|
|Base Mass|机器人质量|负载、质量误差|
|Push|基座运动状态|外部瞬时冲击|

前两项是环境创建阶段的物理参数随机化，Push 是训练运行过程中的状态扰动。

这一题理解很到位。

# 今天的实践练习

今天可以真正开始第一个小型代码验证。

不用修改 Reward、PD 或训练参数。

在你已经成功创建好的环境对象 `env` 可访问的位置，执行：

```python
import torch

print("Num envs:", env.num_envs)
print("Policy dt:", env.dt)

print(
    "Episode max steps:",
    env.max_episode_length)

print(
    "Command resampling steps:",
    int(
        env.cfg.commands.resampling_time / env.dt
    ))

print(
    "Push interval steps:",
    env.cfg.domain_rand.push_interval)

print(
    "Initial commands:",
    env.commands[:4])
```

根据当前官方 G1 配置，预期时间相关结果是：

```text
Policy dt: 0.02
Episode max steps: 1000
Command resampling steps: 500
Push interval steps: 250
```

实际运行结果以你的本地配置为准。

然后进行 5 次零动作环境交互：

```python
actions = torch.zeros(
    env.num_envs,
    env.num_actions,
    device=env.device)

with torch.no_grad():
    for i in range(5):
        obs, privileged_obs, rewards, dones, infos = (
            env.step(actions)
        )

        print(
            f"Step {i}:",
            "obs =", obs.shape,
            "reward =", rewards.shape,
            "done =", dones.sum().item(),
            "episode_length =",
            env.episode_length_buf[:4].tolist()
        )
```

预计观察到：

- Observation 维度为 `[num_envs, 47]`。
- Reward 维度为 `[num_envs]`。
- Done 维度为 `[num_envs]`。
- 每次环境交互，Episode 计数器都会变化。
- 不同环境可以拥有不同的 Command。

> 这只是验证环境接口和张量形状，不代表零动作策略能够正常行走，也不代表训练已经完成。

如果目前没有现成的 `env` 对象，就先完成前面的代码追踪，不需要为了这个实验单独重建整个训练工程。

# 今天的完成标准

今天结束时，希望你能留下以下笔记：

| 模块 | 需要掌握的结论 |
|---|---|
| Termination | 检查接触力、姿态和时间限制 |
| Reset | 通过 `env_ids` 选择性重置环境 |
| DOF Reset | 关节位置随机化，速度归零 |
| Root Reset | 恢复基座状态并随机化初始速度 |
| Command | 每隔约 10 秒重新采样 |
| Friction | 环境创建时随机化摩擦系数 |
| Base Mass | 环境创建时修改基座质量 |
| Push | 训练期间修改基座水平速度进行扰动 |

最重要的三点：

1. Reset 是针对单个并行环境进行的，不意味着整个训练停止。
2. Timeout 和 Fall 都会结束 Episode，但算法应区别处理。
3. Domain Randomization 的目标是让策略适应不同的物理条件，而不是只适应一个完美的仿真模型。