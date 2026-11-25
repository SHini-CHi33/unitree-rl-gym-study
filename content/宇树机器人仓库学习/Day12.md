# 今日任务

今日主题：建立 Baseline，第一次独立运行和记录训练

## **一、今天的核心目标**

前十天，我们一直在研究：

```
Observation
    ↓
Actor / Critic
    ↓
Action
    ↓
PD Controller
    ↓
Physics Simulation
    ↓
Reward
    ↓
Rollout Storage
    ↓
GAE
    ↓
PPO Update
```

但是现在还存在一个问题：

**如果我修改了某个参数，怎么知道机器人表现真的发生了变化？**

在强化学习实验中，不能只凭感觉。

我们需要一个 Baseline（基准实验）。

Baseline 就是：

> 在不修改任何原始算法和环境参数的情况下，使用固定的训练配置运行一次，记录训练过程和结果，作为后续实验的对照组。

今天先不修改任何 Reward。

## **二、今天要阅读的文件**

|文件|需要理解的函数|
|---|---|
|`legged_gym/envs/g1/g1_config.py`|G1 原始配置|
|`legged_gym/scripts/train.py`|`train()`|
|`legged_gym/utils/task_registry.py`|`make_env()`、`make_alg_runner()`|
|`legged_gym/utils/helpers.py`|`update_cfg_from_args()`|
|RSL-RL `on_policy_runner.py`|`learn()`、`log()`、`save()`|

官方代码：

- [train.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/scripts/train.py?utm_source=chatgpt.com)
    
- [task_registry.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/utils/task_registry.py?utm_source=chatgpt.com)
    
- [helpers.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/utils/helpers.py?utm_source=chatgpt.com)
    
- [G1 配置文件](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py?utm_source=chatgpt.com)
    

今天重点理解一条数据流：

```
命令行参数
    ↓
get_args()
    ↓
update_cfg_from_args()
    ↓
task_registry.make_env()
    ↓
task_registry.make_alg_runner()
    ↓
OnPolicyRunner.learn()
    ↓
Rollout + PPO Update
    ↓
TensorBoard Log + Checkpoint
```

官方 README 支持通过 `--num_envs`、`--max_iterations`、`--run_name` 等参数覆盖配置。

## **三、第一部分：检查你的 Baseline 配置**

先打开：

`legged_gym/envs/g1/g1_config.py`

今天不修改文件，先记录：

```python
num_observations = 47
num_privileged_obs = 50
num_actions = 12

action_scale = 0.25
decimation = 4

tracking_lin_vel = 1.0
action_rate = -0.01

push_robots = True
randomize_friction = True
randomize_base_mass = True
```

这些就是我们后续实验的基本对照条件。

接下来在终端执行：

```
git rev-parse --short HEAD

git status --short
```

目的：

- 记录当前仓库 Commit。
    
- 检查当前代码有没有未提交修改。
    

这一步很重要。

因为以后实验结果发生变化，我们必须知道究竟是修改了 Reward，还是代码版本本身发生了变化。

## **四、第二部分：第一次独立运行 Baseline**

今天真正要运行程序。

官方训练入口：

```
python legged_gym/scripts/train.py --task=g1
```

为了第一次独立实验控制资源，我建议使用一个较小的训练规模：

```
python legged_gym/scripts/train.py \
    --task=g1 \
    --headless \
    --num_envs=128 \
    --max_iterations=100 \
    --run_name=day11_baseline
```

这里解释一下：

|参数|作用|
|---|---|
|`--task=g1`|使用 G1 任务|
|`--headless`|不打开实时图形界面|
|`--num_envs=128`|使用 128 个并行环境|
|`--max_iterations=100`|运行 100 轮 PPO Iteration|
|`--run_name`|给实验起一个独立名字|

如果你此前的机器能够稳定运行 4096 个环境，也可以继续使用原来的环境数量。关键是 Day 11 和 Day 12 必须使用相同数量。

这些参数都可以通过当前官方入口和配置覆盖机制传递。

**注意：100 Iterations 不代表训练完成**

当前默认每个环境每轮采集 24 个策略时间步。

因此：

$$ N_{\text{transitions}} =128\times24\times100 = 307200$$

这次运行主要用于建立实验基准和验证训练、日志、模型保存流程，不是为了在 100 次迭代内获得最终稳定行走策略。

## **五、第三部分：观察训练日志**

训练结束后，检查：

```
logs/
└── g1/
    └── 时间戳_day11_baseline/
        └── model_100.pt
```

默认结果目录遵循：

```
logs/<experiment_name>/<date_time>_<run_name>/
```

RSL-RL 的 `OnPolicyRunner.learn()` 在训练结束时会保存当前模型，因此即使 100 小于常规保存间隔，也会保存最终 Checkpoint。

接下来，打开 TensorBoard：

```
tensorboard --logdir=logs/g1
```

在浏览器访问：

```
http://localhost:6006
```

如果没有安装 TensorBoard，需要先检查环境中的依赖。

### **今天重点观察五个指标**

**1. Train/mean****_****reward**

整体平均 Episode Reward。观察是否出现总体改善趋势。

**2. Train/mean****_****episode****_****length**

平均 Episode 长度。机器人如果频繁摔倒，通常会导致这个指标偏低。

**3. Episode/rew****_****tracking****_****lin****_****vel**

观察速度跟踪奖励。

**4. Loss/value****_****function 和 Loss/surrogate**

观察 Critic 和 Actor 的优化情况，但不能只凭 Loss 下降判断机器人性能一定变好。

**5. Perf/total****_****fps**

观察运行效率，帮助判断环境数量是否合适。

当前 RSL-RL v1.0.2 的 `OnPolicyRunner.log()` 会记录这些训练指标，并且还会记录各项 Episode Reward。

# **TensorBoard 训练曲线解读：每个指标从哪里来？什么样才算收敛？**

这是一个非常重要的问题。你现在已经从阅读代码进入实际训练实验阶段，必须学会读懂 TensorBoard，否则即使训练了 10,000 个 Iteration，也不知道机器人究竟学得怎么样。

我先给你一个核心认识：

**TensorBoard 不是单纯记录 PPO Loss 的工具，而是同时记录三件事情：**

1. **机器人表现如何**：Reward、Episode Length、速度跟踪效果。
    
2. **神经网络如何学习**：Policy Loss、Value Loss、Entropy、KL。
    
3. **训练效率如何**：FPS、Learning Time、Iteration Time。
    

而且，这三类指标的收敛标准并不一样。

例如：

- Mean Reward 通常希望逐渐提高并趋于稳定。
    
- Value Loss 通常希望降低并趋于稳定。
    
- Entropy 不一定越低越好。
    
- FPS 根本不是收敛指标。
    
- Episode Length 要结合 Termination 原因来判断。
    

因此，不能只看一条曲线就判断 PPO 收敛。
## **一、先搞清楚：TensorBoard 中的指标究竟从哪里来？**

按照我们一直研究的 Unitree G1 + RSL-RL v1.0.2，TensorBoard 数据主要来自三个地方。

**数据来源一：G1 Environment**

文件：`legged_robot.py`

```python
self.compute_reward()
self.episode_sums[name] += rew
```

记录机器人每一轮 Episode 中各项 Reward 的累计贡献。

**数据来源二：PPO Algorithm**

文件：`rsl_rl/algorithms/ppo.py`

```python
mean_value_loss += value_loss.item()
mean_surrogate_loss += surrogate_loss.item()
```

记录 Critic 和 Actor 的优化损失。

**数据来源三：OnPolicyRunner**

文件：`rsl_rl/runners/on_policy_runner.py`

```python
self.writer.add_scalar(
    'Train/mean_reward',
    statistics.mean(locs['rewbuffer']),
    locs['it']
)
```

将环境指标、算法指标和运行效率指标写入 TensorBoard。

因此，最重要的代码位置其实是：

`rsl_rl/runners/on_policy_runner.py` 中的：

```python
def log(self, locs, width=80, pad=35):
```

我核对了官方 RSL-RL v1.0.2 的实现，下面的指标解释以这个版本为准。

## **二、TensorBoard 中的指标总览**

当前版本主要有四大类：

| TensorBoard 分类 | 主要关注什么             | 是否用于判断收敛   |
| -------------- | ------------------ | ---------- |
| `Train/`       | 整体训练表现             | ==🔴非常重要== |
| `Episode/`     | 各项奖励的具体表现          | ==🔴非常重要== |
| `Loss/`        | Actor、Critic 的优化情况 | ==🔵辅助判断== |
| `Policy/`      | 策略探索程度             | ==🔵辅助判断== |
| `Perf/`        | 训练运行效率             | 不代表收敛      |

下面逐项解释。
## **三、第一类：Train —— 最重要的整体表现指标**

### **1. Train/mean_reward**

==🟢最重要的指标之一==

#### **它是什么？**

每个 Episode 中累计获得的总 Reward，再对最近完成的 Episode 求平均。

看 `on_policy_runner.py`：

```python
rewbuffer = deque(maxlen=100)
```

每完成一个 Episode：

```python
cur_reward_sum += rewards

new_ids = (dones > 0).nonzero(
    as_tuple=False
)

rewbuffer.extend(
    cur_reward_sum[new_ids][:, 0]
    .cpu().numpy().tolist()
)
```

最后：

```python
self.writer.add_scalar(
    'Train/mean_reward',
    statistics.mean(locs['rewbuffer']),
    locs['it']
)
```

这里有三个细节：

- `cur_reward_sum`：每个环境当前 Episode 的累计奖励。
    
- `rewbuffer`：保存最近最多 100 个已经完成的 Episode 的累计奖励。
    
- `mean_reward`：对这个缓冲区中的奖励取平均。
    

它本质上实现了一个“滑动窗口的移动平均（Moving Average）”**。我们把这四个核心步骤拆开，用“考试成绩单”来给你翻译：

##### 1. `rewbuffer = deque(maxlen=100)`：准备一个记录本

- `deque` 是 Python 里的双向队列。`maxlen=100` 意味着这个本子**最多只能记录 100 条数据**。如果记满了 100 条，再来一条新的，最老的那条就会被自动挤掉。
    
- **物理意义**：这是为了保证你看到的分数是你**“最近”**的真实水平。如果你用一百年前的分数来代表今天，那显然是不准的。
    

##### 2. `cur_reward_sum += rewards`：攒硬币

- 环境每走一步（20ms），都会给每个机器人发一点奖励（`rewards`）。
    
- 因为有 4096 个机器人同时在跑，代码里会维护一个长度为 4096 的数组 `cur_reward_sum`。
    
- **物理意义**：这就是你在玩游戏时的“得分累加器”。只要没死，哪怕只拿了 0.01 分，也要一分一分地存起来。
    

##### 3. `new_ids = (dones > 0).nonzero(...)`：抓出“交卷”的人

- 只有当机器人摔死（`done=True`）或者超时的时候，它的这一局分数才算“尘埃落定”。
    
- 这行代码就是精准地把这一帧里，刚刚死掉的那几个机器人的编号（`new_ids`）抓出来。
    

##### 4. `rewbuffer.extend(...)`：登记成绩单

- `cur_reward_sum[new_ids]`：把刚才死掉的那几个机器人的总得分取出来。
    
- `extend(...)`：把这几条总得分塞进刚才那个最多只能存 100 条的 `rewbuffer` 里。同时，底层代码会把那几个死掉机器人的 `cur_reward_sum` 清零，让它们重新开始新一局攒硬币。
    

##### 🎯 最后：算平均分


```python
statistics.mean(locs['rewbuffer'])
```

- 把 `rewbuffer` 里这 100 个**已经完赛的真实总分**，加起来除以 100（如果不足 100 就是实际数量）。
    
- 写入 TensorBoard。
    

##### 💡 灵魂拷问：为什么不直接算当前所有环境的总分平均？

假设直接对当前 4096 个环境的总分（`cur_reward_sum`）求平均，会发生什么？

- 有的机器人才刚出生 2 步（得分 0.1）。
    
- 有的机器人已经活了 1000 步（得分 500）。
    
- 如果你把它们的分数直接平均，得出的数字会受到那些“活得很久的机器人”的严重拉扯，**无法真实反映策略当前的生死概率**。
    

**只有等机器人“死了”，把一条完整的命（Episode）的总得分记录下来，取平均，才是真正的“单局平均回报”。**

##### 💡 灵魂拷问：但是万一机器人都一直不死，那没有返回idx，岂不是reward是零？

直接回答你：**不会变成 0！** 为什么？因为 `deque(maxlen=100)` 这个数据结构极其聪明，同时 PPO 的底层设定也绝对不允许“一直不死”的情况发生。

我们来拆解一下你的疑惑：
##### 📦 情况一：`rewbuffer` 里还有“存货”

`rewbuffer` 就像一个容量为 100 的篮子。  
假设前 100 步，有 50 个机器人陆续摔死了，它们的总得分被扔进了篮子里。  

接下来的 1000 步里，剩下的机器人突然全变成了“走钢丝大师”，一个都没死。  

**此时发生什么？**  

代码确实不会往篮子里放新东西了，但**篮子里的旧东西还在！**  
当 TensorBoard 记录数据时，`statistics.mean(rewbuffer)` 会把这 50 个旧数据拿出来再算一次平均值。  

所以，曲线不会变成 0，它只会变成一条**水平的直线**（因为数字没变过），直到有新的机器人死掉，把旧的挤出去。
##### ⏱️ 情况二：机器人真的能一直不死吗？（绝对不可能）

你忘了一个极其无情的设定：**Episode 超时截断**。  
我们在上一题里讲过 `check_termination` 的第三个条件：

不管你走得多完美，哪怕你是神的化身，只要你走满了规定的时间步（比如 1000 步），系统就会强行判定你 `done=True`，强制你重新开始。  
所以，只要有 `max_episode_length` 这个达摩克利斯之剑悬在头顶，机器人**绝对不可能无限期地活下去**，这个 `new_ids` 就一定会定时刷新出来。

---

#### **它应该呈现什么样子？**

![[Pasted image 20261121170457.png|636]]

不过，这里一定要注意：

**Reward 曲线平稳不等于机器人已经学会走路。**

例如：

- Reward 很稳定，但机器人只会原地站立。
    
- Reward 不断上升，但机器人速度跟踪仍然不好。
    
- Reward 很高，但机器人在某些指令下会摔倒。
    

因此，Mean Reward 只能告诉你策略在当前奖励函数下的整体表现，不能单独证明任务成功。

### **2. Train/mean_episode_length**

它表示最近完成的 Episode 的平均长度。

代码：

```python
cur_episode_length += 1

lenbuffer.extend(
    cur_episode_length[new_ids][:, 0]
    .cpu().numpy().tolist()
)
```
最后：

```python
self.writer.add_scalar(
    'Train/mean_episode_length',
    statistics.mean(locs['lenbuffer']),
    locs['it']
)
```

当前 G1 基础配置：

$$ T_{\max}=20s $$

$$ dt=0.02s=20ms $$

所以：

$$ N_{\max}\approx\frac{20}{0.02}=1000 $$

因此，正常情况下：

**Episode Length 参考**

| 数值      | 可以怎样理解              |
| ------- | ------------------- |
| 100–200 | Episode 较短，可能频繁终止   |
| 400–600 | 能够持续运行一段时间          |
| 800–950 | 接近最大 Episode 时长     |
| 约 1000  | 很多 Episode 已运行至时间上限 |

这些是帮助理解的参考区间，不是 G1 的通用合格标准。

理想趋势通常是：

```
前期：机器人经常摔倒，Episode 很短

中期：能够维持站立和行走，Episode 变长

后期：多数 Episode 达到时间限制
```

但还必须结合 `time_outs` 判断。

如果 Episode Length 达到约 1000，说明机器人没有提前触发终止条件的情况增加了，但不能仅凭这个指标证明它走得好。

**这两个指标应该放在一起看：**

| Mean Reward | Mean Episode Length | 可能的情况              |
| ----------- | ------------------- | ------------------ |
| 上升          | 上升                  | 通常是积极信号            |
| 上升          | 很低                  | 可能学会了部分奖励目标，但仍频繁失败 |
| 平稳且较高       | 接近上限                | 较好的候选收敛状态          |
| 下降          | 下降                  | 可能发生训练退化           |

## **四、第二类：Episode —— 每个 Reward 到底表现如何？**

这部分是我认为你研究 G1 时最值得关注的一组曲线。

因为它能够把总 Reward 拆开，让我们知道机器人究竟在哪个方面进步了，具体reward的含义可以去回看Day6

### **1. 这些曲线从哪里来？**

首先看：

`legged_robot.py`

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

每执行一次策略步，都会累加各项 Reward 的实际加权贡献。

当某个 Episode 结束：

```python
self.extras["episode"] = {}

for key in self.episode_sums.keys():

    self.extras["episode"]["rew_" + key] = (
        torch.mean(self.episode_sums[key][env_ids])
        / self.max_episode_length_s
    )
```

然后 Runner 自动在前面加上 `Episode/`。

例如：

```
Episode/rew_tracking_lin_vel
```

来自：

```python
_reward_tracking_lin_vel()
```

而且它经过了：

1. Reward 函数计算。
    
2. Reward Scale 加权。
    
3. `dt` 时间缩放。
    
4. Episode 内累计。
    
5. 除以最大 Episode 时长。
    

##### 🏭 这 5 个步骤的流水线到底是怎么运转的？

- **步骤 1：Reward 函数计算**  
    `_reward_tracking_lin_vel()` 发现速度追得不错，返回原始数值 `0.8`。
    
- **步骤 2 & 3：Scale 加权 与 dt 时间缩放**  
    在 `_prepare_reward_function` 里已经乘过 `dt`（0.02）了。假设权重是 1.0，那么这一步 `0.8 * 1.0 * 0.02 = 0.016`。  
    _(这是给神经网络训练用的单步奖励)_
    
- **步骤 4：Episode 内累计**  
    机器人每走一步（20ms），就把这 0.016 扔进 `self.episode_sums` 的小金库里。机器人走满 20 秒（1000 步），小金库里就存了 `0.016 * 1000 = 16.0`。
    (在实际中，这里是做了一个mean.（）毕竟不可能每次策略步都获得一样的reward)
    
- **步骤 5：结算与除以最大时长**  
    当这个机器人死了，代码把它的 16.0 拿出来，除以最大存活时长 $T_{\max}=20s$（`16.0 / 20 = 0.8`）。
    

**你看，转了一圈，最后画在 TensorBoard 上的 `Episode/rew_tracking_lin_vel = 0.8`。**  
这 0.8 完美对应了机器人一开始一秒内速度追踪的原始表现！经过这 5 道工艺的“封装”，它变成了一个无论存活长短，都能公平对比的**标准速率**。

##### 💡 灵魂拷问：为什么不能直接比“总分”？

假设你开了一家公司，有两个员工：

- **员工 A（Episode 1）**：干了 **10 秒钟**，拿了 **100 块钱**业绩。
    
- **员工 B（Episode 2）**：干了 **2 秒钟**就猝死了，拿了 **20 块钱**业绩。
    

如果你只看“总业绩”，员工 A（100块）完爆员工 B（20块）。  
但实际上，员工 B 每秒赚 10 块，员工 A 每秒才赚 10 块，他们俩的工作效率一模一样！  
在强化学习中，这就叫**“存活时间不同导致的偏差”**。如果直接看总分，机器人就会觉得“我只要活得久，哪怕在地上蹭，总分也很高”，这会导致网络学出“躺平苟活”的策略。

##### 💡 灵魂拷问：“除以最大时长”在干什么？

为了防止上面那种情况，代码直接把所有人的总成绩，**统一除以一个固定的满分时间尺度**（`max_episode_length_s`，比如设定为 20 秒）。

这个指标就变成了：**“在 20 秒的固定考核期内，你平均每秒能拿多少分（或者说你能拿满多少比例的分）。”**  
刚才那个例子：

- 员工 A：100 / 20 = **5**
    
- 员工 B：20 / 20 = **1**
    

**为什么必须要除以固定的最大时长，而不是除以各自的存活时间？** 
  
如果你除以**实际存活时间**，那个 2 秒钟猝死的员工（20 / 2 = 10），他的得分率居然比满分员工（100 / 20 = 5）还高！这会鼓励网络：“快！快猝死！猝死前随便踢一脚，得分率最高！”  
所以必须除以**固定的最大时长**。早死的人，时间分母是固定的，他的分子很小，所以分数极低；活满的人才能拿到高分。这就逼着网络**必须活得久，且走得好**。


### **2. G1 重点关注哪些 Episode 曲线？**

我帮你分成三组。

#### **A. 机器人行走能力**

| 指标                              | 含义           | 希望看到的趋势   |
| ------------------------------- | ------------ | --------- |
| `Episode/rew_tracking_lin_vel`  | X/Y 线速度跟踪    | 上升并稳定     |
| `Episode/rew_tracking_ang_vel`  | 偏航角速度跟踪      | 上升并稳定     |
| `Episode/rew_contact`           | 步态相位与实际接触匹配  | 根据权重和行为稳定 |
| `Episode/rew_feet_swing_height` | 摆动脚高度误差惩罚    | 通常逐渐接近 0  |
| `Episode/rew_contact_no_vel`    | 接触期间脚部滑动相关惩罚 | 通常逐渐接近 0  |

这里尤其要关注：

```
Episode/rew_tracking_lin_vel
Episode/rew_tracking_ang_vel
```

因为 G1 的核心任务之一就是跟踪运动指令。

对于当前 G1 的线速度跟踪项：

$$ f_{\text{track}} = \exp(-e_v/\sigma) $$

原始函数最大为 1，权重为 1.0，因此理想情况下，按当前实现的归一化方式：

$$ Episode/rew\_tracking\_lin\_vel\leq1 $$

如果长时间保持较高水平，是积极信号。

角速度跟踪的权重是 0.5，因此其对应指标的理想上限约为 0.5。

但实际还要考虑 Episode 提前终止和不同指令难度。

#### **B. 机器人身体稳定性**

| 指标                        | 含义               | 希望看到的趋势 |
| ------------------------- | ---------------- | ------- |
| `Episode/rew_orientation` | 身体姿态偏离惩罚         | 趋近 0    |
| `Episode/rew_base_height` | 基座高度偏差惩罚         | 趋近 0    |
| `Episode/rew_lin_vel_z`   | 机身竖直方向速度惩罚       | 趋近 0    |
| `Episode/rew_ang_vel_xy`  | Roll/Pitch 角速度惩罚 | 趋近 0    |

注意，这些都是负权重的奖励项。

例如：

$$ w_{\text{orientation}}=-1 $$

因此：

- -0.5 变成 -0.1：通常说明惩罚贡献减少。
    
- -0.1 变成 -0.4：通常说明惩罚贡献增加。
    

这里的“趋近 0”是指负奖励贡献的绝对值变小，而不是让所有 Reward 都变成零。

#### **C. 动作控制质量**

| 指标                           | 含义             | 希望看到的趋势  |
| ---------------------------- | -------------- | -------- |
| `Episode/rew_action_rate`    | 相邻 Action 变化惩罚 | 通常趋近 0   |
| `Episode/rew_dof_vel`        | 关节速度惩罚         | 结合运动需求判断 |
| `Episode/rew_dof_acc`        | 关节加速度惩罚        | 通常避免过大   |
| `Episode/rew_dof_pos_limits` | 关节接近限位的惩罚      | 趋近 0     |
| `Episode/rew_hip_pos`        | 髋关节相关位置惩罚      | 通常减少     |

特别提醒：

`Episode/rew_action_rate` 越接近 0，通常表示动作变化惩罚越小，但不能据此单独认定策略更好。

因为策略可能为了避免动作变化而不愿意执行快速动作，反而影响速度跟踪。

这也是 Day 12 我们要做 Action Rate 单因素实验的原因。

另外，G1 配置中的：

```python
feet_air_time = 0.0
collision = 0.0
```

对应奖励项因为 Scale 为零而不会进入普通 Reward 注册流程，所以通常不会看到对应的 `Episode/rew_feet_air_time`、`Episode/rew_collision` 曲线。

---
## **五、第三类：Loss —— 神经网络究竟学得怎么样？**

这一部分和你之前学习的 CleanRL PPO 联系最紧密。

当前 RSL-RL v1.0.2 的 `on_policy_runner.py` 主要记录三个 Loss 类指标，以及一个学习率。

### **1. Loss/value_function**

代表 Critic 的学习情况

它来自 `rsl_rl/algorithms/ppo.py`：

```
value_losses = (
    value_batch - returns_batch
).pow(2)

value_losses_clipped = (
    value_clipped - returns_batch
).pow(2)

value_loss = torch.max(
    value_losses,
    value_losses_clipped
).mean()
```

它衡量的是：

$$ L_V= \mathbb E[ (V_\phi(s_t)-\hat R_t)^2 ] $$

其中：

- $V_\phi(s_t)$：Critic 预测的价值。
    
- $hat R_t$：GAE 计算得到的 Return Target。
    

直观理解：

假设 Critic 预测：

$$ V(s_t)=5 $$

而 GAE 得到：

$$\hat R_t=8 $$

那么它们的预测误差就是：

$$ (5-8)^2=9 $$

Critic 就需要通过梯度更新，让价值预测逐渐靠近 Target。

**应该呈现什么样子？**

<div style="font-family: -apple-system, sans-serif; padding: 20px; border: 1px solid #ebebeb; border-radius: 10px; background-color: #fff; max-width: 650px; margin: 10px 0;">
    <div style="font-size: 16px; font-weight: bold; color: #000; margin-bottom: 20px;">理想的 Value Loss 趋势示意</div>
    <svg viewBox="0 0 600 300" width="100%" height="300" style="overflow: visible;">
        <!-- 坐标轴 -->
        <line x1="40" y1="20" x2="40" y2="240" stroke="#888" stroke-width="1.5" />
        <line x1="40" y1="240" x2="560" y2="240" stroke="#888" stroke-width="1.5" />
        
        <!-- 文本 -->
        <text x="40" y="35" fill="#333" font-size="14" text-anchor="middle">高</text>
        <text x="40" y="235" fill="#333" font-size="14" text-anchor="middle">低</text>
        <text x="40" y="265" fill="#333" font-size="14" text-anchor="middle">Iteration</text>
        <text x="430" y="110" fill="#4285f4" font-size="14">稳定区域</text>
        
        <!-- 折线 -->
        <polyline points="40,60 80,190 110,110 140,210 180,130 220,170 260,125 320,160 400,140 480,145 550,140" fill="none" stroke="#4285f4" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" />
    </svg>
    <div style="font-size: 14px; color: #333; margin-top: 15px;">
        <span style="color: #137333;">理想现象：</span>前期下降，后期保持在相对稳定的范围。
    </div>
</div>

但是有一点必须记住：

**Value Loss 不要求等于 0。**

因为 Critic 面对的是不断变化的数据分布，而且 GAE 的 Return 本身也是估计值。

如果 Value Loss：

- 一直上升：可能意味着 Critic 越来越难预测。
    
- 大幅震荡：可能存在训练不稳定问题。
    
- 逐渐下降后稳定：通常是积极信号。
    
- 很早就接近零：也不一定表示整个策略学得很好。
    

还要注意它受到 Reward 数值尺度的影响，不能脱离其他实验直接比较绝对值。

### **2. Loss/surrogate**

Actor 的策略优化情况

对应 Day 9 学过的 PPO Clipped Surrogate Loss：

```python
ratio = torch.exp(
    actions_log_prob_batch
    - old_actions_log_prob_batch.squeeze()
)

surrogate = -advantages_batch.squeeze() * ratio

surrogate_clipped = (
    -advantages_batch.squeeze()
    * torch.clamp(
        ratio,
        1.0 - self.clip_param,
        1.0 + self.clip_param
    )
)

surrogate_loss = torch.max(
    surrogate,
    surrogate_clipped
).mean()
```

数学上：

$$ L_{\text{policy}} = -\mathbb E[ \min(r_tA_t, \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t) ] $$

这里要注意：

PPO 使用的是最小化 Loss 的形式，所以前面存在负号。

**这条曲线没有一个通用的目标值。**

例如：

- 不能认为 Loss 越接近 0 就一定越好。
    
- 不能认为 Loss 越低，机器人就一定走得越好。
    
- 也不能要求它一直单调下降。
    

通常更值得关注的是：

它是否长期保持有限、平稳，没有突然爆炸，并且与 Reward 的改善趋势相互吻合。

### **3. Loss/learning_rate**

这个名称容易误解。

它虽然在 `Loss/` 分类里面，但它实际上不是 Loss，而是当前 PPO 的学习率。

G1 继承的基础配置：

```python
learning_rate = 1.e-3
schedule = 'adaptive'
desired_kl = 0.01
```

在 PPO Update 中，代码根据 KL Divergence 调节学习率：

```python
if kl_mean > self.desired_kl * 2.0:

    self.learning_rate = max(
        1e-5,
        self.learning_rate / 1.5
    )

elif kl_mean < self.desired_kl / 2.0:

    self.learning_rate = min(
        1e-2,
        self.learning_rate * 1.5
    )
```

具体来说：

|KL 情况|学习率变化|
|---|---|
|KL 大于 0.02|学习率除以 1.5|
|KL 小于 0.005 且大于 0|学习率乘以 1.5|
|介于两者之间|不变|

因此可能出现阶梯变化的曲线

这仅表示可能的趋势，实际也可能升高、降低或长期保持不变。

这里有一个值得注意的版本细节：

在 RSL-RL v1.0.2 中，KL 被用于内部调整学习率，但默认 `OnPolicyRunner.log()` 并没有直接写出 KL 曲线。

所以不要把 `Loss/learning_rate` 当成 KL 曲线。

### **4. 一个特别重要的版本说明：为什么找不到 Entropy 和 KL？**

你之前学习 PPO 时，接触过：

- Policy Entropy；
    
- KL Divergence；
    
- Clip Fraction。
    

这些指标确实可以用于 PPO 训练监控。

但是，我核对了 RSL-RL v1.0.2 的 `OnPolicyRunner.log()`，它实际写入的 Loss 指标是：

```
Loss/value_functionLoss/surrogateLoss/learning_rate
```

另外写入：

```
Policy/mean_noise_std
```

虽然 `ppo.py` 内部计算了 Entropy 和 KL，但默认 Runner 的这个日志函数没有把它们写成 TensorBoard Scalar。

所以：

**如果你没有在 TensorBoard 看到 Loss/entropy 或 Policy/KL，不代表训练没有计算它们。**

如果你的 TensorBoard 中确实出现这些 Tag，则需要检查是不是其他 RSL-RL 版本，或者你本地增加了日志代码。

---
## **六、第四类：Policy —— 策略还在探索吗？**

### **Policy/mean_noise_std**

非常值得关注的指标

来源：

```python
mean_std = self.alg.actor_critic.std.mean()

self.writer.add_scalar(
    'Policy/mean_noise_std',
    mean_std.item(),
    locs['it']
)
```

这就是当前策略高斯分布的平均标准差。

回忆 Day 9：

$$ a_t\sim\mathcal N(\mu_\theta(o_t),\sigma_\theta^2) $$

其中：

- $\mu_\theta$：Actor 输出的动作均值。
    
- $\sigma_\theta$：动作探索标准差。
    

当前 G1 初始配置：

```
init_noise_std = 0.8
```

因此训练初期的标准差从 0.8 开始。

这个指标代表策略的探索幅度。

例如：

$$ \sigma=0.8 $$

意味着策略采样时的随机波动较大。

而：

$$ \sigma=0.1 $$

意味着策略采样结果更加集中在均值附近。

**Noise Std 的直观理解**

<div style="font-family: -apple-system, sans-serif; padding: 20px; border: 1px solid #ebebeb; border-radius: 10px; background-color: #fff; max-width: 650px; margin: 10px 0;">
    <div style="font-size: 16px; font-weight: bold; color: #000; margin-bottom: 15px;">Noise Std 的直观理解</div>
    <div style="display: flex; gap: 15px;">
        
        <!-- 左侧卡片 -->
        <div style="flex: 1; background-color: #f3f6fc; border-radius: 8px; padding: 20px;">
            <div style="font-size: 16px; font-weight: bold; color: #333;">较大标准差</div>
            <div style="font-size: 24px; font-weight: bold; color: #4285f4; margin: 5px 0;">0.8</div>
            <div style="font-size: 13px; color: #666; margin-bottom: 15px;">探索范围较宽</div>
            <svg viewBox="0 0 200 100" width="100%" height="100" style="overflow: visible;">
                <defs>
                    <linearGradient id="gradBlueFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#4285f4" stop-opacity="0.3"/>
                        <stop offset="100%" stop-color="#4285f4" stop-opacity="0.05"/>
                    </linearGradient>
                </defs>
                <path d="M 10 90 Q 90 90 100 40 Q 110 90 190 90 Z" fill="url(#gradBlueFill)" />
                <path d="M 10 90 Q 90 90 100 40 Q 110 90 190 90" fill="none" stroke="#4285f4" stroke-width="3" stroke-linecap="round" />
            </svg>
        </div>

        <!-- 右侧卡片 -->
        <div style="flex: 1; background-color: #eff8f1; border-radius: 8px; padding: 20px;">
            <div style="font-size: 16px; font-weight: bold; color: #333;">较小标准差</div>
            <div style="font-size: 24px; font-weight: bold; color: #137333; margin: 5px 0;">0.1</div>
            <div style="font-size: 13px; color: #666; margin-bottom: 15px;">采样更集中</div>
            <svg viewBox="0 0 200 100" width="100%" height="100" style="overflow: visible;">
                <defs>
                    <linearGradient id="gradGreenFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#137333" stop-opacity="0.3"/>
                        <stop offset="100%" stop-color="#137333" stop-opacity="0.05"/>
                    </linearGradient>
                </defs>
                <path d="M 10 90 Q 95 90 100 20 Q 105 90 190 90 Z" fill="url(#gradGreenFill)" />
                <path d="M 10 90 Q 95 90 100 20 Q 105 90 190 90" fill="none" stroke="#137333" stroke-width="3" stroke-linecap="round" />
            </svg>
        </div>

    </div>
</div>

**什么样算正常？**

通常希望：

- 前期保留一定的探索空间。
    
- 中后期根据学习情况逐步调整。
    
- 后期不要出现突然爆炸或过早坍缩。
    

但它不要求一直下降，也没有一个 G1 通用的标准值。

此外，它是 Policy Action 空间中的标准差，不是关节角度标准差。不要直接乘上或等同于 0.25 的 Action Scale 来解释 TensorBoard 原始数值。

## **七、第五类：Perf —— 训练速度是否正常？**

这一类不决定策略有没有收敛，但对你实际跑实验特别重要。

| 指标                     | 含义                     |
| ---------------------- | ---------------------- |
| `Perf/total_fps`       | 每秒处理的并行环境 Transition 数 |
| `Perf/collection time` | 一次 Rollout 采集耗时        |
| `Perf/learning_time`   | GAE + PPO Update 的耗时   |

代码：

```python
fps = int(
    self.num_steps_per_env * self.env.num_envs
    / (
        locs['collection_time']
        + locs['learn_time']
    )
)
```

例如你使用：

```python
num_envs = 128
num_steps_per_env = 24
```

那么一次 Iteration 收集：

$$ 128\times24=3072 $$

条 Transition。

假设：

- Collection Time = 2 秒；
    
- Learning Time = 1 秒。
    

那么：

$$ FPS=\frac{3072}{2+1}=1024 $$

这意味着每秒处理约 1024 条并行环境 Transition。

注意：

**这里的 FPS 不是 G1 机器人的物理仿真频率 200Hz。**

它是跨所有并行环境统计的吞吐量。

另外，`collection time` 与 `learning time` 如果出现异常增长，需要检查 GPU 显存、仿真环境数量、数据传输和模型计算开销。

FPS 稳定就说明计算效率比较稳定，但不代表策略已经学会走路。

## **八、最关键的问题：究竟什么样的曲线才算收敛？**

现在把前面的指标放在一起。

我建议你不要只用一个指标判断，而是分成三个层次。

### **1. 第一层：机器人是否真正学会任务？**

这是最重要的。

**核心表现指标**

**Train/mean****_****reward**

逐渐提高并稳定

**Train/mean****_****episode****_****length**

逐渐接近合理上限

**Episode/rew****_****tracking****_****lin****_****vel**

较高且稳定

**Episode/rew****_****tracking****_****ang****_****vel**

较高且稳定

**Episode/rew****_****orientation**

惩罚贡献趋近 0

**Episode/rew****_****base****_****height**

惩罚贡献趋近 0

对于 G1 行走任务，我最看重：

- 机器人能不能持续站立和行走。
    
- 是否能准确跟踪 Command。
    
- 是否能转弯和进行侧向运动。
    
- 是否能够承受随机 Push。
    
- 是否在不同摩擦、质量条件下保持稳定。
    

这些才是任务层面的成功。

### **2. 第二层：神经网络是否已经稳定学习？**

|指标|理想现象|
|---|---|
|Value Function Loss|下降后趋于稳定|
|Surrogate Loss|有限、稳定波动|
|Learning Rate|不再剧烈变化，或保持合理自适应|
|Mean Noise Std|探索程度逐渐调整，不突然坍缩|

这一层主要判断优化过程是否稳定。

但是要注意：

**网络训练稳定，不等于机器人一定完成任务。**

例如一个策略可能学会了原地站立，Value Loss 很低，Surrogate Loss 也很稳定，但它根本没有学会按照速度命令行走。

### **3. 第三层：真实行为是否符合预期？**

这一层需要脱离 TensorBoard，运行：

```
python legged_gym/scripts/play.py --task=g1
```

然后检查不同运动指令：

| 测试场景    | 验收目标     |
| ------- | -------- |
| 向前行走    | 能够跟踪前进速度 |
| 向后行走    | 能够稳定后退   |
| 侧向行走    | 不发生明显失衡  |
| 原地转向    | 能够跟踪偏航指令 |
| 不同初始条件  | 不会频繁摔倒   |
| 外部 Push | 能够恢复平衡   |

这一步非常重要。

因为 TensorBoard 只告诉你训练过程中统计指标的变化，不会直接告诉你机器人在实际执行时是不是走得自然。

## **九、我建议你以后用这套标准判断是否收敛**

## **G1 PPO 收敛验收表**

**验收进度**

Mean Reward 已经明显提高，并在后期趋于稳定

Mean Episode Length 较高，且能够确认大多数是正常 Timeout 而不是异常结束

Linear Velocity Tracking Reward 较高且稳定

Angular Velocity Tracking Reward 较高且稳定

Orientation、Base Height 等稳定性惩罚没有明显恶化

Value Function Loss 没有持续发散

Surrogate Loss 保持有限，没有明显训练崩溃

Play 中能够跟踪不同 Command

在随机化和扰动条件下仍然能够稳定运动

这是一套项目验收清单，不是 PPO 数学收敛的充分证明。可以在完成实验后逐项检查。

这里有一个非常重要的判断：

**Reward 长期不再提高，只能说明训练进入平台期，不一定说明学到了最优策略。**

例如：

- Mean Reward 从 10 提高到 100 后稳定：可能是良好的收敛。
    
- Mean Reward 从 10 提高到 20 后稳定：也可能只是陷入一个较差的策略。
    
- Mean Reward 提高，但机器人仍频繁摔倒：可能奖励设计或其他指标存在问题。
    

所以我们追求的是高质量的平台期，而不只是曲线变平。

## **十、你实际打开 TensorBoard 时，应该怎么读？**

我建议按照下面的顺序，不要一上来就看所有曲线。

|顺序|打开的 Tag|你需要回答的问题|
|---|---|---|
|1|`Train/mean_reward`|整体表现有没有提高？|
|2|`Train/mean_episode_length`|机器人是否能活得更久？|
|3|`Episode/rew_tracking_lin_vel`|是否学会跟踪前进速度？|
|4|`Episode/rew_tracking_ang_vel`|是否学会转向？|
|5|`Episode/rew_orientation`|是否保持身体稳定？|
|6|`Episode/rew_action_rate`|是否存在剧烈动作变化？|
|7|`Loss/value_function`|Critic 是否能稳定预测价值？|
|8|`Loss/surrogate`|PPO 更新是否稳定？|
|9|`Policy/mean_noise_std`|探索程度如何变化？|
|10|`Perf/total_fps`|训练速度是否正常？|

TensorBoard 左侧的 Smoothing 滑块建议同时看两种状态：

- 原始曲线：判断真实波动和异常。
    
- 平滑曲线：判断长期趋势。
    

不要只看平滑曲线，否则可能把训练震荡掩盖掉。

另外，当前版本的 `Train/mean_episode_length` 统计的是最近最多 100 个已完成 Episode 的平均步数，而不是所有正在运行的 Episode 长度。`Episode/rew_*` 则由环境 Reset 时产生的 `infos["episode"]` 提供。

如果想准确区分摔倒和 Timeout，建议以后增加自定义的：

```
Train/fall_rate
Train/timeout_rate
```

因为官方 v1.0.2 默认日志并没有直接把这两个比例写入 TensorBoard。

## **十一、最后给你一个很重要的代码阅读任务**

现在你已经知道这些 Tag 分别表示什么，我希望你下一步能够自己追踪它们的来源。

打开：

1. `legged_gym/envs/base/legged_robot.py`
    
2. `rsl_rl/algorithms/ppo.py`
    
3. `rsl_rl/runners/on_policy_runner.py`
    

尝试自己建立这张表：

| TensorBoard Tag              | 从哪里产生                    | 经过什么处理                        |
| ---------------------------- | ------------------------ | ----------------------------- |
| Train/mean_reward            | `rewbuffer`              | 最近最多 100 个 Episode 求均值        |
| Train/mean_episode_length    | `lenbuffer`              | 最近最多 100 个 Episode 求均值        |
| Episode/rew_tracking_lin_vel | `episode_sums`           | 累计、按最大 Episode 时长归一化          |
| Loss/value_function          | `PPO.update()`           | Mini-batch Value Loss 求均值     |
| Loss/surrogate               | `PPO.update()`           | Mini-batch Surrogate Loss 求均值 |
| Loss/learning_rate           | `PPO.learning_rate`      | 记录当前学习率                       |
| Policy/mean_noise_std        | `actor_critic.std`       | 对动作维度求均值                      |
| Perf/total_fps               | Collection、Learning Time | 计算吞吐量                         |

这张表是今天最值得留下的笔记。

### **十二、你下一步做什么？**

你已经学习过 Day 1–10，现在开始真正进入实验阶段。

我建议你下一次把实际训练的 TensorBoard 页面截图发给我，最好包含以下四张图：

- `Train/mean_reward`
    
- `Train/mean_episode_length`
    
- `Loss/value_function` 和 `Loss/surrogate`
    
- `Episode/rew_tracking_lin_vel` 与 `Episode/rew_action_rate`
    

根据你的实际曲线判断：

1. 目前处于学习初期、中期还是平台期。
    
2. Critic 是否正常学习。
    
3. Actor 是否还在有效探索。
    
4. 是否存在 Reward 上升但行走能力没有同步改善的问题。
    
5. 是否已经适合进入 Day 12 的 Action Rate 参数实验。
    

**我希望你以后不只是会看 TensorBoard，而是看到一条异常曲线，就能回到对应 Python 函数，解释它为什么会这样变化。**

**/root/unitree_rl_gym/logs/g1/Nov20_17-12-34_day11_baseline

![[Pasted image 20261121185941.png|347]]



![[Pasted image 20261121185958.png|350]]



![[Pasted image 20261121185951.png|346]]



![[Pasted image 20261121190007.png|342]]






































# 今日问题

1. 为什么 Baseline 实验不应该修改任何 Reward Scale？
    
2. 为什么 Day 11 和 Day 12 必须使用相同的 `num_envs` 和 `max_iterations`？
    
3. 为什么 `Train/mean_reward` 上升，不一定意味着机器人走得更加稳定？
    
4. `Loss/value_function` 和 `Train/mean_episode_length` 分别反映什么？
    
5. 为什么我们需要记录 Git Commit，而不能只记录训练参数？
    

## **七、今天的实践练习**

建立一个实验记录文件：

`experiments/day11_baseline.md`

写入以下内容：

```
# Day 11 Baseline

## 1. Code
- Repository commit:
- RSL-RL version:

## 2. Training Configuration
- Task: G1
- Num Envs: 128
- Max Iterations: 100
- Seed:
- Action Scale: 0.25
- Action Rate: -0.01
- Decimation: 4

## 3. Metrics
- Mean Reward:
- Mean Episode Length:
- Tracking Linear Velocity:
- Value Function Loss:
- Surrogate Loss:
- FPS:

## 4. Observation
- Training behavior:
- Problems:
- Checkpoint path:
```

实际结果由你运行后填写，不要预先编造。

## **八、今天的完成标准**

- 能独立运行 G1 的短训练。
    
- 能找到新生成的 Log 目录。
    
- 能找到 `model_100.pt`。
    
- 能打开 TensorBoard。
    
- 能记录 Baseline 的关键指标。
    
- 没有修改原始 Reward 和控制参数。
    

**Day 11 的核心产出：一份真正由你运行得到的 Baseline 实验记录。**