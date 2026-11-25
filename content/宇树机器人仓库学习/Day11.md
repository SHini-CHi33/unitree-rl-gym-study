# 今日阅读任务

今日主题：Actor-Critic + LSTM——G1 为什么需要记忆？记忆如何进入 PPO？

昨天 ，我们已经研究了 PPO 的核心更新过程：

- Rollout Storage 提供训练数据。
    
- Recurrent Mini-batch Generator 生成训练批次。
    
- Actor 计算 Policy Loss。
    
- Critic 计算 Value Loss。
    
- 最终通过 `backward()` 和 `optimizer.step()` 更新参数。
    

但是昨天留下了一个重要问题：

**G1 为什么不是普通的 Actor-Critic，而是使用 Actor Critic Recurrent？**

今天我们专门研究 LSTM，不再重复昨天的 PPO Loss 推导。

今天结束后，你应该能够解释：一个 G1 环境在连续运行时，LSTM 如何保存历史信息；Episode 结束后如何清空记忆；PPO 更新时又如何把历史记忆重新交给 LSTM。

## **一、先核对代码版本**

今天仍然使用：

- 官方仓库：[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym?utm_source=chatgpt.com)
    
- 强化学习库：`rsl_rl v1.0.2`
    

官方安装文档明确要求切换到 `v1.0.2`。

今天主要研究这五个文件：

|文件|核心函数|
|---|---|
|`legged_gym/envs/g1/g1_config.py`|`G1RoughCfgPPO.policy`|
|`rsl_rl/runners/on_policy_runner.py`|`__init__()`|
|`rsl_rl/modules/actor_critic_recurrent.py`|`ActorCriticRecurrent`、`Memory`|
|`rsl_rl/algorithms/ppo.py`|`act()`、`process_env_step()`|
|`rsl_rl/storage/rollout_storage.py`|`_save_hidden_states()`、`reccurent_mini_batch_generator()`|

今天的核心文件是：

[actor_critic_recurrent.py（v1.0.2）](https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/modules/actor_critic_recurrent.py?utm_source=chatgpt.com)

## **二、第一部分：G1 为什么使用 LSTM？**

打开：

`legged_gym/envs/g1/g1_config.py`

找到：

```python
class G1RoughCfgPPO(LeggedRobotCfgPPO):

    class policy:
        init_noise_std = 0.8

        actor_hidden_dims = [32]
        critic_hidden_dims = [32]

        activation = 'elu'

        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
```

这里有两个部分：

**普通 Actor-Critic 网络：**

```
Actor:
Input → MLP → Action Mean

Critic:
Input → MLP → Value
```

**G1 实际使用的 Recurrent Actor-Critic：**

```
Actor Observation
       ↓
      LSTM
       ↓
      MLP
       ↓
      Action

Critic Observation
       ↓
      LSTM
       ↓
      MLP
       ↓
      Value
```

为什么需要 LSTM？

普通 MLP 的输出只取决于当前输入：

$$ a_t=\pi_\theta(o_t) $$

而 LSTM 可以利用历史观测形成内部记忆：

$$ (h_t,c_t)= \operatorname{LSTM}(o_t,h_{t-1},c_{t-1}) $$

再根据当前输出的隐藏表示产生 Action：

$$ a_t=\pi_\theta(h_t) $$

这里：

- $h_t$：Hidden State。
    
- $c_t$：Cell State。
    
- $o_t$：当前 Observation。
    

例如，机器人当前的关节角度和角速度可能相似，但如果一个机器人正在向前倾倒，另一个机器人正在从倾倒中恢复，仅看当前 Observation 未必能充分区分两者的运动历史。

LSTM 可以通过历史信息帮助策略判断当前处于什么运动状态。

注意：这是一种时序信息建模能力，并不意味着 LSTM 一定能准确恢复所有不可观测的物理状态。

## **三、第二部分：G1 实际上有两个 LSTM**

打开：

`rsl_rl/modules/actor_critic_recurrent.py`

找到：

```python
class ActorCriticRecurrent(ActorCritic):

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        ...
        rnn_type='lstm',
        rnn_hidden_size=256,
        rnn_num_layers=1,
        ...
    ):

        super().__init__(
            num_actor_obs=rnn_hidden_size,
            num_critic_obs=rnn_hidden_size,
            ...
        )

        self.memory_a = Memory(
            num_actor_obs,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size
        )

        self.memory_c = Memory(
            num_critic_obs,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size
        )
```

这里非常重要：

```python
self.memory_a
self.memory_c
```

分别是 Actor 和 Critic 的 LSTM。

为什么不是共用一个 LSTM？

因为 G1 的 Actor 和 Critic 输入并不完全相同。

回忆 Day 3：

|网络|输入|维度|
|---|---|---|
|Actor|普通 Observation|47|
|Critic|Privileged Observation|50|

所以：

$$ o_t^a\in\mathbb R^{47} $$

$$ o_t^c\in\mathbb R^{50} $$

它们需要各自处理自己的输入。

而且，Actor 和 Critic 的学习目标也不同：

- Actor 学习如何选择动作。
    
- Critic 学习如何估计状态价值。
    

因此两个 LSTM 分别拥有独立的参数和隐藏状态。

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; border: 1px solid #eaeaea; border-radius: 12px; background-color: #ffffff; max-width: 750px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); margin: 10px 0;">
    
    <!-- 标题 -->
    <div style="font-size: 18px; font-weight: bold; color: #111; margin-bottom: 20px;">
        G1 的实际网络结构
    </div>

    <!-- 主体布局 -->
    <div style="display: flex; gap: 20px;">
        
        <!-- 左侧：Actor -->
        <div style="flex: 1; background-color: #f4f7fd; border-radius: 8px; padding: 20px;">
            <div style="font-size: 18px; font-weight: bold; color: #111; margin-bottom: 15px;">Actor</div>
            
            <!-- Box 1: Observation -->
            <div style="background-color: transparent; border: 1px solid #d2e3fc; border-radius: 6px; padding: 16px 10px; text-align: center; margin-bottom: 12px;">
                <div style="font-size: 22px; font-weight: bold; color: #1a73e8; margin-bottom: 4px;">47</div>
                <div style="font-size: 14px; color: #5f6368;">Observation</div>
            </div>
            
            <!-- Arrow -->
            <div style="text-align: left; padding-left: 15px; color: #5f6368; font-size: 16px; margin-bottom: 12px;">↓</div>
            
            <!-- Box 2: LSTM -->
            <div style="background-color: transparent; border: 1px solid #d2e3fc; border-radius: 6px; padding: 16px 10px; text-align: center; margin-bottom: 12px;">
                <div style="font-size: 18px; font-weight: bold; color: #111; margin-bottom: 4px;">LSTM</div>
                <div style="font-size: 14px; color: #5f6368;">64 hidden units</div>
            </div>
            
            <!-- Arrow -->
            <div style="text-align: left; padding-left: 15px; color: #5f6368; font-size: 16px; margin-bottom: 12px;">↓</div>
            
            <!-- Box 3: MLP -->
            <div style="background-color: transparent; border: 1px solid #d2e3fc; border-radius: 6px; padding: 16px 10px; text-align: center; margin-bottom: 12px;">
                <div style="font-size: 18px; font-weight: bold; color: #111; margin-bottom: 4px;">MLP</div>
                <div style="font-size: 14px; color: #5f6368;">64 → 32 → 12</div>
            </div>
            
            <!-- Arrow -->
            <div style="text-align: left; padding-left: 15px; color: #5f6368; font-size: 16px; margin-bottom: 12px;">↓</div>
            
            <!-- Box 4: Action -->
            <div style="background-color: #d2e3fc; border-radius: 6px; padding: 16px 10px; text-align: center;">
                <div style="font-size: 18px; font-weight: bold; color: #111;">12 Actions</div>
            </div>
        </div>

        <!-- 右侧：Critic -->
        <div style="flex: 1; background-color: #eff8f1; border-radius: 8px; padding: 20px;">
            <div style="font-size: 18px; font-weight: bold; color: #111; margin-bottom: 15px;">Critic</div>
            
            <!-- Box 1: Privileged Obs -->
            <div style="background-color: transparent; border: 1px solid #ceead6; border-radius: 6px; padding: 16px 10px; text-align: center; margin-bottom: 12px;">
                <div style="font-size: 22px; font-weight: bold; color: #0f9d58; margin-bottom: 4px;">50</div>
                <div style="font-size: 14px; color: #5f6368;">Privileged Obs</div>
            </div>
            
            <!-- Arrow -->
            <div style="text-align: left; padding-left: 15px; color: #5f6368; font-size: 16px; margin-bottom: 12px;">↓</div>
            
            <!-- Box 2: LSTM -->
            <div style="background-color: transparent; border: 1px solid #ceead6; border-radius: 6px; padding: 16px 10px; text-align: center; margin-bottom: 12px;">
                <div style="font-size: 18px; font-weight: bold; color: #111; margin-bottom: 4px;">LSTM</div>
                <div style="font-size: 14px; color: #5f6368;">64 hidden units</div>
            </div>
            
            <!-- Arrow -->
            <div style="text-align: left; padding-left: 15px; color: #5f6368; font-size: 16px; margin-bottom: 12px;">↓</div>
            
            <!-- Box 3: MLP -->
            <div style="background-color: transparent; border: 1px solid #ceead6; border-radius: 6px; padding: 16px 10px; text-align: center; margin-bottom: 12px;">
                <div style="font-size: 18px; font-weight: bold; color: #111; margin-bottom: 4px;">MLP</div>
                <div style="font-size: 14px; color: #5f6368;">64 → 32 → 1</div>
            </div>
            
            <!-- Arrow -->
            <div style="text-align: left; padding-left: 15px; color: #5f6368; font-size: 16px; margin-bottom: 12px;">↓</div>
            
            <!-- Box 4: Value -->
            <div style="background-color: #ceead6; border-radius: 6px; padding: 16px 10px; text-align: center;">
                <div style="font-size: 18px; font-weight: bold; color: #111;">Value</div>
            </div>
        </div>

    </div>

    <!-- 底部注释 -->
    <div style="font-size: 13px; color: #666; margin-top: 20px; line-height: 1.5;">
        这里的 64 是 LSTM 输出维度。由于 ActorCriticRecurrent 调用父类时把 MLP 输入维度设置成 <code style="background-color:#f0f0f0; padding:2px 4px; border-radius:4px; font-family:monospace;">rnn_hidden_size</code>，因此 MLP 实际接收 64 维，而不是原始的 47 或 50 维。
    </div>
</div>

这也是今天需要特别理解的一点：

**LSTM 并不是取代 Actor 和 Critic 的 MLP，而是在 MLP 前面增加一个时序特征提取模块。**

## **四、第三部分：LSTM 的 Memory.forward() 到底做了什么？**

继续在：

`rsl_rl/modules/actor_critic_recurrent.py`

找到：

```python
class Memory(torch.nn.Module):

    def __init__(
        self,
        input_size,
        type='lstm',
        num_layers=1,
        hidden_size=256
    ):

        super().__init__()

        rnn_cls = (
            nn.GRU if type.lower() == 'gru'
            else nn.LSTM
        )

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        self.hidden_states = None
```

注意：

```python
self.hidden_states = None
```

一开始并没有历史记忆。

第一次运行 LSTM 时，PyTorch 会使用默认的零初始 Hidden State 和 Cell State。

##### 🧠 为什么隐藏状态既有 `h` 又有 `c`？

你之前学过普通的 RNN，它只有一个 `h`（Hidden State，隐藏状态）。  

但 **LSTM（长短期记忆网络）** 内部极其复杂，它有两个传送带：

1. **`h` (Hidden State，短时记忆)**：
    
    - 相当于你“当前脑子里在想的事情”**。它直接决定了你这一刻的输出（Action）。
        
    - 它会随着时间步快速变化。比如你现在在看地面，下一秒在看前方。
        
2. **`c` (Cell State，长时记忆)**：
    
    - 相当于你脑子里的“长期经验笔记”**。它像一条传送带，贯穿整个时间序列。
        
    - LSTM 内部有几个“门（Gate）”专门负责往这个 `c` 里加新信息，或者删旧信息。
        
    - 它变化得很慢。比如“我今天刚摔了一跤，我得小心点”，这个记忆会一直存在 `c` 里很久。
        

**在 PyTorch 中，`self.hidden_states` 其实是一个元组 `(h, c)`。** 每次传给 LSTM，它都会把这两个记忆一起送进去，算出新的一步 `(h, c)` 吐出来。


---

有了以上观点，我们再来看forward函数：

```python
def forward(self, input, masks=None, hidden_states=None):

    # 判断当前是在“采数据（Rollout）”还是在“训练更新（Update）”。
    # 如果 masks 不为空，说明是在 Update 阶段（因为采样时不需要传 masks）。
    batch_mode = masks is not None
    
    if batch_mode:
        # 【训练模式】
        # 如果你敢在训练时不把当时的记忆传进来，直接报错！
        if hidden_states is None:
            raise ValueError(...)
        # 把从仓库里取出来的 数据 和 历史记忆，扔进 LSTM 算。
        # 注意：这里不需要存新的 hidden_states，因为更新完网络，这份历史就作废了。
        out, _ = self.rnn(input, hidden_states)
        # 根据 masks 滤网，把那些狗死之后无用的 0 全部剪掉。
        out = unpad_trajectories(out, masks)
        
    else:
        # 【采样模式（Rollout）】
        # 这里的 input 形状是 [4096, 47]，加个 0 维变成 [1, 4096, 47]（时间长度为1）。
        # 把当前输入和历史记忆（self.hidden_states）扔进去。
        # 注意这里：直接把算出来的新记忆覆盖了 self.hidden_states，实现了流式记忆。
        out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
    return out
```

这里出现了一个重要分支：

```python
batch_mode = masks is not None
```

它区分了两种情况。

### **情况 A：Rollout 阶段，真实环境持续运行**

此时：

```python
masks = None
```

执行：

```python
out, self.hidden_states = self.rnn(    input.unsqueeze(0),    self.hidden_states)
```

假设有 $N$ 个并行环境：

原始 Actor Observation：

$$ [N,47] $$

增加时间维度：

$$ [1,N,47] $$

经过 LSTM：

$$ [1,N,64] $$

然后：

```
out.squeeze(0)
```

得到：

$$ [N,64] $$

这就是交给 Actor MLP 的输入。

与此同时，LSTM 保存新的：

$$ (h_t,c_t) $$

供下一个时刻使用。

### **情况 B：PPO Update 阶段，使用历史 Rollout 数据**

此时：

```python
masks != None
```

执行：

```python
out, _ = self.rnn(input, hidden_states)

out = unpad_trajectories(out, masks)
```

这里不再依赖之前实时运行时的内部记忆，而是显式传入：

```python
hidden_states
```

为什么？

因为训练时的数据已经从 Rollout Storage 中取出，原来的实时运行过程已经结束。

所以必须把当时保存的 Hidden State 重新提供给 LSTM，才能正确处理历史轨迹。

#### 🎭 `masks` 到底是个什么东西？

**`masks`（掩码）** 在深度学习里是一个极其常用的概念，它的本质是一个“布尔值（True/False）的滤网”。

为什么 LSTM 训练需要 `masks`？因为**各个环境的“死期”不一样**。  

假设你在 4096 个环境里采了 24 步的数据：

- 1 号狗命长，活满了 24 步。
    
- 2 号狗在第 10 步就摔死了。物理引擎把它重置了，第 11~24 步的数据都是“新的一局”或者全是 0（称为 Padding 填充）。
    

当你把这一批数据（形状 `[24, 4096]`）打乱后扔进 LSTM 训练时，**LSTM 会把这个 0 当成真实的物理状态去算！** 这就把网络教坏了。

**`masks` 的作用就是：**

- 第 1 步：True（有效数据）
    
- 第 10 步：True（2 号狗死前最后一步）
    
- 第 11 步：**False（垃圾数据，LSTM 你不要管它！）**
    

`unpad_trajectories` 这个函数，就是根据 `masks`，把那些为 False 的垃圾时间步全部剪掉（打平），只保留真实有效的记忆轨迹。

##### 💡 灵魂拷问：reset后不就直接接新一局的数据，怎么会等其它环境然后用零padding呢？


兄弟，你这个疑惑非常精准！你的记忆没有错：**在采样（Rollout）阶段，机器人死掉后确实是瞬间 Reset 接着跑的，根本不存在“等别人”的情况。**

那为什么还会出现 padding 和 mask 呢？

因为 **Rollout 阶段（跑数据）** 和 **Update 阶段（训练网络）** 是完全两码事。是 PPO 在训练时，为了喂给 GPU 一个规整的矩阵，**硬生生造出来的 Padding**。

我们用**“考试交卷”**来把这个坑填平：

##### 🏃 1. 采样阶段（Rollout）：绝不等别人

假设你有 4 个机器人，一起跑 24 步。

- **第 1 步**：4 个都在跑。
    
- **第 2 步**：机器人 B 摔死了。环境瞬间把它 Reset 回起点。
    
- **第 3 步**：机器人 A、C、D 还在继续原来的局；机器人 B 已经在新的一局里迈出第 1 步了。
    

这 24 步的数据收集完，存进仓库的矩阵形状是极其规整的：`[24, 4, 47]`（24步，4个环境，47维观测）。**此时完全没有 Padding，也没有 0。** 机器人 B 的轨迹在时间轴上是：死前 2 步 + 新一局 22 步。

##### 🚨 2. 训练阶段（Update）：LSTM 遇到了“记忆断崖”

现在要拿这批数据去更新网络了。如果是普通的 MLP，直接把矩阵拉平算就行。但**如果用了 LSTM，麻烦就来了**。

LSTM 是严格按照时间顺序去算的：第 1 步传给第 2 步，第 2 步传给第 3 步……  
当算到机器人 B 时：**第 2 步是它上一局的死前惨状，第 3 步是它新一局的起点。**  
如果直接连着喂给 LSTM，LSTM 会带着上一局“快摔死”的恐惧记忆，去控制新一局的身体，这不就精神分裂了吗？

**怎么做？必须强行把 B 的轨迹切断！**  
在底层框架（比如 `rsl_rl`）里，遇到 `done=True` 时，会把轨迹切分成两个独立的序列（Segment）：

- **序列 1**：B 的前 2 步。
    
- **序列 2**：B 的后 22 步。
    

##### 🛋️ 3. 为什么要 Padding？是为了“迁就 GPU”

现在情况变成了这样：

- 机器人 A 的交卷：1 个序列，长度 24。
    
- 机器人 B 的交卷：2 个序列，长度分别是 2 和 22。
    
- 机器人 C 的交卷：1 个序列，长度 5。
    
- 机器人 D 的交卷：1 个序列，长度 24。
    

GPU 算矩阵乘法，**要求输入的张量形状必须是严丝合缝的长方形（比如 `[Batch, 24, 47]`）**。它不能接受长度参差不齐的“锯齿状”数组。

为了把所有序列强行塞进一个长方形矩阵里：

- 所有序列都必须对齐到最大长度 **24**。
    
- 机器人 B 长度只有 2 的那个序列，后面必须补 22 个**全 0**。
    
- 机器人 C 长度 5 的序列，后面补 19 个**全 0**。
    
- **这些补齐的 0，就是 Padding。**
    

##### 🎭 4. Mask 闪亮登场

补齐了 0，形状是规整了，但这会带来灾难：LSTM 如果把这些补进去的 0 当成真实的物理状态去算，就会输出错误的梯度。

所以，我们必须配一个 **`masks`**（掩码）。它也是个矩阵，长这样：

- 真实数据的位置：`1` (True)
    
- Padding 补 0 的位置：`0` (False)
    

**`unpad_trajectories` 这个函数的作用，就是拿着这个 `mask` 滤网，把那些补 0 的垃圾时间步直接从计算图里剔除掉。** 告诉 LSTM：“遇到 mask 为 0 的地方，直接跳过，不要影响记忆和梯度。”


## **五、第四部分：Hidden State 是如何保存的？**

打开：

`rsl_rl/algorithms/ppo.py`

找到：

```python
def act(self, obs, critic_obs):

    if self.actor_critic.is_recurrent:
        self.transition.hidden_states = (
            self.actor_critic.get_hidden_states()
        )

    self.transition.actions = (
        self.actor_critic.act(obs).detach()
    )

    self.transition.values = (
        self.actor_critic.evaluate(critic_obs).detach()
    )

    ...
```

这里先执行：

```python
self.actor_critic.get_hidden_states()
```

再执行：

```python
self.actor_critic.act(obs)
```

这个顺序很重要。

它意味着：**Storage 保存的是处理当前 Observation 之前的 Hidden State，也就是当前这一步 LSTM 的初始记忆。**

然后找到 `ActorCriticRecurrent`：

```python
def get_hidden_states(self):

    return (
        self.memory_a.hidden_states,
        self.memory_c.hidden_states
    )
```

因此保存的是：

```
Actor Hidden State
Critic Hidden State
```

对于 LSTM，每个网络的 Hidden State 实际上又是一个二元组：

```
(h, c)
```

所以整体是：

```
(
    (actor_h, actor_c),
    (critic_h, critic_c)
)
```

### **LSTM 的维度**

G1 配置：

```python
rnn_hidden_size = 64
rnn_num_layers = 1
```

对于 \(N\) 个并行环境：

$$ h_a,c_a\in\mathbb R^{1\times N\times64} $$

$$ h_c,c_c\in\mathbb R^{1\times N\times64} $$

注意这里的 Hidden State 不是模型参数。

- LSTM 权重：模型参数，通过 PPO 的梯度更新。
    
- Hidden State / Cell State：运行过程中不断变化的时序状态。
    

两者千万不要混淆。

接着打开：

`rsl_rl/storage/rollout_storage.py`

找到：

```python
def _save_hidden_states(self, hidden_states):

    if hidden_states is None or hidden_states == (None, None):
        return

    hid_a = (
        hidden_states[0]
        if isinstance(hidden_states[0], tuple)
        else (hidden_states[0],)
    )

    hid_c = (
        hidden_states[1]
        if isinstance(hidden_states[1], tuple)
        else (hidden_states[1],)
    )
```

这里把 Actor 和 Critic 的 LSTM 状态拆开保存。

后面再根据 Rollout 的时间步，将它们分别写入：

```python
self.saved_hidden_states_a
self.saved_hidden_states_c
```


## **六、第五部分：Episode 结束后，为什么必须清空 Hidden State？**

这是今天最值得思考的问题。

打开：

`rsl_rl/algorithms/ppo.py`

找到：

```python
def process_env_step(self, rewards, dones, infos):

    ...

    self.storage.add_transitions(
        self.transition
    )

    self.transition.clear()

    self.actor_critic.reset(dones)
```

注意最后一句：

```
self.actor_critic.reset(dones)
```

为什么要执行？

假设某个环境：

```
Episode 1:

o1 → o2 → o3 → done

Episode 2:

o1' → o2' → o3'
```

如果 Episode 1 结束后不清空 LSTM 记忆：

```
Episode 1 的历史记忆
          ↓
Episode 2 的初始状态
```

这就产生了跨 Episode 的记忆污染。

Episode 2 的策略会受到 Episode 1 历史信息的影响。

因此需要清空。

找到：

`actor_critic_recurrent.py`

```python
def reset(self, dones=None):

    self.memory_a.reset(dones)

    self.memory_c.reset(dones)
```

然后找到：

```python
def reset(self, dones=None):

    for hidden_state in self.hidden_states:
        hidden_state[..., dones, :] = 0.0
```

这里有一个重要细节：

因为 LSTM 的：

```python
self.hidden_states
```

包含两个张量：

```
(h, c)
```

所以这段代码会同时清空：

- Hidden State
    
- Cell State
    

而且只清空 `dones=True` 的环境。

假设 4 个并行环境：

**某一时刻的 Episode 结束情况**

| **环境** | **Done** | **LSTM 记忆** |
| ------ | -------- | ----------- |
| Env 0  | True     | **清零**      |
| Env 1  | False    | **保留**      |
| Env 2  | True     | **清零**      |
| Env 3  | False    | **保留**      |

四个环境是并行运行的，一个环境结束不应该影响其他环境的记忆。

这也是为什么不能简单地在每次 `env.step()` 后把所有 LSTM Hidden State 清零。

## **七、第六部分：PPO Update 如何重新构造 LSTM 的历史？**

这是今天最后一个源码重点。

打开：

`rsl_rl/storage/rollout_storage.py`

找到：

```python
def reccurent_mini_batch_generator(
    self,
    num_mini_batches,
    num_epochs=8
):

    padded_obs_trajectories, trajectory_masks = (
        split_and_pad_trajectories(
            self.observations,
            self.dones
        )
    )
```

它首先调用：

```python
split_and_pad_trajectories()
```

打开：

`rsl_rl/utils/utils.py`

找到：

```python
def split_and_pad_trajectories(tensor, dones):
```

它的作用是：

1. 根据 Done 找到 Episode 的边界。
    
2. 将完整 Rollout 拆成多个独立轨迹。
    
3. 将不同长度的轨迹 Padding 到相同长度。
    
4. 创建 Mask，标记哪些位置是真实数据，哪些位置是 Padding。
    

官方代码中有一个示意例子：

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; border: 1px solid #eaeaea; border-radius: 12px; background-color: #ffffff; max-width: 650px; box-shadow: 0 2px 12px rgba(0,0,0,0.03); margin: 10px 0; color: #222;">

    <!-- Padding 前 -->
    <div style="font-size: 16px; font-weight: bold; margin-bottom: 16px;">Padding 前</div>
    
    <!-- Env A -->
    <div style="background-color: #f2f6fe; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; font-size: 15px; display: flex; align-items: center;">
        <span>Env A:&nbsp;</span>
        <span>A1 → A2 → A3 → A4</span>
    </div>
    
    <!-- Env B -->
    <div style="background-color: #f2f9f4; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; font-size: 15px; display: flex; align-items: center;">
        <span>Env B:&nbsp;</span>
        <span>B1 → B2</span>
    </div>
    
    <!-- 注释 -->
    <div style="font-size: 14px; color: #777; margin-bottom: 24px;">
        假设 A、B 两条轨迹都在此处结束。
    </div>

    <!-- 分割线 -->
    <div style="height: 1px; background-color: #f0f0f0; margin: 20px 0;"></div>

    <!-- Padding 后 -->
    <div style="font-size: 16px; font-weight: bold; margin-bottom: 12px;">Padding 后</div>
    
    <table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 15px; margin-bottom: 24px;">
        <thead>
            <tr style="border-bottom: 1px solid #f0f0f0;">
                <th style="padding: 12px 0; text-align: left; font-weight: bold; width: 20%;">轨迹</th>
                <th style="padding: 12px 0; font-weight: bold;">t1</th>
                <th style="padding: 12px 0; font-weight: bold;">t2</th>
                <th style="padding: 12px 0; font-weight: bold;">t3</th>
                <th style="padding: 12px 0; font-weight: bold;">t4</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #f9f9f9;">
                <td style="padding: 12px 0; text-align: left;">A</td>
                <td style="padding: 12px 0;">A1</td>
                <td style="padding: 12px 0;">A2</td>
                <td style="padding: 12px 0;">A3</td>
                <td style="padding: 12px 0;">A4</td>
            </tr>
            <tr style="border-bottom: 1px solid #f9f9f9;">
                <td style="padding: 12px 0; text-align: left;">B</td>
                <td style="padding: 12px 0;">B1</td>
                <td style="padding: 12px 0;">B2</td>
                <td style="padding: 12px 0;">0</td>
                <td style="padding: 12px 0;">0</td>
            </tr>
        </tbody>
    </table>

    <!-- Mask -->
    <div style="font-size: 16px; font-weight: bold; margin-bottom: 12px;">Mask</div>
    
    <table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 15px;">
        <thead>
            <tr style="border-bottom: 1px solid #f0f0f0;">
                <th style="padding: 12px 0; text-align: left; font-weight: bold; width: 20%;">轨迹</th>
                <th style="padding: 12px 0; font-weight: bold;">t1</th>
                <th style="padding: 12px 0; font-weight: bold;">t2</th>
                <th style="padding: 12px 0; font-weight: bold;">t3</th>
                <th style="padding: 12px 0; font-weight: bold;">t4</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #f9f9f9;">
                <td style="padding: 12px 0; text-align: left;">A</td>
                <td style="padding: 12px 0;">True</td>
                <td style="padding: 12px 0;">True</td>
                <td style="padding: 12px 0;">True</td>
                <td style="padding: 12px 0;">True</td>
            </tr>
            <tr style="border-bottom: 1px solid #f9f9f9;">
                <td style="padding: 12px 0; text-align: left;">B</td>
                <td style="padding: 12px 0;">True</td>
                <td style="padding: 12px 0;">True</td>
                <td style="padding: 12px 0;">False</td>
                <td style="padding: 12px 0;">False</td>
            </tr>
        </tbody>
    </table>

</div>

这样就可以让不同长度的 Episode 放在同一个张量里。

接下来：

```python
masks_batch = trajectory_masks[
    :, first_traj:last_traj
]
```

同时提取对应轨迹的初始 Hidden State：

```python
hid_a_batch = [
    saved_hidden_state.permute(
        2, 0, 1, 3
    )[last_was_done][first_traj:last_traj]
    .transpose(1, 0)
    .contiguous()

    for saved_hidden_state
    in self.saved_hidden_states_a
]
```

Critic 也有对应的处理。

最后 Generator 返回：

```python
yield (
    obs_batch,
    critic_obs_batch,
    actions_batch,
    values_batch,
    advantages_batch,
    returns_batch,
    old_actions_log_prob_batch,
    old_mu_batch,
    old_sigma_batch,
    (hid_a_batch, hid_c_batch),
    masks_batch
)
```

你不需要今天完全理解这些 `permute()`、`transpose()` 和索引操作。

只需要理解它们的目标：

**在 PPO Update 时，把每条历史轨迹对应的初始 Hidden State 重新找出来，和轨迹数据一起交给 LSTM。**

然后回到：

`rsl_rl/algorithms/ppo.py`

找到：

```python
self.actor_critic.act(
    obs_batch,
    masks=masks_batch,
    hidden_states=hid_states_batch[0]
)

value_batch = self.actor_critic.evaluate(
    critic_obs_batch,
    masks=masks_batch,
    hidden_states=hid_states_batch[1]
)
```

这就完成了历史记忆的重新输入。

## **八、今天的完整流程**

## **G1 LSTM 生命周期**

**Rollout：环境实时运行**

Actor Obs / Critic Obs

**Actor LSTM**

47 → 64

**Critic LSTM**

50 → 64

**更新各自的 Hidden State / Cell State**

**Episode Done？**

Done 环境的 h、c 清零，其他环境保留

**PPO Update**

Split → Pad → Mask → 恢复轨迹初始 Hidden State

**重新运行 LSTM**

使用保存的 h、c 处理完整轨迹，计算 Actor-Critic Loss

Backward → Optimizer.step()

# 今日问题

今天回答下面五道题。

### **问题 1：为什么 G1 使用两个 LSTM，而不是一个？**

请结合：

```python
self.memory_a
self.memory_c
```

以及：

```python
num_actor_obs
num_critic_obs
```

解释：

- Actor 和 Critic 的输入分别是多少维？
    
- 为什么需要两套独立的 Memory？
    
- 两套 LSTM 的参数是否共享？
    
- 它们最终输出的维度是多少？
    

### **问题 2：为什么 LSTM 有两个状态？**

请结合：

```python
self.hidden_states
```

解释：

1. LSTM 的 $h_t$ 和 $c_t$分别是什么？
    
2. 为什么不能只保存 $h_t$？
    
3. 对于 G1 的一层、64 维 LSTM，两个状态的形状分别是什么？
    
4. Hidden State 和网络权重有什么区别？
    

### **问题 3：为什么 Episode Done 时需要清空 Hidden State？**

请结合：

```python
self.actor_critic.reset(dones)
```

以及：

```python
hidden_state[..., dones, :] = 0.0
```

解释：

- 为什么不应该让新 Episode 继承旧 Episode 的记忆？
    
- 为什么只清空 Done 对应的环境？
    
- 为什么 Hidden State 和 Cell State 都需要清空？
    

### **问题 4：为什么 PPO Update 需要重新提供 Hidden State？**

请结合：

```python
split_and_pad_trajectories()
```

```python
hid_a_batch
hid_c_batch
```

```python
masks_batch
```

解释：

- 为什么不能把所有 Observation 随机打乱？
    
- 为什么要把不同长度的轨迹 Padding？
    
- Padding 位置为什么需要 Mask？
    
- 为什么重新计算 LSTM 输出时还需要原始的 Hidden State？
    

### **问题 5：区分 Rollout 和 PPO Update 中的两种 Memory 使用方式。**

结合：

```python
out, self.hidden_states = self.rnn(
    input.unsqueeze(0),
    self.hidden_states
)
```

和：

```python
out, _ = self.rnn(input, hidden_states)

out = unpad_trajectories(out, masks)
```

解释：

- 第一种方式为什么自己维护 Hidden State？
    
- 第二种方式为什么显式传入 Hidden State？
    
- 两者的输入数据组织有什么不同？
    
- 为什么不能把训练阶段直接当作实时环境推理？
    

# 今日动手实践：手动画出两个 Episode 的记忆变化

不需要运行 Isaac Gym，也不要修改任何代码。

假设有两个并行环境：

```
Env 0:
o0 → o1 → o2 → Done → o0' → o1'

Env 1:
p0 → p1 → p2 → p3 → p4
```

请画出下面三个时间点的 Hidden State：

|时间|Env 0|Env 1|
|---|---|---|
|t0|初始记忆|初始记忆|
|t1|更新后的记忆|更新后的记忆|
|t2|Episode 结束|更新后的记忆|
|t3|新 Episode 的零初始记忆|延续原来的记忆|

然后回答：

**为什么 Env 0 的记忆需要清零，而 Env 1 的记忆必须继续保留？**

最后再画出 PPO Update 阶段如何从 Rollout Storage 中重新提取这两条环境轨迹的 Hidden State。

## **今天的完成标准**

- 能够解释 G1 为什么使用 `ActorCriticRecurrent`。
    
- 能够画出 Actor LSTM 和 Critic LSTM 的独立结构。
    
- 能够区分 Hidden State、Cell State 和网络参数。
    
- 能够解释 `reset(dones)` 的作用。
    
- 能够追踪 `_save_hidden_states()` 的数据来源。
    
- 能够解释 `split_and_pad_trajectories()` 和 Mask。
    
- 能够解释 Rollout 和 PPO Update 两种 LSTM 调用方式。


