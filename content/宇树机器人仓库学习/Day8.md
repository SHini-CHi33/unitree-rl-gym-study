# 今日阅读任务

今日主题：Rollout、Transition、Rollout Storage 与 GAE
## **一、今天的核心问题**

前七天我们已经知道：

```
Observation
    ↓
Actor
    ↓
Action
    ↓
Environment.step()
    ↓
Reward、Done、Next Observation
```

但 PPO 不会在每执行一次 Action 后就立即更新网络。

它会先采集一批数据，然后再统一更新。

今天需要回答：

**一次完整的 Rollout 究竟采集了多少数据？这些数据保存在哪里？GAE 是怎么计算出来的？**

## **二、需要打开的文件**

今天主要阅读 RSL-RL 的三个文件：

|文件|具体位置|
|---|---|
|`rsl_rl/runners/on_policy_runner.py`|`__init__()`、`learn()`|
|`rsl_rl/algorithms/ppo.py`|`act()`、`process_env_step()`、`compute_returns()`|
|`rsl_rl/storage/rollout_storage.py`|`Transition`、`add_transitions()`、`compute_returns()`|

对应源码：

- [OnPolicyRunner](https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/runners/on_policy_runner.py?utm_source=chatgpt.com)
    
- [PPO](https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/algorithms/ppo.py?utm_source=chatgpt.com)
    
- [RolloutStorage](https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/storage/rollout_storage.py?utm_source=chatgpt.com)
    

另外回看：

`legged_gym/envs/g1/g1_config.py`

以及：

`legged_gym/envs/base/legged_robot_config.py`

特别注意：G1 配置继承了基础配置中的一些训练参数。

当前官方配置默认：

| 参数                   | 数值    | 来源        |
| -------------------- | ----- | --------- |
| `num_envs`           | 4096  | 基础环境配置    |
| `num_steps_per_env`  | 24    | 基础 PPO 配置 |
| `num_observations`   | 47    | G1 配置     |
| `num_privileged_obs` | 50    | G1 配置     |
| `num_actions`        | 12    | G1 配置     |
| `max_iterations`     | 10000 | G1 PPO 配置 |

## **三、第一部分：OnPolicyRunner.learn() 到底做什么？**

打开：

`rsl_rl/runners/on_policy_runner.py`

找到：

```python
def learn(self, num_learning_iterations, init_at_random_ep_len=False):
```

重点阅读：

```python
obs = self.env.get_observations()

privileged_obs = self.env.get_privileged_observations()

critic_obs = (
    privileged_obs
    if privileged_obs is not None
    else obs
)
```

这里`num_privileged_obs`为`critic` 50维的输入，`if privileged_obs is not None else obs`则是语法糖，如果没有获取到线速度也可以用47维obs正常输入

接着：

```python
for it in range(
    self.current_learning_iteration,
    tot_iter
):

    with torch.inference_mode():

        for i in range(self.num_steps_per_env):

            actions = self.alg.act(obs, critic_obs)

            obs, privileged_obs, rewards, dones, infos = (
                self.env.step(actions)
            )

            critic_obs = (
                privileged_obs
                if privileged_obs is not None
                else obs
            )

            self.alg.process_env_step(
                rewards, dones, infos
            )

    self.alg.compute_returns(critic_obs)

    mean_value_loss, mean_surrogate_loss = (
        self.alg.update()
    )
```

下面来讲解这段代码：
##### 🏭 第一战役：准备阶段


```python
for it in range(self.current_learning_iteration, tot_iter):
    with torch.inference_mode():
```

- **`for it in range(...)`**：这是最外层的 **Iteration（迭代轮次）**。每一轮迭代，都会收集一批数据，然后更新一次网络。
    
- **`with torch.inference_mode():`**：**极其重要！** 它的作用等同于 `torch.no_grad()`，但速度更快。它告诉 PyTorch：“接下来的所有运算（无论是环境计算还是网络推理），都**不需要**计算梯度，也不建计算图”。这极大地节省了显存，并且提升了采样速度。
    
##### 🏃‍♂️ 第二战役：疯狂采样（Rollout 阶段）

```python
        for i in range(self.num_steps_per_env):
            actions = self.alg.act(obs, critic_obs)
            obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
            critic_obs = privileged_obs if privileged_obs is not None else obs
            self.alg.process_env_step(rewards, dones, infos)
```

这个 `for i` 循环（比如跑 24 步），就是在给大仓库（Rollout Buffer）装满数据。

1. **`actions = self.alg.act(...)`**：  
    策略网络（Actor）根据当前观测 `obs`，输出动作。同时，如果 Actor 和 Critic 是一起跑的，这里也会算出 Critic 的估值。
    
2. **`env.step(actions)`**：  
    把动作发给物理引擎。物理引擎跑 4 次内部循环（decimation），然后返回：
    
    - `obs`：新的 47 维普通观测。
        
    - `privileged_obs`：新的 50 维特权观测。
        
    - `rewards`：计算好的奖励 $r_t$。
        
    - `dones`：这次死没死。
        
3. **`critic_obs = ...`**：  
    这是一个极其精彩的细节。如果环境返回了 `privileged_obs`（特权观测），那就把它喂给 Critic；如果没有（比如环境没开特权模式），那就让 Critic 和 Actor 看一样的 `obs`。
    
4. **`self.alg.process_env_step(...)`**：  
    把刚才环境吐出来的 `(rewards, dones, infos)`，以及刚才产生动作的 `(obs, actions)`，统统存进 `RolloutStorage`（大仓库）里。**此时，数据还不足以训练，继续循环去采。**
    
##### 🧠 第三战役：结算与更新（Update 阶段）

```python
    self.alg.compute_returns(critic_obs)
    mean_value_loss, mean_surrogate_loss = self.alg.update()
```

当 24 步采集完毕后，跳出内层循环，进入更新阶段：

1. **`self.alg.compute_returns(critic_obs)`**：  
    这就是我们之前聊的 **GAE 计算**。代码会把最后一步的 `critic_obs` 塞给 Critic，算出兜底价值。然后倒推算出整条轨迹的 Advantage（优势）和 Return（回报）。  
    _(注意：这一步必须放在 `inference_mode` 里，因为算 GAE 纯属数学推导，不需要更新网络参数)_。
    
2. **`self.alg.update()`**：  
    这是 **PPO 算法的主战场**！它会把刚才 `RolloutStorage` 里装满的 24 步 × 4096 环境的数据，打乱、切分 Minibatch，跑多个 Epoch。计算出 `Surrogate Loss` 和 `Value Loss`，执行 `backward()` 和 `optimizer.step()`。最后返回平均 Loss 给 TensorBoard 打印。

注意：

**一次 Learning Iteration 不等于一个 Episode。**

一个 Episode 可能跨越多个 Learning Iteration，也可能在一次 Rollout 中提前结束并 Reset。

## **四、第二部分：PPO.act() 到底记录了什么？**

打开：

`rsl_rl/algorithms/ppo.py`

找到：

```python
def act(self, obs, critic_obs):
```

关键代码：

```python
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

self.transition.actions_log_prob = (
    self.actor_critic
    .get_actions_log_prob(self.transition.actions)
    .detach()
)

self.transition.action_mean = (
    self.actor_critic.action_mean.detach()
)

self.transition.action_sigma = (
    self.actor_critic.action_std.detach()
)

self.transition.observations = obs

self.transition.critic_observations = critic_obs

return self.transition.actions
```

这里值得特别注意：在 `act()` 中，PPO 不仅仅计算 Action。

它还记录：

| 数据                    | 含义                       | 为什么需要                     |
| --------------------- | ------------------------ | ------------------------- |
| `observations`        | Actor 输入                 | 后面重新计算策略                  |
| `critic_observations` | Critic 输入                | 后面重新计算 Value              |
| `actions`             | 当时执行的动作                  | PPO 更新时使用                 |
| `values`              | 旧 Critic 的价值估计           | GAE 计算                    |
| `actions_log_prob`    | 旧策略动作概率的对数               | Importance Sampling Ratio |
| `action_mean`         | 旧策略高斯均值                  | KL 计算                     |
| `action_sigma`        | 旧策略高斯标准差                 | KL 计算                     |
| `hidden_states`       | Actor、Critic 的 LSTM 隐藏状态 | 后续重建序列计算                  |

这就是为什么 PPO 需要一个 `Transition` 数据结构。

它记录的是某一个环境交互时间步所需要的完整信息。

这里还有一个重要细节：

```python
self.transition.observations = obs
```

发生在：

```python
self.env.step(actions)
```

之前。

因为这条 Transition 对应的是：

$$ (s_t,a_t,r_t,s_{t+1},done_t) $$

其中 Observation 和 Action 属于当前决策时刻，Reward 和 Done 则在环境执行动作后得到。

##### 🧠 1. 处理记忆（LSTM）


```python
if self.actor_critic.is_recurrent:
    self.transition.hidden_states = (
        self.actor_critic.get_hidden_states()
    )
```

- **背景**：如果你开启了 LSTM（RNN 网络），网络是有记忆的。
    
- **动作**：在每一次输出动作前，把当前网络内部的“隐藏状态（Hidden State）”存起来。
    
- **目的**：因为 PPO 在后续更新网络时，需要把这些隐藏状态重新灌入网络，让网络能回忆起过去的时序信息。
    

##### 🎬 2. 生成动作并切断梯度

```python
self.transition.actions = (
    self.actor_critic.act(obs).detach()
)
```

- **动作**：Actor 接收到 `obs`（观测），算出动作并返回。
    
- **`.detach()` 魔法（极其重要）**：这是什么意思？这叫**切断计算图**。在采样阶段，我们只是把数据扔进仓库，**绝对不要计算梯度**。如果不加 `.detach()`，这个动作就会带着整个神经网络的权重矩阵计算图，瞬间把你的显存撑爆。
    

##### ⚖️ 3. 让 Critic 打分


```python
self.transition.values = (
    self.actor_critic.evaluate(critic_obs).detach()
)
```

- **动作**：将 `critic_obs`（注意，这里极有可能包含了只有 Critic 才能看到的特权信息，比如真实线速度）喂给 Critic，算出当前状态的价值 V(s)V(s)。
    
- **注意**：依然要加 `.detach()`，切断梯度。
    

##### 🎲 4. 记录概率与分布参数（PPO 的命根子）


```python
self.transition.actions_log_prob = (
    self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
)
self.transition.action_mean = self.actor_critic.action_mean.detach()
self.transition.action_sigma = self.actor_critic.action_std.detach()
```

- **`actions_log_prob`**：这是动作的对数概率 $log⁡π(a_t∣s_t)$。**没有它，PPO 根本算不出 Ratio(重要性比例)，也就无法做 Clip 更新。**
    
- **`action_mean` 和 `action_sigma`**：记录高斯分布的均值 μμ 和标准差 σ。它们主要用来在 TensorBoard 里监控策略的探索状态（比如你之前截图里看到的 `Mean action noise std: 0.60`）。
    

##### 📦 5. 存下观测状态


```python
self.transition.observations = obs
self.transition.critic_observations = critic_obs
```

- 把 Actor 看到的 `obs` 和 Critic 看到的 `critic_obs` 存下来。后续 `update()` 时，要用这些状态重新计算新的动作概率（算 Ratio）。
    

##### 🚀 6. 返回动作

```python
return self.transition.actions
```
- 把这个动作交给外层的 `env.step(actions)`，去物理引擎里推演下一步。

## **五、第三部分：process_env_step() 如何把数据存进 Storage？**

找到：

```python
def process_env_step(self, rewards, dones, infos):
```

关键代码：

```python
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

今天先关注三个事情。

**1. 保存 Reward 和 Done**

环境返回的 Reward 和 Done 被补充到之前的 Transition 中。

---

**2. Timeout Bootstrap**(这里就和 Day7的问题三 串了起来)

这里对 `time_outs` 做了特殊处理：

$$ r_t\leftarrow r_t+\gamma V_t $$

为什么要加这一段？我们先区分两个概念：

1. **`dones=True` 且 `time_outs=False`**：机器人真的**摔死了**，或者撞墙了。这局彻底完蛋，未来的预期收益是 0。
    
2. **`dones=True` 且 `time_outs=True`**：机器人活得好好的，只是**时间到了（超时）**，被系统强制重启。它其实本来可以继续往前走的。
    

**如果不处理超时，会有什么灾难？**  
在标准的 PPO 中，如果发现 `done=True`，计算 GAE 时会把下一步的价值视为 0。但这对于“超时”来说是极其冤枉的！  
假设机器人走了 1000 步，原本马上要走到终点了，结果因为时间到了被强制结束。系统一看 `done=True`，直接判定它“死了”，不仅没给它未来的奖励，还在算 GAE 时把它的价值砍断。网络会极懵逼：“明明我走得很好，怎么突然就变成死刑了？”

**这段代码的解法（Bootstrap 自举）：**  
如果 `time_outs=True`，说明它不是真死。我们在当前奖励上，**强行加上下一步的预期价值**：$r_t=r_t+γV(s_t)$  
这样后续算 GAE 时，就相当于把超时截断变成了一个连续的过程，网络就不会因为被强制掐断而崩溃。`torch.squeeze` 和 `unsqueeze` 只是为了把维度对齐，让这步加法能顺利广播。

**为什么不能把`time_out`的`done`改为0：**
**绝对不能直接改 `done=0`，因为这会导致极其严重的“物理时空错乱（State Reset Mismatch）”。**

我们用底层的数据流来推演一下，如果你把 `done` 强行改成 0，会发生什么灾难：

##### 💥 灾难现场：物理时空错乱

假设在第 100 步，机器人超时了。

1. **环境（Env）内部**：它不管什么超时不超时，时间到了，它必须执行 `reset_idx()`。所以，**机器人瞬间被瞬移回了出生的起点（完美的站立姿态），所有的速度被清零，关节被重置。**
    
2. 然后，环境把重置后的新状态 `obs`（其实是初始状态 s0s0​）返回给了 PPO 算法。
    
3. 如果你在这里把 `done` 强行改成了 `0`（告诉 PPO 没死）。
    
4. **PPO 的 GAE 计算逻辑**：它看到 `done=0`，就会认为 st+1st+1​（其实是刚刚复活的 s0s0​）是第 100 步的**自然延续**。
    
    - 它会把第 100 步的奖励，和重生的初始状态的价值 V(s0)V(s0​) 加起来算 GAE！
        
    - 但在物理上，第 100 步的机器人（哪怕超时了）和重生的 s0s0​，根本就是两个不同时间线的状态！**这在数学上叫“跨时空污染”，会直接导致 Critic 的价值估算崩溃。**
    - 
---
**3. 保存 Transition**

```python
self.storage.add_transitions(self.transition)
```

进入：

`rsl_rl/storage/rollout_storage.py`

找到：

```python
def add_transitions(self, transition):
```

可以看到：

```python
self.observations[self.step].copy_(
    transition.observations
)

self.actions[self.step].copy_(
    transition.actions
)

self.rewards[self.step].copy_(
    transition.rewards.view(-1, 1)
)

self.dones[self.step].copy_(
    transition.dones.view(-1, 1)
)

self.values[self.step].copy_(
    transition.values
)

self.actions_log_prob[self.step].copy_(
    transition.actions_log_prob.view(-1, 1)
)

self._save_hidden_states(
    transition.hidden_states
)

self.step += 1
```

这里就是将当前时间步的数据写入 Rollout Storage。

## **六、第四部分：G1 一次 Rollout 究竟采集多少数据？**

这是一个你应该亲自计算的数字。

当前：

$$ N_{\text{env}}=4096 $$

$$ T_{\text{rollout}}=24 $$

因此：

$$ N_{\text{transitions}} = 4096\times24 = 98,304 条$$

也就是说，每一个 Learning Iteration，在默认 4096 个并行环境的情况下，会采集 98,304 条环境 Transition。

Storage 可以理解为沿两个维度组织：

$$ [T,N,\ldots] $$

其中：

- \(T=24\)：时间维度。
    
- \(N=4096\)：并行环境维度。
    

例如：

| Tensor                  | 预期形状             |
| ----------------------- | ---------------- |
| Observations            | `[24, 4096, 47]` |
| Privileged Observations | `[24, 4096, 50]` |
| Actions                 | `[24, 4096, 12]` |
| Rewards                 | `[24, 4096, 1]`  |
| Dones                   | `[24, 4096, 1]`  |
| Values                  | `[24, 4096, 1]`  |

实际运行时，Tensor 的设备和具体结构以对应代码为准。

注意：如果使用 `--num_envs` 修改环境数量，98,304 就不再是固定值。

## **七、第五部分：GAE 从哪里开始计算？**

现在找到：

`rsl_rl/storage/rollout_storage.py`

```
def compute_returns(self, last_values, gamma, lam):
```

核心代码：

```python
advantage = 0

for step in reversed(
    range(self.num_transitions_per_env)
):

    if step == self.num_transitions_per_env - 1:
        next_values = last_values

    else:
        next_values = self.values[step + 1]

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

    self.returns[step] = (
        advantage + self.values[step]
    )
```

回顾一下CleanRL PPO中GAE的计算：

```python
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
	                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
```

两者完全可以相互对应

首先：

$$ \delta_t= r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t) $$

然后：

$$ \hat A_t= \delta_t+ \gamma\lambda(1-d_t)\hat A_{t+1} $$

最后：

$$ \hat R_t=\hat A_t+V(s_t) $$

对于 LSTM，Storage 还会保存隐藏状态，并在后续更新时使用按 Episode 分割和补齐的序列。这部分先放到 Day 9。

# 今日问题

1. 为什么 `PPO.act()` 要在环境执行 Action 之前记录 `observations`、`actions_log_prob` 和 `values`？

答：一条 transition$(s_t,a_t,r_t,s_{t+1},done_t)$ 相互对应，执行完Action后，observations等值会被覆盖，导致后续GAE和Loss无法计算


2. 为什么 PPO 不在每次 `env.step()` 后立即执行 `optimizer.step()`，而是先收集 24 个时间步？

答：这是PPO算法相比其它off-policy算法的一大区别，虽然PPO也是off-policy算法，但是重要性采样让他可以成为一个局部on-policy算法，我们可以用旧策略收集一批transition，反复更新策略，这样可以充分利用数据，加速网络收敛

3. 当前默认 4096 个环境、每个环境 24 个策略时间步，一次 Rollout 总共包含多少条 Transition？

答：总共包含$4096 \times 24 = 98304$个Transition

4. 为什么 GAE 使用倒序循环，而不是从第一个时间步正序计算？

答：这是GAE在计算机中的计算技巧，而且从$A_{0}^{(GAE)}$到从$A_{T}^{(GAE)}$我们全部需要，所以倒叙计算也可以方便我们存储所有值，这是正序计算做不到的
 
5. 为什么 `time_outs` 会被单独处理，而 `dones` 仍然会参与 GAE 的终止掩码？

答：`time_outs`虽然是一局游戏结束，但是机器人并不是因为操作失误而死亡，如果同样用done截断奖励是不公平的，容易让机器人获得错误的反馈，时间到了和摔死的反馈一致，那我在时间快到的时候就跑快一点，哪怕摔死，获得的奖励反而可能更多，所以，我们虽然设置done=0，隔绝了下个环境产生的obs等，但是我们会令$r_t\leftarrow r_t+\gamma V_t$，来保留这个值函数。

# 今日实践

今天不需要启动完整训练。

只需要打开 G1 配置及基础配置，自己验证以下内容：

```python
num_envs = 4096
num_steps_per_env = 24
num_transitions = (num_envs * num_steps_per_env)
print(num_transitions)
```

然后画出自己的数据关系图：

```
PPO.act()
    |
    | obs, action, value, log_prob
    v
Transition
    |
    | env.step()
    v
rewards, dones, infos
    |
    v
process_env_step()
    |
    v
Rollout Storage
    |
    | 收集24步
    v
compute_returns()
    |
    v
GAE / Returns / Advantages
```

## **十、今日完成标准**

- 能解释一次 Rollout 和一次 Episode 的区别。
    
- 能说出 Transition 保存了哪些数据。
    
- 能解释 Rollout Storage 的时间维度和环境维度。
    
- 能计算一次默认 Rollout 的样本数量。
    
- 能从代码找到 GAE 的实际计算位置。
    

**今天最重要的结论：PPO 在 Rollout 阶段采集数据、记录旧策略信息，等数据收集完成后再统一计算 Advantage。**