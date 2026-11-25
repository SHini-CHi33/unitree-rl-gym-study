# 今日阅读任务

今日主题：PPO Update、Actor-Critic 与 LSTM
## **一、今天的核心问题**

Day 8 我们已经知道，PPO 可以获得一批数据：

```
Observations
Actions
Old Log Prob
Old Values
Rewards
Dones
Advantages
Returns
```

但是接下来真正重要的是：

**这些数据如何变成 Policy Loss 和 Value Loss，最后通过反向传播更新 Actor 和 Critic？**

今天我们正式进入 PPO 的优化阶段。

## **二、需要打开的文件**

|文件|具体函数|
|---|---|
|`legged_gym/envs/g1/g1_config.py`|`G1RoughCfgPPO`|
|`legged_gym/envs/base/legged_robot_config.py`|`LeggedRobotCfgPPO.algorithm`|
|`rsl_rl/algorithms/ppo.py`|`update()`|
|`rsl_rl/modules/actor_critic_recurrent.py`|`__init__()`、`act()`、`evaluate()`、`Memory`|
|`rsl_rl/modules/actor_critic.py`|`update_distribution()`、`act()`、`act_inference()`|
|`rsl_rl/storage/rollout_storage.py`|`reccurent_mini_batch_generator()`|

源码：

- [PPO Update](https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/algorithms/ppo.py?utm_source=chatgpt.com)
    
- [ActorCriticRecurrent](https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/modules/actor_critic_recurrent.py?utm_source=chatgpt.com)
    
- [ActorCritic](https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/modules/actor_critic.py?utm_source=chatgpt.com)
    
- [RolloutStorage](https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/storage/rollout_storage.py?utm_source=chatgpt.com)
    

## **三、第一部分：先看 G1 的真实网络结构**

打开：

`g1_config.py`

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

    class algorithm(LeggedRobotCfgPPO.algorithm):

        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):

        policy_class_name = "ActorCriticRecurrent"

        max_iterations = 10000

        experiment_name = 'g1'
```

注意：G1 没有重新定义全部 PPO 参数，它继承了 `LeggedRobotCfgPPO`。

因此当前基础配置中：

| 参数                    | 默认值   |
| --------------------- | ----- |
| `num_learning_epochs` | 5     |
| `num_mini_batches`    | 4     |
| `learning_rate`       | 0.001 |
| `clip_param`          | 0.2   |
| `gamma`               | 0.99  |
| `lam`                 | 0.95  |
| `value_loss_coef`     | 1.0   |
| `entropy_coef`        | 0.01  |
| `desired_kl`          | 0.01  |
| `max_grad_norm`       | 1.0   |

今天你需要真正理解的是 G1 的 Actor-Critic 并不是简单的 MLP，而是：

**独立的 Actor LSTM + Actor MLP，以及独立的 Critic LSTM + Critic MLP。**

## **四、第二部分：ActorCriticRecurrent 是怎么构建的？**

打开：

`rsl_rl/modules/actor_critic_recurrent.py`

找到：

```
class ActorCriticRecurrent(ActorCritic):
```

然后看：

```python
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

它建立了两个独立的记忆模块：

- `memory_a`：Actor 的 LSTM。
    
- `memory_c`：Critic 的 LSTM。
    

当前 G1 的尺寸是：

**<div style="max-width: 650px; margin: 20px auto; background-color: #ffffff; border-radius: 12px; border: 1px solid #e5e7eb; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
  
  <div style="font-weight: bold; font-size: 16px; color: #111827; margin-bottom: 16px;">Actor 网络</div>
  
  <div style="background-color: #f0f7ff; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 8px;">
    <div style="font-size: 14px; color: #374151; margin-bottom: 4px;">Observation</div>
    <div style="font-size: 16px; font-weight: 700; color: #3b82f6;">47 维</div>
  </div>
  
  <div style="text-align: center; color: #9ca3af; font-size: 16px; margin: 4px 0; line-height: 1;">↓</div>
  
  <div style="background-color: #f0f7ff; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 8px;">
    <div style="font-size: 14px; color: #374151; margin-bottom: 4px;">LSTM Memory</div>
    <div style="font-size: 16px; font-weight: 700; color: #3b82f6;">64 hidden</div>
    <div style="font-size: 12px; color: #9ca3af; margin-top: 4px;">1 层</div>
  </div>
  
  <div style="text-align: center; color: #9ca3af; font-size: 16px; margin: 4px 0; line-height: 1;">↓</div>
  
  <div style="background-color: #f0f7ff; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 8px;">
    <div style="font-size: 14px; color: #374151; margin-bottom: 4px;">Actor MLP</div>
    <div style="font-size: 16px; font-weight: 700; color: #3b82f6;">32</div>
  </div>
  
  <div style="text-align: center; color: #9ca3af; font-size: 16px; margin: 4px 0; line-height: 1;">↓</div>
  
  <div style="background-color: #f0f7ff; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 8px;">
    <div style="font-size: 14px; color: #374151; margin-bottom: 4px;">Action Mean</div>
    <div style="font-size: 16px; font-weight: 700; color: #3b82f6;">12 维</div>
  </div>

  <div style="height: 1px; background-color: #e5e7eb; margin: 24px 0;"></div>

  <div style="font-weight: bold; font-size: 16px; color: #111827; margin-bottom: 16px;">Critic 网络</div>
  
  <div style="background-color: #ebf9f2; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 8px;">
    <div style="font-size: 14px; color: #374151; margin-bottom: 4px;">Privileged Observation</div>
    <div style="font-size: 16px; font-weight: 700; color: #10b981;">50 维</div>
  </div>
  
  <div style="text-align: center; color: #9ca3af; font-size: 16px; margin: 4px 0; line-height: 1;">↓</div>
  
  <div style="background-color: #ebf9f2; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 8px;">
    <div style="font-size: 14px; color: #374151; margin-bottom: 4px;">LSTM Memory</div>
    <div style="font-size: 16px; font-weight: 700; color: #10b981;">64 hidden</div>
  </div>
  
  <div style="text-align: center; color: #9ca3af; font-size: 16px; margin: 4px 0; line-height: 1;">↓</div>
  
  <div style="background-color: #ebf9f2; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 8px;">
    <div style="font-size: 14px; color: #374151; margin-bottom: 4px;">Critic MLP</div>
    <div style="font-size: 16px; font-weight: 700; color: #10b981;">32</div>
  </div>
  
  <div style="text-align: center; color: #9ca3af; font-size: 16px; margin: 4px 0; line-height: 1;">↓</div>
  
  <div style="background-color: #ebf9f2; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 8px;">
    <div style="font-size: 14px; color: #374151; margin-bottom: 4px;">Value</div>
    <div style="font-size: 16px; font-weight: 700; color: #10b981;">1 维</div>
  </div>

</div>

这里特别提醒：

LSTM 的输出维度为 64，之后才进入 MLP。不要把 `rnn_hidden_size=64` 理解成最终 Actor 的 Action 维度。

## **五、第三部分：为什么 PPO 训练时需要 Gaussian Policy？**

打开：

`rsl_rl/modules/actor_critic.py`

找到：

```python
def update_distribution(self, observations):

    mean = self.actor(observations)

    self.distribution = Normal(
        mean,
        mean * 0. + self.std
    )
```

接着：

```python
def act(self, observations, **kwargs):

    self.update_distribution(observations)

    return self.distribution.sample()
```

这说明当前连续动作策略使用的是高斯分布：

$$ a_t\sim\mathcal N(\mu_\theta(s_t),\sigma_\theta^2) $$
这里：

- Actor 网络输出均值 \($\mu$\)。
    
- `std` 是可学习的动作标准差参数。
    
- 通过分布采样得到实际 Action。
    

所以训练时，Policy 并不是永远输出均值，而是从高斯分布中采样。

这也解释了：

```
init_noise_std = 0.8
```

它是初始动作标准差，而不是 Action Scale。

两者要严格区分：

|参数|作用|
|---|---|
|`init_noise_std=0.8`|初始化策略的探索标准差|
|`action_scale=0.25`|将 Action 转成目标关节角度偏移|

## **六、第四部分：正式进入 PPO.update()**

现在打开：

`rsl_rl/algorithms/ppo.py`

找到：

```python
def update(self):
```

其中第一部分：

```python
if self.actor_critic.is_recurrent:

    generator = (
        self.storage.reccurent_mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs
        )
    )
```

注意官方 v1.0.2 这里的函数拼写是 `reccurent`，不是通常的 `recurrent`。

为什么需要特殊的 Generator？

因为当前 G1 使用 LSTM。

普通 MLP 可以把 Rollout 展平成普通样本进行随机打乱，但 LSTM 需要利用序列顺序和对应的隐藏状态。

因此，Storage 会根据 Done 切分轨迹，进行序列 Padding，并提供：

- Observation 序列；
    
- Critic Observation 序列；
    
- Action；
    
- Advantage；
    
- Return；
    
- Hidden States；
    
- Masks。
    

接下来：

```python
for obs_batch, critic_obs_batch, actions_batch, \
    target_values_batch, advantages_batch, \
    returns_batch, old_actions_log_prob_batch, \
    old_mu_batch, old_sigma_batch, \
    hid_states_batch, masks_batch in generator:
```

然后重新计算当前策略：

```python
self.actor_critic.act(
    obs_batch,
    masks=masks_batch,
    hidden_states=hid_states_batch[0]
)

actions_log_prob_batch = (
    self.actor_critic.get_actions_log_prob(
        actions_batch
    )
)
```

这里与 PPO 理论联系起来了。

旧策略的：

$$ \log\pi_{\theta_{\text{old}}}(a_t|s_t) $$

已经在 Rollout 时保存。

现在重新计算：

$$ \log\pi_\theta(a_t|s_t) $$

然后：

$$ r_t(\theta) = \exp\left( \log\pi_\theta(a_t|s_t) - \log\pi_{\theta_{\text{old}}}(a_t|s_t) \right) $$

对应代码：

```python
ratio = torch.exp(
    actions_log_prob_batch
    - torch.squeeze(old_actions_log_prob_batch)
)
```

这就是 Importance Sampling Ratio。

再看：

```python
surrogate = (
    -torch.squeeze(advantages_batch) * ratio
)

surrogate_clipped = (
    -torch.squeeze(advantages_batch)
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

这就是 PPO Clipped Surrogate Loss 在最小化形式下的实现。

## **七、第五部分：Critic Loss 和真正的参数更新**

继续看：

```python
value_batch = self.actor_critic.evaluate(
    critic_obs_batch,
    masks=masks_batch,
    hidden_states=hid_states_batch[1]
)
```

然后：

```python
value_losses = (value_batch - returns_batch).pow(2)
```

如果启用了 Clipped Value Loss，还会另外计算一份裁剪后的 Value Loss，并取两者最大值。

最后：

```python
loss = (
    surrogate_loss
    + self.value_loss_coef * value_loss
    - self.entropy_coef * entropy_batch.mean()
)
```

到这里就联系上了你之前学习的三项 PPO 损失：

$$ L= L_{\text{policy}} +c_1 L_{\text{value}} -c_2 H(\pi) $$

最后：

```python
self.optimizer.zero_grad()

loss.backward()

nn.utils.clip_grad_norm_(
    self.actor_critic.parameters(),
    self.max_grad_norm
)

self.optimizer.step()
```

这几行就是整个训练过程中真正更新神经网络参数的地方。

### **一个非常重要的计算问题**

当前 G1 默认：

$$ num\_learning\_epochs=5 $$

$$ num\_mini\_batches=4 $$

因此每轮 Rollout 后：

$$ N_{\text{optimizer updates}}=5\times4 = 20 次梯度更新$$

这里不是说每个 Transition 只更新一次，而是收集的这批数据会在多个 Epoch、多个 Mini-batch 中被重复使用。

# 今日问题

1. 为什么 G1 要分别使用 `memory_a` 和 `memory_c`，而不是让 Actor 与 Critic 共用一个 LSTM？
    
2. 为什么 PPO 要在 Rollout 中保存旧策略的 `log_prob`、`mu` 和 `sigma`，而不是 Update 时只重新计算一次？
    
3. 为什么使用 LSTM 后，Mini-batch Generator 要考虑 Done、Mask 和 Hidden States？
    
4. `num_learning_epochs=5` 和 `num_mini_batches=4` 分别代表什么？为什么一次 Rollout 对应 20 次优化步骤？
    
5. `optimizer.step()` 更新的是 Actor、Critic，还是两个网络？请结合 `self.actor_critic.parameters()` 判断。
    

# 今日实践

今天不需要训练。

请先根据 G1 配置，手动整理以下参数：

```python
policy_class_name = "ActorCriticRecurrent"

rnn_type = "lstm"
rnn_hidden_size = 64
rnn_num_layers = 1

actor_hidden_dims = [32]
critic_hidden_dims = [32]

num_learning_epochs = 5
num_mini_batches = 4
```

然后写出：

```
Actor:
47 → LSTM(64) → MLP(32) → 12

Critic:
50 → LSTM(64) → MLP(32) → 1

PPO:
5 epochs × 4 mini-batches = 20 updates
```

最后在 `ppo.py` 中找到：

```python
ratio = torch.exp(...)
```

以及：

```python
loss.backward()self.optimizer.step()
```

在自己的笔记中把它们与 PPO 数学公式一一对应。

# 今日标准

- 能解释 G1 的 Actor 和 Critic 实际网络结构。
    
- 能解释 Gaussian Policy 为什么需要 mean 和 std。
    
- 能从代码找到旧策略和新策略的 Log Probability。
    
- 能解释 PPO Clipped Loss 和 Value Loss。
    
- 能找到真正进行梯度更新的 `optimizer.step()`。
    

**今天最重要的结论：Rollout 提供数据，GAE 提供 Advantage，PPO Update 才是真正改变 Actor 和 Critic 参数的地方。**