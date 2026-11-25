# 今日任务

今日主题：第一次单因素实验——修改 Action Rate Reward**
## **一、今天的核心目标**

昨天我们建立了 Baseline。

今天只修改一个参数：

```
action_rate = -0.01
```

改成：

```
action_rate = -0.02
```

然后重新训练，并与 Day 11 的实验结果对比。

今天希望你学会一种非常重要的研究方法：

**One-factor-at-a-time Experiment（单因素实验）。**

也就是：

一次只改变一个因素，尽可能保持其他条件一致，观察结果变化。

## **二、今天需要阅读的文件**

主要是两个：

|文件|函数|
|---|---|
|`legged_gym/envs/g1/g1_config.py`|`G1RoughCfg.rewards.scales`|
|`legged_gym/envs/base/legged_robot.py`|`_reward_action_rate()`、`_prepare_reward_function()`、`compute_reward()`|

官方代码：

- [G1 Reward 配置](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py?utm_source=chatgpt.com)
    
- [Reward 实现](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/base/legged_robot.py?utm_source=chatgpt.com)
    

## **三、第一部分：先回顾 Action Rate Reward**

找到：

```python
def _reward_action_rate(self):

    return torch.sum(
        torch.square(
            self.last_actions - self.actions
        ),
        dim=1
    )
```

它的数学表达式：

$$ f_{\text{action-rate}} = \sum_i(a_{t-1,i}-a_{t,i})^2 $$

它衡量相邻两个策略时间步中 Action 的变化幅度。

例如：

**情况 A：Action 变化很小**

$$ a_{t-1}=[0.1,0.2,0.3] $$

$$ a_t=[0.11,0.19,0.3] $$

$$ f_{\text{action-rate}}=0.0002 $$

==🟢变化较小

**情况 B：Action 变化较大**

$$ a_{t-1}=[0.1,0.2,0.3] $$

$$ a_t=[0.5,-0.2,0.8] $$

$$ f_{\text{action-rate}}=0.65 $$

==🔴变化较大

这说明 Reward 函数返回的原始数值越大，说明动作变化越剧烈。

但是 G1 配置中：

```python
action_rate = -0.01
```

所以它实际是一个惩罚项。

## **四、第二部分：修改参数后会发生什么？**

我们修改：

```python
# 原始配置
action_rate = -0.01

# 实验配置
action_rate = -0.02
```

注意：

这里不是把惩罚减小了，而是**将惩罚强度增加了一倍**。

回忆之前 Day 6 学习的 `_prepare_reward_function()`：

```python
self.reward_scales[key] *= self.dt
```

所以最终 Action Rate 对总 Reward 的贡献是：

$$ r_{\text{action-rate}} = w\cdot dt\cdot f_{\text{action-rate}} $$

当前：

$$ dt=0.02s $$

假设原始 Action Rate 为：

$$ f=0.1 $$

那么原始配置的贡献是：

$$ r_1=-0.01\times0.02\times0.1 $$

$$ r_1=-0.00002 $$

修改后的贡献：

$$ r_2=-0.02\times0.02\times0.1 $$

$$ r_2=-0.00004 $$

**相同 Action 变化下的惩罚对比:**

|配置|Reward 贡献|
|---|---|
|-0.01|-0.00002|
|-0.02|-0.00004|

惩罚强度增加 100%

因此，我们提出一个实验假设：

> 增大 Action Rate 惩罚后，策略可能会减少相邻时间步之间的 Action 变化，从而使关节目标指令更加平缓。

但不要急着认定效果一定更好。

因为 Reward 中还有：

```python
tracking_lin_vel = 1.0
tracking_ang_vel = 0.5
contact = 0.18
feet_swing_height = -20.0
```

这些奖励彼此存在权衡。

假设惩罚过强，可能导致 Policy 不愿意及时改变动作，反而影响速度跟踪。

**我们的实验目的不是证明修改一定有效，而是验证假设。**

## **五、第三部分：真正修改代码**

先备份原始配置：

```
cp legged_gym/envs/g1/g1_config.py \
   /tmp/g1_config_day11_baseline.py
```

然后打开：

```
legged_gym/envs/g1/g1_config.py
```

找到：

```python
action_rate = -0.01
```

修改为：

```python
action_rate = -0.02
```

除此之外：

**不要修改其他任何参数。**

特别不要修改：

- `action_scale`
    
- `stiffness`
    
- `damping`
    
- `decimation`
    
- `tracking_lin_vel`
    
- `num_envs`
    
- `learning_rate`
    
- `gamma`
    
- `lam`
    

我们今天只研究一个因素。

## **六、第四部分：运行第二次实验**

执行：

```
python legged_gym/scripts/train.py \
    --task=g1 \
    --headless \
    --num_envs=128 \
    --max_iterations=100 \
    --run_name=day12_actionrate_002
```

注意两个非常重要的点：

**第一，不能使用** `--resume`**。**

我们要从新初始化的策略开始训练，而不是接着 Day 11 的 Baseline 继续训练。

否则就变成了：

```
Day 11 训练得到的模型
       ↓
修改 Reward
       ↓
继续训练
```

这无法直接和 Baseline 的独立训练结果进行公平比较。

我们希望的是：

```
                原始初始化
                   |
          ┌────────┴────────┐
          |                 |
          v                 v
      Baseline           Experiment
     action_rate        action_rate
       -0.01              -0.02
          |                 |
          v                 v
      Model A            Model B
          |                 |
          └────────┬────────┘
                   |
                对比结果
```

**第二，保持训练条件一致。**

Day 11 和 Day 12 必须使用相同的：

- 仓库版本；
    
- Num Envs；
    
- Max Iterations；
    
- G1 其他配置；
    
- 随机种子设置；
    
- 训练设备。
    

这样才能尽可能把结果差异归因于 Action Rate 权重。

补充一个复现细节：当前官方代码中环境的 `set_seed()` 在 `make_env()` 阶段执行，而训练配置的命令行覆盖发生在 `make_alg_runner()` 阶段。严格控制随机性时，要检查本地版本中的种子设置顺序；这里建议保持配置文件中的默认 Seed 不变，并记录实际值。

## **七、第五部分：对比两个实验**

现在打开 TensorBoard：

```
tensorboard --logdir=logs/g1
```

找到两个 Run：

- `day11_baseline`
    
- `day12_actionrate_002`
    

我们重点对比：

|指标|Baseline|Experiment|
|---|---|---|
|Train/mean_reward|记录|记录|
|Train/mean_episode_length|记录|记录|
|Episode/rew_tracking_lin_vel|记录|记录|
|Episode/rew_action_rate|记录|记录|
|Loss/value_function|记录|记录|
|Loss/surrogate|记录|记录|

这里有一个非常重要的实验分析细节：

`Episode/rew_action_rate` **本身包含 Reward Scale 和** `dt`**，不是原始动作变化量。**

所以权重从 -0.01 改到 -0.02，即使实际 Action 变化完全一样，该奖励项也会变得更负。

因此，不能只看到 `rew_action_rate` 变得更负，就说策略的动作更加剧烈。

真正要判断动作是否变平滑，需要观察：

$$ \mathbb E\left[ \sum_i(a_{t,i}-a_{t-1,i})^2 \right] $$

也就是原始 Action 差值的均值。

可以进一步在相同命令下运行两组 Play 策略，记录这个原始指标，并比较机器人运动轨迹。

# 今日问题

**1. 为什么修改 `action_rate` 的负权重，会影响 PPO 最终学习到的动作分布？

答：修改action_rate权重，会导致策略输出相同动作时惩罚增加一倍，PPO为了增加reward（减小LOSS）会更加保守或减小相邻动作的变化，随着episode的进行，最终导致学习到的动作分布不同

#### 💡GPT的修正与补充

核心理解正确。

尤其是这句话：

> 相同动作时惩罚增加一倍。

非常准确。

不过需要进一步理解：Reward 并不是直接修改神经网络参数，而是通过 PPO 的优化链路影响网络。

完整过程：

```
action_rate 从 -0.01 变为 -0.02
              |
              v
       Reward 计算
              |
              v
      Episode Return
              |
              v
             GAE
              |
              v
         Advantage
              |
              v
       PPO Surrogate Loss
              |
              v
       Backpropagation
              |
              v
       更新 Actor 参数
              |
              v
       改变 Action 分布
```

数学上：

$$ r_t=\sum_i w_i f_i(s_t,a_t) $$

当 Action Rate 的负权重增加时，意味着：

**对于相同的 Action 变化，策略会受到更大的负奖励。**

这会改变不同动作在训练目标中的相对收益。

但有一点需要修正：

你说 PPO 会更加保守，这只是预期，不是必然。

实际上，PPO 可能出现几种情况：

- 减少相邻 Action 的变化。
    
- 改变 Actor 的均值。
    
- 改变 Actor 的标准差。
    
- 为了保证速度跟踪，依然输出比较大的动作变化。
    

因此，最准确的理解是：

> 增大 Action Rate 惩罚，会改变策略优化目标中平滑性与任务表现之间的权衡，使最终学到的策略分布发生变化，但不保证每个维度的 Action 变化都下降。


**2. 如果增大 Action Rate 惩罚，为什么可能导致机器人对 Command 的跟踪能力下降？

答：增大Action Rate惩罚，会导致机器人减小相邻动作的变化，导致动作变化缓慢，动作变化缓慢导致速度变化缓慢，而到达目标速度的时间增加，看起来就是对Command的跟踪能力下降；从另一个角度理解，Action Rate惩罚增大，势必会导致其它reward作用被稀释，其中包括对Command的跟踪能力

#### 💡GPT的修正与补充

这条因果链基本正确。

不过，我希望你区分两个东西：

**Action 变化不等于机器人速度变化。**

在当前 G1 中：

$$ q_{\text{target},t} = q_{\text{default}}+0.25a_t $$

所以 Action Rate 惩罚实际上约束的是相邻策略时间步之间的目标关节角度变化。

$$ \Delta q_{\text{target}} = 0.25(a_t-a_{t-1}) $$

它不是直接限制：

$$ \Delta v_{\text{base}} $$

机器人实际速度还受到动力学、惯性、地面接触等因素影响。

但是你的结论仍然合理：

例如机器人需要迅速调整步态来达到新的前进速度，如果 Action 变化受到过强惩罚，就可能不愿意快速调整目标关节角度，从而影响速度跟踪。

你第二种解释：

> Action Rate 惩罚增大，势必会导致其它 Reward 作用被稀释。

这个说法要稍微修正。

不是其他 Reward 的数值被直接缩小了，而是：

**在总 Reward 的加权和中，Action Rate 惩罚相对于其他奖励项的影响力增加了。**

举例：

$$ R=r_{\text{tracking}}-0.01f $$

变成：

$$ R=r_{\text{tracking}}-0.02f $$

其他奖励项本身没有变化，但优化目标中的权衡发生了变化。

这个细节很重要。

**3. 为什么 Day 12 必须重新初始化策略，而不能直接 Resume Day 11 的 Checkpoint？

答：因为我们修改了Reward中的action_rate，而在Day11训练的PPO是另一个action_rate，如果Resume Day 11 的 Checkpoint，可能导致网络通过很长时间学习到的经验被打乱，需要重新学习并适应环境，导致各项reward下降，Loss不稳，严重可能导致训练崩溃

#### 💡GPT的修正与补充

你的实验设计方向正确。

但是有一句话需要纠正：

**Resume Day 11 的 Checkpoint，并不意味着之前学到的经验一定被打乱，也不意味着训练一定崩溃。**

实际上，你完全可以：

1. 先用 -0.01 训练一个策略。
    
2. 修改为 -0.02。
    
3. 从旧 Checkpoint 继续训练。
    

这是一种合理的 Fine-tuning 实验。

但它研究的是：

> 一个已经在旧奖励目标下学到策略的模型，如何适应新的奖励目标？

而我们 Day 12 的实验目标不同：

> 从相同的初始条件开始，仅修改 Reward 权重，观察最终学到的策略有什么区别。

所以，为了公平比较，我们应该：

```
相同初始策略
       |
       ├───────────────┐
       |               |
       v               v
Baseline          Experiment
-0.01             -0.02
       |               |
       v               v
独立训练          独立训练
       |               |
       v               v
Model A            Model B
       |               |
       └───────┬───────┘
               |
               v
          结果比较
```

还有一个容易遗漏的实验条件：

最好同时保持：

- 相同的随机种子；
    
- 相同的环境数量；
    
- 相同的训练 Iteration；
    
- 相同的 PPO 超参数；
    
- 相同的代码版本。
    

更严格的研究还应该用多个随机种子重复实验，避免某一个随机种子导致的偶然结果。

**修正后的答案：**

> 不 Resume 旧 Checkpoint，是为了让两组实验从可比的初始策略出发，排除旧奖励目标下已经学到的策略参数和 Optimizer 状态对新实验的影响，从而更清晰地分析 Action Rate 权重这一因素的作用。


**4. 为什么 `Episode/rew_action_rate` 更负，不代表原始 Action 变化一定更大？

答：由于我们通过改变scale增加了一倍action_rate的惩罚，所以即使Action变化相较原来下降了，平均的action_rate可能比原来还小，通过Day11与Day12实验验证后，发现确实如此（其中蓝色线为Day12）
![[Pasted image 20261122162453.png|834]]

#### 💡GPT的修正与补充

你的理解方向正确，而且你的图确实提供了这个假设的支持。

但这里必须严格区分：

| 指标                           | 是否包含 Reward Scale |
| ---------------------------- | ----------------- |
| `_reward_action_rate()` 原始输出 | 否                 |
| `Episode/rew_action_rate`    | 是                 |

图上的蓝线更负，表示：

$$ |\text{加权累计惩罚}_{12}| > |\text{加权累计惩罚}_{11}| $$

但不代表：

$$ f_{12}>f_{11} $$

因为 Day 12 的权重本来就更大。

刚刚我们反算得到，在两组 Episode 时间可比的情况下，蓝线对应的累计原始指标大约低 23%。

所以，你的判断有依据，但还不能说已经完全验证。
#### **怎样真正验证？**

最好的办法是增加一个独立日志：

在 `legged_robot.py` 中定义一个原始 Action Rate 记录量：

```python
raw_action_rate = torch.sum(
    torch.square(
        self.last_actions - self.actions
    ),
    dim=1
)
```

然后独立统计：

```python
self.episode_sums_raw_action_rate += (
    raw_action_rate
)
```

Episode 结束时，除了记录原来的加权 Reward，还记录：

$$ \boxed{ \text{MeanRawActionRate} = \frac{\sum_t f_t}{N_{\text{episode steps}}} } $$

这样就排除了：

- Reward Scale；
    
- 不同 Episode 时长；
    
- 不同累计步数；
    

对比较结果的干扰。

建议下一步做的验证

如果最终结果是：

$$ MeanRawActionRate_{12} < MeanRawActionRate_{11} $$

那么我们就可以更有把握地说：

Day 12 学到的策略确实减少了相邻动作变化。

这里不要求你现在马上修改代码，我建议先把这项验证留作后续实验。

**5. 如果 Mean Reward 上升，但是 Mean Episode Length 下降，你会怎样分析这个结果？

答：可能是机器人在最后时刻为了增加reward，通过加速去接近Command、大跨步去获得更多reward等导致动作不稳而摔倒

#### 💡GPT的修正与补充

这是一个很有价值的假设。

你已经开始考虑：

**策略可能在追求 Reward 的过程中，牺牲了其他方面的表现。**

这可以看作一种 Reward Trade-off，某些情况下也可能出现 Reward Hacking。

但是不能只凭这两个指标就得出这个结论。

还存在其他可能性：

1. 机器人前半段表现很好，但后半段无法维持平衡。
    
2. 速度跟踪奖励增加得足够多，抵消了提前摔倒带来的其他损失。
    
3. 某些奖励项的变化导致总 Reward 提高，但身体稳定性变差。
    
4. 两条曲线的统计窗口受到 Episode 完成数量和时长变化的影响。
    

我建议出现这种情况时，进一步检查：

|指标|重点观察|
|---|---|
|`Episode/rew_tracking_lin_vel`|是否为了追踪速度而更加激进|
|`Episode/rew_action_rate`|相邻动作变化是否增加|
|`Episode/rew_orientation`|身体是否更加倾斜|
|`Episode/rew_base_height`|身体高度是否发生异常|
|`Train/mean_episode_length`|是否提前结束 Episode|
|Fall / Timeout 比例|究竟是摔倒还是正常超时|

还可以直接运行 Play，观察机器人究竟发生了什么。

**你的回答作为一个可能性是成立的，但还需要其他指标和视频来验证。**

# 今日的实践练习

完成两组实验后，建立：

`experiments/day12_action_rate.md`

填写：

```
# Day 12 Action Rate Experiment

## 1. Hypothesis

Increasing action_rate penalty may
reduce action changes and improve smoothness.

## 2. Baseline

action_rate = -0.01

Mean Reward:
Mean Episode Length:
Tracking Lin Vel:

## 3. Experiment

action_rate = -0.02

Mean Reward:
Mean Episode Length:
Tracking Lin Vel:

## 4. Comparison

What changed?

Did action smoothness improve?

Did velocity tracking get worse?

Did episode length improve?

## 5. Conclusion

Does the result support the hypothesis?
What else should be tested?
```

这里最重要的不是写一个漂亮的结论，而是如实记录数据。

## **十、今天的完成标准**

- 成功备份原始配置。
    
- 只修改 `action_rate` 一项。
    
- 成功运行第二组训练。
    
- 得到两个独立的 Checkpoint。
    
- 在 TensorBoard 中找到两个 Run。
    
- 能够分析 Reward 与动作平滑性之间的关系。
    
- 实验结束后恢复原始配置。
    

**Day 12 的核心产出：一次完整的单因素强化学习实验，以及一份 Baseline / Experiment 对比记录。**

## **最后，我对这两天有一个特别要求**

不要追求 100 次 Iteration 就能训练出一台稳定行走的 G1。

我们这两天要验证的是：

**你能不能独立完成一套强化学习实验流程。**

即使结果是：

- Reward 变低了；
    
- 机器人没有更平滑；
    
- Tracking Lin Vel 反而下降；
    

这依然是有价值的实验，因为你能够解释参数为什么产生这样的影响。

等这两天结束，我希望你不再只是说：

> 我能够运行 Unitree RL Gym。

而是能够说：

> 我基于官方 G1 PPO 工程建立了 Baseline，理解了 Reward 对策略优化的影响，并完成了一个控制动作平滑性的单因素实验。

这才是一个真正可以写入简历、也经得起技术面试追问的项目经历。

# 拔高实验：闭环运动控制与MuJoCo

**你训练出来的策略能够根据机器人实时状态，不断调整控制动作，让机器人完成指定运动任务。**

这就是我们接下来应该补充的实验。

我建议你做一个：

最推荐的实验方向

**Unitree G1 闭环运动控制与扰动恢复实验**

**Closed-loop Locomotion Control and Disturbance Rejection**

使用训练好的 G1 策略，在 MuJoCo 中完成速度指令跟踪，并通过改变运动指令和施加外部扰动，验证策略能否利用实时状态反馈调整动作、维持平衡。

## **一、什么是闭环实验？**

先理解一个最基本的区别：开环控制（Open-loop）和闭环控制（Closed-loop）。

### **1. 开环控制：给机器人一个动作，让它一直执行**

例如：

```
预先设置一组关节角度
       ↓
机器人执行
       ↓
继续执行预先设定的动作
       ↓
结束
```

这种情况下，机器人执行动作时，不会根据最新的身体状态重新调整下一步控制指令。

假设我们预先生成一段行走动作，即使机器人被推了一下、身体已经倾斜，仍然继续执行原来的动作序列。

这就是开环控制的典型问题：

**没有利用执行结果进行反馈修正。**

### **2. 闭环控制：根据机器人当前状态持续调整动作**

G1 的闭环控制过程是：

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; border: 1px solid #eaeaea; border-radius: 12px; background-color: #ffffff; max-width: 650px; box-shadow: 0 2px 12px rgba(0,0,0,0.03); margin: 10px 0; display: flex; flex-direction: column; align-items: center;">

    <!-- 1. 运动指令 Command -->
    <div style="width: 100%; background-color: #eef4fd; border-radius: 8px; padding: 16px 10px; text-align: center; box-sizing: border-box;">
        <div style="font-size: 16px; font-weight: bold; color: #111; margin-bottom: 4px;">运动指令 Command</div>
        <div style="font-size: 14px; color: #555;">例如：希望机器人以 0.5 m/s 向前走</div>
    </div>

    <!-- 箭头 -->
    <div style="color: #999; font-size: 18px; margin: 10px 0; line-height: 1;">↓</div>

    <!-- 2. 实时 Observation -->
    <div style="width: 100%; background-color: #eef4fd; border-radius: 8px; padding: 16px 10px; text-align: center; box-sizing: border-box;">
        <div style="font-size: 16px; font-weight: bold; color: #111; margin-bottom: 4px;">实时 Observation</div>
        <div style="font-size: 14px; color: #555;">关节角度、关节速度、姿态、角速度等</div>
    </div>

    <!-- 箭头 -->
    <div style="color: #999; font-size: 18px; margin: 10px 0; line-height: 1;">↓</div>

    <!-- 3. Actor / Policy -->
    <div style="width: 100%; background-color: #eef4fd; border-radius: 8px; padding: 16px 10px; text-align: center; box-sizing: border-box;">
        <div style="font-size: 16px; font-weight: bold; color: #111; margin-bottom: 4px;">Actor / Policy</div>
        <div style="font-size: 14px; color: #555;">根据当前状态生成 12 维 Action</div>
    </div>

    <!-- 箭头 -->
    <div style="color: #999; font-size: 18px; margin: 10px 0; line-height: 1;">↓</div>

    <!-- 4. PD Controller -->
    <div style="width: 100%; background-color: #f0f7f2; border-radius: 8px; padding: 16px 10px; text-align: center; box-sizing: border-box;">
        <div style="font-size: 16px; font-weight: bold; color: #111; margin-bottom: 4px;">PD Controller</div>
        <div style="font-size: 14px; color: #555;">目标关节角度 → 关节力矩</div>
    </div>

    <!-- 箭头 -->
    <div style="color: #999; font-size: 18px; margin: 10px 0; line-height: 1;">↓</div>

    <!-- 5. 机器人 / MuJoCo 物理仿真 -->
    <div style="width: 100%; background-color: #f0f7f2; border-radius: 8px; padding: 16px 10px; text-align: center; box-sizing: border-box;">
        <div style="font-size: 16px; font-weight: bold; color: #111; margin-bottom: 4px;">机器人 / MuJoCo 物理仿真</div>
        <div style="font-size: 14px; color: #555;">机器人实际产生运动</div>
    </div>

    <!-- 箭头 -->
    <div style="color: #999; font-size: 18px; margin: 10px 0; line-height: 1;">↓</div>

    <!-- 6. 反馈新的机器人状态 -->
    <div style="width: 100%; background-color: #f4f0f8; border-radius: 8px; padding: 16px 10px; text-align: center; box-sizing: border-box;">
        <div style="font-size: 16px; font-weight: bold; color: #111; margin-bottom: 4px;">反馈新的机器人状态</div>
        <div style="font-size: 14px; color: #555;">重新生成 Observation，进入下一次策略决策</div>
    </div>

</div>

数学表达：

$$ o_t\rightarrow\pi_\theta(o_t) \rightarrow a_t \rightarrow \text{Robot} \rightarrow o_{t+1} $$

这里最重要的是：

$$ \boxed{a_t=\pi_\theta(o_t)} $$

下一次 Action 不是提前写好的，而是根据机器人最新的 Observation 重新计算。

### **3. 其实你之前学过的 PPO 已经包含闭环交互**

例如：

```python
while training:

    actions = policy(obs)

    obs, rewards, dones, infos = env.step(actions)
```

这已经是策略与环境之间的闭环交互。

而 PD 本身也构成一个更底层的反馈控制回路：

$$ \tau=K_p(q_{\text{target}}-q)-K_d\dot q $$

所以要区分两个层次：

| 层次           | 反馈内容                 | 作用           |
| ------------ | -------------------- | ------------ |
| 高层 Policy 闭环 | Observation → Action | 根据机身状态调整运动策略 |
| 底层 PD 闭环     | 关节位置、速度 → Torque     | 根据关节状态跟踪目标角度 |

师兄希望你补充的实验，我更建议展示这两个回路共同作用的结果，而不只是展示 PPO 训练曲线。

## **二、为什么推荐使用 MuJoCo 做这个实验？**

官方 Unitree RL Gym 给出的流程是：

$$ Train\rightarrow Play\rightarrow Sim2Sim\rightarrow Sim2Real $$

其中，Sim2Sim 就是把在 Isaac Gym 中训练得到的策略放到 MuJoCo 中运行，检验策略是否过度依赖原仿真器的特性。

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; border: 1px solid #eaeaea; border-radius: 12px; background-color: #ffffff; max-width: 650px; box-shadow: 0 2px 12px rgba(0,0,0,0.03); margin: 10px 0; display: flex; flex-direction: column; align-items: center;">

    <!-- 标题 -->
    <div style="font-size: 18px; font-weight: bold; align-self: flex-start; margin-bottom: 20px; width: 100%;">
        你的实验路线
    </div>

    <!-- 1. Isaac Gym -->
    <div style="width: 100%; background-color: #f5f8fc; border-radius: 8px; padding: 16px 10px; text-align: center; box-sizing: border-box;">
        <div style="font-size: 16px; font-weight: bold; color: #111; margin-bottom: 4px;">Isaac Gym</div>
        <div style="font-size: 14px; color: #555;">训练得到 G1 PPO 策略</div>
    </div>

    <!-- 箭头 -->
    <div style="color: #999; font-size: 18px; margin: 10px 0; line-height: 1;">↓</div>

    <!-- 2. Checkpoint -->
    <div style="width: 100%; background-color: #f5f8fc; border-radius: 8px; padding: 16px 10px; text-align: center; box-sizing: border-box;">
        <div style="font-size: 16px; font-weight: bold; color: #111; margin-bottom: 4px;">Checkpoint</div>
        <div style="font-size: 14px; color: #555;">保存训练参数</div>
    </div>

    <!-- 箭头 -->
    <div style="color: #999; font-size: 18px; margin: 10px 0; line-height: 1;">↓</div>

    <!-- 3. Export Actor -->
    <div style="width: 100%; background-color: #f5f8fc; border-radius: 8px; padding: 16px 10px; text-align: center; box-sizing: border-box;">
        <div style="font-size: 16px; font-weight: bold; color: #111; margin-bottom: 4px;">Export Actor</div>
        <div style="font-size: 14px; color: #555;">导出 <span style="font-family: monospace; background-color: #eaeaea; padding: 2px 4px; border-radius: 4px; font-size: 13px;">policy_lstm_1.pt</span></div>
    </div>

    <!-- 箭头 -->
    <div style="color: #999; font-size: 18px; margin: 10px 0; line-height: 1;">↓</div>

    <!-- 4. MuJoCo Sim2Sim -->
    <div style="width: 100%; background-color: #f2f9f4; border-radius: 8px; padding: 16px 10px; text-align: center; box-sizing: border-box;">
        <div style="font-size: 16px; font-weight: bold; color: #111; margin-bottom: 4px;">MuJoCo Sim2Sim</div>
        <div style="font-size: 14px; color: #555;">加载策略，实时反馈和控制 G1</div>
    </div>

</div>

官方 README 明确提供了 MuJoCo 部署入口：

```python
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

并支持把自己的导出策略替换默认模型。

注意：官方 `g1.yaml` 默认使用的是：

```
policy_path: "{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/g1/motion.pt"
```

这是仓库提供的预训练策略路径。

**你如果要展示自己训练的结果，应该将它改成自己的模型：**

**1. 准备好Day 11 模型**

假设你的文件是：

```
logs/
└── g1/
    └── 时间戳_day11_baseline/
        └── model_100.pt
```

它是 PPO Checkpoint，包含 Actor、Critic 和优化器等训练状态，还不是 MuJoCo 直接使用的导出策略。

**2. Isaac Gym Play**

运行：

```
python legged_gym/scripts/play.py \
    --task=g1 \
    --load_run=你的Day11实验文件夹名称 \
    --checkpoint=100
```

这里 `load_run` 填的是 `model_100.pt` 所在文件夹的名称。

官方 `play.py` 会加载对应 Checkpoint，并自动导出 Actor。

导出文件：

```
logs/g1/exported/policies/policy_lstm_1.pt
```

这一步先观察机器人能否稳定站立、行走和跟踪指令。官方 README 也明确采用 Train → Play → Sim2Sim 的流程。

**3. MuJoCo 闭环实验**

将导出的文件路径填入：

```
deploy/deploy_mujoco/configs/g1.yaml
```

修改：

```
policy_path: "你的路径/policy_lstm_1.pt"
```

然后运行：

```
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

这样运行的就是你 Day 11 自己训练的策略，而不是官方预训练策略

```
policy_path: "{LEGGED_GYM_ROOT_DIR}/logs/g1/你的实验目录/exported/policies/policy_lstm_1.pt"
```

否则你展示的只是官方预训练模型推理，不能称为自己训练的策略在 Sim2Sim 中验证。

## **三、我建议你具体做什么闭环实验？**

我建议分成三个实验，按难度逐渐增加。不需要一开始就做得很复杂。

### **实验一：固定速度指令下的闭环行走**

必须完成 · 基础实验

目的：验证 G1 能否根据实时 Observation，持续产生 Action，并跟踪指定速度。

设置：

```
cmd = [0.5, 0.0, 0.0]
```

含义：

- 向前速度：0.5 m/s。
    
- 侧向速度：0。
    
- 偏航角速度：0。
    

实验流程：

```
输入前进速度 0.5m/s
       ↓
Policy 获取实时 Observation
       ↓
生成 Action
       ↓
转换为关节目标角度
       ↓
PD 控制
       ↓
MuJoCo 仿真
       ↓
读取新的机器人速度和姿态
       ↓
再次输入 Policy
       ↓
持续循环
```

观察：

- G1 能否稳定行走？
    
- 实际前进速度是否接近 0.5 m/s？
    
- 身体是否出现明显倾斜？
    
- 是否会摔倒？
    

如果只是设置一个固定 Action，让机器人执行，不算你要展示的策略反馈闭环。

## **实验二：改变 Command，观察速度跟踪响应**

推荐完成 · 核心实验

这是我最推荐你向师兄展示的实验。

我们不再一直输入固定的 0.5 m/s，而是中途改变目标速度。

例如：

|仿真时间|目标前进速度|
|---|---|
|0–4 s|0.3 m/s|
|4–8 s|0.5 m/s|
|8–12 s|0.3 m/s|

这里的数值是建议的实验设置，需要确认处于策略训练时的 Command 范围内。

在 `deploy_mujoco.py` 中，找到：

```python
if counter % control_decimation == 0:

    # create observation
```

在构造 Observation 之前，根据当前仿真时间更新 `cmd`。

示意：

```python
sim_time = counter * simulation_dt

if sim_time < 4.0:
    cmd[:] = [0.3, 0.0, 0.0]

elif sim_time < 8.0:
    cmd[:] = [0.5, 0.0, 0.0]

else:
    cmd[:] = [0.3, 0.0, 0.0]
```

然后原来的代码：

```python
obs[6:9] = cmd * cmd_scale
```

就会把新的运动指令输入策略。

注意：这里应该在每次策略更新时构造 Observation，而不是仅在程序启动时读取一次 Command。

### **你应该看到什么？**

理想情况下：

1. 收到 0.3 m/s 指令，机器人逐渐调整到对应运动状态。
    
2. 指令改变到 0.5 m/s，机器人通过改变关节目标角度调整步态。
    
3. 指令重新降低到 0.3 m/s，机器人再次调整运动状态。
    

这里真正要验证的是：

**机器人是否根据 Command 的改变，重新计算动作，并通过状态反馈逐步调整实际运动。**

### **必须记录一张曲线**

<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; border: 1px solid #eaeaea; border-radius: 12px; background-color: #ffffff; max-width: 700px; box-shadow: 0 2px 12px rgba(0,0,0,0.03); margin: 10px 0;">

    <!-- 标题 -->
    <div style="font-size: 16px; font-weight: bold; margin-bottom: 20px;">
        期望记录的速度跟踪曲线
    </div>

    <!-- SVG 绘图区 -->
    <div style="position: relative; width: 100%;">
        <svg viewBox="0 0 700 350" width="100%" height="100%" style="overflow: visible; font-family: -apple-system, sans-serif;">
            
            <!-- Y轴 -->
            <line x1="80" y1="50" x2="80" y2="300" stroke="#888" stroke-width="1.5" />
            <!-- X轴 -->
            <line x1="80" y1="300" x2="650" y2="300" stroke="#888" stroke-width="1.5" />
            
            <!-- Y轴文字 -->
            <text x="70" y="55" fill="#333" font-size="14" text-anchor="end">速度</text>
            <text x="70" y="120" fill="#333" font-size="16" text-anchor="end">0.5</text>
            <text x="70" y="240" fill="#333" font-size="16" text-anchor="end">0.3</text>
            
            <!-- X轴文字 -->
            <text x="80" y="330" fill="#333" font-size="16" text-anchor="middle">0s</text>
            <text x="240" y="330" fill="#333" font-size="16" text-anchor="middle">4s</text>
            <text x="440" y="330" fill="#333" font-size="16" text-anchor="middle">8s</text>
            <text x="630" y="330" fill="#333" font-size="16" text-anchor="middle">12s</text>

            <!-- Command 虚线 (橙色) -->
            <!-- 0-4s 0.3; 4-8s 0.5; 8-12s 0.3 -->
            <polyline points="80,240 240,240 240,120 440,120 440,240 630,240" 
                      fill="none" stroke="#d97706" stroke-width="5" stroke-dasharray="10, 8" stroke-linejoin="round" />

            <!-- Actual 实线 (绿色) -->
            <!-- 平滑曲线：起点0.3 -> 缓升 -> 越过4s继续升到0.5 -> 8s后下降至0.3 -->
            <path d="M 80,240 
                     C 130,240 160,220 190,210 
                     C 230,190 240,140 300,120 
                     C 350,105 400,120 440,125 
                     C 480,130 520,160 560,200 
                     C 590,230 610,240 630,240" 
                  fill="none" stroke="#10b981" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />

            <!-- 图例 -->
            <!-- Command -->
            <line x1="500" y1="60" x2="530" y2="60" stroke="#d97706" stroke-width="5" stroke-dasharray="8, 6" />
            <text x="540" y="66" fill="#333" font-size="16">Command</text>
            
            <!-- Actual -->
            <line x1="500" y1="90" x2="530" y2="90" stroke="#10b981" stroke-width="5" stroke-linecap="round" />
            <text x="540" y="96" fill="#333" font-size="16">Actual</text>

        </svg>
    </div>

    <!-- 底部注释 -->
    <div style="font-size: 14px; color: #777; margin-top: 20px;">
        示意图：橙色为目标速度，绿色为实际速度。实际响应会有延迟和波动，不要求完全重合。
    </div>

</div>

这张图比单纯展示机器人行走视频更有说服力，因为它能够量化控制效果。

**实验结果：

![[g1_velocity_tracking.png]]

但实际速度仍存在周期性波动和瞬态跟踪误差，表明策略在跟踪精度和响应特性方面仍有改进空间。
## **实验三：外部扰动下的闭环恢复**

进阶实验 · 强烈推荐

目的：

验证机器人遭受外部干扰后，能否通过反馈控制恢复原来的运动状态。

例如：

```
机器人正常向前行走
       ↓
施加一个短暂的侧向扰动
       ↓
身体发生偏移
       ↓
Observation 发生变化
       ↓
Policy 产生新的 Action
       ↓
PD 调整关节力矩
       ↓
机器人尝试恢复平衡
```

你可以在 MuJoCo 中使用 GUI 的交互扰动，或在确认机器人基座 Body ID 后，使用 `d.xfrc_applied` 对基座施加一个短时的小力，并及时清零。

### 如何实现

运行：

```
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

等 G1 在 MuJoCo 窗口中开始运动后：

1. 用鼠标左键双击机器人躯干或骨盆附近，选中对应的刚体。
    
2. 按住 `Ctrl`，同时按住鼠标右键拖动，对机器人施加外力。
    
3. 松开鼠标，结束扰动。
    

常用操作：

|操作|效果|
|---|---|
|双击鼠标左键|选中刚体|
|`Ctrl` + 鼠标右键拖动|施加垂直方向的力|
|`Ctrl + Shift` + 鼠标右键拖动|施加水平方向的力|
|`Ctrl` + 鼠标左键拖动|施加扭矩|
|`F1`|查看 MuJoCo 操作帮助|

这是 MuJoCo 原生 Viewer 的交互方式。

代码也可以直接操作

看你使用的官方文件：

`deploy/deploy_mujoco/deploy_mujoco.py`

其中有：

```
with mujoco.viewer.launch_passive(m, d) as viewer:
```

这说明它使用的是 MuJoCo 的被动式 Viewer。

而在仿真循环末尾还有：

```
viewer.sync()
```

这个操作会同步 GUI 和仿真状态，也会接收 GUI 的鼠标扰动输入。因此，当前脚本已经具备接收鼠标扰动的条件。

我建议你先做一个简单的抗扰动实验：

**G1 闭环抗扰动实验**

1. **正常行走**
    
    让 G1 使用你训练的 Day11 模型正常行走，观察它能否维持运动。
    
2. **施加外力**
    
    在机器人行走过程中，使用 `Ctrl + 鼠标右键拖动`，短暂地向侧面施加扰动。
    
3. **观察恢复**
    
    松开鼠标，观察机器人是否能恢复平衡、继续跟踪速度指令，还是发生摔倒。
    
4. **记录结果**
    
    保存扰动前、扰动中、扰动后的三个阶段的视频或截图。
    

这里需要注意：鼠标扰动适合先做定性验证，但施加的力大小和持续时间不容易精确控制。后续如果要形成严谨的实验结果，可以用 `d.xfrc_applied` 在代码中设置固定大小、固定持续时间的外力。

不建议第一次就施加很大的推力。

你要观察：

- 受到扰动后身体是否倾斜？
    
- Policy 是否产生新的控制动作？
    
- 机器人能否恢复？
    
- 恢复需要多长时间？
    
- 是否出现摔倒？
    

这个实验体现的是：

**Disturbance Rejection（扰动抑制能力）和闭环反馈控制能力。**

它也和你之前学习的 Domain Randomization 中的 `push_robots` 直接对应。

## **四、具体应该从哪个代码开始？**

我已经核对了官方目前的 MuJoCo 部署实现。

重点文件：

`deploy/deploy_mujoco/deploy_mujoco.py`

以及：

`deploy/deploy_mujoco/configs/g1.yaml`

官方 G1 MuJoCo 配置中：

```
simulation_dt: 0.002

control_decimation: 10

action_scale: 0.25

num_actions: 12

num_obs: 47
```

因此：

$$ dt_{\text{physics}}=0.002s $$

$$ dt_{\text{policy}}=10\times0.002=0.02s $$

也就是：


| 控制层         | 频率     |
| ----------- | ------ |
| MuJoCo 物理仿真 | 500 Hz |
| PD 力矩计算     | 500 Hz |
| Actor 推理    | 50 Hz  |

注意，这里和 Isaac Gym 的 5ms 物理步不同：MuJoCo 当前配置使用 2ms 物理步，但策略仍然每 20ms 更新一次。

在 `deploy_mujoco.py` 中，核心闭环代码是：

```python
while viewer.is_running():

    tau = pd_control(
        target_dof_pos,
        d.qpos[7:],
        kps,
        np.zeros_like(kds),
        d.qvel[6:],
        kds
    )

    d.ctrl[:] = tau

    mujoco.mj_step(m, d)

    counter += 1

    if counter % control_decimation == 0:

        # 读取当前关节状态、姿态和速度

        # 构造 Observation

        obs_tensor = torch.from_numpy(
            obs
        ).unsqueeze(0)

        # Policy 根据最新状态推理

        action = policy(
            obs_tensor
        ).detach().numpy().squeeze()

        # 更新目标关节角度

        target_dof_pos = (
            action * action_scale
            + default_angles
        )
```

这段代码就是你要重点展示的高层 Policy 与底层 PD 的双层闭环。

## **五、实验最终需要产出什么？**

我建议你最终整理出四个成果。

|产出|内容|作用|
|---|---|---|
|1. 运行视频|G1 在 MuJoCo 中跟踪不同 Command|直观展示闭环控制|
|2. 速度曲线|Command vs Actual Velocity|量化跟踪能力|
|3. 扰动恢复曲线|扰动时间、姿态偏差、恢复过程|展示鲁棒性|
|4. 实验报告|参数、实验条件、结果、分析|可写进 GitHub 和简历|

建议至少计算三个指标：

**速度跟踪 RMSE：**

$$ RMSE_v= \sqrt{ \frac{1}{N} \sum_{t=1}^{N} (v_t^{cmd}-v_t^{actual})^2 } $$

**扰动恢复时间：**

从扰动发生开始，到机器人恢复到预先设定的跟踪误差范围并持续稳定一段时间所需的时间。

**摔倒率：**

$$ FallRate= \frac{N_{\text{fall}}}{N_{\text{episodes}}} $$

这些指标不需要一开始就全部做得很复杂。先完成 Command 跟踪，再增加扰动测试。

注意，MuJoCo 中的实际基座速度需要和 Command 转换到相同坐标系再比较，不能直接将不同坐标系的速度分量混在一起。

## **六、我建议你的实际执行顺序**

### **闭环实验实施路线**

1.  **第一步：确认策略模型**
    
    确认自己训练得到的 `policy_lstm_1.pt` 是否存在，并且先在 Isaac Gym 的 Play 中验证策略表现。
    
2. **第二步：Sim2Sim 基础运行**
    
    修改 `g1.yaml` 的 `policy_path`，将自己的导出模型接入 MuJoCo。首先保持默认 Command，确认机器人能够稳定运行。
    
3. **第三步：Command 阶跃实验**
    
    修改 `deploy_mujoco.py`，按时间改变前进速度指令，记录 Command 与实际速度。
    
4. **第四步：扰动恢复实验**
    
    在机器人行走过程中施加小幅外部扰动，记录身体姿态和恢复过程。
    
1.  **最终产出：实验分析**
    
    绘制速度跟踪曲线、恢复曲线，整理视频和定量指标。
    

如果你还没有训练出稳定行走的自有策略，可以先用官方预训练模型检查 MuJoCo 闭环流程，但必须明确标注为官方预训练策略验证，不能作为你自己训练结果的证明。

此外，我建议先完成 Sim2Sim，不要直接上真实 G1。实机部署需要经过单独的安全验证、限幅和急停检查，并由熟悉机器人硬件的人监督。

## **七、这个实验写进简历会有什么价值？**

完成之后，你原来的项目描述就可以升级为：

基于 Unitree RL Gym 完成 G1 人形机器人 PPO 强化学习训练与控制研究，分析 Observation/Action、PD 控制及 Reward 机制，自主开展奖励参数对比实验，并将训练策略部署至 MuJoCo，通过速度指令跟踪与外部扰动恢复实验验证策略的闭环控制能力。

这比单纯写“运行宇树机器人训练代码”更有技术含量。

**我现在最建议你先做的不是马上修改代码，而是检查一个关键条件：**

你目前有没有一个自己训练得到、并且已经在 Isaac Gym 的 `play.py` 中验证能够稳定行走的 `policy_lstm_1.pt`？

这是决定我们能否直接进入自有策略闭环实验的关键。官方预训练策略可以先用来验证整套 Sim2Sim 管线，但最终实验最好使用你自己的策略。

这里说的“输出页面”是机器人运行时弹出的 **MuJoCo 3D 窗口**，不是终端打印日志的黑色窗口。

### **1. 具体怎么操作？**

运行：

```
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

等 G1 在 MuJoCo 窗口中开始运动后：

1. 用鼠标左键双击机器人躯干或骨盆附近，选中对应的刚体。
    
2. 按住 `Ctrl`，同时按住鼠标右键拖动，对机器人施加外力。
    
3. 松开鼠标，结束扰动。
    

常用操作：

|操作|效果|
|---|---|
|双击鼠标左键|选中刚体|
|`Ctrl` + 鼠标右键拖动|施加垂直方向的力|
|`Ctrl + Shift` + 鼠标右键拖动|施加水平方向的力|
|`Ctrl` + 鼠标左键拖动|施加扭矩|
|`F1`|查看 MuJoCo 操作帮助|

这是 MuJoCo 原生 Viewer 的交互方式。

MuJoCo Documentation+1

### **2. 为什么你当前的代码可以直接操作？**

看你使用的官方文件：

`deploy/deploy_mujoco/deploy_mujoco.py`

其中有：

```
with mujoco.viewer.launch_passive(m, d) as viewer:
```

这说明它使用的是 MuJoCo 的被动式 Viewer。

而在仿真循环末尾还有：

```
viewer.sync()
```

这个操作会同步 GUI 和仿真状态，也会接收 GUI 的鼠标扰动输入。因此，当前脚本已经具备接收鼠标扰动的条件。

### **3. 你应该怎么做这个实验？**

我建议你先做一个简单的抗扰动实验：

**G1 闭环抗扰动实验**

1. **正常行走**
    
    让 G1 使用你训练的 Day11 模型正常行走，观察它能否维持运动。
    
2. **施加外力**
    
    在机器人行走过程中，使用 `Ctrl + 鼠标右键拖动`，短暂地向侧面施加扰动。
    
3. **观察恢复**
    
    松开鼠标，观察机器人是否能恢复平衡、继续跟踪速度指令，还是发生摔倒。
    
4. **记录结果**
    
    保存扰动前、扰动中、扰动后的三个阶段的视频或截图。
    

这里需要注意：鼠标扰动适合先做定性验证，但施加的力大小和持续时间不容易精确控制。后续如果要形成严谨的实验结果，可以用 `d.xfrc_applied` 在代码中设置固定大小、固定持续时间的外力。

**我的建议是：先用 GUI 做第一次测试，不要急着改代码。** 这样你能直观看到 Policy 是否根据新的机器人状态调整 Action，并通过 PD 控制让机器人恢复平衡。这才是我们想验证的闭环反馈过程。