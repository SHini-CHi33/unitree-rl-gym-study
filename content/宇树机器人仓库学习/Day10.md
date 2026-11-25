# 今日阅读任务

今日主题：Train、Checkpoint、Play 与策略推理
## **一、今天的核心问题**

前两天我们已经理解：

- Day 8：PPO 如何采集数据。
    
- Day 9：PPO 如何更新策略网络。
    

今天我们研究：

**训练结束后，神经网络参数保存在哪里？Play 是如何找到训练好的模型，并让 G1 再次走起来的？**

特别注意：

**Play 是策略推理，不是重新训练。**

## **二、今天需要阅读的文件**

|文件|具体函数|
|---|---|
|`legged_gym/scripts/train.py`|`train()`|
|`legged_gym/scripts/play.py`|`play()`|
|`legged_gym/utils/helpers.py`|`export_policy_as_jit()`、`get_load_path()`|
|`rsl_rl/runners/on_policy_runner.py`|`save()`、`load()`、`get_inference_policy()`|
|`rsl_rl/modules/actor_critic.py`|`act_inference()`|
|`rsl_rl/modules/actor_critic_recurrent.py`|`act_inference()`、`Memory`|

源码链接：

- [Unitree train.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/scripts/train.py?utm_source=chatgpt.com)
    
- [Unitree play.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/scripts/play.py?utm_source=chatgpt.com)
    
- [Unitree helpers.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/utils/helpers.py?utm_source=chatgpt.com)
    
- [RSL-RL OnPolicyRunner](https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/runners/on_policy_runner.py?utm_source=chatgpt.com)
    

## **三、第一部分：Train.py 实际做了什么？**

打开：

`legged_gym/scripts/train.py`

官方代码非常短：

```
def train(args):    env, env_cfg = task_registry.make_env(        name=args.task,        args=args    )    ppo_runner, train_cfg = (        task_registry.make_alg_runner(            env=env,            name=args.task,            args=args        )    )    ppo_runner.learn(        num_learning_iterations=train_cfg.runner.max_iterations,        init_at_random_ep_len=True    )
```

这里实际上只有三个步骤：

1. 创建机器人环境。
    
2. 创建 PPO Runner。
    
3. 启动 `learn()`。
    

真正复杂的训练工作都在 RSL-RL 的 `OnPolicyRunner.learn()` 和 `PPO.update()` 中完成。

因此：

```
ppo_runner.learn(...)
```

并不是一个简单的函数调用，它启动了此前 Day 8、Day 9 学习的整个训练闭环。

## **四、第二部分：Checkpoint 是怎么保存的？**

打开：

`rsl_rl/runners/on_policy_runner.py`

找到：

```
def save(self, path, infos=None):    torch.save({        'model_state_dict':            self.alg.actor_critic.state_dict(),        'optimizer_state_dict':            self.alg.optimizer.state_dict(),        'iter':            self.current_learning_iteration,        'infos':            infos,    }, path)
```

GitHub

这里保存了四类信息：

|内容|用途|
|---|---|
|`model_state_dict`|保存神经网络参数|
|`optimizer_state_dict`|保存优化器状态|
|`iter`|保存当前训练迭代次数|
|`infos`|保存附加信息|

为什么要保存 Optimizer State？

因为恢复训练时，不只是需要恢复 Actor 和 Critic 的参数，还可能需要恢复 Adam 优化器的内部状态，例如一阶、二阶动量估计。

继续看：

```
def load(self, path, load_optimizer=True):    loaded_dict = torch.load(path)    self.alg.actor_critic.load_state_dict(        loaded_dict['model_state_dict']    )    if load_optimizer:        self.alg.optimizer.load_state_dict(            loaded_dict['optimizer_state_dict']        )    self.current_learning_iteration = (        loaded_dict['iter']    )
```

这样就可以恢复训练。

GitHub

### **Checkpoint 保存在哪里？**

当前官方 README 给出的默认结果目录形式是：

```
logs/
└── g1/
    └── 时间戳_run_name/
        ├── model_50.pt
        ├── model_100.pt
        ├── model_150.pt
        └── ...
```

实际保存间隔由 `save_interval` 控制。当前基础配置默认每 50 次 Learning Iteration 检查一次保存，并且训练调用结束时还会保存最终模型。

GitHub+1

## **五、第三部分：Play.py 如何加载训练好的策略？**

打开：

`legged_gym/scripts/play.py`

找到：

```
def play(args):    env_cfg, train_cfg = (        task_registry.get_cfgs(name=args.task)    )
```

这里先加载环境配置和训练配置。

接着：

```
env_cfg.env.num_envs = min(    env_cfg.env.num_envs,    100)env_cfg.terrain.num_rows = 5env_cfg.terrain.num_cols = 5env_cfg.terrain.curriculum = Falseenv_cfg.noise.add_noise = Falseenv_cfg.domain_rand.randomize_friction = Falseenv_cfg.domain_rand.push_robots = Falseenv_cfg.env.test = True
```

GitHub

这里有一个非常重要的工程思想：

Play 是用于观察训练结果的测试程序，因此会简化环境设置：

- 并行环境最多 100 个。
    
- 关闭 Observation Noise。
    
- 关闭摩擦随机化。
    
- 关闭随机推机器人。
    
- 关闭地形 Curriculum。
    

接下来：

```
env, _ = task_registry.make_env(    name=args.task,    args=args,    env_cfg=env_cfg)
```

拿到环境初始 Observation：

```
obs = env.get_observations()
```

然后：

```
train_cfg.runner.resume = Trueppo_runner, train_cfg = (    task_registry.make_alg_runner(        env=env,        name=args.task,        args=args,        train_cfg=train_cfg    ))
```

注意这里：

```
train_cfg.runner.resume = True
```

意味着需要加载已有训练模型，而不是从随机初始化开始训练。

官方 README 也说明，Play 默认加载对应实验目录中最近一次运行的模型，可以通过 `load_run` 和 `checkpoint` 指定其他模型。

GitHub

## **六、第四部分：get_inference_policy() 和 act_inference()**

现在来到一个非常重要的区别。

在 `on_policy_runner.py` 中找到：

```
def get_inference_policy(self, device=None):    self.alg.actor_critic.eval()    if device is not None:        self.alg.actor_critic.to(device)    return self.alg.actor_critic.act_inference
```

这里做了两件事：

1. 将网络切换到 Eval Mode。
    
2. 返回 Actor 的推理接口。
    

GitHub

为什么不是继续调用 `act()`？

回顾 Day 9：

训练时：

```
def act(self, observations):    self.update_distribution(observations)    return self.distribution.sample()
```

这里会从高斯分布中采样。

而推理时：

```
def act_inference(self, observations):    actions_mean = self.actor(observations)    return actions_mean
```

直接返回策略分布的均值。

GitHub

所以：

|阶段|Action 生成方式|
|---|---|
|Train|从 Gaussian Policy 采样|
|Play|使用 Actor 输出的均值进行推理|

这能让推理过程具有更强的确定性。

### **但 G1 还有一个 LSTM 的特殊情况**

当前 G1 使用：

```
policy_class_name = "ActorCriticRecurrent"
```

因此 `act_inference()` 实际会先经过 LSTM：

```
def act_inference(self, observations):    input_a = self.memory_a(observations)    return super().act_inference(        input_a.squeeze(0)    )
```

也就是说：

```
47维 Observation
       ↓
Actor LSTM
       ↓
64维隐藏特征
       ↓
Actor MLP
       ↓
12维 Action Mean
```

而 LSTM 的隐藏状态会在推理过程中持续更新。

这里我给你留一个值得检查的代码细节：

当前官方 `play.py` 展示的主循环是直接调用 `policy(obs)`，之后执行 `env.step(actions)`，没有显式调用 `actor_critic.reset(dones)`。

而训练时，`PPO.process_env_step()` 明确会根据 Done 重置 LSTM 状态。

因此阅读 Play 时，还要检查 Episode Reset 之后 LSTM 状态如何处理。不要默认环境 Reset 就一定会自动清除策略网络的隐藏状态。

## **七、第五部分：导出的 policy_lstm_1.pt 是什么？**

最后进入：

`legged_gym/utils/helpers.py`

找到：

```
def export_policy_as_jit(actor_critic, path):    if hasattr(actor_critic, 'memory_a'):        exporter = PolicyExporterLSTM(actor_critic)        exporter.export(path)    else:        ...
```

这段代码区分两种情况：

- MLP Policy：导出 `policy_1.pt`。
    
- LSTM Policy：导出 `policy_lstm_1.pt`。
    

当前 G1 是 LSTM，因此会导出：

```
policy_lstm_1.pt
```

继续看：

```
class PolicyExporterLSTM(torch.nn.Module):    def __init__(self, actor_critic):        self.actor = copy.deepcopy(            actor_critic.actor        )        self.memory = copy.deepcopy(            actor_critic.memory_a.rnn        )
```

它只复制 Actor 侧的：

- LSTM；
    
- Actor MLP。
    

而不是完整复制 Actor-Critic 和 PPO Optimizer。

再看：

```
self.register_buffer(    'hidden_state',    torch.zeros(...))self.register_buffer(    'cell_state',    torch.zeros(...))
```

这两个变量就是 LSTM 的隐藏状态和细胞状态。

在前向推理中：

```
def forward(self, x):    out, (h, c) = self.memory(        x.unsqueeze(0),        (self.hidden_state, self.cell_state)    )    self.hidden_state[:] = h    self.cell_state[:] = c    return self.actor(out.squeeze(0))
```

这意味着导出的策略文件不仅包含神经网络参数，还包含了维护 LSTM 记忆状态的逻辑。

另外还提供：

```
def reset_memory(self):    self.hidden_state[:] = 0.    self.cell_state[:] = 0.
```

这个接口在 Sim2Sim 部署中很重要：Episode 结束、需要重置策略记忆时，可以调用它。

GitHub

## **八、Checkpoint 和导出策略有什么区别？**

这是今天最重要的知识总结。

|对比|`model_*.pt`|`policy_lstm_1.pt`|
|---|---|---|
|Actor 参数|有|有|
|Critic 参数|有|无|
|Optimizer 状态|有|无|
|训练迭代信息|有|无|
|Actor LSTM|有|有|
|Critic LSTM|有|无|
|主要用途|恢复训练、评估|部署推理|

可以理解成：

```
                 训练完成
                    |
                    v
              model_10000.pt
                    |
           ┌────────┴────────┐
           |                 |
           v                 v
      继续训练              导出 Actor
           |                 |
           v                 v
      Resume Training   TorchScript
                             |
                             v
                     policy_lstm_1.pt
                             |
                             v
                      MuJoCo 推理
```

官方 README 明确说明，导出的 RNN 策略文件可以配置到 MuJoCo 部署程序中使用。

GitHub

## **九、今天的引导问题**

1. 为什么 `model_*.pt` 需要保存 Optimizer State，而 `policy_lstm_1.pt` 不需要？
    
2. 为什么训练时使用 Gaussian Distribution 采样 Action，而 Play 时通常直接使用 Action Mean？
    
3. 为什么当前 G1 的导出文件是 `policy_lstm_1.pt`，而不是 `policy_1.pt`？
    
4. `PolicyExporterLSTM` 为什么需要同时保存 `hidden_state` 和 `cell_state`？
    
5. 为什么训练和部署时，都必须正确处理 LSTM Hidden State 的 Episode Reset？
    

## **十、今日实践（10–15 分钟）**

今天建议你做一个真正的文件检查。

首先打开你自己的训练日志目录：

```
logs/
└── g1/
    └── 你的实验目录/
```

找一个已有的 Checkpoint。

然后找到：

```
exported/
└── policies/
    └── policy_lstm_1.pt
```

如果你此前没有执行过 Play 导出，先检查已有目录，不需要重新训练。

再运行：

```
python legged_gym/scripts/play.py --task=g1
```

官方 README 给出了这个启动方式。

GitHub

运行期间观察：

- 是否成功加载模型；
    
- 是否输出导出策略的路径；
    
- 是否能够正常调用 Actor 推理；
    
- 机器人是否能在仿真中执行动作。
    

你不需要判断今天的策略一定能稳定行走，因为这取决于已经训练的 Checkpoint。

**今天主要验证的是：训练好的策略可以脱离 PPO Update 流程，独立进行推理。**

## **十一、今天的完成标准**

- 能解释 `train.py` 和 `play.py` 的职责区别。
    
- 能解释 `model_*.pt` 保存的内容。
    
- 能解释 `get_inference_policy()` 的作用。
    
- 能解释 `act()` 与 `act_inference()` 的区别。
    
- 能解释 LSTM 导出策略为什么需要隐藏状态。
    
- 能找到自己的训练 Checkpoint 和导出策略文件。