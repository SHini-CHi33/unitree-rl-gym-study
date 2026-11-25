# 今日阅读任务
## 主题：训练入口是如何启动整个强化学习系统的？

今天不研究 PPO 数学原理，也不研究机器人动作控制。

只解决一个问题：

当你执行 `python legged_gym/scripts/train.py --task=g1` 时，程序究竟是如何找到 G1 环境、创建机器人、初始化 PPO，并开始训练的？

## 1. 今天需要阅读的文件

按顺序打开：

|文件|今天重点|
|---|---|
|`legged_gym/scripts/train.py`|训练入口|
|`legged_gym/envs/__init__.py`|G1 任务注册|
|`legged_gym/utils/task_registry.py`|环境及 PPO Runner 创建|
|`legged_gym/utils/helpers.py`|命令行参数和配置覆盖|

代码链接：

- [train.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/scripts/train.py?utm_source=chatgpt.com)
    
- [envs/__init__.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/__init__.py?utm_source=chatgpt.com)
    
- [task_registry.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/utils/task_registry.py?utm_source=chatgpt.com)
    
- [helpers.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/utils/helpers.py?utm_source=chatgpt.com)
    

## 2. 先看最重要的代码

打开 `train.py`，你会看到整个训练程序实际上只有几个关键步骤：

```python
def train(args):

    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
```



这里有三个核心动作：

1. `make_env()`：创建机器人仿真环境。
    
2. `make_alg_runner()`：创建 PPO 训练器。
    
3. `learn()`：开始强化学习训练。
    

这就是今天需要理解的主线。

## 3. 把代码调用关系拆开

train.py

程序入口

`get_args()`

envs/init.py

导入时注册任务

`task_registry.register("g1", ...)`

task_registry.make_env()

查找 G1 类、读取配置、创建环境

`G1Robot(...)`

task_registry.make_alg_runner()

读取 PPO 配置，初始化训练器

`OnPolicyRunner(...)`

ppo_runner.learn()

进入训练循环

Rollout → Advantage → PPO Update

### 重点 A：为什么代码里没有直接创建 G1？

你会发现，`train.py` 里面没有：

```
env = G1Robot(...)
```

而是：

```
env = task_registry.make_env(    name=args.task,    args=args)
```

原因是项目采用了任务注册机制。

在 `envs/__init__.py` 中，官方提前注册了：

```
task_registry.register(    "g1",    G1Robot,    G1RoughCfg(),    G1RoughCfgPPO())
```

这相当于建立了一张映射表：

```
"g1"
  ├── G1Robot
  ├── G1RoughCfg
  └── G1RoughCfgPPO
```

所以当 `args.task == "g1"` 时，程序就能找到对应的机器人环境类和配置类。



### 重点 B：make_env 到底做了什么？

进入 `task_registry.py` 的：

```
def make_env(...)
```

重点追踪以下步骤：

- `get_task_class(name)`：根据任务名称找到环境类。
    
- `get_cfgs(name)`：取得环境配置。
    
- `update_cfg_from_args(...)`：用命令行参数覆盖部分配置。
    
- `set_seed(...)`：设置随机种子。
    
- `parse_sim_params(...)`：解析物理仿真参数。
    
- `task_class(...)`：实例化机器人环境。
    

最终返回：

```
return env, env_cfg
```

注意，返回的不仅仅是环境，还有环境配置。


### 重点 C：PPO Runner 是如何创建的？

进入：

```
def make_alg_runner(...)
```

找到这一句：

```
runner = OnPolicyRunner(    env,    train_cfg_dict,    log_dir,    device=args.rl_device)
```

这里有三个很重要的输入：

- `env`：刚刚创建的机器人环境。
    
- `train_cfg_dict`：PPO 训练配置。
    
- `device`：强化学习计算设备。
    

之后，如果开启 `resume`，会加载之前的模型；否则直接返回初始化好的 Runner。

最后由 `train.py` 调用：

```
ppo_runner.learn(...)
```

开始训练。

# Python 中 `args` (Arguments) 的核心作用与原理

## 📌 一句话总结
`args` 是你在终端命令行敲入的所有指令的**“打包小票”**。它实现了代码与配置的**彻底解耦**。

---

## 🎯 核心作用（为什么大型项目必用）

### 1. 免除“改代码”的痛苦（解耦）
*   **传统硬编码**：`def func(size=256):`。每次想换参数，都要打开代码、找到位置、修改数字、保存、重新运行。
*   **使用 args**：代码写死 `def func(size=args.size):`。你只需要在终端敲 `--size 512` 即可瞬间切换，**一行代码都不用改**。

### 2. 避免“参数位置传错”的灾难
*   大型项目的函数可能有 20 个参数（比如物理引擎的重力、摩擦、并行数等）。
*   通过 `args` 传参，使用的是**关键字传参**（`func(gravity=args.gravity, num_envs=args.num_envs)`），哪个参数对应哪个名字，绝对不会传错顺序。

### 3. 实验管理与复现（科研必备）
*   通过命令行传入参数，每次实验产生的配置（比如 `--num_envs 1024 --lr 3e-4`）可以被系统自动记录下来，保存为 `config.yaml`。
*   方便你在 TensorBoard 上对比不同参数（Ablation Study），也方便把代码发给别人（对方只需复制你的终端命令即可复现）。

---

## 💻 底层实现原理

在 Python 中，通常使用标准库 `argparse` 来生成和管理 `args`：

```python
import argparse

# 1. 实例化解析器
parser = argparse.ArgumentParser(description="强化学习训练脚本")

# 2. 注册参数（定义默认值）
parser.add_argument("--task", type=str, default="h1", help="机器人型号")
parser.add_argument("--num_envs", type=int, default=1024, help="并行环境数")
parser.add_argument("--headless", action="store_true", help="是否无头模式")

# 3. 解析终端输入，生成“小票” args
args = parser.parse_args()

# 4. 在代码任何地方调用
print(args.task)       # 输出: h1
print(args.num_envs)   # 输出: 1024

```

## 🚀 在强化学习/深度学习项目中的实战

以 `unitree_rl_gym` 为例：

```python

python legged_gym/scripts/train.py --task=h1 --headless --num_envs=1024 --run_name test

def train(args):

    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
```
这段命令会在train.py代码内部被解析为：

- `args.task = 'h1'`
    
- `args.headless = True`
    
- `args.num_envs = 1024`

而在legged_gym/envs/__init__.py中，官方提前配置好了

```python

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())

task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())

task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())

task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
```

## 最后传入--task = h1之后发生了什么

这是一个极其经典的软件设计模式——**“工厂模式（Factory Pattern）”**。在 `legged_gym` 这种大型框架里，主程序 `train.py` 是绝对不允许直接 `import H1Robot` 的（否则每次加一个新机器人，主程序都要改代码）。

`task_registry` 就是一个“**注册中心（字典工厂）**”。当你传入 `--task=h1` 后，代码内部发生了一场极其优雅的“魔法”。

我们分 **三个步骤**，把这套操作扒得干干净净：

---

### 第一步：建档案（注册阶段）

在你运行 `train.py` 的瞬间，Python 会执行 `import legged_gym.envs`。此时，框架会默默打开 `legged_gym/envs/__init__.py` 这个文件。

在里面，代码长这样：

```python

from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
### 核心动作：在字典里存下 H1 的名字



task_registry.register(
    name='h1', 
    task_class=H1Robot,          # 存下环境的类（图纸）
    task_cfg=H1RoughCfg,         # 存下环境的配置（比如重力、摩擦）
    train_cfg=H1RoughCfgPPO       # 存下算法配置（比如 PPO 的 clip 参数）
)
```

_此时，`task_registry` 内部就有了一个字典：`{"h1": [H1Robot, H1RoughCfg, H1RoughCfgPPO]}`。_

---

### 第二步：查字典（解析阶段）

回到 `train.py` 的入口处，代码读取到了你的命令 `--task=h1`。

```python

args.task = 'h1'

主程序拿着 `'h1'` 这个字符串，去呼叫注册中心：


### 制造环境

env, env_cfg = task_registry.make_env(name=args.task, args=args)
```
---

### 第三步：招兵买马（实例化阶段）

这是最精彩的地方。我们点开 `legged_gym/utils/task_registry.py` 里的 `make_env` 函数（我精简了核心逻辑）：

```python

class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    def register(self, name, task_class, task_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = task_cfg
        self.train_cfgs[name] = train_cfg
    def make_env(self, name, args):
    
        # 1. 拿图纸：根据 'h1' 拿到 H1Robot 类和 H1RoughCfg
        env_class = self.task_classes[name]
        env_cfg = self.env_cfgs[name]
        
        # 2. 改图纸：用 args 覆盖默认配置
        env_cfg = self._parse_cfg(args, env_cfg) 
        
        # 3. 造实体：把配置喂给类，生成真正的 H1 环境对象！
        env = env_class(cfg=env_cfg, args=args)
        
        return env, env_cfg
        
# make global task registry(实例化) 后面在train.py中直接导入实例使用 from legged_gym.utils import get_args, task_registry

task_registry = TaskRegistry()  
```
---

### 💡 核心逻辑总结

1. **防止代码污染**：`train.py` 根本不知道 H1 是什么东西，它只认识字符串 `'h1'`。
    
2. **动态实例化**：`task_registry` 通过字典查表，把 `H1Robot` 这张图纸找出来，然后把 `H1RoughCfg` 里的参数（重力、奖励权重）塞进去，最后 `new` 出一个真正的环境实例。
    
3. **配置覆盖**：如果你在命令行敲了 `--num_envs 1024`，`make_env` 里面会有一行代码，强行把 `H1RoughCfg.env.num_envs` 改成 `1024`，然后再实例化。



# 今日问题

**问题 1： `train.py` 中为什么要先执行 `from legged_gym.envs import *`？它和 `task_registry.register()` 有什么关系？

答：train中利用注册制度（`task_registry.register()`）引入task，必须先导入envs，envs内包含h1、g2等模型、环境及其训练器的参数和信息(`H1Robot, H1RoughCfg(), H1RoughCfgPPO()`)，这样在--task=h1，导入时，task_ registry. `register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())`才能正确索引

#### 💡GPT的修正与补充

理解基本正确。

这里涉及 Python 模块导入的一个重要知识点：

**Python 导入模块时，不仅可以使用模块中的类和变量，还会执行模块顶层的代码。**

例如：

```python
from legged_gym.envs import *
```

会触发：

```text
legged_gym/envs/__init__.py
```

的执行。

在 __init__.py 中，程序会：

1. 导入不同机器人的环境类。
2. 导入对应的环境配置类。
3. 导入对应的 PPO 训练配置类。
4. 通过 task_registry.register() 注册任务。

例如：

```python
task_registry.register(
    "h1",
    H1Robot,
    H1RoughCfg(),
    H1RoughCfgPPO()
)

task_registry.register(
    "g1",
    G1Robot,
    G1RoughCfg(),
    G1RoughCfgPPO()
)
```

可以把注册过程理解成建立三张映射表：

```python
task_classes["h1"] = H1Robot

env_cfgs["h1"] = H1RoughCfg()

train_cfgs["h1"] = H1RoughCfgPPO()
```

这样后续就可以通过任务名称查找对应的环境和配置。

### 核心知识点

- Python 模块导入机制
- 模块导入时的顶层代码执行
- 注册表模式（Registry Pattern）
- 初始化副作用（Import Side Effect）

---
**问题 2： 如果执行：

```
python legged_gym/scripts/train.py --task=g1
```

程序是如何知道应该使用 `G1Robot`，而不是 `H1Robot` 的？

答：--task=g1，g1传回train.py中的arg.task，而envs/ __init__ . py中，官方注册好了，

```python
task_ registry. register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())

task_ registry. register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
```

所以task_registry会直接传回对应环境和PPO训练器


#### 💡GPT的修正与补充

理解正确，但需要明确：

**注册表负责查找对应的类和配置，真正创建环境是在 make_env() 中完成的（用make_env() 实例化环境）。**

完整调用流程：

```text
命令行输入 --task=g1
          |
          v
       get_args()
          |
          v
       args.task
          |
          v
 task_registry.make_env(name=args.task)
          |
          v
 get_task_class("g1")
          |
          v
       G1Robot
          |
          v
   实例化 G1 环境
```

Task_Registry 内部主要维护三个字典：

```python
self.task_classes = {}
self.env_cfgs = {}
self.train_cfgs = {}
```

分别保存：

- task_classes：环境类
- env_cfgs：环境配置
- train_cfgs：训练配置

### 核心知识点

--task=g1 并不是直接创建 G1Robot，而是通过任务名称查询注册表，找到 G1Robot 类和对应配置，再实例化环境。

---


**问题 3： `make_env()` 返回的 `env` 和 `env_cfg` 分别是什么？为什么需要同时返回？

答：env是环境模型，而env_cgf是环境配置，记录了环境的摩擦系数、光照等参数

#### 💡GPT的修正与补充

理解基本正确，需要补充


##### 1. env 和 env_cfg 分别是什么？

**env：环境实例（Environment Instance）

env 是根据机器人环境类和配置创建出来的实际仿真环境，负责运行环境和维护动态状态。

主要包括：
- 机器人模型及仿真实例。
- 当前 Observation。
- 当前 Action。
- 当前关节角度和角速度。
- 当前机身姿态和速度。
- Reward 计算。
- Episode 状态与 Reset。
- 环境交互逻辑。

**env_cfg：环境配置对象（Environment Configuration）

env_cfg 是描述环境应该如何创建和运行的参数集合。

主要包括：

| 配置类别 | 具体内容 |
|---|---|
| env | 并行环境数量、Observation 维度、Action 维度 |
| terrain | 地形类型、摩擦系数 |
| commands | 机器人速度指令范围 |
| init_state | 初始位置、姿态、关节角度 |
| control | PD 参数、Action Scale、Decimation |
| asset | 机器人 URDF、碰撞设置 |
| domain_rand | 摩擦系数、质量随机化、外力扰动 |
| rewards | 奖励函数及对应权重 |
| noise | Observation 噪声 |
| sim | 仿真时间步长、重力、PhysX 参数 |

##### 2. 为什么要将 cfg 和 env 分离？

核心原因：**将环境运行逻辑与具体配置参数分离，降低代码耦合，提高代码复用性和实验灵活性。**

假设把所有参数都直接写在环境类中：

~~~python
class G1Robot:

    def __init__(self):
        self.num_envs = 4096
        self.num_actions = 12
        self.action_scale = 0.25
        self.decimation = 4
        self.friction = 0.8
~~~

如果增加 H1 机器人，或者希望修改不同机器人的控制参数，就可能需要不断修改环境类。

将配置独立出来：

~~~python
class G1RoughCfg:

    class env:
        num_envs = 4096
        num_actions = 12

    class control:
        action_scale = 0.25
        decimation = 4
~~~

创建环境时传入对应配置：

~~~python
env_g1 = G1Robot(cfg=G1RoughCfg())
env_h1 = H1Robot(cfg=H1RoughCfg())
~~~

这样环境实现和具体参数就可以分开管理。

##### 3. Unitree RL Gym 中的实际代码关系

###### （1）task_registry.py

在 make_env() 中：

~~~python
env_cfg, _ = self.get_cfgs(name)

env_cfg, _ = update_cfg_from_args(
    env_cfg, None, args
)

env = task_class(
    cfg=env_cfg,
    sim_params=sim_params,
    physics_engine=args.physics_engine,
    sim_device=args.sim_device,
    headless=args.headless
)

return env, env_cfg
~~~

这里发生了三个步骤：

1. 根据 task 名称获取对应的环境配置。
2. 根据命令行参数更新配置。(如在终端中输入python legged_gym/scripts/train.py --task=h1  ==🔴--num_envs=1024==)
3. 将配置传入环境类，创建实际环境实例。

###### （2）legged_robot.py

在环境初始化函数中：

~~~python
def __init__(self, cfg, sim_params, ...):

    self.cfg = cfg

    self.sim_params = sim_params

    self._parse_cfg(self.cfg)

    ...

    self._init_buffers()

    self._prepare_reward_function()
~~~

关键是：

~~~python
self.cfg = cfg
~~~

这意味着环境实例内部也保存着配置对象。

因此，后续环境运行时可以通过：

~~~python
self.cfg.xxx
~~~

读取对应配置。

例如：

~~~python
for _ in range(self.cfg.control.decimation):

    self.torques = self._compute_torques(
        self.actions
    )

    self.gym.simulate(self.sim)
~~~

这里环境根据 cfg 中的 decimation 决定每次策略动作对应多少个物理仿真步。

##### 4. cfg 和 env 是否可以理解为不变量与动态变量？

可以作为一个初步的概念模型：

- cfg：一轮训练运行期间通常保持稳定的配置参数。
- env：按照这些配置运行，并维护随着机器人与环境交互而不断变化的状态。

例如：

| cfg：通常固定的配置 | env：动态变化的状态 |
|---|---|
| num_envs = 4096 | 当前机器人状态 |
| action_scale = 0.25 | 当前 Action |
| decimation = 4 | 当前关节角度 |
| PD 控制参数 | 当前关节角速度 |
| 初始关节角度 | 当前机身姿态 |
| Reward 权重 | 当前 Reward |
| Episode 最大长度 | 当前 Episode 运行步数 |

但是需要注意两个例外：

###### 例外一：cfg 不是绝对不变量

例如，不同实验可以修改：

~~~python
cfg.terrain.static_friction = 0.3
~~~

此外，某些配置也可能通过 Curriculum 等机制在训练过程中动态调整。

所以 cfg 更准确地说是配置参数，而不是永远不能改变的常量。

###### 例外二：cfg 可以规定动态变量的变化规则（域随机化）

例如：

~~~python
cfg.domain_rand.randomize_friction = True

cfg.domain_rand.friction_range = [0.1, 1.25]
~~~

这里：

- randomize_friction：是否启用摩擦系数随机化。
- friction_range：实际摩擦系数的随机化范围。
- 每个仿真环境实际采样得到的摩擦系数：运行时参数。

也就是说，cfg 可以规定 env 中某些动态变量如何变化。

##### 5. 为什么 make_env() 要同时返回 env 和 env_cfg？

~~~python
return env, env_cfg
~~~

实际上，PPO 进行环境交互时，主要使用 env：

~~~python
obs = env.reset()

obs, rewards, dones, infos = env.step(actions)
~~~

并不需要每一步额外传入 env_cfg，因为 env 内部已经保存了 self.cfg。

额外返回 env_cfg 是为了方便外部代码：

1. 查看最终生效的环境配置。
2. 分析环境参数。
3. 获取经过命令行参数覆盖后的配置。
4. 供其他训练、测试和管理代码使用。

因此，env_cfg 不是 PPO 与环境交互时必须额外携带的对象。

##### 6. 最终知识总结

**env_cfg：规定环境应该如何创建和运行。**

**env：根据配置创建出来的实际环境，负责执行环境逻辑并维护动态状态。**

两者分离的原因：

将环境的运行逻辑与具体参数分开，使不同机器人能够复用环境框架，也方便调整参数和开展实验。

最重要的一句话：

> cfg 定规则，env 执行规则并维护动态状态。

注意：env 内部也保存着 cfg，即 self.cfg，因此两者不是完全独立的两个对象。

**问题 4： PPO 算法在哪里创建？为什么不直接在 `train.py` 里面写 PPO 网络？

答：在LeggedRobotCfgPPO中，由于要在h1_config.py中具体配置PPO的参数，每个模型的PPO参数设置不同，最后由--task = h1 将 H1RoughCfgPPO传入train.py

#### 💡GPT的修正与补充

###### 1.首先区分：配置 PPO ≠ 创建 PPO

这里需要区分三个概念：

| 名称                  | 实际作用                |
| ------------------- | ------------------- |
| `LeggedRobotCfgPPO` | PPO 训练配置的基类         |
| `H1RoughCfgPPO`     | H1 任务对应的 PPO 配置     |
| `OnPolicyRunner`    | 负责组织和执行训练流程的 Runner |

例如：

```python
class H1RoughCfgPPO(LeggedRobotCfgPPO):

    class policy:
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        rnn_type = 'lstm'
        rnn_hidden_size = 64

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
```

这里主要定义：

- Actor 和 Critic 的网络结构。
- 是否使用 LSTM。
- PPO 算法超参数。
- Runner 的训练参数。


### 配置继承关系

```text
LeggedRobotCfgPPO（父类）
        |
        | 继承基础 PPO 配置
        v
H1RoughCfgPPO（子类）
        |
        | 覆盖 H1 需要修改的参数
        v
获得 H1 对应的 PPO 配置
```

没有被子类覆盖的参数，继续使用基类中的配置。

注意：

**H1RoughCfgPPO 覆盖的是基础 PPO 配置类的参数，而不是 task_registry 中的参数。**

###### 2. task_registry 在这里起什么作用？

在 `legged_gym/envs/__init__.py` 中，官方提前注册了不同机器人的任务：

```python
task_registry.register(
    "h1",
    H1Robot,
    H1RoughCfg(),
    H1RoughCfgPPO()
)

task_registry.register(
    "g1",
    G1Robot,
    G1RoughCfg(),
    G1RoughCfgPPO()
)
```

TaskRegistry 内部维护三个字典：

```python
self.task_classes = {}
self.env_cfgs = {}
self.train_cfgs = {}
```

分别保存：

- `task_classes`：机器人环境类。
- `env_cfgs`：机器人环境配置。
- `train_cfgs`：PPO 训练配置。

因此，注册表的作用是建立任务名称与环境类、环境配置、训练配置之间的映射关系。

注意：

**task_registry 是任务注册表，不是一个存储默认参数的配置类。**


###### 3. PPO Algorithm 和 Runner 到底在哪里创建？

真正创建 Runner 的位置在：

```text
legged_gym/utils/task_registry.py
```

核心代码：

```python
train_cfg_dict = class_to_dict(train_cfg)

runner = OnPolicyRunner(
    env,
    train_cfg_dict,
    log_dir,
    device=args.rl_device
)

return runner, train_cfg
```

这里：

- `env`：已经创建好的机器人环境。
- `train_cfg_dict`：转换后的 PPO 训练配置。
- `log_dir`：训练日志保存路径。
- `device`：强化学习使用的计算设备。

随后 `OnPolicyRunner` 根据训练配置初始化 Actor-Critic 和 PPO 算法。

最后由：

```python
ppo_runner.learn(
    num_learning_iterations=train_cfg.runner.max_iterations,
    init_at_random_ep_len=True
)
```

启动训练。

###### 完整创建流程

```text
H1RoughCfgPPO
    |
    | 提供网络结构和 PPO 参数
    v
TaskRegistry.make_alg_runner()
    |
    | 获取并转换训练配置
    v
OnPolicyRunner(...)
    |
    | 初始化 Actor-Critic 和 PPO Algorithm
    v
ppo_runner.learn()
    |
    v
正式开始训练
```


######  4.Runner 负责什么？

Runner 主要负责：

- 管理训练循环。
- 与环境交互。
- 收集 Rollout 数据。
- 管理训练迭代次数。
- 调用 PPO Algorithm 计算和更新。
- 记录训练日志。
- 保存和加载模型 Checkpoint。

###### 5.可以将runner类比为PPO的train.py吗

可以，而且这是一个很好的类比。

之前研究 CleanRL PPO 时，通常可以在一个 `train.py` 中看到完整训练流程：

```python
def train():

    # 初始化环境和网络
    agent = Agent()

    for iteration in range(num_iterations):

        # Rollout
        collect_data()

        # GAE
        compute_advantage()

        # PPO Update
        update_policy()
```

而 Unitree 使用 RSL-RL，将这些职责拆分为不同模块：

```text
Unitree train.py
      |
      | 创建环境、加载配置
      v
OnPolicyRunner
      |
      | 管理整个训练循环
      |
      ├── Rollout
      |
      ├── GAE
      |
      └── 调用 PPO.update()
              |
              ├── 计算 Loss
              ├── Backpropagation
              └── Optimizer.step()
```

两者的对应关系：

| CleanRL PPO | Unitree / RSL-RL |
|---|---|
| `train.py` | `OnPolicyRunner`（训练流程层面） |
| `Agent` | `ActorCritic` |
| PPO 更新逻辑 | `PPO` |
| `torch.optim.Adam` | PPO 内部的 Optimizer |

注意：

Unitree 自己的 `train.py` 主要负责初始化和启动 Runner，并不等同于 CleanRL 的完整训练主循环。

所以：

**OnPolicyRunner 可以理解为封装好的 PPO 训练主程序（Trainer），类似 CleanRL PPO 中的 train.py。**

它是可复用的 Python 类，而不是一个普通训练脚本。



###### 6. 为什么不直接在 train.py 里面写 PPO 网络？

因为采用了分层设计，使各个模块的职责更加清晰：

| 模块 | 职责 |
|---|---|
| `train.py` | 训练入口，解析参数，启动训练 |
| `task_registry` | 根据任务选择环境和训练配置，创建环境与 Runner |
| `H1RoughCfgPPO` | 提供 H1 对应的 PPO 配置 |
| `OnPolicyRunner` | 组织整个训练流程 |
| `ActorCritic` | 定义策略网络和价值网络 |
| `PPO` | 实现 PPO 优化算法 |
| `Optimizer` | 根据梯度更新网络参数 |

这样 H1、G1 等不同机器人可以使用各自的配置，同时复用相同的 Runner 和 PPO 训练框架。

###### 7. 命令行参数还可以覆盖配置

例如：

```bash
python legged_gym/scripts/train.py \
    --task=h1 \
    --max_iterations=2000 \
    --num_envs=512
```

其中：

- `--task=h1`：选择 H1 任务。
- `--max_iterations=2000`：覆盖 H1 PPO 配置中的最大训练迭代次数。
- `--num_envs=512`：覆盖环境配置中的并行环境数量。

配置覆盖由相应的配置更新函数完成。

因此需要区分：

1. 配置类继承：子类覆盖基类中的默认配置。
2. 任务注册：建立任务名称与环境、配置之间的映射。
3. 任务选择：通过 `--task=h1` 选择对应任务。
4. 命令行覆盖：通过参数修改最终生效的配置。
5. 实例化：创建环境实例和 Runner 实例。

###### 8. 最终知识总结

***总而言之，一句话，所有实例都是由task_registry的函数创建，env用task_registry.make_env()，runner用task_registry.make_alg_runner()，而里面具体参数由--task=h1，输入后查找到具体的参数替换，这样便于使训练流程可以复用，机器人任务可以灵活切换

###### 核心概念

1. `LeggedRobotCfgPPO`：PPO 配置基类。
2. `H1RoughCfgPPO`：H1 对应的 PPO 配置，不是 PPO 算法实例。
3. `task_registry`：负责注册和查找任务，不负责执行 PPO。
4. `OnPolicyRunner`：负责管理和组织 PPO 训练流程。
5. `ActorCritic`：负责定义 Actor 和 Critic 网络。
6. `PPO`：负责实现 PPO 策略优化逻辑。
7. `Optimizer`：负责根据梯度更新网络参数。
8. `runner.learn()`：正式启动训练循环。


###### 整体代码调用关系

```text
LeggedRobotCfgPPO
        |
        v
H1RoughCfgPPO
        |
        | 继承并覆盖基础配置
        v
task_registry.register()
        |
        | 注册 H1 对应的环境与配置
        v
train.py
        |
        | --task=h1
        v
task_registry.make_env()
        |
        v
创建 H1Robot 环境实例
        |
        v
task_registry.make_alg_runner()
        |
        v
创建 OnPolicyRunner
        |
        ├── 初始化 ActorCritic
        |
        └── 初始化 PPO Algorithm
                  |
                  └── 创建 Optimizer
        |
        v
runner.learn()
        |
        ├── Rollout
        ├── GAE / Returns
        ├── PPO Update
        └── 模型保存
```



**问题 5： `ppo_runner.learn()` 中的 `max_iterations` 是从哪里来的？它和 PPO 中一次更新所用的 minibatch 数量是不是同一个概念？

答：/root/unitree_rl_gym/legged_gym/envs/h1/h1_config.py 中

```python
class runner( LeggedRobotCfgPPO.runner ):

        policy_class_name = "ActorCritic"

        max_iterations = 10000

        run_name = ''

        experiment_name = 'h1'
```

和PPO更新中minibatch不是同一概念，最大迭代次数是模型总的训练次数，而minibatch是PPO算法在rollout阶段要用多少批数据进行更新，这是PPO作为局部offpolicy策略的性质，这个数量不宜过大

#### 💡GPT的修正与补充

###### 1. max_iterations 从哪里来？

以 H1 为例，在：

```text
legged_gym/envs/h1/h1_config.py
```

可以看到：

```python
class H1RoughCfgPPO(LeggedRobotCfgPPO):

    class runner(LeggedRobotCfgPPO.runner):

        policy_class_name = "ActorCriticRecurrent"

        max_iterations = 10000

        run_name = ''

        experiment_name = 'h1'
```

这里的 `max_iterations` 表示最大外层训练迭代次数。

它继承自基础 PPO 配置类：

```python
LeggedRobotCfgPPO
```

H1 在此基础上将最大训练迭代次数设置为 10000。

最终在 `train.py` 中传入：

```python
ppo_runner.learn(
    num_learning_iterations=train_cfg.runner.max_iterations,
    init_at_random_ep_len=True
)
```

因此，训练器最多执行 10000 次外层训练迭代。

注意：`max_iterations` 也可以被命令行参数覆盖，例如：

```bash
python legged_gym/scripts/train.py \
    --task=h1 \
    --max_iterations=2000
```

这样最终使用的就是 2000，而不是配置文件中的 10000。

---

###### 2. max_iterations 和 minibatch 有什么区别？

两者不是同一个概念。

###### max_iterations：外层训练迭代次数

一次 iteration 通常包含：

1. 收集 Rollout 数据。
2. 计算 GAE 和 Returns。
3. 使用 PPO 算法更新策略。
4. 进入下一次 iteration。

例如：

```text
Iteration 1
    |
    ├── Rollout
    ├── GAE
    ├── PPO Update
    └── 更新策略
           |
           v
Iteration 2
    |
    ├── Rollout
    ├── GAE
    ├── PPO Update
    └── 更新策略
           |
           v
        ...
```

###### num_mini_batches：每个 Epoch 将 Rollout 数据划分为多少个 Minibatch

例如：

```python
num_learning_epochs = 5
num_mini_batches = 4
```

含义：

- `num_learning_epochs = 5`：对同一批 Rollout 数据重复优化 5 轮。
- `num_mini_batches = 4`：每个 Epoch 将数据划分为 4 个 Minibatch。

因此：

理论上每次外层 iteration 会进行 20 次 Minibatch 参数更新。

---

###### 4. Minibatch 设置成 4 批或 5 批有什么区别？

假设一次 Rollout 收集 98,304 条 transition：

```python
num_learning_epochs = 5
```

分别设置：

```python
num_mini_batches = 4
```

和：

```python
num_mini_batches = 5
```

对普通 Feedforward 策略，理论上的划分如下：

| 对比 | 4 个 Minibatch | 5 个 Minibatch |
|---|---:|---:|
| Rollout 总数据量 | 98,304 | 98,304 |
| 每个 Minibatch | 24,576 | 约 19,660 |
| 每个 Epoch 更新次数 | 4 | 5 |
| Epoch 数量 | 5 | 5 |
| 总参数更新次数 | 20 | 25 |

注意：整数除法可能导致少量数据不能被完整划分，具体取决于数据生成器的实现。

###### Minibatch 数量对训练的影响

**Minibatch 较少：**

- 每次更新使用的数据更多。
- 梯度估计通常更加稳定。
- 每个 Epoch 的参数更新次数更少。
- 单次更新的显存需求通常更大。

**Minibatch 较多：**

- 每次更新使用的数据更少。
- 梯度估计通常具有更高的随机性。
- 每个 Epoch 的参数更新次数更多。
- 单次更新的显存需求通常更小。

---

## 5. 为什么不直接把 num_mini_batches 设置为 1？

这是一个很有意思的优化问题。

我的想法是：

既然数据量越大，计算出来的平均梯度通常越稳定，那么直接让一个 Minibatch 使用所有 Rollout 数据，不就能降低梯度估计方差，同时减少参数更新次数、节约时间吗？

###### 🚨 1. 最直接的原因：显存（VRAM）会爆炸

假设你跑了 1024 个环境，每个环境走 24 步。  
这批采集的数据总量 `batch_size = 1024 * 24 = 24576`。  
如果你不切分，把 24576 个样本一次性扔进神经网络（尤其是带有 LSTM 或大隐藏层的网络）进行前向传播和反向传播。

- 你需要存储这 24576 个样本的计算图。
    
- 你的 8GB 显存会瞬间 **Out of Memory (OOM)**，代码直接崩溃。  
    切成 `minibatch_size = 256` 的小块，每次只算 256 个样本，显存占用极低，稳稳跑满。
    

---

###### 🧮 2. 数学原因：随机梯度下降（SGD）的“玄学”优势

在深度学习里，**全量梯度下降（Full Batch）**虽然方向最准，但极其容易陷入**鞍点（Saddle Point）**或局部最优。  
如果你每次用全部数据算梯度，方向是确定的，网络会直勾勾地卡死在一个坑里。  
切成 Minibatch，每次算出来的梯度带有一点点**“==🔴随机==噪声”**。这种随机性就像给网络加了一点“抖动”，能帮它从坑里跳出来，泛化能力更强。

###### 🎲 随机性来源一：数据子集的抽样（Subsampling）

你有一大锅汤（固定的 10 万条数据），你每次只舀一勺（256 条）去尝味道。

- 第一勺（Minibatch 1）：可能刚好全是“左脚踩地”的数据，算出来的梯度是往左偏的。
    
- 第二勺（Minibatch 2）：可能刚好全是“右脚踩地”的数据，算出来的梯度是往右偏的。  
    **因为每次喂给网络的 256 个样本不同，算出来的梯度方向就会有差异。** 这就是我们常说的“梯度噪声”。这就是为什么代码里必须有 `np.random.shuffle(b_inds)` 的原因——如果不打乱，每次挑出的可能都是高度相似的数据，网络就学偏了。
    

###### ⚡ 随机性来源二：参数在不断变化（Online Update）

这也是最核心的一点！  
假设你运气极好，在 Epoch 1 的第一个 Minibatch 挑到了数据 A，算完梯度，**更新了一次网络参数**。  
接着，你在 Epoch 1 的第二个 Minibatch 里，**又挑到了数据 A**（因为在 10 个 Epoch 里，数据是会被反复抽到的）。  
这时候，**数据 A 没变，但网络的参数已经变了！** 同样一组数据 A，在新的网络参数下，产生的误差和梯度也就完全不同了。

###### 🤔 为什么非要这种“随机性”？（对比全量梯度）

假设你不切 Minibatch，每次都用**全量 10 万条数据**算梯度。

- **结果**：你算出来的梯度是非常“完美”的，精确指向了当前地形的最低点（这叫 Full-batch Gradient Descent）。
    
- **致命缺点**：如果你刚好站在一个“小坑（局部最优）”里，这个完美的梯度会告诉你：“往下走！”于是你就永远卡死在这个小坑里了。
    

**而 Minibatch 产生的“随机噪声”呢？**  
因为每次算的梯度方向有偏差（噪声），这就好比你在下山的时候，走的不是一条完美的直线，而是**左摇右晃**的。  
正是这种左摇右晃（随机性），让你在不小心掉进一个小坑时，有概率“晃”出这个坑，从而找到更大的平原（全局最优）。


---

######  ⚡ 3. 工程原因：更新次数暴增，疯狂“薅羊毛”

这是 PPO 的最大特色（样本利用率高）。  
假设你的 `batch_size = 10000`，`update_epochs = 10`。

- **如果不切分**：在一个 Epoch 里，你把 10000 个数据算一次，执行 **1 次** `optimizer.step()`。10 个 Epoch 只更新 10 次参数。
    
- **如果切分**（假设 `minibatch_size = 1000`）：10000 个数据切成了 10 块。在一个 Epoch 里，你会执行 **10 次** `optimizer.step()`。10 个 Epoch 下来，参数更新了 **100 次**！  
    这样一来，同一批数据被反复利用，网络的收敛速度会快得飞起。
    

---

###### 🏎️ 4. GPU 效率：矩阵乘法的最优解

GPU 最喜欢的运算尺寸，既不是极大（像一座大山，算起来太慢），也不是极小（像一滴水，并行能力浪费）。  
现代 GPU 架构（Tensor Cores）在计算 `256 x 64` 这种规模的小矩阵乘法时，吞吐量是最高的。切分 Minibatch 能完美榨干 GPU 的并行算力。

---


###### 6. 特别注意：G1/H1 使用的是循环策略

在 Unitree G1/H1 配置中可以看到：

```python
policy_class_name = "ActorCriticRecurrent"
```

这意味着使用的是带 LSTM 的循环 Actor-Critic。

在 RSL-RL v1.0.2 中，这类策略使用：

```python
self.storage.reccurent_mini_batch_generator(
    self.num_mini_batches,
    self.num_learning_epochs
)
```

而不是普通 Feedforward 策略的扁平化 Minibatch Generator。

因此，两者的数据划分方式有所不同：

### Feedforward 策略

主要将 Rollout 中的 transition 样本划分成多个 Minibatch。

### Recurrent 策略

需要保留轨迹的时间顺序和循环网络所需的初始隐藏状态，主要按照环境轨迹进行分组。

例如：

```text
4096 个并行环境

num_mini_batches = 1
    → 每批使用全部环境的轨迹

num_mini_batches = 4
    → 每批约使用 1024 个环境的轨迹
```

因此，对于 G1/H1，不能简单地把 98,304 除以 Minibatch 数量，就认为这一定是实际每批的数据组织方式。

具体还需要考虑轨迹长度、Padding 和数据生成器实现。

---

## 8. 三个参数之间的关系

| 参数                    | 决定什么                 |
| --------------------- | -------------------- |
| `max_iterations`      | 外层总训练迭代次数            |
| `num_learning_epochs` | 每批 Rollout 数据重复优化多少轮 |
| `num_mini_batches`    | 每个 Epoch 划分多少个小批次    |

对于普通 Feedforward 策略，可以理解为：


###### 最终知识总结

1. `max_iterations`：决定外层总训练迭代次数，不等于环境交互步数。
2. `num_learning_epochs`：决定同一批 Rollout 数据重复优化多少轮。
3. `num_mini_batches`：决定每个 Epoch 将 Rollout 数据划分为多少个小批次。
4. Minibatch 数量越少，单次梯度估计通常越稳定，但更新次数更少、显存压力更大。
5. Minibatch 数量越多，单次梯度估计通常具有更高随机性，但参数更新更加频繁。
6. 一个 Epoch 中，不论划分多少个 Minibatch，原则上都是在使用同一批 Rollout 数据。
7. PPO 仍然是 On-Policy 算法，只是在一次策略更新阶段重复利用当前 Rollout 数据。
8. G1/H1 使用 LSTM 循环策略，需要使用 Recurrent Minibatch Generator，不能完全套用普通 MLP 的数据划分理解。




# 进阶问题

###### 一、题目

假设 H1/G1 的训练配置如下：

```python
num_envs = 4096
num_steps_per_env = 24

num_learning_epochs = 5
num_mini_batches = 4
max_iterations = 10000
```

**问题：一次外层 PPO iteration，理论上收集多少条 Transition？**

选项：

- A. 4096 × 24 = 98,304 条
- B. 4096 × 24 × 4 = 393,216 条
- C. 4096 × 24 × 5 = 491,520 条
- D. 10000 × 4096 = 40,960,000 条

我的答案：A。

---

###### 二、正确答案及计算过程

正确答案：**A，98,304 条 Transition。**


### 1. 为什么这样计算？

```python
num_envs = 4096
num_steps_per_env = 24
```

两个参数的含义：

- `num_envs = 4096`：同时运行 4096 个并行机器人环境。
- `num_steps_per_env = 24`：每个环境收集 24 步数据。

可以理解为：

| 环境 | 收集的数据量 |
|---|---:|
| Env 1 | 24 |
| Env 2 | 24 |
| Env 3 | 24 |
| ... | ... |
| Env 4096 | 24 |
| 总计 | 98,304 |

每个环境每执行一步策略交互，就产生一条 Transition。

通常可以表示为：


(s_t,a_t,r_t,s_{t+1},done_t)


其中：

- \(s_t\)：当前状态。
- \(a_t\)：策略输出的动作。
- \(r_t\)：环境返回的奖励。
- \(s_{t+1}\)：下一时刻状态。
- \(done_t\)：当前 Episode 是否结束。

注意：即使某个 Episode 中途结束，环境自动 Reset 后继续采集，正常情况下该步仍然计入 Rollout 数据量。

---

## 三、为什么不乘以其他参数？

这是这道题真正考察的知识点。

| 参数 | 是否参与计算 | 原因 |
|---|---|---|
| `num_envs = 4096` | 是 | 决定并行环境数量 |
| `num_steps_per_env = 24` | 是 | 决定每个环境收集的步数 |
| `num_learning_epochs = 5` | 否 | 决定同一批数据重复优化多少轮 |
| `num_mini_batches = 4` | 否 | 决定每个 Epoch 将数据划分成多少个小批次 |
| `max_iterations = 10000` | 否 | 决定外层训练循环的次数 |

关键区别：

**Rollout 是收集数据，Minibatch 和 Epoch 是利用已经收集的数据进行优化。**

它们属于两个不同阶段：

```text
        PPO 一次外层 Iteration
                  |
                  v
        ┌──────────────────┐
        │  阶段一：Rollout  │
        │                  │
        │ 4096 个环境       │
        │ 每个环境 24 步    │
        │                  │
        │ 共收集 98,304 条  │
        │ Transition       │
        └──────────────────┘
                  |
                  v
        ┌──────────────────┐
        │ 阶段二：PPO Update│
        │                  │
        │ 5 个 Epoch       │
        │ 每个 Epoch        │
        │ 划分 4 个 Batch   │
        │                  │
        │ 5 × 4 = 20 次    │
        │ 参数更新          │
        └──────────────────┘
```

因此，不能将 Minibatch 数量或 Epoch 数量乘进 Rollout 数据量。

因为它们不会产生新的环境交互数据，而是重复使用当前收集的 Rollout 数据。

---

###### 四、扩展一：一次 PPO Iteration 中，Rollout 和 Update 有什么区别？

###### 1. Rollout 阶段

Rollout 负责与环境交互并收集数据。

例如：

```python
num_envs = 4096
num_steps_per_env = 24
```

得到：

\[
4096\times24=98304
\]

条 Transition。

###### 2. Update 阶段

假设：

```python
num_learning_epochs = 5
num_mini_batches = 4
```


---

###### 五、扩展二：Decimation 与 RL Step 有什么区别？

这是机器人强化学习与普通 PPO 实现之间一个非常重要的区别。

假设配置为：

```python
num_envs = 4096
num_steps_per_env = 24

decimation = 4
```

这里有两个不同层次的步数：

###### 1. RL Step（策略交互步）

指策略网络输出一次 Action，环境执行这个 Action，并返回下一次 Observation 和 Reward。

每个环境收集 24 个 RL Step。

###### 2. Physics Simulation Step（物理仿真步）

指物理引擎实际推进一次物理状态。

当：

```python
decimation = 4
```

意味着每个策略动作对应 4 个物理仿真子步（类似于微积分，物理仿真需要实时计算现场情况，即将时间拆分为一小段如t=2m
s，分析碰撞、受力，最后计算出新状态，详细可以看BV1mRHW6uEzZ）。

例如：

```text
Policy 输出 Action
        |
        v
    RL Step 1
        |
        ├── Physics Step 1
        ├── Physics Step 2
        ├── Physics Step 3
        └── Physics Step 4
        |
        v
    RL Step 2
        |
        ├── Physics Step 5
        ├── Physics Step 6
        ├── Physics Step 7
        └── Physics Step 8
```



每个环境对应 24 × 4 = 96 个物理仿真子步。

4096 个并行环境合计：

4096\96=393216

个环境物理仿真子步。

注意：

- 98,304 是 RL Transition 数量。
- 393,216 是按并行环境累计的物理仿真子步数量。

两者不是同一个概念。

---


# 注：什么是物理仿真步（接进阶问题）

物理仿真步（Physics Simulation Step），就是物理引擎每推进一次仿真时间，重新计算机器人及环境物理状态的过程。

它不仅包括碰撞检测，还包括动力学计算和状态更新。

###### 1. 一个物理仿真步具体做什么？

例如 G1 正在行走，物理引擎每推进一个 Physics Step，通常需要处理：

1. 碰撞检测（Collision Detection）：检查机器人各个碰撞体是否发生接触。
    
2. 接触约束计算（Contact Constraints）：计算脚与地面等接触关系。
    
3. 受力计算（Force Calculation）：计算重力、关节力矩、摩擦力、接触力等。
    
4. 动力学计算（Dynamics）：根据合力和合力矩计算加速度。
    
5. 状态积分（Integration）：根据加速度更新速度、位置和姿态。
    

例如：

$$F=ma $$

$$ v_{t+\Delta t}=v_t+a_t\Delta t 

$$$$ x_{t+\Delta t}=x_t+v_{t+\Delta t}\Delta t $$

这里是简化的物理积分示意，实际机器人还涉及多刚体动力学、关节约束和接触求解。

###### 2. 那么 Decimation=4 是什么意思？

这就联系到我们之前学的内容了。

假设：

```
sim.dt = 0.005

decimation = 4
```

那么：

- 物理仿真每隔 0.005 秒推进一次。
    
- 策略输出一次 Action 后，物理引擎连续推进 4 次。
    
- 这 4 次物理仿真使用同一个策略 Action。
    

###### Decimation：物理仿真频率与策略控制频率

假设：

```python
sim.dt = 0.005
decimation = 4
```

含义：

- `sim.dt = 0.005`：物理引擎每次推进 0.005 秒（5 ms）。
- `decimation = 4`：每次策略输出 Action 后，物理引擎连续推进 4 次。
- 这 4 次物理仿真使用同一个策略 Action。
- 完成 4 次物理仿真后，策略获取新的 Observation，再输出下一个 Action。

###### 3. 一个策略控制周期

```text
          Policy 输出 Action
                  |
                  v
        ┌──────────────────┐
        │    Physics 1     │
        │      5 ms        │
        └──────────────────┘
                  |
        ┌──────────────────┐
        │    Physics 2     │
        │      5 ms        │
        └──────────────────┘
                  |
        ┌──────────────────┐
        │    Physics 3     │
        │      5 ms        │
        └──────────────────┘
                  |
        ┌──────────────────┐
        │    Physics 4     │
        │      5 ms        │
        └──────────────────┘
                  |
                  v
      Policy 获取新的 Observation
                  |
                  v
          输出下一个 Action
```


###### 4. 两种频率的区别

| 对比 | 物理仿真 | 策略控制 |
|---|---:|---:|
| 时间间隔 | 0.005 s | 0.02 s |
| 频率 | 200 Hz | 50 Hz |
| 每秒执行次数 | 200 次 | 50 次 |
| 执行主体 | 物理仿真引擎 | Policy 网络 |
| 主要工作 | 动力学计算、碰撞检测、状态更新 | 根据 Observation 生成 Action |

###### 5. 为什么要这样设计？

因为机器人动力学变化很快。

假设机器人脚掌正在接触地面：

- 策略不需要每 0.005 秒都重新计算一次动作。
    
- 但物理引擎需要更频繁地计算接触力、摩擦力、加速度和机器人运动状态。
    
- 因此采用较低频率的策略控制，配合较高频率的物理仿真。
    

这样可以减少策略网络的调用频率，同时保持物理模拟的时间分辨率。

最终记住：

- `sim.dt`：物理引擎每次推进多少仿真时间。
    
- `decimation`：一个策略 Action 对应多少个物理仿真步。
    
- `sim.dt × decimation`：一个策略控制周期。
    
- `1 / (sim.dt × decimation)`：策略控制频率。
    

这也就是为什么 Unitree 的 `decimation` 会直接影响机器人控制频率，而不是 PPO 的训练迭代次数。

###### 6. 最终记忆

```python
sim.dt = 0.005
decimation = 4
```

**物理仿真步长：**

$$sim.dt=0.005s$$

**每个策略 Action 对应的物理仿真步数：**

$$decimation=4$$

**策略控制周期：**

$$T_{\text{policy}}=sim.dt\times decimation$$

**物理仿真频率：**

$$f_{\text{physics}}=\frac{1}{sim.dt}=200Hz$$

**策略控制频率：**

$$f_{\text{policy}}
=\frac{1}{sim.dt\times decimation}
=50Hz$$

核心理解：

- `sim.dt`：物理引擎每次推进多少仿真时间。
- `decimation`：一个策略 Action 对应多少个物理仿真步。
- `sim.dt × decimation`：一个策略控制周期。
- `1 / sim.dt`：物理仿真频率。
- `1 / (sim.dt × decimation)`：策略控制频率。

**策略每秒输出 50 次 Action，而物理引擎每秒推进 200 次仿真。**

---

###### 7. 问：policy需要obs，输出act，只要环境给他obs就开始计算不就好了，为什么需要几步几步的

**答：这是把 Policy 输出 Action 和 物理引擎推进时间 当成了同一件事。

实际上，它们是两个不同的过程。

你说的这句话完全正确：

> Policy 需要 Observation，拿到 Observation 后就计算 Action。

但是，你漏掉了一个问题：

Policy 输出 Action 之后，机器人什么时候运动？下一次 Observation 又是怎么产生的？

这就是物理仿真需要存在的原因。

###### 1. 先看一个最简单的闭环

假设机器人现在站在原地。

在 \(t=0\) 时刻：

```
环境提供 Observation
        |
        v
Policy 接收 Observation
        |
        v
Policy 输出 Action
        |
        v
机器人执行 Action
        |
        v
物理引擎模拟机器人运动
        |
        v
得到新的机器人状态
        |
        v
生成新的 Observation
        |
        v
Policy 再次输出 Action
```

注意这里：

Policy 本身不会让机器人运动。它只是输出动作指令。

真正让机器人产生运动的是物理引擎。

例如 Policy 输出：

```
action = [0.2, -0.3, 0.5, ...]
```

这只是一个动作指令，并不意味着机器人关节立即转到了目标角度。

机器人究竟会怎样运动，需要物理引擎根据：

- 当前关节角度；
    
- 当前关节速度；
    
- 关节力矩；
    
- 重力；
    
- 机器人惯性；
    
- 地面接触力；
    

计算出下一时刻的状态。

然后才能得到新的 Observation。

###### 2. 为什么要推进 4 次？

假设：

```
sim.dt = 0.005
decimation = 4
```

我们希望 Policy 每隔 20ms 决策一次。

那么过程是这样的：

<h3>时间从 0ms 推进到 20ms</h3>

<div style="background-color: #ebf3fc; padding: 16px; text-align: center; border-radius: 8px;">
  <p style="font-weight: bold; margin: 0 0 8px 0; color: #333;">0ms：获取 Observation 0</p>
  <p style="font-weight: bold; font-size: 20px; color: #2a609e; margin: 0;">Policy 输出 Action A</p>
</div>

<div style="text-align: center; margin: 12px 0; color: #666;">
  <div>↓</div>
  <div style="font-size: 14px;">Action A 保持不变</div>
</div>

<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <strong style="font-size: 16px;">Physics Step 1</strong>
    <span style="background-color: #e6f4ea; color: #1e8e3e; padding: 2px 10px; border-radius: 12px; font-size: 13px;">0–5ms</span>
  </div>
  <div style="color: #666; font-size: 14px; margin-top: 4px;">物理引擎计算动力学，更新机器人状态</div>
</div>

<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <strong style="font-size: 16px;">Physics Step 2</strong>
    <span style="background-color: #e6f4ea; color: #1e8e3e; padding: 2px 10px; border-radius: 12px; font-size: 13px;">5–10ms</span>
  </div>
  <div style="color: #666; font-size: 14px; margin-top: 4px;">物理引擎计算动力学，更新机器人状态</div>
</div>

<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <strong style="font-size: 16px;">Physics Step 3</strong>
    <span style="background-color: #e6f4ea; color: #1e8e3e; padding: 2px 10px; border-radius: 12px; font-size: 13px;">10–15ms</span>
  </div>
  <div style="color: #666; font-size: 14px; margin-top: 4px;">物理引擎计算动力学，更新机器人状态</div>
</div>

<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <strong style="font-size: 16px;">Physics Step 4</strong>
    <span style="background-color: #e6f4ea; color: #1e8e3e; padding: 2px 10px; border-radius: 12px; font-size: 13px;">15–20ms</span>
  </div>
  <div style="color: #666; font-size: 14px; margin-top: 4px;">物理引擎计算动力学，更新机器人状态</div>
</div>

<div style="text-align: center; margin: 12px 0; color: #666;">
  ↓
</div>

<div style="background-color: #ebf3fc; padding: 16px; text-align: center; border-radius: 8px;">
  <p style="font-weight: bold; margin: 0 0 8px 0; color: #333;">20ms：获取新的 Observation 1</p>
  <p style="font-weight: bold; font-size: 20px; color: #2a609e; margin: 0;">Policy 输出 Action B</p>
</div>


你看：

- Policy 只调用了两次：Action A 和 Action B。
    
- 物理引擎调用了四次，完成了从 0ms 到 20ms 的仿真推进。
    
- 四次物理仿真期间，Policy 的 Action A 没有改变。
    

这里的四步不是让 Policy 等四次，而是让机器人在两次策略决策之间真实地演化 20ms 的模拟时间。

###### 3. 那为什么不每隔 5ms 就调用一次 Policy？

当然可以！

这正是你一开始提出的方案，对应：

```
sim.dt = 0.005
decimation = 1
```

此时：

```
0ms：Observation 0 → Policy → Action A
                         ↓
                     Physics 1

5ms：Observation 1 → Policy → Action B
                         ↓
                     Physics 2

10ms：Observation 2 → Policy → Action C
                         ↓
                     Physics 3

15ms：Observation 3 → Policy → Action D
                         ↓
                     Physics 4
```

两种方案对比如下：

|对比|decimation=1|decimation=4|
|---|---|---|
|物理仿真步长|5ms|5ms|
|策略决策间隔|5ms|20ms|
|物理仿真频率|200Hz|200Hz|
|Policy 调用频率|200Hz|50Hz|
|每次 Action 保持时间|5ms|20ms|

所以，你提出的方案完全成立。

只是 Unitree 选择让策略以 50Hz 决策，而不是 200Hz 决策。

这样可以降低策略网络的调用频率，同时让物理引擎仍然以 200Hz 模拟机器人运动。

###### 4. 最关键的地方：如果不推进物理仿真，会发生什么？

假设 Policy 拿到一个 Observation：

```
obs = [0.1, 0.2, 0.3, ...]
```

输出：

```
action = policy(obs)
```

现在假设你不调用物理引擎，直接再次调用：

```
action = policy(obs)
```

那么你得到的还是基于同一个机器人状态的动作。

机器人没有运动，关节角度没有变化，机身姿态没有变化，Observation 也没有更新。

这样就无法形成真正的闭环控制。

物理引擎推进时间，本质上就是为了让 Action 对机器人产生实际影响，从而得到新的状态反馈。

###### 7.总结

***说白了就是，你想让policy获得什么时候的obs，毕竟我们PPO算法过于理论或者说我们在学习强化学习时，用的例子：grid world example过于简单，状态是离散的，实际上我们需要考虑，利用policy采样 $(s_{t},a_{t},r_{t},s_{t+1})$ 中的 $s_{t+1}$ 到底是什么,在物理仿真中，动作没有真正意义上的做完，只能保证做到一定范围，而且，不同动作执行的时间也不同，无法设定一个动态时间，做完某个动作再去拿 $s_{t+1}$，所以选择固定一个时间（5ms一次物理仿真，一次做四步仿真，那么policy的固定时间就是20ms）

###### 8.担忧：在PPO中难道不是要求 $(s_{t},a_{t},r_{t},s_{t+1})$，如果我都不能保证动作有没有做完，我怎么知道rt就是at对应的reward，如果每一个动作对应的reward都不能很好的记录，我们怎么利用ppo更新到最优策略

你的担忧很合理，但这里有一个关键概念需要拆开：**“Action 没有执行完成”不等于“这条 transition 的 reward 记录错了”**。PPO 需要的不是“动作完成后的成绩”，而是：

$$
(s_t,a_t,r_t,s_{t+1},done_t)
$$

其中 $a_t$ 的语义是：**在从 $t$ 到 $t+\Delta t$ 这段时间内持续生效的控制指令/控制目标**，而不是“一个必须执行到终点的独立任务”。所以它不需要“做完”才有 reward。

---

###### 1. PPO 中的 action 不是“任务”，而是“一个控制周期内的决策”

在 G1 行走里，Policy 输出的 `action` 通常会被映射成关节目标位置、目标速度或残差动作。它表达的是：

> 接下来 20ms 内，各关节应该朝什么目标去控制。

它不是：

> 把左腿抬到 0.5 rad，完成后再通知我。

因此一次交互是：

$$
s_t \xrightarrow{a_t \text{ 持续 } \Delta t} s_{t+1}
$$

然后环境在 $$t+\Delta t$$ 时刻计算：

$$
r_t = r(s_t,a_t,s_{t+1})
$$

注意：**reward 是定义在这个状态转移上的，不是定义在“动作完成事件”上的。**

在 Unitree 代码里就是：

```python
for _ in range(decimation):
    self.torques = self._compute_torques(self.actions)
    self.gym.set_dof_actuation_force_tensor(...)
    self.gym.simulate(self.sim)

self.post_physics_step()  # 这里 compute_reward() 和 compute_observations()
```

`self.actions` 在 4 个物理子步中保持不变，PD 每个子步根据最新关节状态重新算力矩。20ms 后调用 `compute_reward()`，得到 $$r_t$$，再得到 $$s_{t+1}$$。

所以这条数据是明确绑定的：

$$
(s_t,\ a_t,\ r_t,\ s_{t+1})
$$

并不是“我不知道 r_t 是不是 a_t 的 reward”。它是：**在 $s_t$ 下选择 $a_t$，让 $a_t$ 生效 20ms，环境演化到 $s_{t+1}$，并返回这段时间对应的奖励 $r_t$。**

---

###### 2. “动作没做完”为什么不是问题？

因为这里的“动作”本来就不要求“做完”。

举个简单例子。假设 Policy 输出：

$$
a_t = q_{\text{target}} = 0.5
$$

PD 控制器每 5ms 根据当前角度计算力矩：

$$
\tau = K_p(q_{\text{target}}-q_t)-K_d\dot q_t
$$

20ms 后关节可能只到 0.3，还没到 0.5。那 $$a_t$$ 算执行完了吗？

按“任务完成”角度看：没完成。  

按 RL 控制角度看：**它已经执行完了它该执行的东西——它在 20ms 内作为目标==🔴持续生效==了。**

环境返回的 $r_t$ 不是“关节到 0.5 的奖励”，而可能是：

- 机身速度是否接近目标；
- 姿态是否稳定；
- 足端接触是否合理；
- 力矩是否过大；
- 关节速度是否异常；
- 是否跌倒；
- 跟踪误差是否小；
- 等等。

这些奖励是**20ms 后机器人状态和这段转移的函数**，不是“这个动作是否独立完成”的函数。

所以你说“我都不能保证动作有没有做完，我怎么知道 r_t 就是 a_t 对应的 reward”，这里需要把“对应”理解成：

> $r_t$ 是在 $s_t$ 下执行 $a_t$ 一个控制周期后环境给出的即时反馈。

而不是：

> $r_t$ 是 $a_t$ 这个独立任务完成后的唯一因果结果。

RL 从不要求 reward 是某个 action 的唯一因果结果。机器人状态还受重力、惯性、接触、历史动作影响。PPO 估计的是策略梯度：

$$
\nabla_\theta J(\theta)
=
\mathbb E\left[
\nabla_\theta \log \pi_\theta(a_t|s_t) \hat A_t
\right]
$$

它要求的是：这条轨迹来自当前策略，并且 $a_t$、$s_t$、$r_t$、$s_{t+1}$ 的时间顺序和策略概率记录正确。它不要求“动作完成事件”。

---

###### 3. 真正会出问题的是“动作语义和奖励定义不匹配”

你的担心在一种情况下是完全正确的：

如果你把 action 定义成：

> “完成抬腿任务”

而 reward 定义成：

> “抬腿任务完成后给 1”

那你就必须知道任务何时完成，否则 reward 与 action 的对应关系确实会模糊。这时标准固定步长 MDP 就不够自然，需要用：

- Options / 分层强化学习；
- Semi-Markov Decision Process，SMDP；
- 事件触发控制；
- 或者显式加入“完成信号/终止信号”。

但 Unitree 的行走控制不是这种建模。它的 action 是**周期性持续控制目标**，reward 是每个控制周期根据机器人状态计算的即时项。因此不需要“动作完成”。

---

###### 4. PPO 不要求“每个动作有一个清晰独立的 reward”，它用的是回报和 Advantage

即使某个 reward 不是单独由 $$a_t$$ 造成，PPO 也能更新。因为它不直接做：

> 这个动作得了 1 分，所以提高它的概率。

它做的是：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
\hat A_t = \delta_t + \gamma\lambda \delta_{t+1} + \cdots
$$

然后：

$$
L^{CLIP} = \mathbb E[
\min(
\rho_t \hat A_t,
\text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
)]
$$

这里的 $\hat A_t$ 表示：

> 在 $s_t$ 下执行 $a_t$ 后，实际观察到的后续回报，相对于 Critic 基线 $V(s_t)$ 好多少。

所以即使“左脚落地成功”的奖励出现在后面几步，GAE 也会把==🔴后续奖励向前传播==，做信用分配。

---

###### 5. 固定 20ms 的作用：让 MDP 的时间尺度定义清楚

固定 `decimation=4`、`sim.dt=0.005` 后：

$$
\Delta t = 4 \times 0.005 = 0.02\text{s}
$$

每次 PPO 交互就是：

$$
s_t \rightarrow a_t \rightarrow \text{20ms 物理演化} \rightarrow r_t,s_{t+1}
$$

这使得：

- 每条 transition 的时间间隔一致；
- $\gamma$ 的折扣尺度一致；
- reward scale 可以按 `dt` 缩放；
- 4096 个并行环境容易同步；
- PPO 的 rollout 数据组织简单；
- GAE 的时间步定义清晰。

这不是说“物理引擎只能 20ms 算一次”。物理引擎仍然 5ms 算一次，PD 也 5ms 更新一次。`decimation=4` 只是说：**Policy 每 20ms 决策一次，环境每 20ms 返回一次 observation 和 reward。**

---

###### 最核心的一句话

在 Unitree 这类控制任务中：

> **Action 执行完成 ≠ 控制周期结束；控制周期结束 ≠ 机器人动作完成。**

PPO 记录的是：$(s_t,a_t,r_t,s_{t+1})$

其中 $a_t$ 是“20ms 内持续生效的控制目标”，$r_t$ 是这段状态转移的即时反馈。它不需要 $a_t$ 是一个“做完的任务”。  

如果你把 action 理解成“独立任务”，你的担心完全成立；但一旦把 action 理解成“让这个动作持续20ms”，而reward就是"==🔴这个动作持续20ms后产生的奖励=="，这个矛盾就消失了。