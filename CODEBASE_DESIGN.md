# ICL_Policy codebase 设计提案

日期：2026-09-20。状态：规划；尚未迁移 policy、转换数据或启动训练。

## 1. 目标与边界

项目根目录使用 `/home/qikang/ICL_codebase/ICL_Policy`。面向 Piper 单臂真机数据的训练、离线推理和可供真机调用的模型接口；不依赖 RoboTwin、XPolicyLab 或任何 benchmark 环境。

共享数据读取、物理量定义、数据划分和实验记录；保留每个方法自己的模型、样本组织、训练循环、配置系统、checkpoint 和推理逻辑。根目录不提供统一 train.py，也不设计统一 Trainer 基类。

用户已明确：human video 是独立的人类任务示范，作为 ICL prompt，不需要与机器人轨迹逐帧同步。它拥有自己的时间轴，通过任务/示范关系关联机器人 query，而不是作为机器人的第三个同步相机。

## 2. 本次实际检查结果

### 2.1 当前数据

当前只有 `data/pap_rope`，是自定义 `piper_hdf5_external_av1_v1` 格式，不是 LeRobot v3。未发现 Parquet 或 LeRobot 的 meta/info.json。

| 项目 | 检查结果 |
| --- | --- |
| 轨迹 | 100 条，每条 0.hdf5、0.valid_range.json 和两路 MP4 |
| 完整行数 | 63,478；每条 457–1,077 帧 |
| 标注有效行数 | 35,987；每条 242–614 帧 |
| 数值字段 | action: [T,7]；observations/qpos: [T,7]；observations/eef: [T,6] |
| 图像 | HDF5 中已外置；third/wrist 两路 MP4，样本为 AV1、640×480、标称 30 fps |
| 语义元数据 | action 为 master native control frame；state 为 slave feedback |
| 一致性 | 100 条数值字段长度匹配且均有限；200 个 MP4 声明帧数与对应 HDF5 一致 |
| 裁剪 | 所有 valid_range 都合法，total_frames 与当前 HDF5 长度相等 |

这里保留的是完整录制。应按 `[valid_start, valid_end)` 读取有效部分；以前 trimmed 数据的 35,987 帧不能直接当成当前文件的总帧数。需要明确保存裁剪状态和原始行索引，避免重复裁剪。

检查范围：遍历了全部数值数据和 200 个视频的容器帧数元数据，实际试解码了一个视频的首帧；尚未逐帧解码所有视频，也没有验证控制量单位、坐标系或物理标定。

时间相关问题：

- 有效区间内，观测和相机时间戳严格递增；完整数据中存在一个零时间戳及相应回跳，落在有效区间外。
- 有效区间内，action 时间戳有 1,652 次重复，无倒退。可能是同一控制命令被多次采样，需要确认采集逻辑，不能直接按重复行删除。
- 观测时间间隔中位数约 33.37 ms，99 分位约 123.44 ms；标称 30 fps 不等于采集无抖动。
- 同行观测时间减 action 时间，中位数约 2.47 ms，99 分位约 288.74 ms，最大约 4.99 s。应检查命令保持/延迟语义，不自动整体平移 action。
- MP4 播放时间、原始采集时间和控制采样时间必须分别保留。相同帧数只能证明索引长度相容，不能证明物理同步准确。

### 2.2 三个原始仓库

本次检查时三个仓库工作区干净，版本分别为 ICRT `e0c9588`、ICLR `12f9290`、BPP `ec29e62`。

| 方法 | 当前原生训练入口 | 样本组织 | 主要接入工作 |
| --- | --- | --- | --- |
| ICRT | policy/ICRT/scripts/train.py | 按任务组织多条 observation/proprio/action 序列；EOS、prompt/weight mask、action chunks | 增加数据读取后端和 Piper 配置，保留序列拼接与因果建模 |
| ICLR | policy/ICLR/scripts_real/train_real_visual_trace.py | 类似 ICRT，增加 image-space visual trace 监督和预测 | 除数据适配外，补 Piper visual trace 的生成、坐标变换与无 teacher forcing 推理 |
| BPP | policy/BPP/behavior_prompting/train_network/train.py | 当前机器人观测与 behavior prompt 分开；prompt 含分段动作轨迹；相对位姿处理 | 适配 Piper 的双相机、EEF/夹爪、任务/示范索引；保留原始 prompting sampler |

已发现不能原样沿用的默认值：

- ICRT、ICLR 都有 maximum_length=450。若直接用完整轨迹，会过滤全部 100 条；即使按有效区间裁剪，也会过滤 6 条。应把轨迹长度限制配置化并完整报告过滤情况，不能默默丢掉长轨迹。
- BPP UMI 配置预设部分数据 60 Hz、部分相机 10 Hz，再下采样到 20 Hz；不能直接套在本数据上。
- BPP UmiTaskDataset.get_normalizer() 内部硬编码 num_workers=32，且累积低维 batch；需要改成可配置、有内存上限的统计过程。
- ICLR 实机脚本引用的 config/dataset_config_real_visual_trace.json 当前不存在；GPU 编号、输出路径和 WandB entity 也是作者机器的设置。
- ICLR README 重点介绍 LIBERO，但本地确实有实机训练与模型代码；它们仍需单独 smoke test，不能据此声称 Piper 已可直接运行。
- ICLR 的默认依赖包含 LIBERO/MuJoCo；新工程的训练环境应按实机执行路径拆出所需依赖，不把全部仿真依赖装进公共包。

### 2.3 XPolicyLab 和已有 DP

XPolicyLab 主要提供 policy 适配契约、服务端、checkpoint 命名/查找、数据转换入口与环境接线，各 policy 仍使用各自的训练实现。

`utils/data_loader.py` 面向其 xspark/RoboDojo HDF5，递归使用 item[()] 读取数据，不是通用、按窗口惰性读取的 LeRobot v3 后端。不能拿它作为新工程的数据基础。

已有 DP 的 train.py 调用 diffusion_policy.workspace，真实数据使用自定义 RealPiperImageDataset 和 DP 自己的 ReplayBuffer/SequenceSampler/normalizer。它是基于 DP 的本地改版，不是使用统一 XPolicyLab Trainer，也不是完全未修改的上游 DP。

可以移植：DP 模型与已验证的空间特征头配置、policy 自包含组织方式、少量日志/检查工具。应去掉 benchmark/env_cfg/robot_info 的路径耦合。XPolicyLab 的特殊图片解码规则只适用于它自己的历史数据，不应照搬到标准 MP4 解码。

## 3. 推荐目录

保留目前 policy 下三个原始仓库的位置，在各仓库内部增加小规模适配。以下为目标布局，并非已创建的框架：

```text
ICL_Policy/
  pyproject.toml                    # 仅安装轻量公共包，不捆绑所有 policy
  src/icl_data/
    schema.py                      # Episode/Context/ActionSpec 等数据契约
    readers/
      piper_hdf5_video.py           # 当前数据，按区间惰性读
      lerobot_v3.py                 # 未来主格式，按 metadata 定位 shard
    video.py                       # 解码、时间/帧索引、有限缓存
    geometry.py                    # 显式坐标系、旋转与相对位姿转换
    indexing.py                    # episode、task、context 关系
    splits.py                      # 持久化数据划分与泄漏检查
    statistics.py                  # 只在训练集合拟合，可流式累计
    provenance.py                  # 源数据、预处理和配置版本
  configs/
    datasets/pap_rope.yaml          # 数据位置、字段映射、裁剪约定
    robots/piper_single.yaml       # 经确认的物理约定
    protocols/                     # prompt/query 和数据划分规则
  policy/
    ICRT/                          # 原模型、scripts/train.py 保留
      adapters/                    # 该方法的数据/推理适配
      configs_piper/
      infer_piper.py               # 按需添加薄入口，调用原生模型
      UPSTREAM.md                  # 来源 commit、许可、本地变更
    ICLR/                          # 原 scripts_real/ 和 iclr/ 保留
      adapters/
      configs_piper/
      infer_piper.py
      UPSTREAM.md
    BPP/                           # 原 behavior_prompting/ 保留
      adapters/
      configs_piper/
      infer_piper.py
      UPSTREAM.md
    DP/                            # 可选：迁入之前的无 prompt 对照
    MyMethod/                      # 自己的 train/infer/model/dataset
  data/
    pap_rope/                      # 当前数据保留原位
    lerobot/                       # 未来的正式 v3 数据集
    human/                         # 独立 human demonstration videos
    annotations/                   # tasks、context 关系、标定、trace
    manifests/                     # 版本化 episode/context 索引
    splits/                        # 固定的 train/val/test 及 context pools
    cache/                         # 可重建的 BPP zarr、trace 等
  tools/                           # inspect/validate/index/export/profile
  tests/                           # 契约、对齐和 policy 接入检查
  runs/<policy>/<run_id>/           # 权重、最终配置、统计、日志
  docs/
```

每个 policy 继续用自己的 Python/Conda/uv 环境；公共包避免强依赖 Torch、Hydra、Transformers，视频和 LeRobot 后端使用可选依赖。版本固定后逐个安装测试，不能假设三个仓库能共用一套依赖。

三个 policy 目前是独立 Git 仓库。工程纳入版本管理时保留 fork/submodule 的来源和固定 commit；本地修改先在对应 fork 中提交，再更新父仓库引用，避免父仓库只记录一个无法复现的工作树。数据、cache、权重不入 Git。提取上游代码保留许可和署名。

## 4. 公共数据接口：共享读取能力，不强制统一训练 batch

第一版只需很小的 EpisodeStore 接口：

```python
store.list_episodes(split_spec)
store.describe(episode_id)                 # 长度、任务、字段、时间轴、来源
store.read_steps(episode_id, indices, fields)
store.read_video_frames(stream_ref, indices)
```

内部可支持连续 slice 和离散 indices，按需读字段。公共层返回显式声明布局的 RGB uint8、未归一化低维量、时间戳和有效性信息；policy 自己决定 CHW/TVC、resize/crop、增强、normalization、collate 和 GPU 转移。

LeRobot v3 后端优先封装固定版本官方 reader，按元数据定位跨文件 episode/video offset；不使用“一个 episode 对应一个 MP4/Parquet”的旧假设。当前 HDF5 后端先跑通，不要求为使用框架而先复制整套视频。

EpisodeRecord 至少包括：dataset_id、episode_id、task_id、source_session_id、robot_id、可用字段、完整/有效区间、原始行映射、各时间轴、源版本。不要根据文件夹名字猜任务细分或把未知单位写成已确认。

逻辑训练样本分为：

- context_refs：一个或多个示范引用，可来自 robot demo 或 human video。
- query_ref：机器人 episode 与当前时刻/窗口。
- target_spec：要监督的动作语义、时间范围、有效 mask。

query 的未来 action/trace 属于训练目标，不能因它们也在同一个 dict 中就被编码为当前可见输入。这个接口描述数据权限与来源，不规定所有 policy 的张量结构。

## 5. ICL 数据组织与 human video

ContextRecord 建议包含：context_id、source_kind、task_id、source_session_id、video_refs、valid_range、available_modalities、可选 annotation_refs。机器人示范可以有 state/action；human video 没有这些字段就是 absent，不能用全零伪装成有效机器人轨迹。

另外维护 context-to-query 的关系表：task-level compatible、显式 demonstration pair、或研究中定义的其它关系。允许同一 human video 对应多条 robot demo，也允许多个人类示范对应一个任务。没有标注时不能自动建立可靠的语义配对。

独立 human video 使用自身 fps/时间戳与有效区间。模型 adapter 决定均匀采样、关键帧、分段或视觉 token；不插值成人为逐帧对齐的机器人时间轴。记录原始时刻和 temporal mask。若未来研究需要学习阶段对齐，由方法实现，并把派生对齐结果保存在 cache。

已完成的独立示范可以完整提供给模型；在线 query 只能看到当前和历史机器人观测。训练目标使用 query 未来数据是允许的，但推理输入不能访问真实未来动作或未来 trace。

公共层只约束可选哪些示范、任务关系、划分和随机种子；每个方法保留自己的 prompt 帧采样、序列拼接、chunk 和 mask。固定验证 context/query 配对，训练可以在合规池内随机抽样并记录种子。

必须区分两个实验系列：

1. 原方法数据适配：给 ICRT/ICLR/BPP 各自需要的 robot demonstrations，保持方法核心设计。
2. human-video 改进：在原方法上新增或改变 prompt 编码。明确标成如 ICRT+HumanVideo，而非声称原版原生支持纯视频 prompt。

ICRT 原生 prompt 包含观测、state、action；ICLR 还使用 trace；BPP 本地 UMI 分支的 prompt 使用相对动作片段。iPhUMI 的“人类示范”带有专门采集的操作信息，不能等同于任意普通 RGB 视频。

若做 human-video 信息预算下的对比，所有方法可获得的信息应列成矩阵；某个 baseline 无法消费某种模态时，记录其适配方式或明确作为信息不等价的参考，不能给它偷偷增加机器人标签。

## 6. ActionSpec：先固定物理含义，再适配模型表示

公共存储保留 raw master action、slave qpos、slave eef 三种独立量。机械臂为单臂并不意味着所有模型内部必须是 7 维：xyz+rotation6d+gripper 可以是 10 维，ICRT/ICLR 还可有 EOS。内部表示必须能映射回明确的控制量。

ActionSpec 至少声明：

```yaml
space: joint | eef
source: master_command | fk_master_command | achieved_next_state
representation: absolute | relative_to_query | incremental
translation_frame: base | tool
rotation_rep: euler | axis_angle | quaternion | rotation_6d
rotation_convention: explicit_required
units: explicit_required
gripper_semantics: width | normalized_opening | binary
gripper_open_direction: explicit_required
target_offset: explicit_required
control_dt: explicit_required
```

这里的枚举是设计表达，不是可直接运行的配置。

EEF 目标有两条可行路线，需要由机器人采集/控制语义决定：

- 从 master joint command 经正确的 Piper FK、零位/方向映射和工具标定得到 commanded EEF target。它更接近控制命令，但必须验证 FK 与 slave EEF 的参考系、工具和控制映射。
- 用相邻或未来 slave EEF 构造 achieved pose target。当前已有 DP 用的就是相邻实际 EEF 差分加 master gripper；这能定义学习任务，但它不是原始 commanded action，包含系统响应和采样周期影响。

初始数值范围提示关节和姿态像弧度、位置和夹爪像米，但本次未找到采集程序来确认，不能据此固化单位。夹爪原始 action 有约 -0.0006 的小负值；记录并确认处理规则，不在不同 adapter 中各自偷偷裁剪。

旋转转换使用显式约定的 SO(3)/SE(3) 运算；简单 wrapped Euler 差分不等于一般的相对旋转。区分“每一步相对前一步的增量”和“整个 chunk 相对 query 当前姿态的目标”。后者不能在执行时反复累加。

建议先实现一条经验证的公共物理目标定义，再让各 policy 做本地编码/解码。保留原方法配置与受控对比配置两组；若要求严格对比，统一原始监督来源、控制时间尺度和可用信息，记录仍保留的方法差异。

horizon 32 可用于自己的实验配方，但不作为公共数据格式的常量。分别记录 observation history、prompt 长度、prediction horizon、execution horizon 和 control_dt。32 帧在 30 Hz 约 1.07 秒；重采样后按真实时间比较。执行步数属于推理策略，应限制在模型可预测范围内并确认各方法缓存/历史更新逻辑。

## 7. 各 policy 的落地方式

### ICRT

保留 scripts/train.py、icrt 模型和原 sequence construction。增加可选 reader factory，让取 episode/image/state/action 的位置转到 EpisodeStore；训练入口只需支持选择原生 dataset 或 Piper adapter。保持 EOS、prompt mask、multi-step loss 的语义，逐项验证跨 episode 边界。

现有 HDF5 文件在 dataset 初始化时打开的做法不适合直接跨多 worker 共享；新 reader 按 worker 惰性创建文件/视频句柄。长轨迹改为窗口采样或显式配置过滤，不把全部 episode 长度当成必须小于 context 长度。

### BPP

初版优先导出可重建的原生 Zarr cache，复用其 ReplayBuffer、任务索引、SequenceSampler、pose conversion 和 diffusion workspace。这样不必先重写最复杂的 prompt sampler。导出过程按 episode/有限视频块顺序处理，图像尺寸与 cache 容量可配置。

生成真实的 task/episode 索引与固定 split；Piper 两路相机分别配置为 third/wrist，去掉 UMI 60 Hz/10 Hz 的假设。EEF Euler 先按确认的约定转换为其期望的 axis-angle/rotation6d，不能直接改字段名。将 normalizer 的 worker 数和统计内存控制参数化。

后续若磁盘或导出耗时成为问题，再为其 ReplayBuffer 接口增加惰性后端。先测量瓶颈，不预先建设通用存储框架。

### ICLR

保留 scripts_real 入口、原 visual-trace 序列和模型，补实机 dataset 配置与日志配置。raw data 增加不了现成 trace，必须生成或标注。

如果已获得相机内外参和 EEF 工具点标定，可以把未来 EEF 轨迹投影到第三视角；否则采用视觉跟踪/上游对应方法，记录标注工具、置信度和失败样本。投影与视觉伪标注属于不同监督来源，应作为实验条件记录。

resize/crop/flip 必须同步变换 trace 坐标和相机几何。示范 trace 可以由完整已录制示范产生；query 的未来 trace 只作监督，在线推理走模型预测 trace 的非 teacher-forcing 路径。不能用离线真实未来 trace 评估然后把结果当成真实部署能力。

### DP 与自己的方法

DP 可迁入作为无 ICL prompt 的基本对照，保留已有空间特征头；接入共享 source/split/ActionSpec，重新检查训练集合统计和解码一致性。已有权重保持原来的预处理与 action 语义，不能静默改配方后继续加载部署。

自己的方法可自由设计 human video encoder、融合和损失，复用 EpisodeStore、context 索引和数据划分。公共包不实现这些模型层。

## 8. 推理接口与复现记录

每个 policy 保留自己的推理入口和模型调用；可额外实现轻量可选协议：

```python
session.reset()
session.set_context(contexts)
chunk = session.predict(robot_observation)
```

这是接口示意，不要求所有上游模型继承统一基类。session 管理各自的 KV cache、prompt feature cache、历史动作和归一化。reset 必须清掉上一个任务的上下文。耗时的人类视频编码可在 set_context 时完成。

输出 ActionChunk 至少携带 values、ActionSpec、有效长度、目标时间/间隔和参考姿态。checkpoint 包含或伴随 normalizer、camera mapping、预处理配置、动作定义与 source/split 版本。离线推理使用同一解码路径还原物理量。

真机控制器独立于 policy 包，未来消费明确的 ActionChunk 并负责时间调度和设备命令。工程现在只提供模型输出契约和离线回放，不为了训练引入 ROS/CAN 或仿真环境。

WandB 在各原生训练循环里接入。统一 optimizer_step 为横轴，并另记 micro_step、epoch、samples/tokens 和 wall time，避免不同梯度累积方式把 step 含义混淆。保留各方法自己的 loss 名称，同时记录采样后、反归一化后的物理量误差和推理耗时；不同模型的原生 loss 数值不能直接比较。

运行记录保存：所有有效配置、各仓库 commit 与 dirty diff、环境锁定、数据/切分/标注版本、目标定义、normalizer、seed、预训练来源、resume 状态。路径从项目/配置目录解析，支持 ICL_DATA_ROOT，避免再出现 workspace_robotwin 的绝对路径残留。

## 9. ICL 实验协议

当前 pap_rope 是一个任务的数据，只能用于数据管线验证、动作拟合和同任务 prompt 敏感性测试。不能用它单独证明对未见任务的 ICL 泛化；需要后续多任务或可定义的任务变体。

统一原始数据并不要求统一所有 sampler。应统一 task 定义、split、可用 context 池、可见信息、动作物理目标及比较预算。

- 先按 task/session/object 等实验单元确定 train/val/test，再在集合内建立 support/query，避免随机切帧泄漏。
- 新任务测试允许提供该测试任务的 support demos，但它们不参与梯度训练；query 来自独立 episode/session。
- robot query 不能把自身完整轨迹作为测试 prompt；训练中的上游 repeated-trajectory augmentation 若保留，记录其作用范围。
- human video 的同源片段/重复导出共享 source_id，不把同一拍摄的视频片段放到互相隔离的集合。
- normalizer、可学习预处理和数据驱动统计只使用训练池。固定验证 prompt，避免模型选择时采样变化掩盖退化。
- 同任务常量 prompt 可能被忽略；安排正确 prompt、错误任务 prompt、去掉 prompt、不同示范数量等可解释对照，确认模型是否利用 context。单任务时错误任务对照需等多任务数据到位。

## 10. 内存、解码与缓存

初始 worker=0，解码线程=1，小 batch 测峰值，再试 1–2 workers；normalizer 不能另开 32 workers。限制 BLAS/OpenMP/视频解码线程和 prefetch，统计主进程与 workers 的总 RSS。

每个 worker 自己管理有限数量的视频/HDF5 句柄与 LRU 缓存；按视频分组、合并相邻帧请求，避免一个 prompt 对同一个 AV1 文件反复随机 seek。只加载实际需要的图像步数和字段，不按 action horizon 盲目分配所有 RGB。

AV1 可以读，但长 prompt 随机访问可能昂贵。先 profiling，然后按需生成短 GOP 视频或 policy 专属图像 cache；不默认给每个 baseline 复制完整分辨率数据。

cache key 包含源版本、有效区间、采样/时间定义、图像处理、动作定义、几何/trace 版本；涉及统计或派生采样时包含 split。换配置后不能误用旧 cache。cache 可删除重建，源数据只读，导出支持断点和原子完成标志。

## 11. 实施顺序与验收

1. 数据底座：把本次检查固化为 inspect/validate 工具、数据清单、Piper source reader 与有效区间映射。验收为全部 100 条可索引、有效行数 35,987、不跨 episode、不重复裁剪、时间问题有报告。
2. 物理约定与划分：确认采集单位/控制映射/坐标系，选择首个 EEF 目标定义，建立 split/context 池。验收为训练与推理动作往返一致、旋转边界检查通过、统计不使用 held-out 数据。
3. 先接通 ICRT：它无需新增 trace 标注，能验证共享 reader 与原生 ICL 序列接口。验收为 batch 语义检查、小样本拟合、保存/加载一致和有状态离线推理。
4. 接通 BPP：生成有限容量 Zarr cache、映射任务索引与双相机、修正 worker/fps；验收为导出与源片段数值一致、prompt/query 不串数据、采样推理回到正确物理尺度。
5. 接通 ICLR：准备 trace 标注和必要几何，验证增广同步，测试无真实未来 trace 的推理。不能仅有一次 forward 就宣布完成接入。
6. 加入独立 human videos：先完善 context 标注/配对，再实现自己的视频旁路；按原版与修改版分别记录消融结果。DP 可作为阶段 2–3 的可选简单对照。

本轮只完成调查和设计文档。下一步首个可交付版本应是数据检查工具、轻量 reader、明确的 ActionSpec 和一个 baseline 的端到端接入；之后用真实接入需求扩展公共层。

## 12. 依据

- 本地 XPolicyLab/README.md、utils/data_loader.py、DP/train.py 和 diffusion_policy/dataset/real_piper_image_dataset.py。
- 本地 ICRT/icrt/data/dataset.py、DATASET.md、scripts/train.py。
- 本地 ICLR/iclr/data/dataset_real_visual_trace.py、scripts_real/train_real_visual_trace.py、iclr/models/policy_real/。
- 本地 BPP/behavior_prompting/train_network/dataset/umi_task_dataset.py、config/task/umi.yaml 和 docs/iphumi.md。
- [LeRobot v3 官方格式说明](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)：Parquet/MP4 分片和 metadata 索引。
- [ICRT 论文](https://arxiv.org/abs/2408.15980)：sensorimotor trajectory prompting。
- [ICLR 论文](https://arxiv.org/abs/2603.07530)：visual reasoning trace 与 action 联合预测。
- [BPP 论文](https://arxiv.org/abs/2606.30457)：behavior prompting 与 iPhUMI。
