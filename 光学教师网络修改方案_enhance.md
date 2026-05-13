# 光学教师网络修改方案 Enhance

## 1. 当前文档定位

这份文档作为当前分支唯一继续维护的增强方案记录，后续只以这里为准。

本次已做的处理：

1. 不再使用 `教师网络修改方案_StrengTea_补充_20260430.md`
2. 清理旧方案里没有实际落地、或已经不再采用的讨论项
3. 只保留当前代码中已经真正实现的修改

## 2. 当前主要问题判断

结合最近几轮训练日志和可视化，当前主问题不是“完全学不会位置”，而是：

1. 定位有一定基础，但 `Precision` 很低
2. 同一目标上容易出现多个重复框、小碎框
3. 可视化看起来目标附近有响应，但 `mAP50` 和 `mAP50:95` 仍然偏低
4. 说明当前瓶颈更偏向：
   - 正负样本分配过宽
   - 背景抑制不够
   - 检测头后处理对重复框压制不足

因此这一轮修改主线不是继续放大召回，而是优先压误检、减重复框、提高有效框质量。

## 3. optical_teacher_yolo.py 已落地修改

## 3.1 训练轮数重分配

当前总轮数已改为 `500`：

1. `teacher = 120`
2. `detector = 240`
3. `joint = 140`

对应代码：

- `TEACHER_STAGE_EPOCHS = 120`
- `DETECTOR_STAGE_EPOCHS = 240`
- `JOINT_FINETUNE_EPOCHS = 140`

这样分配的目的：

1. teacher 阶段先把教师塑形稳定下来
2. detector 阶段给最多轮数，重点解决当前检测精度不足问题
3. joint 阶段再用较长时间做联合收敛和平衡

## 3.2 面向低精度的参数收紧

这一组修改的核心目标是减少“一个 GT 扩散成很多正样本”的情况。

当前已调整为：

1. `NOOBJ_WEIGHT_BASE = 0.60`
2. `CLS_WEIGHT_BASE = 0.55`
3. `MAX_POSITIVE_ANCHORS = 1`
4. `NOOBJ_IGNORE_IOU = 0.60`
5. `NEIGHBOR_ASSIGN_MARGIN = 0.12`
6. `POSITIVE_SCALE_IOU_RATIO = 0.85`

对应作用：

1. 每个 GT 只保留更少、更严格的正样本锚框
2. 减少邻域扩散分配，缓解一个目标周围长出很多小框
3. 让更多“靠近但不该算正样本”的框重新回到背景抑制中
4. 提高分类和背景约束，避免只会“到处报目标”

## 3.3 检测头辅助热图约束减弱

为了避免 detector 被热图辅助项拉成“大面积发亮，大量边缘框”，当前已下调：

1. `DETECTOR_HEAT_AUX_WEIGHT = 0.05`
2. `DETECTOR_FEATURE_AUX_WEIGHT = 0.02`
3. `DETECTOR_OBJ_HEAT_AUX_WEIGHT = 0.003`

目的：

1. 保留教师特征对检测头的辅助引导
2. 但不让辅助热图继续主导检测头输出形态
3. 让 detector 更关注框质量、类别和目标性本身

## 3.4 重复框后处理补偿

针对“一个目标上多个框”的现象，当前已在后处理里加入加权框融合。

流程为：

1. 先按类别执行 NMS
2. 再执行同类框 `Weighted Box Fusion`

当前参数：

1. `USE_WEIGHTED_BOX_FUSION = True`
2. `WBF_IOU_THRESH = 0.60`
3. `WBF_SKIP_CONF_THRESH = 0.05`

说明：

1. 这不是从训练根源上解决重复框
2. 但对于“框大致都落在目标附近，只是太碎太多”的情况，通常比单纯 NMS 更稳
3. 这也是对你前面提出“同一目标多个框能否融合”的正式落地

## 4. 训练加速与吞吐优化

## 4.1 当前已启用的后端加速

`optical_teacher_yolo.py` 和 `optical_teacher.py` 已同步启用：

1. `USE_TF32 = True`
2. `USE_CUDNN_BENCHMARK = True`
3. `USE_CHANNELS_LAST = True`

实际生效内容包括：

1. `torch.backends.cudnn.benchmark = True`
2. `torch.backends.cudnn.allow_tf32 = True`
3. `torch.backends.cuda.matmul.allow_tf32 = True`
4. 模型使用 `channels_last`
5. 输入 batch 在送入 GPU 前转为 `channels_last contiguous`
6. GPU 搬运使用 `non_blocking`

适用前提：

1. 当前输入尺寸固定为 `640x640`
2. 主体结构以卷积为主
3. 单 GPU 训练不受影响

## 4.2 DataLoader 参数已统一增强

当前已加入：

1. `NUM_WORKERS = 8`
2. `PIN_MEMORY = True`
3. `PERSISTENT_WORKERS = True`
4. `PREFETCH_FACTOR = 4`

并统一通过 `build_dataloader_kwargs(...)` 构建训练和验证加载参数。

另外，原先不利于 Windows `spawn` 的匿名 `collate_fn` 已替换为顶层函数：

1. `identity_collate`

这样做的目的：

1. 让 `num_workers` 在当前 Windows / PowerShell 环境下更稳定生效
2. 减少 GPU 等待数据的空转
3. 在不改训练逻辑的前提下提升吞吐

### 4.2.4 Windows 下多日志文件问题与修复

这一轮还顺手修了一个容易忽略的 bug：

1. 当 `NUM_WORKERS = 8` 时，训练开始后会生成 `9` 份日志文件

原因不是单卡 `4090` 本身有问题，而是：

1. Windows 下 DataLoader 多进程默认使用 `spawn`
2. worker 进程会重新导入 `optical_teacher_yolo.py`
3. 如果脚本顶层就执行了 `Config.print_config()`、`init_log_file()`、`log_all_parameters()` 这一类带副作用代码
4. 那么每个 worker 都会各自再生成一份新的带时间戳日志

因此会出现：

1. 主进程生成 `1` 份
2. `8` 个 worker 再各生成 `8` 份
3. 最终总共 `9` 份

当前已修复为：

1. 顶层只保留 `Config.initialize()`
2. `print_config`
3. 后端开关初始化
4. `init_log_file`
5. `log_all_parameters`

全部移动到 `train()` 主进程入口中执行。

这样后续无论：

1. 单卡 `4090`
2. 单卡 `A6000`
3. 双卡 `A6000`
4. 只要 `num_workers > 0`

都不会再因为 worker 重新导入脚本而重复创建日志文件。

### 4.2.5 matplotlib / tkinter 多线程析构问题

在 Windows 下，如果训练脚本里用了 `matplotlib.pyplot`，而默认后端又落在 `TkAgg` 这一类交互后端上，那么配合 DataLoader 多 worker 时，容易出现：

1. `RuntimeError: main thread is not in main loop`
2. `Tcl_AsyncDelete: async handler deleted by the wrong thread`

这不是模型训练本身出错，而是：

1. worker 或后台线程在退出时清理到了 `tkinter` / `Tk` 相关对象
2. 但这些对象只能在主线程图形循环中安全销毁

当前已修复为：

1. 在 `optical_teacher_yolo.py`
2. 在 `optical_teacher.py`

都显式加入：

1. `import matplotlib`
2. `matplotlib.use("Agg")`

这会强制使用纯离屏后端，只负责保存图像，不再依赖 `Tk` 图形界面。

这样适合当前训练场景，因为我们需要的是：

1. 保存 loss 曲线
2. 保存训练可视化
3. 不需要弹窗交互显示

后续如果在其他分支也用了 `matplotlib.pyplot`，建议同样在导入 `pyplot` 之前显式设置 `Agg` 后端。

### 4.2.1 pin_memory 的作用

`pin_memory` 不是一个“继续调大”的数值参数，而是一个布尔开关。

当前含义是：

1. `False`：CPU 侧 batch 使用普通内存
2. `True`：DataLoader 预先把 batch 放到页锁定内存

它的主要作用不是提升精度，而是提升 CPU 到 GPU 的数据搬运效率，尤其是在下面这种场景更明显：

1. 使用 CUDA 训练
2. DataLoader 有多个 worker
3. 训练循环里配合 `tensor.to(device, non_blocking=True)` 一起使用

当前代码里已经是这一套组合：

1. `PIN_MEMORY = True`
2. 数据搬运使用 `non_blocking`

所以这一项在本分支中已经落地，不需要再额外“调大”。

### 4.2.2 pin_memory 是否会提高训练速度

结论是：

1. 对 `GPU` 训练，通常会有帮助
2. 对 `CPU` 训练，基本没有意义
3. 它主要提升的是吞吐和等待时间，不直接提升 `mAP`

更准确地说，它可能带来的收益是：

1. 单 step 更平稳
2. GPU 等数据的空转更少
3. 在 `num_workers` 生效时，整体训练速度更容易提升

但它也有边界：

1. 会额外占用一部分主机内存
2. 如果数据读取本身不是瓶颈，收益不会特别大
3. 如果是小 batch、低分辨率、CPU 训练，提升可能不明显

### 4.2.3 后续在其他分支如何迁移

如果你待会在其他分支同步这部分，建议直接保持：

1. `PIN_MEMORY = True`
2. `num_workers > 0`
3. `tensor.to(device, non_blocking=True)`

不要把 `pin_memory` 理解成需要继续增大的超参数，它更像是“GPU 训练的数据搬运加速开关”。

## 5. optical_teacher.py 已同步修改

为了避免教师检测联合训练和 student 训练两条链路配置割裂，`optical_teacher.py` 已同步加入：

1. TF32 / cudnn benchmark / channels_last
2. 统一的 DataLoader worker 配置
3. `non_blocking` 数据传输
4. `build_dataloader_kwargs(...)`
5. `configure_runtime_backends(...)`

这意味着：

1. 后续 student 训练不会继续沿用旧的低吞吐配置
2. 当前两条主训练链路在运行时策略上保持一致

## 6. 本轮修改后的重点观察项

下一轮训练后，优先看下面几项是否改善：

1. `Precision` 是否明显高于之前接近 0 的状态
2. `Recall` 即使略降，也要换来更有效的 `mAP50`
3. 同一目标上的重复框、小碎框是否减少
4. 大目标是否更容易被单个主框覆盖，而不是很多小框拼出来
5. `mAP50` 是否先从当前低位抬升到一个可持续上升区间
6. 每 step 训练速度和 GPU 利用率是否更稳定

## 7. 当前方案的边界

需要明确的是，这一轮修改主要解决的是：

1. 重复框多
2. 误检多
3. 精度低于可视化观感

它还没有单独针对“教师特征纹理不够细、内部结构还偏糊”的问题做新一轮结构级改造。

也就是说，当前这版更偏向：

1. 先把检测头输出质量和评估指标拉起来
2. 再决定是否继续加大教师特征细节塑形

## 8. 后续继续修改时的原则

后续如果还要继续改，建议按下面顺序推进，而不是再次同时改很多方向：

1. 先看这一轮是否真正压住了误检和重复框
2. 如果 `Recall` 高但 `Precision` 仍极低，继续收紧正样本分配和背景抑制
3. 如果重复框明显下降但 `mAP50` 仍不高，再继续看分类分数和框回归质量
4. 只有在检测头输出已经基本稳定时，再继续单独加强教师纹理塑形

## 9. 结论

当前 `Enhance` 方案已经从“讨论稿”整理成“已落地配置说明”。

后续再看训练结果时，优先将：

1. 日志指标变化
2. 重复框数量变化
3. 大目标是否仍被很多小框替代
4. 训练速度是否明显更稳

作为是否继续下一轮参数调整的依据。

## 10. optical_teacher.py 相位坍塌问题

针对 `output\OpticalTeacher_enhance_teacher` 这一条训练链路，当前已经确认一个比“检测头权重”更根本的问题：

1. 学生相位在 `student_only` 前十几轮后快速收缩
2. `PhaseSpan` 从接近 `2π` 很快掉到约 `0.08~0.09 rad`
3. `PhaseStd` 也掉到约 `0.008~0.009`

这不是“范围稍小但还能用”，而是明显的相位坍塌。

### 10.1 为什么这会导致后面几乎完全失真

一旦相位层接近常数相位：

1. 两层 SLM 几乎不再提供有效波前调制
2. 学生输出退化成低纹理、低判别性的响应图
3. 后面的检测头即使继续训练，也是在拟合一个已经失真的输入分布
4. 所以会出现：
   - 位置不准
   - 类别不对
   - 单独加载 student 权重后几乎完全不可用

### 10.2 这次确认到的两个直接诱因

第一类是训练目标本身把相位往常数解压：

1. `feature loss + response loss` 可以通过更平滑、更低变化的学生输出快速降低
2. 原先的 `phase diversity` 只约束 `std`，而且权重偏小
3. 因此不足以阻止相位迅速收缩

第二类是训练和推理分布不一致：

1. `student.enable_norm` 只是运行时属性，不在 `state_dict` 里
2. 训练后半段 student 可能开启了归一化
3. 但单独加载 `optical_student_best.pth` 推理时，新建 student 默认还是另一种归一化状态
4. 这会进一步放大学生特征和检测头输入分布失配

### 10.3 当前已落地的修复

这次已在 `optical_teacher.py` 中加入四类修复：

1. student 从训练开始就保持归一化一致：
   - `NORM_ENABLE_EPOCH = 0`
   - 不再让训练后半段才突然切换 student 输出分布

2. student checkpoint 会额外保存：
   - `student_enable_norm`
   - 加载学生权重时会恢复这个标志

3. phase diversity 约束增强：
   - 不再只约束 `phase std`
   - 同时加入 `phase span` 下限约束
   - 并提高相位多样性损失权重

4. phase1 增加检测代理监督：
   - `student_only` 阶段虽然冻结检测头参数
   - 但允许检测损失通过冻结检测头反传到 student
   - 这样 student 不会只学成“低纹理亮斑”，而会保留对检测有用的结构

### 10.4 当前还额外加的保护

为了避免再次把“已经坍塌的学生”保存成 best：

1. 训练里增加了 phase collapse 检测
2. 当 `PhaseStd` 或 `PhaseSpan` 低于阈值时，日志会报警
3. `student_only` 阶段不会再把这种坍塌状态当成 `phase1 best`

### 10.5 后续看什么算有效

下一轮重新训练后，优先看：

1. `PhaseSpan` 是否还能长期保持在明显高于 `1 rad` 的范围
2. `PhaseStd` 是否不再在十几轮内掉到 `0.01` 附近
3. `student_only` 阶段的特征图是否不再那么快退化成平滑亮斑
4. 单独加载 `optical_student_best.pth` 后，`Optical_SLM_yolo_model.py` 推理是否不再几乎全失真

如果这四项恢复，才说明这条链路开始重新具备“可部署到 SLM 上”的基础。

## 11. optical_teacher.py 新一轮修正

### 11.1 `num_workers=12` 生成 `13` 个日志文件的原因

这和前面 `optical_teacher_yolo.py` 修过的问题本质一样：

1. Windows 下 DataLoader 多进程使用 `spawn`
2. worker 会重新导入 `optical_teacher.py`
3. 而 `optical_teacher.py` 之前仍然在顶层执行了：
   - `Config.print_config()`
   - `configure_runtime_backends()`
   - `init_log_file()`
   - `log_to_file(...)`

因此：

1. 主进程写 `1` 份
2. `12` 个 worker 各再写 `12` 份
3. 最终就是 `13` 份日志

当前已修复为：

1. 顶层只保留 `Config.initialize()`
2. 所有日志初始化与配置打印移入 `train()` 主进程入口

这样即使 `num_workers=12`，也只会生成一份日志。

### 11.2 当前不是“完全坍塌”，而是“相位有变化但检测不可用”

你这次描述的现象说明：

1. 相位层已经不再像之前那样迅速掉到 `0.08 rad`
2. 但 student 输出对检测头而言仍然不够判别
3. 所以后面会表现为：
   - 位置偏
   - 类别乱
   - 可视化看着有响应，但检测结果不准确

这说明现在下一步重点不是只防止“完全坍塌”，而是要让相位变化真正服务于检测任务。

### 11.3 相位 span 至少保持在 `pi ~ 2pi`

当前已进一步收紧目标：

1. `PHASE_SPAN_TARGET = 5.20`
2. `PHASE_STD_TARGET = 1.20`

并且增加了 best 保存约束：

1. `PHASE_BEST_MIN_SPAN = pi`
2. `PHASE_BEST_MIN_STD = 0.45`

作用：

1. student-only 阶段不会再把“相位跨度太低”的模型保存成 `phase1 best`
2. 这能避免后续 detector-only 和 joint 阶段从一个物理上已经不合格的相位状态继续训练

### 11.4 关于相位层学习率是否太低

你的思路是对的：当前相位层原本跟着普通 student 参数一起用较低学习率，确实容易导致：

1. 相位层更新太慢
2. 其他部分先把 feature loss 压下去
3. student 更倾向学成“平滑响应”，而不是有效波前调制

但不建议直接把整条 student 学习率粗暴拉到 `0.01`。

原因是：

1. `0.01` 对这类复数传播 + 相位参数通常过大
2. 很容易造成相位震荡、训练不稳、loss 抖动
3. 更稳的方式是只给相位参数更高学习率

当前已改为分组学习率：

1. 普通 student 参数：
   - `LEARNING_RATE = 1e-3`
   - `JOINT_STUDENT_LR = 2e-4`

2. 相位参数 `phase_raw`：
   - `PHASE_PARAM_LR = 2e-3`
   - `JOINT_PHASE_PARAM_LR = 5e-4`

也就是说：

1. 相位层学习率已经进入你建议的 `0.001 ~ 0.01` 区间低段
2. 但没有冒进到 `0.01`
3. 这样更适合当前这条 SLM 学生网络

### 11.5 本轮后的核心观察点

下一轮训练时，重点看下面三件事：

1. `PhaseSpan` 是否长期保持在至少 `pi` 以上，而不是又掉回很低
2. 检测可视化是否从“有响应但框不准”逐渐变成“主框更贴近目标”
3. `phase1 best` 是否不再保存那些相位跨度过低的状态

如果相位 span 已经稳定在 `pi ~ 2pi`，但检测仍然明显不准，那么下一步重点就该转到：

1. 冻结检测头反传给 student 的代理检测监督是否还要增强
2. detector-only 阶段是否需要更强的分类 / objectness 约束
3. student 输出是否还需要额外的局部纹理监督

## 12. optical_teacher_yolo.py 指标接近 0 的原因

针对 `output\OpticalTeacherYOLO_enhance_teacher` 这一条链路，当前已经确认：

1. 这次 `mAP` 几乎为 `0`
2. 不是单一由 WBF 或可视化设置导致
3. 更核心的原因是训练条件已经偏离了之前能跑出 `0.07` 左右 `mAP` 的版本

### 12.1 当前日志里最关键的异常点

从日志可以直接看到：

1. `Batch size = 64`
2. `Teacher init mode = scratch`
3. `Detector init mode = scratch`
4. `Max positive anchors = 1`
5. `Neighbor assign margin = 0.12`
6. `Positive scale IoU ratio = 0.85`
7. `Noobj weight = 0.60`
8. `Class weight = 0.55`

而 detector 阶段的验证结果基本长期是：

1. `Precision = 0`
2. `Recall = 0`
3. `mAP50 = 0`

说明 detector 根本没有真正学会输出有效正样本。

### 12.2 为什么会比之前更差

主要是三类因素叠加：

第一类：初始化被改成了 `scratch`

1. 之前能跑出 `0.07` 左右 `mAP` 的前提之一，是 teacher / detector 至少有较稳定的起点
2. 现在 teacher 和 detector 都从随机初始化开始
3. 但同时你又把正样本分配收得更严
4. 结果就是 detector 在前期几乎拿不到足够可学习的正样本

第二类：正样本分配收得过头

1. `MAX_POSITIVE_ANCHORS = 1`
2. `NEIGHBOR_ASSIGN_MARGIN = 0.12`
3. `POSITIVE_SCALE_IOU_RATIO = 0.85`

这三项一起会让一个 GT 的正样本锚框数量太少，尤其在训练早期很容易导致：

1. 没有足够 anchor 愿意成为正样本
2. detector loss 在降，但预测几乎全是背景
3. 最终验证时 `Precision/Recall/mAP` 都接近 0

第三类：背景抑制和分类惩罚过强

1. `NOOBJ_WEIGHT_BASE = 0.60`
2. `CLS_WEIGHT_BASE = 0.55`

在 detector 还没站稳前，这会更偏向让模型“宁可不报，也别报错”，结果就容易变成：

1. train loss 下降
2. val 上几乎没有有效框

### 12.3 为什么这不是 WBF 能解决的

你提到的“多个框融合后是否更接近 GT”这个思路本身没问题，但它适合：

1. 离线分析
2. 推理后处理补偿

不适合作为训练主因判断，原因是：

1. 推理阶段并不知道 GT
2. 不能在线上运行时根据“离 GT 更远了就取消融合”
3. 如果训练本身已经学成“几乎不出有效框”，那 WBF 根本无框可融

所以当前优先级应该是：

1. 先恢复 detector 的基本出框能力
2. 再考虑是否需要做更细的框融合策略

### 12.4 当前已回调的修改

这次已经把 `optical_teacher_yolo.py` 往“先恢复可学习性”方向拉回：

1. `BATCH_SIZE: 64 -> 8`
2. `TEACHER_INIT_MODE: scratch -> checkpoint`
3. `TEACHER_INIT_CHECKPOINT -> output\oty_m1\teacher_final.pth`
4. `DETECTOR_INIT_MODE: scratch -> checkpoint`
5. `SKIP_TEACHER_STAGE_IF_INIT: False -> True`
6. `MAX_POSITIVE_ANCHORS: 1 -> 2`
7. `POSITIVE_ANCHOR_IOU: 0.30 -> 0.25`
8. `NOOBJ_IGNORE_IOU: 0.60 -> 0.72`
9. `NEIGHBOR_ASSIGN_MARGIN: 0.12 -> 0.25`
10. `POSITIVE_SCALE_IOU_RATIO: 0.85 -> 0.70`
11. `NOOBJ_WEIGHT_BASE: 0.60 -> 0.40`
12. `CLS_WEIGHT_BASE: 0.55 -> 0.40`
13. `SMALL_OBJ_WEIGHT: 0.8 -> 1.0`
14. detector 辅助热图损失适度回调增强
15. `USE_WEIGHTED_BOX_FUSION: True -> False`

这套回调的核心不是“直接追求最终最优”，而是：

1. 先把 detector 从“几乎全 0 输出”拉回到“至少会稳定出框”
2. 让训练重新回到有正样本、可学习的区间

### 12.5 后续怎么判断这轮是否回正

下次重新训练后，重点看：

1. detector 阶段验证不应再长期 `Precision=0, Recall=0, mAP50=0`
2. 即使 `Precision` 低一点，也应该先恢复到至少能稳定有非零 `Recall`
3. `mAP50` 应先回到之前的 `0.05 ~ 0.07` 量级，再谈继续优化

只有先回到这个区间，后面再讨论：

1. 是否重新启用 WBF
2. 是否设计更复杂的框融合策略
3. 是否再收紧正样本分配

### 12.6 补充澄清：`scratch` 不是要改掉的主线

这里再单独澄清一次，避免后面误读：

1. `optical_teacher_yolo.py` 这条主线原本就是从 `scratch` 训练
2. 因此不需要为了这次分析，把 teacher / detector 初始化永久改成 `checkpoint`
3. 当前已经恢复为：
   - `TEACHER_INIT_MODE = scratch`
   - `DETECTOR_INIT_MODE = scratch`
   - `SKIP_TEACHER_STAGE_IF_INIT = False`

也就是说，这次接近 `0` 指标的真正问题，不是 `scratch` 本身，而是：

1. `scratch` 起点下
2. 同时叠加了过严的正样本分配
3. 过强的背景抑制
4. 再加上过大的 batch

才把 detector 压成了“几乎全背景输出”。

### 12.7 保守同类框融合

你之前在另外一个分支提的“保守同类框融合”思路是合理的，当前已经同步到这条主线代码里，但默认仍保持关闭，方便你后面自己决定是否开启。

当前逻辑不是普通的无脑 WBF，而是：

1. 先经过 NMS
2. 只在同类框内部尝试融合
3. 必须同时满足：
   - `IoU` 足够高
   - 中心距离足够近
   - 宽高比例差不能太大

也就是说，它的目标不是“只要重叠就合”，而是只融合那些本来就像是在描述同一个目标的候选框，尽量避免把偏框、小碎框、尺寸明显不一致的框硬并后反而离真实目标更远。

当前相关配置为：

1. `USE_WEIGHTED_BOX_FUSION = False`
2. `WBF_IOU_THRESH = 0.60`
3. `WBF_SKIP_CONF_THRESH = 0.05`
4. `WBF_CENTER_DIST_RATIO_THRESH = 0.25`
5. `WBF_SIZE_RATIO_THRESH = 1.35`

说明：

1. 这部分现在只是代码能力已具备
2. 默认不参与当前训练/评估主流程
3. 等后续 detector 恢复基本出框能力后，你再按需要手动开启更合适

## 13. `checkpoint + skip` 跳阶段说明

`optical_teacher_yolo.py` 里下面三个配置不要混在一起理解：

1. `TEACHER_INIT_MODE`
   - 决定 teacher 是否尝试从 checkpoint 初始化

2. `SKIP_TEACHER_STAGE_IF_INIT`
   - 决定 teacher 成功从 checkpoint 初始化后，是否跳过 teacher 阶段

3. `SAVE_TEACHER_WEIGHTS`
   - 只决定是否保存 `teacher_best.pth / teacher_final.pth`
   - 不参与是否跳过 teacher 阶段的判断

真正会跳过 teacher 阶段的条件是：

1. `TEACHER_INIT_MODE = checkpoint`
2. `SKIP_TEACHER_STAGE_IF_INIT = True`
3. teacher checkpoint 实际加载成功

如果第 3 条不成立，例如：

1. 路径写错
2. 文件不存在
3. 权重不兼容

那么程序不会跳过 teacher 阶段。

这次日志里属于这种情况：

1. 你配置了 `checkpoint + skip`
2. 但 `Teacher checkpoint not found`
3. 所以程序回退到了 `scratch`
4. 然后正常进入 teacher 阶段训练

当前已进一步改进：

1. 当 `TEACHER_INIT_MODE = checkpoint`
2. 且 `SKIP_TEACHER_STAGE_IF_INIT = True`
3. 但 checkpoint 没有加载成功时

程序会直接报错停止，而不是静默回退到 `scratch`。

这样后面你只要看到训练没启动，就能立刻排查 checkpoint 路径，不会再出现“本来想跳过，结果实际上偷偷从头训了 teacher”的情况。
