```Python

"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
```


nanoGPT 是一个极其精简且高效的 PyTorch 实现，它重新构建了 GPT（Generative Pre-trained Transformer）模型的训练流程。我们将深入分析提供的 `train.py` 代码，从配置、数据加载、分布式训练、优化策略到具体的时间步循环，剖析其中的技术细节和工程设计。

### 1. 配置与初始化

代码的开头定义了大量的配置参数，这些参数覆盖了 I/O、数据集、模型架构、优化器设置以及分布式训练（DDP）的选项。

*   **Model Configuration**: 默认配置旨在复现 GPT-2 (124M) 模型。
    *   `n_layer = 12`, `n_head = 12`, `n_embd = 768`：定义了 Transformer 的深度和宽度。
    *   `bias = False`：这是一个现代优化，许多现代 LLM（如 Llama）在 Linear 层和 LayerNorm 中去除了 bias，以减少计算量并 potentially 提高稳定性。
*   **Data Configuration**:
    *   `block_size = 1024`：上下文窗口大小，即序列的最大长度（Context Length）。
    *   `gradient_accumulation_steps = 5 * 8`：这是一个关键参数，用于在显存有限的情况下模拟更大的 Batch Size。通过多次反向传播累积梯度，使得有效 Batch Size 变为 `micro_batch_size * accumulation_steps`。
*   **System Configuration**:
    *   `dtype = 'bfloat16'`：在支持 BFloat16 的 GPU（如 Ampere 架构 A100 或更新）上，BF16 是首选，因为它具有与 Float32 相同的动态范围，不容易出现溢出或下溢，且不需要 Loss Scaling。如果不支持，则回退到 Float16。
    *   `compile = True`：利用 PyTorch 2.0 的 `torch.compile` 技术。这会通过 kernel fusion 等技术将模型图编译为更高效的执行形式，通常能带来显著的加速。

### 2. 分布式训练

代码中包含了处理 DDP (Distributed Data Parallel) 的逻辑，这是在多卡或多节点上训练 LLM 的标准方式。

*   **Process Group Initialization**:
    ```python
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend) # backend 通常是 'nccl'
    ```
    通过检测环境变量 `RANK` 来判断是否处于分布式环境中。
*   **Gradient Accumulation Adjustment**:
    ```python
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
    ```
    这里做了巧妙的数学调整。如果有 4 张卡，原本计划累积 8 步，每张卡只需负责 `8 // 4 = 2` 步的局部累积。因为 DDP 在反向传播结束时会自动对全局梯度做 All-Reduce（求和平均），所以总的累积效果不变，但每张卡的负载被分摊了。

**DDP 通信优化技巧**：
在训练循环中，你会看到 `model.require_backward_grad_sync` 的设置。这是为了减少通信开销。在梯度累积的微步中，禁止梯度同步，只有当所有微步完成，即最后一次反向传播时，才进行跨 GPU 的梯度同步。

### 3. 数据加载策略: "Poor Man's Data Loader"

nanoGPT 的数据加载器非常高效，避免了复杂数据加载器的开销，直接利用 `numpy.memmap`。

*   **Memory Mapping**:
    ```python
    data = np.memmap(..., dtype=np.uint16, mode='r')
    ```
    `np.memmap` 允许系统将文件直接映射到内存地址空间，而不需要将整个文件读入 RAM。这意味着你可以训练比物理内存大得多的数据集。
*   **Data Type Optimization**:
    使用 `np.uint16` 存储数据。GPT-2 的词表大小是 50257，这完全可以用 16 位无符号整数表示（最大 65535）。相比标准的 `int64`，这节省了 75% 的磁盘空间和内存带宽。
*   **Pinned Memory**:
    ```python
    x.pin_memory().to(device, non_blocking=True)
    ```
    使用 `pin_memory` 将页面锁定在内存中，防止操作系统将其换出到磁盘。这使得数据从 CPU 传输到 GPU 时可以异步进行，避免 CPU-GPU 传输成为瓶颈。

### 4. 模型初始化策略

代码支持三种初始化模式：
1.  **`scratch`**: 从零开始随机初始化权重。
2.  **`resume`**: 从检查点恢复训练，包括优化器状态（这对于 AdamW 等自适应优化器至关重要，因为它们维护了一阶和二阶矩估计）。
3.  **`gpt2*`**: 加载 OpenAI 发布的预训练权重。这通常用于微调或作为初始化起点以加快收敛。

**Technical Detail**: 注意 `state_dict` 的处理逻辑，特别是移除 `_orig_mod.` 前缀。这是因为使用 `torch.compile` 后，PyTorch 会在模型参数名前加上这个前缀，为了兼容旧版 checkpoint 需要手动清理。

### 5. 混合精度训练与优化器

*   **GradScaler**:
    ```python
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    ```
    仅在使用 `float16` 时启用。Float16 的数值范围很小，梯度容易变成 0（下溢）。`GradScaler` 动态地将 Loss 缩放（例如乘以 65536），使其保持在 FP16 的有效范围内，在反向传播后再将梯度 Unsclaing 回来。
*   **AdamW Optimizer**:
    AdamW 是 Transformer 训练的事实标准。
    *   参数：`weight_decay=1e-1`（L2 正则化），`beta1=0.9`, `beta2=0.95`。
    *   模型通常包含权重衰减的参数和不包含权重衰减的参数（如 LayerNorm 和 Bias），nanoGPT 的 `model.configure_optimizers` 内部会处理这种分组。

### 6. 学习率调度

代码实现了一个带有 Warmup 的 Cosine Decay Schedule。

**公式解析**：
给定 `learning_rate` ($\eta_{max}$), `min_lr` ($\eta_{min}$), `warmup_iters` ($T_{warm}$), `lr_decay_iters` ($T_{max}$):

1.  **Warmup Phase** (当 $iter < T_{warm}$):
    $$ \eta_t = \eta_{max} \cdot \frac{iter + 1}{T_{warm} + 1} $$
    线性增长，防止初始步长过大破坏预训练特征。

2.  **Decay Phase** (当 $T_{warm} \le iter \le T_{max}$):
    首先计算衰减比例:
    $$ \text{decay\_ratio} = \frac{iter - T_{warm}}{T_{max} - T_{warm}} $$
    然后应用 Cosine 函数:
    $$ \eta_t = \eta_{min} + \frac{1}{2}(1 + \cos(\pi \cdot \text{decay\_ratio})) \cdot (\eta_{max} - \eta_{min}) $$
    余弦衰减可以让学习率在训练末期平滑下降到最小值，有助于模型收敛到更优的局部极小值。

### 7. 训练循环深度解析

这是代码的核心部分，展示了高效的 GPU 利用技巧。

*   **Latency Hiding (隐藏延迟)**:
    ```python
    # inside the micro_step loop
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
    X, Y = get_batch('train') # Fetch next batch ASAP
    scaler.scale(loss).backward()
    ```
    注意 `get_batch('train')` 的位置。它在计算完当前 Batch 的 Loss 之后、进行反向传播之前就被调用了。这意味着当 GPU 忙于计算当前 Batch 的反向传播（通常计算量较大）时，CPU 可以在后台并行地准备下一个 Batch 的数据（IO 操作和解码）。这是一种经典的 Pipeline 重叠技巧。

*   **Gradient Accumulation Implementation**:
    ```python
    loss = loss / gradient_accumulation_steps
    ```
    这里通过直接除以累积步数来缩放 Loss。在求导时，梯度也会相应缩小。当累积结束后进行 Optimizer Step，这相当于对这 N 个微步的 Loss 求平均后进行更新。

*   **Gradient Clipping**:
    ```python
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    ```
    在 `scaler.step` 之前必须 `unscale_`。梯度裁剪可以防止训练过程中出现梯度爆炸，这是 RNN 和 Transformer 训练中常见的稳定性手段。

### 8. 性能监控: MFU (Model Flops Utilization)

代码中通过 `raw_model.estimate_mfu(...)` 计算 MFU。
*   这是一个衡量硬件效率的指标。
*   理论 TFLOPS 是 GPU 的峰值性能。
*   实际 TFLOPS 是根据模型参数量、Batch Size 和时间计算出来的。
*   **MFU = 实际 TFLOPS / 理论 TFLOPS**。对于 A100/H100 上训练 Transformer，通常能达到 30%-50% 甚至更高的 MFU，说明 Kernel 优化得很好。

### 技术扩展与联想

nanoGPT 虽然代码短，但它触及了现代 LLM 训练的所有关键点。与其相关的技术联想包括：

1.  **FlashAttention**: 虽然这个 `train.py` 主要关注训练循环，但 nanoGPT 的核心 `model.py` 通常会集成 FlashAttention。这是一种针对注意力机制的 IO 精确算法，通过将 Attention 计算分块到 SRAM 中，显著减少 HBM（高带宽内存）的读写次数，从而大幅提升训练速度并降低显存占用。
2.  **Chinchilla Scaling Laws**: 代码中的注释 `# should be ~= max_iters per Chinchilla` 提到了 Chinchilla 法则。该法则指出了为了最优计算效率，模型参数量和训练数据量之间的比例关系（大约 20:1 的 token 数量）。`min_lr` 的设置也遵循了类似的经验法则。
3.  **Tensor Parallelism vs Pipeline Parallelism**: nanoGPT 使用的是 DDP（数据并行）。对于参数量极大的模型（如 GPT-3 175B），单卡显存放不下，就需要使用 Tensor Parallelism（TP，切分矩阵乘法）或 Pipeline Parallelism（PP，切分层），而这些通常需要 DeepSpeed 或 Megatron-LM 这样的框架支持。

### 参考链接

*   **nanoGPT GitHub Repository (Source of Truth)**: [https://github.com/karpathy/nanogpt](https://github.com/karpathy/nanogpt)
*   **PyTorch DDP Documentation**: [https://pytorch.org/docs/stable/ddp.html](https://pytorch.org/docs/stable/ddp.html)
*   **FlashAttention Paper (Fast and Memory-Efficient Exact Attention)**: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
*   **PyTorch 2.0 Compile Documentation**: [https://pytorch.org/get-started/pytorch-2.0/](https://pytorch.org/get-started/pytorch-2.0/)
*   **AdamW Optimizer Paper**: [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
*   **Chinchilla Scaling Laws Paper**: [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)