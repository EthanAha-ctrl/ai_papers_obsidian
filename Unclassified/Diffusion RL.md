
Start with one basic problem: words depend on each other. After choosing “cats,” the next verb should usually be “are.” Choosing “cats” and “is” independently can produce an error.

For the examples below, assume each word is one token—a unit of text processed by the model. Actual tokens can be parts of words. `[MASK]` means an unknown token. **The examples are illustrations, not recorded model outputs.**

**1. Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models**

Use this prompt:

```
Describe the animal:
```

Set the block size to three tokens. One possible generation path is:

```
                         Current block
Start                    [MASK] [MASK] [MASK]
Diffusion step 1         The    [MASK] sleeps
Diffusion step 2         The    cat    sleeps
```

Each diffusion step reads the whole current block and predicts missing tokens. Thus, “cat” can use both “The” and “sleeps” as context.

Complete the block, then start another block:

```
Completed text           Current block
The cat sleeps           [MASK] [MASK] [MASK]
The cat sleeps           on     [MASK] mat
The cat sleeps           on     the    mat
```

The result is:

```
The cat sleeps on the mat
```

**Autoregression controls the order of blocks. Diffusion fills each block.** Completed blocks stay fixed. There is no separate autoregressive check of each generated token.

During training, the model receives completed earlier blocks and a partly masked current block. The model learns to recover missing tokens. The paper also improves the masking schedule to reduce variation in training updates. [Paper, Sections 3–5](https://arxiv.org/html/2503.09573v2)

**2. TiDAR: Think in Diffusion, Talk in Autoregression**

TiDAR means “Think in Diffusion, Talk in Autoregression.”

Use the same prompt. Diffusion prepares three candidate tokens in one step:

```
Input slots              [MASK] [MASK] [MASK]
Draft                    The    cat    sleep
```

For a simple illustration, use the highest-probability autoregressive choice:

|Context used for checking|Draft token|Autoregressive choice|Result|
|---|---|---|---|
|Prompt|The|The|Accept|
|Prompt + The|cat|cat|Accept|
|Prompt + The cat|sleep|sleeps|Reject draft token; use sleeps|

The committed text becomes:

```
The cat sleeps
```

Only the accepted draft prefix survives. Draft tokens after the first rejection are discarded.

Why can checking run in parallel? **The candidate words already exist.** One model evaluation can calculate each next-token prediction with access restricted to earlier candidate words.

The same evaluation also prepares new diffusion drafts for the possible accepted prefixes. The selected continuation becomes the next draft. One shared model performs both jobs.

The table simplifies checking; probabilistic generation uses rejection sampling to preserve the model’s autoregressive output distribution. [Paper, Section 3](https://arxiv.org/html/2511.08923v1)

|Paper|Diffusion’s job|What makes text final|
|---|---|---|
|Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models|Fill the current block|Completion of the diffusion process|
|TiDAR: Think in Diffusion, Talk in Autoregression|Propose several tokens together|Autoregressive acceptance and sampling|


---

A diffusion model can serve as the actor and sample actions conditioned on the current state. Researchers studied this directly in 2022.
Reinforcement learning (RL) supplies the reward objective. The diffusion model supplies a policy—the distribution used to select actions.
A conventional multilayer perceptron (MLP) actor can output action probabilities or the parameters of a Gaussian distribution. A diffusion actor generates an action through several denoising steps:

```
Observe the environment state
            ↓
Sample random noise
            ↓
Denoise repeatedly, using the state at every step
            ↓
Send the final action to the environment
            ↓
Receive reward and the next state
```

The denoising network can itself be a multilayer perceptron. Diffusion changes how the actor represents and samples the action distribution.

**The papers that directly address your question**

|Paper|What the paper establishes|
|---|---|
|**[Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/abs/2208.06193), 2022**|Uses a diffusion actor. Training combines learning actions from a fixed dataset with an objective that favors actions with higher predicted return. This is the closest early paper to your idea.|
|**[Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137), 2023**|Generates robot actions and action sequences through diffusion. Demonstrates distributions with several distinct action modes. The main training method learns from demonstrations.|
|**[Diffusion Policy Policy Optimization](https://arxiv.org/abs/2409.00588), 2024**|Uses reward feedback to improve diffusion policies through policy gradients. Treats the denoising chain as part of the decision process. This is particularly relevant to learning through environment interaction.|

Suppose a continuous action is a steering command. At one state, two successful strategies require commands near `−0.8` and `+0.8`. A distribution with two peaks can represent both strategies. A single Gaussian has one peak and may place substantial probability between the strategies.

Diffusion can learn separate peaks. Different random samples can produce different valid actions. The 2022 paper explicitly tests this issue with an action distribution that has multiple modes. 

However, multiple valid actions alone do not establish a need for diffusion. A categorical distribution already represents arbitrary probabilities over a small finite action set.

Continuous LunarLander is a better first test of a continuous diffusion actor. 