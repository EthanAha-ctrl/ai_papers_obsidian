# 《用于 LLM 可解释性的稀疏自动编码器直观解释》

Jun 11, 2024

Sparse Autoencoders (SAEs) have recently become popular for interpretability of machine learning models (although sparse dictionary learning has been around since [1997](https://www.sciencedirect.com/science/article/pii/S0042698997001697)). Machine learning models and LLMs are becoming more powerful and useful, but they are still black boxes, and we don’t understand how they do the things that they are capable of. It seems like it would be useful if we could understand how they work.  
稀疏自编码器（SAEs）最近因机器学习模型的可解释性而变得流行（尽管稀疏字典学习自 1997 年以来就已存在）。机器学习模型和 LLMs 变得越来越强大和有用，但它们仍然是黑箱，我们仍不清楚它们是如何完成它们所具备的能力的。如果我们能够理解它们的工作原理，这似乎会很有用。

Using SAEs, we can begin to break down a model’s computation into [understandable components](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html). There are [several](https://transformer-circuits.pub/2023/monosemantic-features#setup-autoencoder) [existing](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab) explanations of SAEs, and I wanted to create a brief writeup from a different angle with an intuitive explanation of how they work.  
使用 SAEs，我们可以开始将模型的计算分解为可理解的部分。目前已有几种对 SAEs 的解释，我想从一个不同的角度，以直观的方式解释它们是如何工作的。

## Challenges with interpretability  
可解释性方面的挑战

The most natural component of a neural network is individual neurons. Unfortunately, individual neurons do not conveniently correspond to single concepts. An [example neuron](https://transformer-circuits.pub/2023/monosemantic-features) in a language model corresponded to academic citations, English dialogue, HTTP requests, and Korean text. This is a concept called [superposition](https://transformer-circuits.pub/2022/toy_model/), where concepts in a neural network are represented by combinations of neurons.  
神经网络中最自然的组成部分是单个神经元。不幸的是，单个神经元并不方便地对应于单个概念。语言模型中的一个示例神经元对应于学术引用、英语对话、HTTP 请求和韩语文本。这是一个称为叠加的概念，其中神经网络中的概念由神经元的组合表示。

This may occur because many variables existing in the world are naturally sparse. For example, the birthplace of an individual celebrity may come up in less than one in a billion training tokens, yet modern LLMs will learn this fact and an extraordinary amount of other facts about the world. Superposition may emerge because there are more individual facts and concepts in the training data than neurons in the model.  
这可能是因为世界上许多变量天然就是稀疏的。例如，一个知名人士的出生地可能在一亿个训练 token 中只出现不到一次，但现代 LLMs 会学习这个事实以及关于世界的大量其他事实。叠加现象可能产生的原因是训练数据中的个体事实和概念比模型中的神经元更多。

Sparse autoencoders have recently gained popularity as a technique to break neural networks down into understandable components. SAEs were inspired by the [sparse coding](https://en.wikipedia.org/wiki/Autoencoder#Sparse_autoencoder_\(SAE\)) hypothesis in neuroscience. Interestingly, SAEs are one of the most promising tools to interpret artificial neural networks. SAEs are similar to a standard autoencoder.  
稀疏自编码器最近作为一种将神经网络分解为可理解组件的技术而受到欢迎。SAEs 受到神经科学中稀疏编码假说的影响。有趣的是，SAEs 是解释人工神经网络的几种最有前景的工具之一。SAEs 与标准自编码器相似。

A regular autoencoder is a neural network designed to compress and then reconstruct its input data. For example, it may receive a 100 dimensional vector (a list of 100 numbers) as input, feed this input through an encoder layer to compress the input to a 50 dimensional vector, and then feed the compressed encoded representation through the decoder to produce a 100 dimensional output vector. The reconstruction is typically imperfect because the compression makes the task challenging.  
一个常规自编码器是一种设计用来压缩并重建其输入数据的神经网络。例如，它可能接收一个 100 维向量（一个包含 100 个数字的列表）作为输入，将这个输入通过编码层压缩成一个 50 维向量，然后将压缩后的编码表示通过解码器产生一个 100 维的输出向量。重建通常是不完美的，因为压缩使得任务具有挑战性。

![Diagram of a standard autoencoder](https://adamkarvonen.github.io/images/sae_intuitions/autoencoder.png)

Diagram of a standard autoencoder with a 1x4 input vector, 1x2 intermediate state vector, and 1x4 output vector. The cell colors indicate activation value. The output is an imperfect reconstruction of the input.  
标准自编码器的示意图，具有 1x4 输入向量、1x2 中间状态向量和 1x4 输出向量。单元格颜色表示激活值。输出是对输入的不完美重建。

## Sparse Autoencoder Explanation  
稀疏自编码器解释

### How Sparse Autoencoders Work  
稀疏自编码器的工作原理

A sparse autoencoder transforms the input vector into an intermediate vector, which can be of higher, equal, or lower dimension compared to the input. When applied to LLMs, the intermediate vector’s dimension is typically larger than the input’s. In that case, without additional constraints the task is trivial, and the SAE could use the identity matrix to perfectly reconstruct the input without telling us anything interesting. As an additional constraint, we add a sparsity penalty to the training loss, which incentivizes the SAE to create a sparse intermediate vector. For example, we could expand the 100 dimensional input into a 200 dimensional encoded representation vector, and we could train the SAE to only have ~20 nonzero elements in the encoded representation.  
稀疏自动编码器将输入向量转换为中间向量，该中间向量的维度可以高于、等于或低于输入维度。当应用于 LLMs 时，中间向量的维度通常大于输入维度。在这种情况下，如果没有额外的约束，任务将是平凡的，SAE 可以使用单位矩阵来完美地重建输入，而不会告诉我们任何有趣的信息。作为额外的约束，我们向训练损失添加稀疏性惩罚，这激励 SAE 创建稀疏的中间向量。例如，我们可以将 100 维输入扩展为 200 维的编码表示向量，并训练 SAE 使其在编码表示中仅包含~20 个非零元素。

![Diagram of a sparse autoencoder](https://adamkarvonen.github.io/images/sae_intuitions/SAE_diagram.png)

Diagram of a sparse autoencoder. Note that the intermediate activations are sparse, with only 2 nonzero values.  
稀疏自编码器的示意图。请注意，中间激活是稀疏的，只有 2 个非零值。

We apply SAEs to the intermediate activations within neural networks, which can be composed of many layers. During a forward pass, there are intermediate activations within and between each layer. For example, [GPT-3](https://arxiv.org/abs/2005.14165) has 96 layers. During the forward pass, there is a 12,288 dimensional vector (a list of 12,288 numbers) for each token in the input that is passed from layer to layer. This vector accumulates all of the information that the model uses to predict the next token as it is processed by each layer, but it is opaque and it’s difficult to understand what information is contained within.  
我们将稀疏自编码器（SAEs）应用于神经网络中的中间激活，这些网络可能由多个层组成。在正向传播过程中，每一层内部和层与层之间都存在中间激活。例如，GPT-3 有 96 层。在正向传播时，输入的每个标记都会从一个 12,288 维的向量（一个包含 12,288 个数字的列表）传递到每一层。这个向量随着每一层的处理，累积了模型用于预测下一个标记的所有信息，但它是不透明的，很难理解其中包含的信息。

We can use SAEs to understand this intermediate activation. An SAE is basically a matrix -> ReLU activation -> matrix[1](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:1)[2](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:2). As an example, if our GPT-3 SAE has an expansion factor of 4, the input activation is 12,288 dimensional and the SAE’s encoded representation is 49,512 dimensional (12,288 x 4). The first matrix is the encoder matrix of shape (12,288, 49,512) and the second matrix is the decoder matrix of shape (49,512, 12,288). By multiplying the GPT’s activation with the encoder and applying the ReLU, we produce a 49,512 dimensional SAE encoded representation that is sparse, as the SAE’s loss function incentivizes sparsity. Typically, we aim to have less than 100 numbers in the SAE’s representation be nonzero. By multiplying the SAE’s representation with the decoder, we produce a 12,288 dimensional reconstructed model activation. This reconstruction doesn’t perfectly match the original GPT activation because our sparsity constraint makes the task difficult.  
我们可以使用稀疏自编码器（SAE）来理解这个中间激活。一个 SAE 基本上是一个矩阵 -> ReLU 激活 -> 矩阵 [1](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:1) [2](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:2) 。例如，如果我们的 GPT-3 SAE 的扩展因子是 4，输入激活是 12,288 维的，而 SAE 的编码表示是 49,512 维的（12,288 x 4）。第一个矩阵是形状为（12,288, 49,512）的编码器矩阵，第二个矩阵是形状为（49,512, 12,288）的解码器矩阵。通过将 GPT 的激活与编码器相乘并应用 ReLU，我们生成了一个 49,512 维的稀疏 SAE 编码表示，因为 SAE 的损失函数会激励稀疏性。通常，我们希望 SAE 的表示中非零数字少于 100 个。通过将 SAE 的表示与解码器相乘，我们生成了一个 12,288 维的重建模型激活。这个重建并不完美地匹配原始的 GPT 激活，因为我们的稀疏性约束使得任务变得困难。

We train individual SAEs on only one location in the model. For example, we could train a single SAE on intermediate activations between layers 26 and 27. To analyze the information contained in the outputs of all 96 layers in GPT-3, we would train 96 separate SAEs - one for each layer’s output. If we also wanted to analyze various intermediate activations within each layer, this would require hundreds of SAEs. Our training data for these SAEs comes from feeding a diverse range of text through the GPT model and collecting the intermediate activations at each chosen location.  
我们在模型的单个位置上训练独立的 SAE。例如，我们可以在第 26 层和第 27 层之间的中间激活上训练一个 SAE。为了分析 GPT-3 中所有 96 层输出的信息，我们需要训练 96 个独立的 SAE——每个层一个。如果我们还想分析每层内的各种中间激活，这将需要数百个 SAE。这些 SAE 的训练数据来自于将各种文本输入 GPT 模型，并在每个选定位置收集中间激活。

I’ve included a reference SAE Pytorch implementation. The variables have shape annotations following [Noam Shazeer’s tip](https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd). Note that various SAE implementations will often have various bias terms, normalization schemes, or initialization schemes to squeeze out additional performance. One of the most common additions is some sort of constraint on decoder vector norms. For more details, refer to various implementations such as [OpenAI’s](https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py#L16), [SAELens](https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py#L97), or [dictionary_learning](https://github.com/saprmarks/dictionary_learning/blob/main/dictionary.py#L30).  
我提供了一个参考 SAE Pytorch 实现。变量遵循 Noam Shazeer 的建议添加了形状注释。请注意，不同的 SAE 实现通常会有不同的偏置项、归一化方案或初始化方案，以榨取额外的性能。最常见的添加之一是对解码器向量范数的某种约束。更多细节，请参考 OpenAI、SAELens 或 dictionary_learning 等实现。

```
import torch
import torch.nn as nn

# D = d_model, F = dictionary_size
# e.g. if d_model = 12288 and dictionary_size = 49152
# then model_activations_D.shape = (12288,) and encoder_DF.weight.shape = (12288, 49152)

class SparseAutoEncoder(nn.Module):
    """
    A one-layer autoencoder.
    """
    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        self.encoder_DF = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder_FD = nn.Linear(dict_size, activation_dim, bias=True)

    def encode(self, model_activations_D: torch.Tensor) -> torch.Tensor:
        return nn.ReLU()(self.encoder_DF(model_activations_D))
    
    def decode(self, encoded_representation_F: torch.Tensor) -> torch.Tensor:
        return self.decoder_FD(encoded_representation_F)
    
    def forward_pass(self, model_activations_D: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_representation_F = self.encode(model_activations_D)
        reconstructed_model_activations_D = self.decode(encoded_representation_F)
        return reconstructed_model_activations_D, encoded_representation_F
```

The loss function for a standard autoencoder is based on the accuracy of input reconstruction. To introduce sparsity, early SAE implementations added a sparsity penalty to the SAE’s loss function. This most common form of this penalty is calculated by taking the L1 loss of the SAE’s encoded representation (not the SAE weights) and multiplying it by an L1 coefficient. The L1 coefficient is a crucial hyperparameter in SAE training, as it determines the trade-off between achieving sparsity and maintaining reconstruction accuracy.  
标准自编码器的损失函数基于输入重建的准确性。为了引入稀疏性，早期的 SAE 实现向 SAE 的损失函数中添加了稀疏性惩罚。这种惩罚最常见的计算方式是取 SAE 编码表示的 L1 损失（不是 SAE 权重），并将其乘以一个 L1 系数。L1 系数是 SAE 训练中的一个关键超参数，因为它决定了在实现稀疏性和保持重建准确性之间的权衡。

Note that we aren’t optimizing for interpretability. Instead, we obtain interpretable SAE features as a side effect of optimizing for sparsity and reconstruction.  
请注意，我们并不是在优化可解释性。相反，我们在优化稀疏性和重建时，将可解释的 SAE 特征作为副作用获得。

Here is a reference loss function.  
这里是一个参考损失函数。

```
# B = batch size, D = d_model, F = dictionary_size

def calculate_loss(autoencoder: SparseAutoEncoder, model_activations_BD: torch.Tensor, l1_coeffient: float) -> torch.Tensor:
    reconstructed_model_activations_BD, encoded_representation_BF = autoencoder.forward_pass(model_activations_BD)
    reconstruction_error_BD = (reconstructed_model_activations_BD - model_activations_BD).pow(2)
    reconstruction_error_B = einops.reduce(reconstruction_error_BD, 'B D -> B', 'sum')
    l2_loss = reconstruction_error_B.mean()

    l1_loss = l1_coefficient * encoded_representation_BF.sum()
    loss = l2_loss + l1_loss
    return loss
```

UPDATE 11/29/2024: I think the Vanilla ReLU SAE is fairly outdated and should not be used except as a baseline. My preferred SAE is the [BatchTopK](https://www.alignmentforum.org/posts/Nkx6yWZNbAsfvic98/batchtopk-a-simple-improvement-for-topk-saes) SAE, as it significantly improves on the sparsity / reconstruction accuracy trade-off, the desired sparsity can be directly set without tuning a sparsity penalty, and it has good training stability. The BatchTopK SAE is very similar to the ReLU SAE. Instead of a ReLU and a sparsity penalty, you simply retain the top k activation values and zero out the rest. In this case, the k hyperparameter directly sets the desired sparsity. An example BatchTopK implementation can be seen [here](https://github.com/saprmarks/dictionary_learning/blob/main/trainers/batch_top_k.py). Other strong alternative approaches are the [TopK](https://cdn.openai.com/papers/sparse-autoencoders.pdf) and [JumpReLU](https://arxiv.org/abs/2407.14435) SAEs.  
更新于 2024 年 11 月 29 日：我认为 Vanilla ReLU SAE 已经相当过时，除了作为基准外不应使用。我更倾向于使用 BatchTopK SAE，因为它显著改善了稀疏性/重建精度之间的权衡，所需的稀疏性可以直接设置，无需调整稀疏性惩罚，并且具有良好的训练稳定性。BatchTopK SAE 与 ReLU SAE 非常相似。它不是使用 ReLU 和稀疏性惩罚，而是简单地保留前 k 个激活值并将其余值置零。在这种情况下，k 超参数直接设置所需的稀疏性。一个 BatchTopK 的实现示例可以在这里看到。其他强大的替代方法包括 TopK 和 JumpReLU SAE。

![Diagram of a sparse autoencoder forward pass](https://adamkarvonen.github.io/images/sae_intuitions/SAE_forward_pass.png)

A single Sparse Autoencoder forward pass. We begin with a 1x4 model vector. We multiply it with a 4x8 encoder matrix to produce a 1x8 encoded vector, then apply the ReLU to zero out negative values. The encoded vector is sparse. We multiply it with a 8x4 decoder matrix to produce a 1x4 imperfectly reconstructed model activation.  
一个稀疏自编码器的单次前向传播。我们从一个 1x4 的模型向量开始。将其与一个 4x8 的编码器矩阵相乘，得到一个 1x8 的编码向量，然后应用 ReLU 函数将负值置零。编码向量是稀疏的。将其与一个 8x4 的解码器矩阵相乘，得到一个 1x4 的不完美重建的模型激活。

### A Hypothetical SAE Feature Walkthrough  
一个假设的 SAE 特征遍历

Hopefully, each active number in the SAE’s representation corresponds to some understandable component. As a hypothetical example, assume that the 12,288 dimensional vector `[1.5, 0.2, -1.2, ...]` means “Golden Retriever” to GPT-3. The SAE decoder is a matrix of shape (49,512, 12,288), but we can also think of it as a collection of 49,512 vectors, with each vector being of shape (1, 12,288). If the SAE decoder vector 317 has learned the same “Golden Retriever” concept as GPT-3, then the decoder vector would approximately equal `[1.5, 0.2, -1.2, ...]`. Whenever element 317 of the SAE’s activation is nonzero, a vector corresponding to “Golden Retriever” (and scaled by element 317’s magnitude) will be added to the reconstructed activation. In the jargon of mechanistic interpretability, this can be succinctly described as “decoder vectors correspond to linear representations of features in residual stream space”.  
希望 SAE 的表示中每个活跃的数字都对应某个可理解的部分。作为一个假设性的例子，假设 12,288 维向量 `[1.5, 0.2, -1.2, ...]` 对 GPT-3 来说意味着“金毛猎犬”。SAE 解码器是一个形状为(49,512, 12,288)的矩阵，但也可以将其视为一个包含 49,512 个向量的集合，每个向量形状为(1, 12,288)。如果 SAE 解码器向量 317 学习了与 GPT-3 相同的“金毛猎犬”概念，那么解码器向量将大致等于 `[1.5, 0.2, -1.2, ...]` 。每当 SAE 的激活元素 317 非零时，一个对应“金毛猎犬”（并按元素 317 的幅度缩放）的向量将被加到重建的激活中。在机制可解释性的术语中，这可以简洁地描述为“解码器向量对应于残差流空间中特征的线性表示”。

This makes intuitive sense when we consider the mathematics of vector-matrix multiplication. Multiplying a vector with a matrix is essentially a weighted sum of the matrix’s rows (or columns, depending on the multiplication order), where the weights are the elements of the vector. In our case, the SAE’s sparse encoded representation serves as these weights, selectively activating and scaling the relevant decoder vectors (matrix rows) to reconstruct the original activation.  
当我们考虑向量-矩阵乘法的数学原理时，这一点就变得直观易懂。向量与矩阵相乘本质上是对矩阵的行（或列，取决于乘法顺序）的加权求和，权重即为向量的元素。在我们的案例中，SAE 的稀疏编码表示充当这些权重，选择性地激活和缩放相关的解码器向量（矩阵行），以重建原始激活。

We can also say that our SAE with a 49,512 dimensional encoded representation has 49,512 features. A feature is composed of the corresponding encoder and decoder vectors. The role of the encoder vector is to detect the model’s internal concept while minimizing interference with other concepts, while the decoder vector’s role is to represent the “true” feature direction. Empirically, we find that encoder and decoder vectors for each feature are different, with a [median cosine similarity](https://www.alignmentforum.org/posts/fKuugaxt2XLTkASkk/open-source-replication-and-commentary-on-anthropic-s) of 0.5. In the below diagram, the three red boxes correspond to a single feature.  
我们也可以说，我们这个具有 49,512 维编码表示的 SAE 有 49,512 个特征。一个特征由相应的编码器向量和解码器向量组成。编码器向量的作用是检测模型内部的概念，同时尽量减少对其他概念的影响，而解码器向量的作用是表示“真实”的特征方向。经验上，我们发现每个特征的编码器向量和解码器向量是不同的，其中中位数余弦相似度为 0.5。在下方的图中，三个红色框对应于一个特征。

![Diagram of a sparse autoencoder with bolded feature 1](https://adamkarvonen.github.io/images/sae_intuitions/SAE_feature_diagram.png)

The three red boxes correspond to SAE feature 1, and the green boxes correspond to feature 4. Per feature, there is a 1x4 encoder vector, 1x1 feature activation, and 1x4 decoder vector. The reconstructed activation is only constructed from the decoder vectors from SAE features 1 and 4. If red represents the color red, and green represents a sphere, then the model could be representing a red sphere.  
三个红色框对应 SAE 特征 1，绿色框对应特征 4。每个特征都有一个 1x4 编码向量、1x1 特征激活和 1x4 解码向量。重建的激活仅由 SAE 特征 1 和 4 的解码向量构建。如果红色代表红色，绿色代表一个球体，那么模型可能表示一个红球。

How do we know what our hypothetical feature 317 represents? The current practice is to just look at the inputs that maximally activate the feature and give a gut reaction on their interpretability. The inputs each feature activates on are frequently interpretable. For example, Anthropic trained SAEs on Claude Sonnet and found separate SAE features that activate on text and images related to the [Golden Gate Bridge, neuroscience, and popular tourist attractions](https://transformer-circuits.pub/2024/scaling-monosemanticity/). Other features activate on concepts that aren’t immediately obvious, such as a feature from a SAE trained on Pythia that activates [“on the final token of relative clauses or prepositional phrases which modify a sentence’s subject”](https://x.com/saprmarks/status/1758253577888493901).  
我们如何知道我们的假设特征 317 代表什么？目前的做法是观察最大化激活该特征的输入，并对其可解释性进行直觉反应。每个特征激活的输入通常是可解释的。例如，Anthropic 在 Claude Sonnet 上训练 SAE，并发现不同的 SAE 特征分别激活与金门大桥、神经科学和热门旅游景点相关的文本和图像。其他特征激活的概念并不立即明显，例如一个在 Pythia 上训练的 SAE 的特征激活“句子主语修饰的从句或介词短语的最后一个词”。

Because the SAE decoder vectors match the shape of the LLMs intermediate activations, we can perform causal interventions by simply adding decoder vectors to model activations. We can scale the strength of the intervention by multiplying the decoder vector with a scaling factor. When Anthropic researchers added the [Golden Gate Bridge](https://www.anthropic.com/news/golden-gate-claude) SAE decoder vector to Claude’s activations, Claude was compelled to mention the Golden Gate Bridge in every response.  
因为稀疏自动编码器解码器向量与 LLMs 中间激活的形状相匹配，我们可以通过简单地将解码器向量添加到模型激活中来进行因果干预。我们可以通过将解码器向量与缩放因子相乘来调整干预的强度。当 Anthropic 研究人员将金门大桥的 SAE 解码器向量添加到 Claude 的激活中时，Claude 被迫在每次回应中都提到金门大桥。

Here is a reference implementation of a causal intervention using our hypothetical feature 317[3](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:3). This very simple intervention would compel our GPT-3 model to mention Golden Retrievers in every response, similar to `Golden Gate Bridge Claude`.  
这里是一个使用我们假设的特征 317 [3](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:3) 的因果干预参考实现。这个非常简单的干预会迫使我们的 GPT-3 模型在每次回应中都提到金毛猎犬，类似于 `Golden Gate Bridge Claude` 。

```
def perform_intervention(model_activations_D: torch.Tensor, decoder_FD: torch.Tensor, scale: float) -> torch.Tensor:
    intervention_vector_D = decoder_FD[317, :]
    scaled_intervention_vector_D = intervention_vector_D * scale
    modified_model_activations_D = model_activations_D + scaled_intervention_vector_D
    return modified_model_activations_D
```

## Challenges with Sparse Autoencoder Evaluations  
稀疏自动编码器评估的挑战

One of the main challenges with using SAEs is in evaluation. We are training sparse autoencoders to interpret language models, but we don’t have a measurable underlying ground truth in natural language. Currently, our evaluations are subjective, and basically correspond to “we looked at activating inputs for a range of features and gave a gut reaction on interpretability of the features”. This is a [major limitation](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#discussion-limitations/) in the field of interpretability.  
使用 SAEs 的主要挑战之一在于评估。我们正在训练稀疏自编码器来解释语言模型，但在自然语言中我们没有可测量的潜在真实情况。目前，我们的评估是主观的，基本上对应于“我们观察了一系列特征的激活输入，并对特征的可解释性给出了直觉反应”。这是可解释性领域的一个主要局限性。

Researchers have found common proxy metrics that seem to correspond to feature interpretability. The most commonly used are `L0` and `Loss Recovered`. `L0` is the average number of nonzero elements in the SAE’s encoded intermediate representation. `Loss Recovered` is where we replace the GPT’s original activation with our reconstructed activation and measure the additional loss from the imperfect reconstruction. There is typically a trade-off between these two metrics, as SAEs may choose a solution that decreases reconstruction accuracy to increase sparsity.  
研究人员发现了一些常见的代理指标，似乎与特征可解释性相对应。最常用的指标是 `L0` 和 `Loss Recovered` 。 `L0` 是 SAE 编码中间表示中非零元素的平均数量。 `Loss Recovered` 是指我们将 GPT 的原始激活替换为我们的重建激活，并测量由于重建不完美而产生的额外损失。这两个指标之间通常存在权衡，因为 SAE 可能会选择一个解决方案，以降低重建精度来增加稀疏性。

A common comparison of SAEs involves graphing these two variables and examining the tradeoff[4](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:4). Many new SAE approaches, such as Deepmind’s Gated SAE and OpenAI’s TopK SAE, modify the sparsity penalty to improve on this tradeoff. The following graph is from Google Deepmind’s [Gated SAE paper](https://arxiv.org/abs/2404.16014). The red line for Gated SAEs is closer to the top left of the graph, meaning that it performs better on this tradeoff.  
SAE 的常见比较涉及绘制这两个变量并检查权衡 [4](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:4) 。许多新的 SAE 方法，如 Deepmind 的 Gated SAE 和 OpenAI 的 TopK SAE，通过修改稀疏性惩罚来改进这种权衡。以下图表来自 Google Deepmind 的 Gated SAE 论文。Gated SAE 的红色线更接近图表的左上角，这意味着它在这种权衡上表现更好。

![Gated SAE L0 vs Loss Recovered](https://adamkarvonen.github.io/images/sae_intuitions/L0_vs_loss_recovered.jpeg)

There’s several layers to difficulties with measurements in SAEs. Our proxy metrics are `L0` and `Loss Recovered`. However, we don’t use these when training as `L0` isn’t differentiable and calculating `Loss Recovered` during SAE training is computationally expensive[5](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:5). Instead, our training loss is determined by an L1 penalty and the accuracy of reconstructing the internal activation, rather than its effect on downstream loss.  
在 SAEs 中，测量存在多个层面的困难。我们的代理指标是 `L0` 和 `Loss Recovered` 。然而，我们在训练时并不使用这些指标，因为 `L0` 不可微分，并且在 SAE 训练期间计算 `Loss Recovered` 计算成本高昂 [5](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fn:5) 。相反，我们的训练损失由 L1 惩罚和内部激活重建的准确性决定，而不是其对下游损失的影响。

Our training loss function doesn’t directly correspond to our proxy metrics, and our proxy metrics are only proxies for our subjective evaluations of feature interpretability. There’s an additional layer of mismatch, as our subjective interpretability evaluations are proxies for our true goal of “how does this model work”. There’s a possibility that some important concepts within LLMs are not easily interpretable, and we could ignore these by blindly optimizing interpretability.  
我们的训练损失函数并不直接对应我们的代理指标，而我们的代理指标只是特征可解释性主观评估的代理。由于我们的主观可解释性评估又是“这个模型是如何工作的”这一真正目标的代理，因此存在另一层不匹配。LLMs 中某些重要概念可能难以解释，而我们可能会在盲目优化可解释性的过程中忽略这些概念。

For a more detailed discussion of SAE evaluation methods and an evaluation approach using board game model SAEs, refer to my blog post on [Evaluating Sparse Autoencoders with Board Game Models](https://adamkarvonen.github.io/machine_learning/2024/06/12/sae-board-game-eval.html).  
关于 SAE 评估方法的更详细讨论以及使用棋盘模型 SAE 的评估方法，请参考我的博客文章《使用棋盘模型评估稀疏自动编码器》。

## Conclusion  结论

The field of interpretability has a long way to go, but SAEs represent real progress. They enable interesting new applications, such as an unsupervised method to find steering vectors like the “Golden Gate Bridge” steering vector. SAEs have also made it easier to find circuits in language models, which can potentially be used to [remove unwanted biases](https://arxiv.org/abs/2403.19647) from the internals of the model.  
可解释性领域还有很长的路要走，但 SAE 代表了真正的进步。它们能够实现有趣的新应用，例如一种无监督方法来寻找类似“金门大桥”转向向量的转向向量。SAE 还使在语言模型中寻找电路变得更加容易，这些电路有可能用于从模型的内部移除不想要的偏见。

The fact that SAEs find interpretable features, even though their objective is merely to identify patterns in activations, suggests that they are uncovering something meaningful. This is also evidence that LLMs are learning something meaningful, rather than just memorizing surface-level statistics.  
SAEs 能够发现可解释的特征，尽管其目标仅仅是识别激活中的模式，这表明它们正在揭示某些有意义的东西。这也是 LLMs 正在学习某些有意义的东西，而不仅仅是记忆表面层次的统计数据的证据。

They also represent an early milestone that companies such as Anthropic have aimed for, which is [“An MRI for ML models”](https://www.dwarkeshpatel.com/p/dario-amodei). They currently do not offer perfect understanding, but they may be useful to detect unwanted behavior. The challenges with SAEs and SAE evaluations are not insurmountable and are the subject of much ongoing research.  
它们也代表了 Anthropic 等公司所追求的早期里程碑，即“机器学习模型的 MRI”。它们目前并不提供完美的理解，但可能有助于检测不希望的行为。SAEs 和 SAE 评估的挑战并非不可克服，并且是当前许多正在进行的研究的主题。

For further study of Sparse Autoencoders, I recommend [Callum McDougal’s Colab notebook](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab).  
对于进一步研究稀疏自动编码器，我推荐 Callum McDougal 的 Colab 笔记本。

Acknowledgements: I am grateful to Justis Mills, Can Rager, Oscar Obeso, and Slava Chalnev for their valuable feedback on this post.  
致谢：我感谢 Justis Mills、Can Rager、Oscar Obeso 和 Slava Chalnev 对这篇文章的宝贵反馈。

1. The ReLU activation function is simply `y = max(0, x)`. That is, any negative input is set to 0. [↩](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fnref:1)  
    ReLU 激活函数很简单，就是 `y = max(0, x)` 。也就是说，任何负输入都会被设置为 0。 ↩
    
2. There are typically also bias terms at various points, including the encoder and decoder layers. [↩](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fnref:2)  
    通常在各个位置，包括编码器和解码器层，也会有偏差项。 ↩
    
3. Note that this function would intervene on a single layer and that the SAE should have been trained on the same location as the model activations. For example, if the intervention was performed between layers 6 and 7 then the SAE should have been trained on the model activations between layers 6 and 7. Interventions can also be performed simultaneously on multiple layers. [↩](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fnref:3)  
    请注意，这个函数只作用于单层，而 SAE 应该在模型激活的相同位置进行训练。例如，如果干预是在第 6 层和第 7 层之间进行的，那么 SAE 应该在模型第 6 层和第 7 层之间的激活上进行训练。干预也可以同时作用于多层。 ↩
    
4. It’s worth noting that this is only a proxy and that improving this tradeoff may not always be better. As mentioned in the recent [OpenAI TopK SAE paper](https://cdn.openai.com/papers/sparse-autoencoders.pdf), an infinitely wide SAE could achieve a perfect `Loss Recovered` with an `L0` of 1 while being totally uninteresting. [↩](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fnref:4)  
    值得注意的是，这只是一个代理指标，而改善这种权衡并不总是更好。正如最近 OpenAI 的 TopK SAE 论文中提到的，一个无限宽的 SAE 可以在 `L0` 为 1 的情况下实现完美的 `Loss Recovered` ，但会变得毫无意义。 ↩
    
5. Apollo Research recently released [a paper](https://arxiv.org/abs/2405.12241) that used a loss function that aimed to produce the same output distribution, rather than reconstruct a single layer’s activation. It works better but is also more computationally expensive. [↩](https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html#fnref:5)  
    Apollo Research 最近发布了一篇论文，该论文使用了一种旨在产生相同输出分布的损失函数，而不是重建单层的激活。它效果更好，但也更耗费计算资源。 ↩