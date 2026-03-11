```Python
import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional


from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.config import VLMConfig

from data.processors import get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        """
        Replace every image-token placeholder in `input_ids` with the corresponding slice
        from `image_embd`. Supports an arbitrary number of image-token placeholders per sample.
        The first example in the batch might have 2 images and the second none.
        """
        # Clone the original embeddings to avoid in-place issues
        updated_token_embd = token_embd.clone()

        # Build a mask of all image-token positions: shape [B, T_seq]
        mask = (input_ids == self.tokenizer.image_token_id)
        updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype) # torch flattens before assigning

        return updated_token_embd

    def _process_images(self, images, device):
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]

            if not images:  # Handle cases with no images
                return None
            else:
                return torch.cat(images, dim=0).to(device)
        return images # Already a tensor

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids) # [B, T_sequence, D_lm]

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)  # [num_images, mp_image_token_length, D_lm]
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        logits, _ = self.decoder(token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits) # Apply LM head
            # Loss is calculated over all tokens, but `targets` (labels) will have -100 for non-answer tokens.
            # No need to slice logits based on image embedding size here, as the target mask handles it.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids) # [B, T_prompt_text, D_lm]

        if images_tensor is not None:
            # 1. Process image if present
            image_embd = self.vision_encoder(images_tensor) # [B, T_img_feat, D_model]
            image_embd = self.MP(image_embd)      # [B, mp_image_token_length, D_lm]
            # 2. Combine image and text embeddings
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        current_total_seq_len = token_embd.size(1)
        batch_size = input_ids.size(0) # Or token_embd.size(0)
        
        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = self.decoder(
            token_embd,
            attention_mask=attention_mask, # Use the provided attention mask
            kv_cache=None,
            start_pos=0
        )
        
        last_token_output_from_prefill = prefill_output[:, -1, :] 
        
        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill) 
        else:
            current_logits = last_token_output_from_prefill 

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            newly_generated_ids_list.append(next_token_id)
            
            # Embed the newly generated token
            next_token_embed = self.decoder.token_embedding(next_token_id) # [B, 1, D_lm]
            
            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            # With KV cache: only process the new token
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
      
            last_token_output = decode_step_output[:, -1, :] 
            
            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output
        
        if not newly_generated_ids_list: # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size,0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0: # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (generated_ids == self.tokenizer.eos_token_id) # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(seq_len, device=device) # Create column indices [0, 1, ..., seq_len-1]
            
            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1) 

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            
            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            
            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            
            generated_ids[replace_mask] = self.tokenizer.eos_token_id
        
        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```



基于您提供的 `VisionLanguageModel` 代码文件，我们可以深入剖析 **Vision Language Model (VLM)** 的架构设计与实现细节。这个代码库 `nanoVLM` 展示了一个典型的、轻量级的基于 **Transformer** 的多模态模型。

以下是对该模型架构、核心机制以及相关联想的技术深度解析。

### 1. Architecture Overview (架构概览)

这个 `VisionLanguageModel` 遵循了经典的 **"Encoder-Decoder"** (或者更准确地说是 **"Frozen Encoder + Tuned Decoder"**) 范式，类似于 LLaVA 或 InstructBLIP 的核心架构。它主要由三个核心 `Component` 组成：

1.  **VisionEncoder (ViT)**: 负责处理 Raw Image 数据。
2.  **ModalityProjector (MP)**: 负责 Alignment，将 Visual features 映射到 Language Model 的 Embedding 空间。
3.  **Language Model (Decoder)**: 负责多模态 Reasoning 和 Text generation。

#### 架构图解
```mermaid
graph LR
    SubGraph Input [Input Layer]
        Image[Raw Image]
        Text[Text Input IDs]
    end

    SubGraph Vision [Visual Path]
        ViT[Vision Encoder / ViT]
        Features[Image Features]
    end

    SubGraph Projection [Alignment Layer]
        MP[Modality Projector / MLP]
        ImageTokens[Image Token Embeddings]

    end

    SubGraph Language [Language Path]
        LM[LLM Decoder / Causal LM]
        Output[Generated Text]
    end

    Image --> ViT
    ViT --> Features
    Features --> MP
    MP --> ImageTokens
    Text --> LM
    ImageTokens -->|Inject & Replace| LM
    LM --> Output
```

---

### 2. Deep Dive into Code Components (代码组件深度解析)

#### A. Vision Encoder (`vision_encoder`)
代码中调用了 `ViT.from_pretrained(cfg)`，且根据 `MODEL_CARD_TEMPLATE`，该模型使用了 **SigLIP-B/16-224-85M** 作为 Vision Backbone。

*   **Technical Detail**: **SigLIP** (Sigmoid Loss for Language Image Pre-training) 是一种改进的 CLIP。与 CLIP 使用 **Global Contrastive Loss** (对比整个图像和整个文本的余弦相似度) 不同， SigLIP 将问题建模为 **Binary Classification** 任务。
*   **Formula**:
    对于一对图像-文本 $(I, T)$：
    $$ L = -\frac{1}{N} \sum_{i=1}^{N} \log \sigma(z_{i, y_i}) - \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq y_i} \log (1 - \sigma(z_{i,j})) $$
    其中 $z$ 是 image-text similarity 的 logit，$\sigma$ 是 sigmoid 函数。这种 Loss function 使得训练更稳定，且对 **Batch Size** 不那么敏感。
*   **Hallucination/Association**: 这里可以联想到 **DINOv2** 或 **MAE**，它们是另一种自监督学习范式，但在 VLM 中，像 SigLIP 这样与文本对齐的 Encoder 通常作为 Visual Feature Extractor 的首选。

#### B. Modality Projector (`MP`)
`ModalityProjector` 是连接 Visual Space 和 Textual Space 的 Bridge。
代码中的调用：`image_embd = self.MP(image_embd)`。

*   **Function**: 将 Vision Encoder 的输出维度 $D_{vit}$ 映射到 LLM 的维度 $D_{lm}$。
*   **Design Possibilities**:
    *   **Linear Layer**: 最简单的映射，$H_{lm} = W \cdot H_{vit} + b$。
    *   **MLP**: 2-layer MLP (通常中间维度大一些，然后降维)，可以增加非线性的表达能力。
    *   **Q-Former** (来自 BLIP-2): 使用可学习的 Query 来提取视觉特征，减少 Token 数量。
*   **nanoVLM Context**: 考虑到它是 Minimalist 的，很可能使用的是 **Linear Layer** 或简单的 **2-layer MLP**。这一层通常是 VLM 训练过程中更新的主要部分（如果冻住了 ViT 和 LLM 的参数，这就是 **Adapter**）。

#### C. Modality Fusion Mechanism: Token Replacement (模态融合机制：Token 替换)
这是代码中 `_replace_img_tokens_with_embd` 方法最核心的逻辑，也是 LLaVA 系列最流行的 Fusion 方式。

*   **Mechanism**:
    1.  在 Tokenization 阶段，插入特殊的 Placeholder Token (例如 `<image>`)。
    2.  获取整个 Prompt 的 Text Embeddings。
    3.  检测 `input_ids` 中等于 `image_token_id` 的位置。
    4.  **Direct Replacement**: 将这些位置的 Text Embedding 直接用 Projector 输出的 Image Embedding 覆盖。

*   **Code Analysis**:
    ```python
    mask = (input_ids == self.tokenizer.image_token_id)
    updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1))
    ```
    这种方式简单高效。它将 Image 理解为一种特殊的 "Visual Language"，直接输入到 LLM 的 **Self-Attention** 层中进行交互。

*   **Alternative Methods (联想)**:
    *   **Cross-Attention (Flamingo)**: LLM 增加 Cross-Attention 层来读取 Visual Features。这种方式更复杂，但可能更高效。
    *   **Prefix Tuning / Soft Prompt**: 将 Image Embedding 作为 Prefix 拼接在 Text 前面，而不是替换特定的 Token 位置。

#### D. Language Model (`decoder`)
根据注释，使用的是 **SmolLM2-135M**。
这是一个轻量级的 **Causal Language Model (CLM)**。

*   **Training Phase (`forward`)**:
    *   计算 **Cross Entropy Loss**。
    *   `ignore_index=-100`: 这是一个关键的技巧。在多模态训练中，我们通常只计算 **Answer** 部分的 Loss，而忽略 **Instruction** 和 **Image Tokens** 的 Loss。
    *   **Formula**:
        $$ L = - \sum_{t \in \text{answer\_tokens}} \log P(x_t | x_{<t}, I) $$

*   **Inference Phase (`generate`)**:
    *   代码实现了标准的 **Autoregressive Decoding**。
    *   **KV-Cache Optimization**: 这是提高 LLM 推理速度的关键。
        *   在 `prefill` 阶段处理长序列，并存储 $K^l, V^l$ (Keys and Values for each layer)。
        *   在 `decode` 阶段，只处理当前生成的一个 token，复用之前的 KV-Cache。
    *   **Sampling Strategies**:
        *   **Greedy**: `torch.argmax`。
        *   **Top-k / Top-p (Nucleus Sampling)**: `top_k_top_p_filtering`。这解决了生成文本重复或无意义的问题。
        *   **Temperature**: $P(w) \propto \exp(\text{logits}(w) / T)$。$T < 1$ 使分布更 sharp，$T > 1$ 更 random。

---

### 3. Advanced Concepts & Associations (高级概念与扩展联想)

基于您提供的代码逻辑，我们可以扩展联想更广泛的 VLM 训练技术和变体：

#### A. Training Stages (VLM 训练阶段)
通常 `nanoVLM` 这样的模型会经历两个阶段：
1.  **Pre-training / Feature Alignment**: 使用大量的 **Image-Text Pairs** (如 CC3M, LAION) 来训练 Modality Projector，让 LLM 理解图像内容。
    *   *Target*: Image Captioning 任务。
2.  **Supervised Fine-Tuning (SFT)**: 使用 **Instruction Tuning Data** (如 LLaVA-Instruct) 来增强模型的对话能力。
    *   *Target*: VQA (Visual Question Answering) 任务。

#### B. Data Efficiency Techniques (数据效率技术)
由于 SmolLM2 是一个小模型 (135M)，其 Capacity 有限。
*   **LoRA (Low-Rank Adaptation)**: 代码中虽未直接体现，但在实际微调 VLM 时，为了节省显存，通常会对 LLM 的 Attention 层应用 LoRA，只训练低秩矩阵，而不是更新所有权重。
*   **Hallucination Mitigation**: 小模型容易出现 "Object Hallucination"（描述图中不存在的物体）。解决方法包括使用 **RLHF (Reinforcement Learning from Human Feedback)** 或 **DPO (Direct Preference Optimization)** 来对齐模型生成。

#### C. High-Resolution Adaptation (高分辨率适应)
代码中使用的是 `224x224` 的分辨率（由 SigLIP-B/16 决定）。
*   对于 Optical Character Recognition (OCR) 或细粒度识别，这太小了。
*   **Advanced Architecture联想**: 像 **LLaVA-NeXT** 或 **Qwen2-VL** 采用了 **Dynamic Resolution** 或 **Spatial Partition** (例如切分大图为多个 Sub-image)，并通过 2D Absolute Position Embedding 来保持空间关系。

#### D. System Integration (系统集成)
代码使用了 `safetensors` 格式。
*   **Safety**: 相比于 `pickle` (PyTorch 默认)，`safetensors` 更安全，防止加载恶意模型时执行任意代码代码。
*   **Hugging Face Hub**: `from_pretrained` 和 `push_to_hub` 方法展示了模型如何无缝集成到 MLOps 流程中。

---

### 4. Reference Links (参考链接)

为了更深入地理解上述技术，请参考以下资源：

*   **SigLIP Paper**: Zhai, X., et al. "Sigmoid loss for language image pre-training." ICCV 2023.
    *   https://arxiv.org/abs/2303.15343
*   **LLaVA Paper**: Liu, H., et al. "Visual instruction tuning." NeurIPS 2023. (架构最接近本代码)
    *   https://arxiv.org/abs/2304.08485
*   **BLIP-2 Paper**: Li, J., et al. "BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models." ICML 2023. (介绍了 Q-Former)
    *   https://arxiv.org/abs/2301.12597
*   **nanoVLM Repository**: 模型卡中提到的代码库。
    *   https://github.com/huggingface/nanovlm
*   **SmolLM2**: 轻量级语言模型。
    *   https://huggingface.co/HuggingFaceTB/SmolLM2-135M
*   **Top-p Sampling**: Holtzman, E., et al. "The curious case of neural text degeneration." ICLR 2020.
    *   https://arxiv.org/abs/1904.09751

### 总结
您提供的代码是一个教科书级别的 **Minimal VLM Implementation**。它揭示了现代多模态大模型最核心的原理：**Modality Alignment via Projector** 和 **Token-level Fusion**。通过将图像特征视为特殊的词元，VLM 成功地将视觉信息“翻译”为语言模型可以理解的格式，从而实现了复杂的多轮对话和视觉问答。