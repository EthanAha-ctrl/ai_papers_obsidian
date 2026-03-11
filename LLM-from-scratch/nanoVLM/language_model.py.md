```Python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L69
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input across the last dimension using RMS normalization,
    which scales the input without subtracting the mean. Commonly used as a
    lighter alternative to LayerNorm in transformer models.

    Args:
        cfg: A configuration object containing:
            - lm_hidden_dim (int): The dimensionality of the model hidden states. 
            - lm_rms_eps (float): A small constant to avoid division by zero.
    """
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, lm_hidden_dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute inverse of RMS: square the tensor element-wise, mean is computed across lm_hidden_dim.
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) # inverse of RMS
        x = x * irms * self.weight

        return x

# Multiple derivates of Rotary Embeddings by now, this is a basic one with linear scaling to context length
# e.g. https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L190
class RotaryEmbedding(nn.Module):
    """
        Compute Rotary Embedding to introduce positional dependency to input sequence without additional training parameters and 
        relative distance of token position ids through angle rotation.

        Args:
            cfg: Configuration object containing:
                - lm_hidden_dim (int): Hidden dimension size.
                - lm_n_heads (int): Number of attention heads.
                - lm_re_base (float): Base for rotary embedding frequencies.
                - lm_max_position_embeddings (int): Max sequence length supported for rotary embedding.
                - lm_attn_scaling (float): Attention scaling factor.
        """
    
    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads # dim of each head
        self.base = cfg.lm_re_base
        self.max_seq_len = cfg.lm_max_position_embeddings
        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling = cfg.lm_attn_scaling

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings (cosine and sine components).

        Args:
            position_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing position indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors (cos, sin), each of shape
                                  (batch_size, seq_len, dim), representing rotary embeddings.
        """

        batch_size, seq_len = position_ids.shape
        # Dynamic scaling for longer sequences
        # Divide the angle frequency to fit more rotation into the embedding space.
        max_seq = position_ids.max() + 1
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = self.inv_freq / scale
        else:
            inv_freq = self.inv_freq
            
        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flat_position_ids = position_ids.reshape(-1).float()
        
        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)
        
        # Reshape to include batch dimension
        freqs = freqs.reshape(batch_size, seq_len, -1)
        
        # Now create interleaved pattern
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Compute cos and sin
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the input by dividing the hidden dimension to two, then swapping and negating dimensions.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# Apply rotary position embeddings to queries and keys.
def apply_rotary_pos_embd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim:int=1)-> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors in attention mechanisms.

    Rotary positional embeddings inject position-dependent rotations into query and key vectors,
    enabling transformers to encode positional information effectively without explicit positional encoding.

    Args:
        q (torch.Tensor): Query tensor with shape [batch_size, num_heads, seq_len, head_dim].
        k (torch.Tensor): Key tensor with shape [batch_size, num_heads, seq_len, head_dim].
        cos (torch.Tensor): Precomputed cosine positional embeddings with shape [batch_size, seq_len, head_dim].
        sin (torch.Tensor): Precomputed sine positional embeddings with shape [batch_size, seq_len, head_dim].
        unsqueeze_dim (int, optional): Dimension index to unsqueeze `cos` and `sin` to enable broadcasting.
                                      Defaults to 1 (typically the heads dimension).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors (`q_embed`, `k_embed`), 
                                           each with the same shape as the input tensors.

    How it works:
        - `cos` and `sin` tensors are unsqueezed at `unsqueeze_dim` to broadcast across attention heads.
        - Rotary embeddings apply a complex number rotation in the embedding space using:
            rotated = (original * cos) + (rotate_half(original) * sin)
        - `rotate_half` performs a specific half-dimension rotation on the input tensor.
        - This operation encodes relative position information in q and k without adding explicit positional vectors.

    Example:
        q_embed, k_embed = apply_rotary_pos_embd(q, k, cos, sin)

    """

    # We need to make sure cos and sin can be properly broadcast
    # to the shape of q and k by adding the heads dimension
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    
    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L214
# https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L382
class LanguageModelGroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) as used in some transformer-based language models.

    GQA reduces computation by using fewer key-value heads than query heads,
    grouping multiple query heads to share the same key-value heads.

    Args:
        cfg: Configuration object containing:
            - lm_n_heads (int): Number of query heads.
            - lm_n_kv_heads (int): Number of key-value heads.
            - lm_hidden_dim (int): Hidden embedding dimension.
            - lm_dropout (float): Dropout rate.
    """
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads

        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available, using standard attention in LM.")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask=None, block_kv_cache=None) -> tuple[torch.Tensor, dict]:
        """
        Forward pass for grouped query attention.

        Args:
            x (Tensor): Input tensor of shape (B, T_curr, C), where
                        B = batch size,
                        T_curr = current sequence length,
                        C = embedding dimension.
            cos (Tensor): Rotary embedding cosines, shape compatible with q and k.
            sin (Tensor): Rotary embedding sines, shape compatible with q and k.
            attention_mask (Tensor, optional): Attention mask tensor of shape (B, total_kv_length),
                                               with 1 for tokens to attend to and 0 for padding.
            block_kv_cache (dict, optional): Cache dict with 'key' and 'value' tensors for autoregressive decoding.

        Returns:
            tuple[Tensor, dict]:
                - Output tensor after attention and projection, shape (B, T_curr, C).
                - Updated block_kv_cache dict for caching key-value states.
        """
        is_prefill = block_kv_cache is None

        B, T_curr, C = x.size() # T_curr is the sequence length of the current input x

        q_curr = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T_curr, head_dim)
        k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)
        v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)

        # Apply rotary embeddings to the current q and k
        q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

        # Check if we can use cached keys and values
        if not is_prefill and block_kv_cache['key'] is not None:
            # Concatenate with cached K, V
            # k_rotated and v_curr are for the new token(s)
            k = block_kv_cache['key']
            v = block_kv_cache['value']
            k = torch.cat([k, k_rotated], dim=2)
            v = torch.cat([v, v_curr], dim=2)
            block_kv_cache['key'] = k
            block_kv_cache['value'] = v
        else:
            # No cache, this is the first pass (prefill)
            k = k_rotated
            v = v_curr
            block_kv_cache = {'key': k, 'value': v}

        # Repeat K, V for Grouped Query Attention
        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
        
        T_kv = k_exp.size(2) # Total sequence length of keys/values

        # Prepare attention mask for SDPA or manual path
        # attention_mask is (B, T_kv_total_length), 1 for attend, 0 for pad
        additive_attn_mask = None
        if attention_mask is not None:
            # The current `attention_mask` parameter is assumed to be `[B, total_sequence_length_kv]`
            # Let's make it `[B, 1, 1, T_kv]` for SDPA.
            mask_for_keys = attention_mask[:, :T_kv] # Ensure mask matches key length [B, T_kv]
            additive_attn_mask = (1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
            # This additive_attn_mask shape is [B, 1, 1, T_kv]

        if self.sdpa and x.device.type != 'mps':
            # During decode, no additional masking needed as [1, T_kv] is naturally causal
            is_causal = (T_curr == T_kv and T_curr > 1)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=additive_attn_mask, 
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Manual attention implementation
            attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, n_heads, T_curr, T_kv)
            # During decode: no additional masking needed as [1, T_kv] is naturally causal
            if T_curr == T_kv and T_curr > 1:
                causal_mask_val = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
                attn = attn.masked_fill(~causal_mask_val, float('-inf'))

            if additive_attn_mask is not None: # Additive padding mask
                # additive_attn_mask is [B,1,1,T_kv], needs to be broadcast to [B, n_heads, T_curr, T_kv]
                attn = attn + additive_attn_mask 

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v_exp
            
        y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y, block_kv_cache

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
class LanguageModelMLP(nn.Module):
    """
    Implements the feed-forward network (MLP) block used in transformer-based language models.

    This MLP uses a gated activation mechanism where two separate linear projections
    are applied to the input: one passed through an activation function (gate_proj),
    and the other as is (up_proj). Their element-wise product is then projected back
    to the embedding dimension (down_proj).

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): The embedding dimension size.
            - lm_inter_dim (int): The intermediate dimension size for the MLP.

    Attributes:
        activation_fn (Callable): The activation function used (SiLU).
        gate_proj (nn.Linear): Linear projection for gating pathway.
        up_proj (nn.Linear): Linear projection for upscaling pathway.
        down_proj (nn.Linear): Linear projection for downscaling back to embedding dim.
    """

    def __init__(self, cfg):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim
        self.inter_dim = cfg.lm_inter_dim

        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)

    def forward(self, x):
        """
        Forward pass through the gated MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embd_dim),
                    after gated MLP transformation.
        """
        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)

        return x

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg) # Input Norm
        self.norm2 = RMSNorm(cfg) # Post Attention Norm
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor=None, block_kv_cache: dict=None):
        """
        Forward pass of the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            cos (Tensor): Cosine positional embeddings for rotary embedding, shape
                matching sequence length and head dimension.
            sin (Tensor): Sine positional embeddings for rotary embedding, same shape as cos.
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, total_kv_length),
                with 1 indicating tokens to attend to and 0 for padding tokens.
            block_kv_cache (dict, optional): Key-value cache dict for cached keys and values
                during decoding. If None, no cache is used.

        Returns:
            Tuple[Tensor, dict]: Output tensor after the block (same shape as input),
                and the updated key-value cache dictionary.
        """
        res = x
        x = self.norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = res + x

        return x, block_kv_cache

# https://github.com/meta-llama/llama3/blob/main/llama/model.py#L251
class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights

        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList([
            LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)
        ])
        self.norm = RMSNorm(cfg) # Final Norm
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor=None, kv_cache: list[dict]=None, start_pos: int=0):
        """
        Performs a forward pass through the language model.

        Args:
            x (Tensor): Input tensor. If `lm_use_tokens` is True, this should be
                token indices with shape (batch_size, sequence_length).
                If False, it should be embeddings of shape (batch_size, sequence_length, hidden_dim).
            attention_mask (Tensor, optional): Mask tensor for attention to
                specify which tokens to attend to, typically of shape
                (batch_size, sequence_length). Default is None.
            kv_cache (list[dict], optional): List of key-value caches for each transformer
                block to enable efficient autoregressive decoding.
                If None, no cache is used and new ones are created. Default is None.
            start_pos (int, optional): The starting position index for the current input
                sequence. Used to compute rotary positional embeddings correctly,
                especially for cached sequences during generation. Default is 0.

        Returns:
            Tuple:
                - Tensor: Output logits with shape (batch_size, sequence_length, vocab_size)
                if `lm_use_tokens` is True, otherwise the hidden state embeddings
                (batch_size, sequence_length, hidden_dim).
                - list: Updated list of key-value caches, one for each transformer block,
                useful for autoregressive decoding and incremental generation.

        Behavior:
            - If `lm_use_tokens` is True, the input token indices are first embedded.
            - Rotary positional embeddings are generated for the current input positions,
            which are passed along to each transformer block.
            - For each transformer block, the input is processed along with
            rotary embeddings, attention mask, and optional cached key-values.
            - After processing all blocks, a final RMS normalization is applied.
            - If tokens are used, the normalized hidden states are projected to logits
            over the vocabulary.
            - The method returns the logits or embeddings along with the updated
            cache for efficient decoding.
        """
        if self.lm_use_tokens:
            x = self.token_embedding(x)

        # T_curr is the length of the current input sequence
        B, T_curr, _ = x.size()
        
        # Create position_ids for the current sequence based on start_pos
        current_position_ids = torch.arange(start_pos, start_pos + T_curr, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_embd(current_position_ids) # Get rotary position embeddings for current tokens

        # Initialize new KV cache if none provided
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])

        x = self.norm(x)

        # Compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens: 
            x = self.head(x) 

        return x, kv_cache


    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int=20):
        """
        Generate tokens autoregressively from a given input sequence.

        Args:
            inputs (torch.Tensor): Input tensor containing token indices or embeddings.
                Shape: (batch_size, sequence_length) or (sequence_length,) for a single sequence.
            max_new_tokens (int): Number of new tokens to generate after the input sequence.

        Returns:
            torch.Tensor: The generated sequence, including the original inputs and newly generated tokens.
                Shape: (batch_size, sequence_length + max_new_tokens)
        """
        # Add batch dimension if needed
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()

        prompt_output, kv_cache_list = self.forward(
            generated_outputs, 
            attention_mask=None,
            kv_cache=None,
            start_pos=0
        )
        last_output = prompt_output[:, -1, :]

        # Decode Phase with KV cache
        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                # Now the model outputs logits
                next_output = torch.argmax(last_output, dim=-1, keepdim=True)
            else:
                # Now the model outputs embeddings
                next_output = last_output.unsqueeze(1)

            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)
            
            # The token being processed is `next_token`. Its position is `generated_outputs.size(1) - 1`.
            current_token_start_pos = generated_outputs.size(1) - 1

            if i == max_new_tokens - 1: 
                break

            decode_step_output, kv_cache_list = self.forward(
                next_output, 
                attention_mask=None,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
            last_output = decode_step_output[:, -1, :] 
    
        return generated_outputs

    # Load the model from a pretrained HuggingFace model (we don't want to have to train the Language Backbone from scratch)
    @classmethod
    def from_pretrained(cls, cfg):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
        import json
        from huggingface_hub.utils import EntryNotFoundError
                
        # Load the HuggingFace config
        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)
        
        # Store original HF vocab size before we modify it
        original_vocab_size = hf_config.vocab_size
        # print(f"Original vocabulary size from pretrained model: {original_vocab_size}")
        
        # Configure model parameters from HF config
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg, 'lm_vocab_size'):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
            # print(f"Using vocabulary size: {cfg.lm_vocab_size}")
        else:
            # If not specified, use the original
            cfg.lm_vocab_size = original_vocab_size
            # print(f"Using original vocabulary size: {cfg.lm_vocab_size}")
        
        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers
        
        # Create our model with potentially larger vocabulary
        model = cls(cfg)
        
        try:
            index_path = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors.index.json")
            with open(index_path, 'r') as f:
                index = json.load(f)
            # Get unique filenames from weight map
            safetensors_filenames = sorted(list(set(index['weight_map'].values())))
            # Download all the sharded files
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename=fn) for fn in safetensors_filenames]
        except EntryNotFoundError:
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")]

        sd = model.state_dict()
        
        mapping = {
            'model.embed_tokens.weight': 'token_embedding.weight',
            'model.norm.weight': 'norm.weight'
        }
        
        for i in range(cfg.lm_n_blocks):
            layer_prefix = f'model.layers.{i}.'
            block_prefix = f'blocks.{i}.'
            
            mapping.update({
                f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight"
            })
        
        # Special handling for token embeddings with extended vocabulary
        has_extended_embeddings = False
        loaded_keys = set()
        
        for safetensors_file in safetensors_files:
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded_keys:
                        continue
                    
                    if hf_key in f.keys() and our_key in sd:
                        tensor = f.get_tensor(hf_key)
                        
                        # Special handling for token embeddings if vocab sizes differ
                        if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
                            has_extended_embeddings = True
                            print(f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")
                            
                            # Copy existing embeddings to the beginning of our larger embedding matrix
                            sd[our_key][:tensor.shape[0]].copy_(tensor)
                            
                            # Initialize the new embeddings using the same approach as the original model
                            std = 0.02  # Common value, but you might want to adjust based on model
                            init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)
                            
                            print(f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings")
                            sd['head.weight'].copy_(sd[our_key])  # Update the head weights as well
                        elif tensor.shape == sd[our_key].shape:
                            sd[our_key].copy_(tensor)
                        else:
                            print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                        
                        loaded_keys.add(our_key)

        for hf_key, our_key in mapping.items():
            if our_key not in loaded_keys:
                if our_key in sd:
                    print(f"Warning: Key {our_key} not found in any safetensors file (HF key: {hf_key})")
        
        # Load the state dict
        model.load_state_dict(sd)
        
        # Handle output projection / language modeling head
        if has_extended_embeddings and hasattr(model, 'head') and 'head.weight' in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            lm_head_loaded = False
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                    if 'lm_head.weight' in f.keys():
                        lm_head = f.get_tensor('lm_head.weight')
                        if lm_head.shape[0] != sd['head.weight'].shape[0]:
                            print(f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")
                            # Copy existing weights
                            sd['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                            # Initialize new weights
                            std = 0.02
                            init.normal_(sd['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)
                            # Load updated weights
                            model.load_state_dict(sd)
                        lm_head_loaded = True
                        break
        
        # Handle weight tying (if needed)
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")
        
        print(f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model
```


这段代码实现了一个基于 **Llama 3** 架构的 **Language Model (LM)** Backbone（主干网络）。在 **Vision Language Model (VLM)** 的上下文中，这部分代码扮演着“大脑”的角色，负责处理来自 **Vision Encoder**（如 CLIP ViT 或 SigLIP）的视觉特征以及用户的文本输入，并进行推理生成。

虽然提供的代码中没有包含 **Vision Encoder** 或 **Projector（连接器）**，但它是 VLM 核心的 **Text Decoder** 部分。这段代码通过 `from_pretrained` 方法加载 HuggingFace 的权重，并支持 **KV Cache** 和 **Grouped Query Attention (GQA)**，这对于处理 VLM 中通常包含大量图像 Patch 导致的长上下文至关重要。

以下是详细的代码技术讲解与 VLM 相关架构扩展联想：

---

### 1. **RMSNorm (Root Mean Square Layer Normalization)**

**Class:** `RMSNorm`

在 VLM 中，视觉特征和文本特征的分布差异很大，归一化层对于稳定训练和收敛至关重要。`RMSNorm` 是 **LayerNorm** 的一种变体，去掉了均值中心化的操作，计算量更小，且在 Llama 系列模型中被证明效果优异。

#### 技术细节与公式
相比于 LayerNorm $y = \frac{x - \mu}{\sigma} \cdot \gamma$，RMSNorm 的公式如下：

$$ \bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{n} \sum_{j=1}^{n} x_j^2 + \epsilon}} \cdot \gamma_i $$

其中：
*   $x$ 是输入 Tensor。
*   $n$ 是 hidden dimension (`lm_hidden_dim`)。
*   $\epsilon$ 是为了数值稳定性添加的小常数 (`cfg.lm_rms_eps`)。
*   $\gamma$ 是可学习的缩放参数 (`self.weight`)。

#### 代码解析
```python
irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
x = x * irms * self.weight
```
*   `torch.rsqrt` 计算平方根的倒数，即 $1 / \sqrt{\dots}$。
*   这种实现方式比 `LayerNorm` 更高效，因为它不需要减去均值。

---

### 2. **Rotary Positional Embedding (RoPE)**

**Class:** `RotaryEmbedding`

VLM 输入通常包含两部分：Image Tokens（图像 Patch 展开后的序列）和 Text Tokens。**RoPE** 通过旋转矩阵将位置信息注入到 Query 和 Key 向量中，具有较好的外推性。

#### 技术细节
RoPE 的核心思想是将位置索引 $m$ 映射为旋转角度 $\theta$。
频率计算公式：
$$ \theta_i = 10000^{-2i/d}, \quad i \in [0, d/2] $$
对于位置 $m$，其旋转角度为 $m\theta$。

代码中实现了 **Linear Scaling（线性缩放）** 来处理超过训练长度的序列（这在处理高分辨率图像、产生大量 Patch 时非常有用）：
```python
if max_seq > self.original_max_seq_len:
    scale = max_seq / self.original_max_seq_len
    inv_freq = self.inv_freq / scale
```

#### VLM 中的联想
在 VLM（如 LLaVA, InternVL）中，如果输入一张 $448 \times 448$ 的图像，经过 ViT 处理可能变成 256 甚至 1024 个 Token。加上文本 prompt，序列长度很容易超过 LLM 原本的 2048 或 4096 限制。这里的 Scaling 机制允许模型在不重新训练的情况下容纳更长 Vision Context。

---

### 3. **Grouped Query Attention (GQA)**

**Class:** `LanguageModelGroupedQueryAttention`

**GQA** 是 **Multi-Head Attention (MHA)** 和 **Multi-Query Attention (MQA)** 的折中方案。在 VLM 推理阶段，**KV Cache** 的大小是显存占用的主要瓶颈。

#### 架构解析
*   **Query Heads** (`lm_n_heads`): 数量较多，保持模型强大的表达能力。
*   **KV Heads** (`lm_n_kv_heads`): 数量较少。例如，Llama-3-8B 中，n_heads=32, n_kv_heads=8。
*   **Grouping**: 每 $G = n\_heads / n\_kv\_heads$ 个 Query Head 共享一组 key 和 value。

#### 代码实现
代码通过 `repeat_interleave` 将 KV 扩展以匹配 Q 的维度：
```python
# Repeat K, V for Grouped Query Attention
k_exp = k.repeat_interleave(self.n_kv_groups, dim=1) 
v_exp = v.repeat_interleave(self.n_kv_groups, dim=1) 
```

#### VLM 性能影响表
| 机制 | KV Cache 显存占用 | 推理速度 | 模型表达能力 |
| :--- | :--- | :--- | :--- |
| MHA (Multi-Head) | 高 (100%) | 慢 | 强 |
| **GQA (Grouped)** | **低** | **快** | **较强** |
| MQA (Multi-Query) | 极低 | 极快 | 较弱 |

在 VLM 处理高分辨率图像（大量 Vision Tokens）时，GQA 能显著降低显存消耗，使得单卡推理成为可能。

---

### 4. **SwiGLU MLP Activation**

**Class:** `LanguageModelMLP`

这是 Transformer FFN 层的一种改进激活函数结构，相比标准的 ReLU 或 GeLU，SwiGLU 通常能带来更好的性能。

#### 数学公式
$$ \text{FFN}_{SwiGLU}(x) = \text{Down}(\text{SiLU}(\text{Gate}(x)) \cdot \text{Up}(x)) $$

代码中对应：
```python
gate = self.activation_fn(self.gate_proj(x)) # SiLU(Gate(x))
x = self.up_proj(x)
x = self.down_proj(gate * x)               # Down(SiLU(...) * Up(x))
```
这种门控机制增强了模型对特征的非线性变换能力，对于理解复杂的图像-文本语义对齐非常有帮助。

---

### 5. **Language Model 主体与 VLM 对齐 (`LanguageModel` & `from_pretrained`)**

**Class:** `LanguageModel`

这是 VLM 的核心调度器。特别值得注意的是 `from_pretrained` 方法中的 **Embedding Extension**（词表扩展）逻辑。

#### VLM 中的 Token 扩展逻辑
在 VLM 训练中，我们通常需要引入特殊的 **Token** 来代表图像或特定的指令起始位置，例如 `<image>`, `<pad>` 等。这会导致我们需要扩展现有的 LLM 词表。

代码片段详细展示了如何从 HuggingFace 加载权重并处理词表不匹配的问题：
```python
if hf_key == 'model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
    has_extended_embeddings = True
    # 1. 复制原有的词向量
    sd[our_key][:tensor.shape[0]].copy_(tensor)
    # 2. 初始化新增的词向量（例如，<image> token 的 embedding）
    std = 0.02
    init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)
```

####联想：Projector 的作用
虽然这段代码没有包含 **Projector**，但在 VLM 流程中，`LanguageModel` 的输入往往经过一个 Projector（通常是一个简单的 MLP 或 Q-Former）。例如：
1.  **Image** -> **Vision Encoder (ViT)** -> **Image Features** (dim=1024/4096)
2.  **Image Features** -> **Projector (MLP)** -> **Projected Embeddings** (dim=4096, matches `lm_hidden_dim`)
3.  **Projected Embeddings** -> **`LanguageModel.forward`**

这段代码的 `forward` 函数支持 `lm_use_tokens=False`，这意味着你可以直接传入处理好的 Image Embeddings 而不是 Token IDs，这正是 VLM 推理时的关键接口。

#### Generation Loop with KV Cache
```python
decode_step_output, kv_cache_list = self.forward(
    next_output, 
    kv_cache=kv_cache_list, # 传入之前的 KV Cache
    start_pos=current_token_start_pos # 计算正确的位置
)
```
*   **Prefill 阶段**: 处理整个 Prompt（包含大量 Image Tokens），一次性填满 KV Cache。
*   **Decode 阶段**: 逐个生成 Token，直接读取 Cache，避免重复计算 Image Tokens 的 Attention。

---

### 6. **参考资料与 Further Reading**

为了对 VLM 和该代码涉及的架构有更深入理解，请参考以下链接：

1.  **Llama 3 Model Card (Architecture Source)**:
    *   https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md
    *   *讲解该代码所基于的原始架构，包括 GQA 和 RoPE 的官方定义。*

2.  **LLaVA (Large Language-and-Vision Assistant)**:
    *   https://github.com/haotian-liu/LLaVA
    *   *经典的 VLM 架构，展示了如何将 CLIP ViT 连接到类似的 LLM Backbone（如 Vicuna/Llama）。*

3.  **RoPE (Rotary Positional Embeddings) Paper**:
    *   https://arxiv.org/abs/2104.09864
    *   *详细了解旋转位置编码的数学原理。*

4.  **GQA (Grouped-Query Attention) Paper**:
    *   https://arxiv.org/abs/2305.13245
    *   *理解代码中 KV Cache 优化的理论基础。*

5.  **HuggingFace Transformers (Source of Pretrained Weights)**:
    *   https://huggingface.co/docs/transformers/model_doc/llama
    *   *查看 `from_pretrained` 方法兼容的标准模型格式。*

6.  **ViT-22B (Vision Encoder Context)**:
    *   https://arxiv.org/abs/2302.05442
    *   *了解在 VLM 中常用的视觉编码器架构，这是 `LanguageModel` 输入数据的来源之一。*

### 总结
这段代码是一个高效、现代化的 **LLM Decoder** 实现。
*   **RMSNorm** 保证了训练稳定性。
*   **RoPE** 处理了包含图像和文本的长序列位置关系。
*   **GQA** 和 **KV Cache** 优化了推理速度，使得处理高分辨率图像成为可能。
*   **Embedding Extension** 机制支持了 VLM 特有的特殊 Token 需求。

在 VLM 系统中，这个模块充当中央处理器，它接收经过编码的图像信息和文本指令，通过复杂的注意力机制融合视觉与语言信息，最终生成人类可读的文本回复。