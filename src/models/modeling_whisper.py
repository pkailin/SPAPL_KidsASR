# coding=utf-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Whisper model."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import copy
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from models.configuration_whisper import WhisperConfig
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "WhisperConfig"
_CHECKPOINT_FOR_DOC = "openai/whisper-tiny"


WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/whisper-base",
    # See all Whisper models at https://huggingface.co/models?filter=whisper
]


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


def _median_filter(inputs: torch.Tensor, filter_width: int) -> torch.Tensor:
    """
    Applies a median filter of width `filter_width` along the last dimension of the input.

    The `inputs` tensor is assumed to be 3- or 4-dimensional.
    """
    if filter_width <= 0 or filter_width % 2 != 1:
        raise ValueError("`filter_width` should be an odd number")

    pad_width = filter_width // 2
    if inputs.shape[-1] <= pad_width:
        return inputs

    # Pad the left and right edges.
    inputs = nn.functional.pad(inputs, (pad_width, pad_width, 0, 0), mode="reflect")

    # sort() is faster than torch.median (https://github.com/pytorch/pytorch/issues/51450)
    result = inputs.unfold(-1, filter_width, 1).sort()[0][..., pad_width]
    return result


def _dynamic_time_warping(matrix: np.ndarray):
    """
    Measures similarity between two temporal sequences: the input audio and the output tokens. Used to generate
    token-level timestamps.
    """
    output_length, input_length = matrix.shape
    cost = np.ones((output_length + 1, input_length + 1), dtype=np.float32) * np.inf
    trace = -np.ones((output_length + 1, input_length + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, input_length + 1):
        for i in range(1, output_length + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = matrix[i - 1, j - 1] + c
            trace[i, j] = t

    # backtrace
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    text_indices = []
    time_indices = []
    while i > 0 or j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise RuntimeError(
                f"Internal error in dynamic time warping. Unexpected trace[{i}, {j}]. Please file a bug report."
            )

    text_indices = np.array(text_indices)[::-1]
    time_indices = np.array(time_indices)[::-1]
    return text_indices, time_indices


class WhisperPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0):
        return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[1]]


class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.peft_type = None

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def setup_peft(self, peft_config):
        self.peft_type = peft_config.peft_type
        from models.peft import LoRALinear

        lora_dim = peft_config.lora_dim
        lora_alpha = peft_config.lora_alpha
        dropout = peft_config.dropout

        self.lora_v_proj = LoRALinear(self.v_proj, lora_dim, lora_alpha, dropout)
        self.lora_q_proj = LoRALinear(self.q_proj, lora_dim, lora_alpha, dropout)

        self.weights_merged = False

    def merge_lora_weights(self):
        assert self.weights_merged == False

        self.v_proj.weight.data += self.lora_v_proj.lora_B @ self.lora_v_proj.lora_A * self.lora_v_proj.scaling
        self.q_proj.weight.data += self.lora_q_proj.lora_B @ self.lora_q_proj.lora_A * self.lora_q_proj.scaling

        self.weights_merged = True

    def substract_lora_weights(self):
        assert self.weights_merged == True

        self.v_proj.weight.data -= self.lora_v_proj.lora_B @ self.lora_v_proj.lora_A * self.lora_v_proj.scaling
        self.q_proj.weight.data -= self.lora_q_proj.lora_B @ self.lora_q_proj.lora_A * self.lora_q_proj.scaling

        self.weights_merged = False

    # Copied from transformers.models.bart.modeling_bart.BartAttention.forward with BART->whisper
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) 

        if self.peft_type == "lora" and not self.weights_merged:
            query_states += (self.lora_q_proj.lora_dropout(hidden_states) @ self.lora_q_proj.lora_A.transpose(0, 1) @ 
                                self.lora_q_proj.lora_B.transpose(0, 1)) * self.lora_q_proj.scaling
        
        query_states = query_states * self.scaling
            
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)

            value_states = self.v_proj(key_value_states)
            
            if self.peft_type == "lora" and not self.weights_merged:
                value_states += (self.lora_v_proj.lora_dropout(key_value_states) @ self.lora_v_proj.lora_A.transpose(0, 1) @ 
                                self.lora_v_proj.lora_B.transpose(0, 1)) * self.lora_v_proj.scaling

            value_states = self._shape(value_states, -1, bsz)

        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)

            value_states = self.v_proj(hidden_states)
            
            if self.peft_type == "lora" and not self.weights_merged:
                value_states += (self.lora_v_proj.lora_dropout(hidden_states) @ self.lora_v_proj.lora_A.transpose(0, 1) @ 
                                self.lora_v_proj.lora_B.transpose(0, 1)) * self.lora_v_proj.scaling

            value_states = self._shape(value_states, -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self.v_proj(hidden_states)
            
            if self.peft_type == "lora" and not self.weights_merged:
                value_states += (self.lora_v_proj.lora_dropout(hidden_states) @ self.lora_v_proj.lora_A.transpose(0, 1) @ 
                                self.lora_v_proj.lora_B.transpose(0, 1)) * self.lora_v_proj.scaling

            value_states = self._shape(value_states, -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper
class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def setup_peft(self, peft_config):
        # only setup lora weighst for the attention layer
        self.self_attn.setup_peft(peft_config)

    def merge_lora_weights(self):
        self.self_attn.merge_lora_weights()

    def substract_lora_weights(self):
        self.self_attn.substract_lora_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer with MBart->Whisper
class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def setup_peft(self, peft_config):
        # only setup lora weighst for the attention layer
        self.self_attn.setup_peft(peft_config)
        self.encoder_attn.setup_peft(peft_config)

    def merge_lora_weights(self):
        self.self_attn.merge_lora_weights()
        self.encoder_attn.merge_lora_weights()

    def substract_lora_weights(self):
        self.self_attn.substract_lora_weights()
        self.encoder_attn.substract_lora_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class WhisperPreTrainedModel(PreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WhisperEncoderLayer", "WhisperDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths


WHISPER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`WhisperConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

WHISPER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing *SpecAugment* data augmentation on padding token indices. Mask values selected in
            `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            Whisper uses the `decoder_start_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read
            [`modeling_whisper._prepare_decoder_attention_mask`] and modify to your needs. See diagram 1 in [the BART
            paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

WHISPER_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class WhisperEncoder(WhisperPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.embed_dim = embed_dim
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self.peft_type = None

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def setup_peft(self, peft_config):
        self.peft_type = peft_config.peft_type   # "lora" or "adapter"

        if self.peft_type == "lora":
            
            from models.peft import LoRAConv
            lora_dim = peft_config.lora_dim
            lora_alpha = peft_config.lora_alpha
            dropout = peft_config.dropout

            if 0 in peft_config.peft_encoder_layers:
                self.lora_conv1 = LoRAConv(self.conv1, lora_dim, lora_alpha, dropout)
                self.lora_conv2 = LoRAConv(self.conv2, lora_dim, lora_alpha, dropout)
            else:
                self.lora_conv1 = None
                self.lora_conv2 = None

            for idx, layer in enumerate(self.layers):
                if idx + 1 in peft_config.peft_encoder_layers:
                    layer.setup_peft(peft_config)
                    layer.contain_lora = True
                else:
                    layer.contain_lora = False

            self.weights_merged = False
        
        if self.peft_type == "adapter":
            from models.peft import ResidualAdapter

            bottleneck_dim = peft_config.bottleneck_dim
            dropout = peft_config.dropout
            
            self.adapters = nn.ModuleList([ResidualAdapter(self.embed_dim, bottleneck_dim, dropout) if i in peft_config.peft_encoder_layers
                                else lambda x: x for i in range(len(self.layers)+1)])  

        if self.peft_type == "prompt_tuning":
            self.n_tokens = peft_config.prompt_n_tokens[0]
            random_range = peft_config.prompt_random_range
            init_prompt_value = torch.FloatTensor(self.n_tokens, self.embed_dim).uniform_(
                -random_range, random_range
            )
            self.soft_prompt = nn.Embedding(self.n_tokens, self.embed_dim)
            # Initialize weight
            self.soft_prompt.weight = nn.Parameter(init_prompt_value)

        if self.peft_type == "prefix_tuning":
            self.prefix_seq_len = peft_config.prefix_seq_len[0]
            self.prefix_wte = nn.Embedding(self.prefix_seq_len, self.embed_dim)
            self.prefix_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, peft_config.prefix_hidden_dim),
                nn.Tanh(),
                nn.Linear(peft_config.prefix_hidden_dim, peft_config.prefix_n_layer * self.embed_dim),
            )
            self.prefix_dropout = nn.Dropout(peft_config.prefix_dropout_rate)


    def merge_lora_weights(self):
        assert self.weights_merged == False, "The lora weights are already merged."

        if self.lora_conv1 is not None:
            self.conv1.weight.data += self.lora_conv1.lora_B @ self.lora_conv1.lora_A.view(self.conv1.weight.shape) * self.lora_conv1.scaling 
            self.conv2.weight.data += self.lora_conv2.lora_B @ self.lora_conv2.lora_A.view(self.conv2.weight.shape) * self.lora_conv2.scaling

        for layer in self.layers:
            if layer.contain_lora:
                layer.merge_lora_weights()
        
        self.weights_merged = True

    def substract_lora_weights(self):
        assert self.weights_merged == True, "The lora weights are already substracted!"

        if self.lora_conv1 is not None:
            self.conv1.weight.data -= self.lora_conv1.lora_B @ self.lora_conv1.lora_A.view(self.conv1.weight.shape) * self.lora_conv1.scaling 
            self.conv2.weight.data -= self.lora_conv2.lora_B @ self.lora_conv2.lora_A.view(self.conv2.weight.shape) * self.lora_conv2.scaling

        for layer in self.layers:
            if layer.contain_lora:
                layer.substract_lora_weights()

        self.weights_merged = False

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.peft_type == "lora" and not self.weights_merged:
            inputs_embeds = nn.functional.gelu(self.conv1._conv_forward(
                input_features, 
                self.conv1.weight + (self.lora_conv1.lora_B @ self.lora_conv1.lora_A).view(self.conv1.weight.shape) * self.lora_conv1.scaling,
                self.conv1.bias,
                ))
            inputs_embeds = nn.functional.gelu(self.conv2._conv_forward(
                inputs_embeds, 
                self.conv2.weight + (self.lora_conv2.lora_B @ self.lora_conv2.lora_A).view(self.conv2.weight.shape) * self.lora_conv2.scaling,
                self.conv2.bias,
                ))
        else:
            inputs_embeds = nn.functional.gelu(self.conv1(input_features))
            inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        if self.peft_type == "adapter":
            inputs_embeds = self.adapters[0](inputs_embeds)

        embed_pos = self.embed_positions.weight
        hidden_states = inputs_embeds + embed_pos

        if self.peft_type == "prompt_tuning":
            # [batch_size, n_tokens, n_embds
            learned_embeds = self.soft_prompt.weight.repeat(hidden_states.size(0), 1, 1)
            hidden_states = torch.cat([learned_embeds, hidden_states], dim=1)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        if self.peft_type == "prefix_tuning":
            learned_embeds = self.prefix_wte.weight
            all_embeds = self.prefix_dropout(self.prefix_mlp(learned_embeds)).repeat(hidden_states.size(0), 1, 1)

        for idx, encoder_layer in enumerate(self.layers):
            if self.peft_type == "prefix_tuning":
                start, end = idx * self.embed_dim, (idx + 1) * self.embed_dim
                hidden_states = torch.cat([all_embeds[:,:,start:end], hidden_states], dim=1)

            if output_hidden_states:
                
                if self.peft_type == "prompt_tuning":
                    hidden_states_save = hidden_states[:,self.n_tokens:,:]
                elif self.peft_type == "prefix_tuning":
                    hidden_states_save = hidden_states[:,self.prefix_seq_len:,:]
                else:
                    hidden_states_save = hidden_states

                encoder_states = encoder_states + (hidden_states_save,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        use_reentrant=False,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

                if self.peft_type == "adapter":
                    hidden_states = self.adapters[idx+1](hidden_states)
                
                if self.peft_type == "prefix_tuning":
                    hidden_states = hidden_states[:,self.prefix_seq_len:,:]

            if output_attentions:
                # to do: to be compatible with prefix tuning and prompt tuning
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        
        if self.peft_type == "prompt_tuning":
            hidden_states = hidden_states[:,self.n_tokens:,:]
        
        if self.peft_type == "prefix_tuning":
            hidden_states = hidden_states[:,self.prefix_seq_len:,:]
       
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class WhisperDecoder(WhisperPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.embed_dim = config.d_model
        self.n_head = config.decoder_attention_heads
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList([WhisperDecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self.peft_type = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def setup_peft(self, peft_config):
        self.peft_type = peft_config.peft_type   # "lora" or "adapter"

        if self.peft_type == "lora":

            for idx, layer in enumerate(self.layers):
                if idx + 1 in peft_config.peft_decoder_layers:
                    layer.setup_peft(peft_config)
                    layer.contain_lora = True
                else:
                    layer.contain_lora = False

            self.weights_merged = False
        
        if self.peft_type == "adapter":
            from models.peft import ResidualAdapter

            bottleneck_dim = peft_config.bottleneck_dim
            dropout = peft_config.dropout
            
            self.adapters = nn.ModuleList([ResidualAdapter(self.embed_dim, bottleneck_dim, dropout) if i in peft_config.peft_decoder_layers
                                else lambda x: x for i in range(1, len(self.layers)+1)])    
            
        if self.peft_type == "prompt_tuning":
            self.n_tokens = peft_config.prompt_n_tokens[1]
            self.soft_prompt = nn.Embedding(self.n_tokens, self.embed_dim)

            if peft_config.prompt_init_vocab:
                init_prompt_value = self.embed_tokens.weight[:self.n_tokens].clone().detach()
            else:
                random_range = peft_config.prompt_random_range
                init_prompt_value = torch.FloatTensor(self.n_tokens, self.embed_dim).uniform_(
                    -random_range, random_range
                )
                # Initialize weight
            self.soft_prompt.weight = nn.Parameter(init_prompt_value)

        if self.peft_type == "prefix_tuning":
            self.prefix_seq_len = peft_config.prefix_seq_len[1]
            self.prefix_n_layer = peft_config.prefix_n_layer
            self.prefix_wte = nn.Embedding(self.prefix_seq_len, self.embed_dim)
            self.prefix_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, peft_config.prefix_hidden_dim),
                nn.Tanh(),
                nn.Linear(peft_config.prefix_hidden_dim, peft_config.prefix_n_layer * 2 * self.embed_dim),
            )
            self.prefix_dropout = nn.Dropout(peft_config.prefix_dropout_rate)

    def merge_lora_weights(self):
        assert self.weights_merged == False, "The lora weights are already merged."

        for layer in self.layers:
            if layer.contain_lora:
                layer.merge_lora_weights()
        
        self.weights_merged = True

    def substract_lora_weights(self):
        assert self.weights_merged == True, "The lora weights are already substracted!"

        for layer in self.layers:
            if layer.contain_lora:
                layer.substract_lora_weights()

        self.weights_merged = False

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if self.peft_type == "prompt_tuning" and past_key_values is None:
            # [batch_size, n_tokens, n_embds]
            learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)
            inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)
            input_shape = inputs_embeds.size()[:-1]

        if self.peft_type == "prefix_tuning" and past_key_values is None:
            
            learned_embeds = self.prefix_wte.weight
            all_embeds = self.prefix_mlp(learned_embeds).repeat(inputs_embeds.size(0), 1, 1)
            
            past_key_values = all_embeds.view(
                inputs_embeds.size(0),
                self.prefix_seq_len,
                self.prefix_n_layer * 2,
                self.n_head,
                int(self.embed_dim / self.n_head),
            )
            # Dropout
            past_key_values = self.prefix_dropout(past_key_values)       
            # Transpose -> [match_n_layer*2, batch_size, match_n_head, prefix_seq_len, match_n_embd]
    
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)          
            past_key_values_length = past_key_values[0][0].shape[2]
            self.gradient_checkpointing = False

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        
        # embed positions
        if input_ids is not None:
            if self.peft_type == "prompt_tuning" and past_key_values is None:
                positions = self.embed_positions(inputs_embeds, past_key_values_length=past_key_values_length)
            elif self.peft_type == "prompt_tuning":
                positions = self.embed_positions(inputs_embeds, past_key_values_length=past_key_values_length-self.n_tokens)
            else:
                positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)
        else:
            positions = self.embed_positions(inputs_embeds, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if self.peft_type == "adapter":
                hidden_states = self.adapters[idx](hidden_states)

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if self.peft_type == "prompt_tuning" and past_key_values is None:
            hidden_states = hidden_states[:,self.n_tokens:,:]
       
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare Whisper Model outputting raw hidden-states without any specific head on top.",
    WHISPER_START_DOCSTRING,
)
class WhisperModel(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.encoder._freeze_parameters()

    def freeze_decoder(self):
        self.decoder._freeze_parameters()

    def setup_peft(self, peft_config):
        # using peft for encoder only
        if peft_config.to_encoder:
            self.encoder.setup_peft(peft_config)
        else:
            self.encoder._requires_grad = False

        if peft_config.to_decoder:
            self.decoder.setup_peft(peft_config)
        else:
            self.decoder._requires_grad = False

    def _mask_input_features(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features
        
        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        perturbed_features: Optional[torch.FloatTensor] = None,
        pif_layer: Optional[int] = 0,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, WhisperModel
         >>> from datasets import load_dataset

         >>> model = WhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            if perturbed_features is not None:
                perturbed_features = self._mask_input_features(perturbed_features, attention_mask=attention_mask)
                perturbed_encoder_outputs = self.encoder(
                    perturbed_features,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                # MSE_loss
                pif_loss = F.mse_loss(encoder_outputs.hidden_states[int(pif_layer)], perturbed_encoder_outputs.hidden_states[int(pif_layer)], reduction='none')
                print(attention_mask)
                attention_mask = attention_mask[:,::2].unsqueeze(-1)
                pif_loss = pif_loss.masked_fill(attention_mask==0, 0).sum() / attention_mask.sum() / pif_loss.size(-1)


        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
      
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        y = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        if perturbed_features is not None:
            return y, pif_loss
        else:
            return y


@add_start_docstrings(
    "The Whisper Model with a language modeling head. Can be used for automatic speech recognition.",
    WHISPER_START_DOCSTRING,
)
class WhisperForConditionalGeneration(WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if hasattr(config, "peft_config"):
            self.setup_peft(config.peft_config)
            
        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    def freeze_decoder(self):
        self.model.decoder._freeze_parameters()

    def setup_peft(self, peft_config):
        from dataclasses import fields

        peft_config_dict = {}
        for field in fields(peft_config):
            value = getattr(peft_config, field.name)
            peft_config_dict[field.name] = value

        self.config.peft_config = peft_config_dict
        self.model.setup_peft(peft_config)

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        perturbed_features: Optional[torch.FloatTensor] = None,
        pif_layer: Optional[int] = 0,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if perturbed_features is not None:
            output_hidden_states = True

        outputs = self.model(
            input_features,
            perturbed_features=perturbed_features,
            pif_layer=pif_layer,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if perturbed_features is not None:
            pif_loss = outputs[1]
            outputs = outputs[0]

        lm_logits = self.proj_out(outputs[0])
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

            if perturbed_features is not None:
                loss += self.config.pif_loss_alpha * pif_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        prompt_ids: Optional[torch.Tensor] = None,
        num_segment_frames: Optional[int] = None,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: int = 0.02,
        return_dict_in_generate: Optional[bool] = None,
        **kwargs,
    ):
        """
        Transcribes or translates passed mel input features to a sequence of token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            return_timestamps (`bool`, *optional*):
                Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
            task (`str`, *optional*):
                Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
                will be updated accordingly.
            language (`str`, *optional*):
                Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. You can
                find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
            is_multilingual (`bool`, *optional*):
                Whether or not the model is multilingual.
            prompt_ids (`torch.Tensor`, *optional*):
                Rank-1 tensor of token IDs created by passing text to [`~WhisperProcessor.get_prompt_ids`] that is
                provided as a prompt to each chunk. This can be used to provide or "prompt-engineer" a context for
                transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
                correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
            return_token_timestamps (`bool`, *optional*):
                Whether to return token-level timestamps with the text. This can be used with or without the
                `return_timestamps` option. To get word-level timestamps, use the tokenizer to group the tokens into
                words.
            return_segments (`bool`, *optional*, defaults to `False`):
                Whether to additionally return a list of all segments. Note that this option can only be enabled
                when doing long-form transcription.
            attention_mask (`torch.Tensor`, *optional*):
                `attention_mask` needs to be passed when doing long-form transcription using a batch size > 1.
            time_precision (`int`, *optional*, defaults to 0.02):
                The duration of output token in seconds. *E.g.* 0.02 means that a generated token on average accounts
                for 20 ms.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of just returning the generated tokens.
                Note that when doing long-form transcription, `return_dict_in_generate` can only be enabled when
                `return_segments` is set True. In this case the generation outputs of each segment is added to each
                segment.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor` or `Dict[str, Any]`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor` or a dict of segments when `return_segments=True`.

                If the passed input is > 30 seconds / > 3000 mel input features and `return_segments=True` then a dictionary of generated sequence ids, called `sequences` and a list of each generated segment is returned.

                else if the passed input is <= 30 seconds / >= 3000 mel input features, the possible [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]

                else only the generated output sequence ids are returned.

        Example:

        - *Longform transcription*: To transcribe or translate audios longer than 30 seconds, process the audio files without truncation and pass all mel features at once to generate.

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset, Audio

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        >>> model.cuda()

        >>> # load audios > 30 seconds
        >>> ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
        >>> # resample to 16kHz
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        >>> # take first 8 audios and retrieve array
        >>> audio = ds[:8]["audio"]
        >>> audio = [x["array"] for x in audio]

        >>> # make sure to NOT truncate the input audio, to return the `attention_mask` and to pad to the longest audio
        >>> inputs = processor(audio, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000)
        >>> inputs = inputs.to("cuda", torch.float32)

        >>> # transcribe audio to ids
        >>> generated_ids = model.generate(**inputs)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        >>> transcription[0]
        ' Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories developing the central headline pawns, definitely maneuvering an oso topical night to F6, fainting a classic Sicilian, nade door variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a fisher's shows in Lip Nitsky attack that culminates in the elegant lethal slow-played, all-passant checkmate that is my nightly monologue. But sometimes, sometimes, folks, I. CHEERING AND APPLAUSE Sometimes I startle away, cubside down in the monkey bars of a condemned playground on a super fun site. Get all hept up on goofballs. Rummage that were discarded tag bag of defective toys. Yank out a fist bowl of disembodied doll limbs, toss them on a stained kid's place mat from a defunct dennies. set up a table inside a rusty cargo container down by the Wharf and challenged toothless drifters to the godless bughouse blitz of tournament that is my segment. Meanwhile!'
        ```

        - *Shortform transcription*: If passed mel input features are < 30 seconds, the whole audio will be transcribed with a single call to generate.

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```

        """
        
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")

        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)

        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        if num_segment_frames is None:
            num_segment_frames = input_stride * self.config.max_source_positions

        # 1. Check whether we're in shortform or longform mode
        if input_features is not None:
            total_input_frames = input_features.shape[-1]
        elif "encoder_outputs" in kwargs:
            encoder_outputs_shape = (
                kwargs["encoder_outputs"][0].shape
                if isinstance(kwargs["encoder_outputs"], BaseModelOutput)
                else kwargs["encoder_outputs"].shape
            )
            total_input_frames = encoder_outputs_shape[1] * input_stride
        else:
            raise ValueError("Make sure to provide either `input_features` or `encoder_outputs` to `generate`.")

        is_shortform = total_input_frames <= num_segment_frames

        # 2. Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        if return_timestamps is True:
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You are trying to return timestamps, but the generation config is not properly set. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                )
            generation_config.return_timestamps = return_timestamps
        elif not is_shortform:
            if return_timestamps is False:
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the model to predict timestamp tokens. Please either pass `return_timestamps=True` or make sure to pass no more than 3000 mel input features."
                )

            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the generation config to have `no_timestamps_token_id` correctly. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                    "or make sure to pass no more than 3000 mel input features."
                )

            # logger.info("Setting `return_timestamps=True` for long-form generation.")
            # for the finetuning models, they cannot predict the timestamps, we 
            logger.info("We don't do timestamps for the finetuning models.")
            generation_config.return_timestamps = False
        else:
            generation_config.return_timestamps = False
        
        # 3. Make sure to correctly set language-related parameters
        if is_multilingual is not None:
            if not hasattr(generation_config, "is_multilingual"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `is_multilingual` argument "
                    "to `generate`. Please update the generation config as per the instructions "
                    "https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.is_multilingual = is_multilingual

        if hasattr(generation_config, "is_multilingual") and not generation_config.is_multilingual:
            if task is not None or language is not None:
                raise ValueError(
                    "Cannot specify `task` or `language` for an English-only model. If the model is intended to be "
                    "multilingual, pass `is_multilingual=True` to generate, or update the generation config."
                )

        if language is not None:
            if not hasattr(generation_config, "lang_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `language` argument "
                    "to `generate`. Either set the language using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            language = language.lower()
            generation_config.language = language
        if task is not None:
            if not hasattr(generation_config, "task_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `task` argument "
                    "to `generate`. Either set the task using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.task = task

        # 4. Add forced decoder ids depending on passed `language`, `task`,`prompt_ids`, `return_token_timestamps` and `return_timestamps`
        forced_decoder_ids = None
        # Legacy code for backward compatibility
        if hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids is not None:
            forced_decoder_ids = self.config.forced_decoder_ids
        elif (
            hasattr(self.generation_config, "forced_decoder_ids")
            and self.generation_config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = self.generation_config.forced_decoder_ids
        else:
            forced_decoder_ids = kwargs.get("forced_decoder_ids", None)

        if task is not None or language is not None or (forced_decoder_ids is None and prompt_ids is not None):
            forced_decoder_ids = []
            if hasattr(generation_config, "language"):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
                elif generation_config.language in TO_LANGUAGE_CODE.values():
                    language_token = f"<|{generation_config.language}|>"
                else:
                    is_language_code = len(generation_config.language) == 2
                    raise ValueError(
                        f"Unsupported language: {generation_config.language}. Language should be one of:"
                        f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                    )
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))  # automatically detect the language

            if hasattr(generation_config, "task"):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(
                        f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                    )
            elif hasattr(generation_config, "task_to_id"):
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
            if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        if forced_decoder_ids is not None:
            generation_config.forced_decoder_ids = forced_decoder_ids
        
        if prompt_ids is not None:
            if kwargs.get("decoder_start_token_id") is not None:
                raise ValueError(
                    "When specifying `prompt_ids`, you cannot also specify `decoder_start_token_id` as it gets overwritten."
                )
            prompt_ids = prompt_ids.tolist()
            decoder_start_token_id, *text_prompt_ids = prompt_ids
            # Slicing the text prompt ids in a manner consistent with the OpenAI implementation
            # to accomodate context space for the prefix (see https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/decoding.py#L599)
            text_prompt_ids = text_prompt_ids[-self.config.max_target_positions // 2 - 1 :]
            # Set the decoder_start_token_id to <|startofprev|>
            kwargs.update({"decoder_start_token_id": decoder_start_token_id})

            # If the user passes `max_new_tokens`, increase its number to account for the prompt
            if kwargs.get("max_new_tokens", None) is not None:
                kwargs["max_new_tokens"] += len(text_prompt_ids)
                if kwargs["max_new_tokens"] >= self.config.max_target_positions:
                    raise ValueError(
                        f"The length of the sliced `prompt_ids` is {len(text_prompt_ids)}, and the `max_new_tokens` "
                        f"{kwargs['max_new_tokens'] - len(text_prompt_ids)}. Thus, the combined length of the sliced "
                        f"`prompt_ids` and `max_new_tokens` is: {kwargs['max_new_tokens']}. This exceeds the "
                        f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                        "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                        f"so that their combined length is less that {self.config.max_target_positions}."
                    )

            # Reformat the forced_decoder_ids to incorporate the prompt
            non_prompt_forced_decoder_ids = (
                kwargs.pop("forced_decoder_ids", None) or generation_config.forced_decoder_ids
            )
            forced_decoder_ids = [
                *text_prompt_ids,
                generation_config.decoder_start_token_id,
                *[token for _rank, token in non_prompt_forced_decoder_ids],
            ]
            forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_decoder_ids)]
            generation_config.forced_decoder_ids = forced_decoder_ids

        if return_token_timestamps:
            kwargs["output_attentions"] = True
            return_dict_in_generate = True

            if getattr(generation_config, "task", None) == "translate":
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            if not hasattr(generation_config, "alignment_heads"):
                raise ValueError(
                    "Model generation config has no `alignment_heads`, token-level timestamps not available. "
                    "See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
                )

            if kwargs.get("num_frames") is not None:
                generation_config.num_frames = kwargs.pop("num_frames")

        if generation_config.return_timestamps is True:
            last_forced_decoder_ids = (
                generation_config.forced_decoder_ids[-1][-1]
                if hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids
                else None
            )
            if last_forced_decoder_ids == self.generation_config.no_timestamps_token_id:
                # remove no_timestamp to be forcefully generated if we want to return timestamps
                # this is also important to make sure `WhisperTimeStampLogitsProcessor` functions correctly
                forced_decoder_ids = generation_config.forced_decoder_ids[:-1]
                # Make sure that if list is empty we set it to None
                generation_config.forced_decoder_ids = None if len(forced_decoder_ids) == 0 else forced_decoder_ids

            timestamp_processor = [WhisperTimeStampLogitsProcessor(generation_config)]
            logits_processor = (
                timestamp_processor if logits_processor is None else timestamp_processor + logits_processor
            )
        
        # 5. If we're in shortform mode, simple generate the whole input at once and return the output
        if is_shortform:
            outputs = super().generate(
                input_features,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

            if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                num_frames = getattr(generation_config, "num_frames", None)
                outputs["token_timestamps"] = self._extract_token_timestamps(
                    outputs, generation_config.alignment_heads, num_frames=num_frames
                )

            return outputs

        # 6. Else we're in longform mode which is more complex. We need to chunk the audio input depending on when the model generated
        # timestamp tokens
        # 6.1 Set running parameters for while loop
        if not return_segments and return_dict_in_generate:
            raise ValueError(
                "Make sure to set `return_segments=True` to return generation outputs as part of the `'segments' key.`"
            )

        # if input is longer than 30 seconds we default to long-form generation
        timestamp_begin = self.generation_config.no_timestamps_token_id + 1
        # input stride is mel frames per encoder output vector which is the product of all conv strides
        batch_size = input_features.shape[0]

        if batch_size > 1 and attention_mask is None:
            raise ValueError(
                "When doing long-form audio transcription, make sure to pass an `attention_mask`. You can retrieve the `attention_mask` by doing `processor(audio, ..., return_attention_mask=True)` "
            )
        elif batch_size > 1:
            max_frames = attention_mask.sum(-1).cpu().to(torch.long)
            seek = torch.zeros((batch_size,), dtype=torch.long)
        else:
            max_frames = torch.ones((1,), dtype=torch.long) * total_input_frames
            seek = torch.zeros((1,), dtype=torch.long)

        current_segments = [[] for _ in range(batch_size)]
        cur_to_prev_index_map = list(range(batch_size))

        # batch size can decrease during the run
        cur_bsz = prev_bsz = batch_size

        # 6.2 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            prev_bsz = cur_bsz

            # 6.3 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            new_cur_to_prev_index_map = []
            for i in range(prev_bsz):
                prev_i = cur_to_prev_index_map[i]
                if seek[prev_i] >= max_frames[prev_i]:
                    cut_index = i + (cur_bsz - prev_bsz)
                    cur_bsz -= 1
                    input_features = torch.cat([input_features[:cut_index], input_features[cut_index + 1 :]], dim=0)
                else:
                    # cut out index that goes away
                    new_cur_to_prev_index_map.append(prev_i)

            # 6.4  Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            cur_to_prev_index_map = new_cur_to_prev_index_map
            time_offset = seek * time_precision / input_stride
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.5 Make sure that all inputs are padded to the same input length
            segment_input = []
            for i in range(cur_bsz):
                prev_i = cur_to_prev_index_map[i]
                segment_input_slice = input_features[
                    i : i + 1, :, seek[prev_i] : seek[prev_i] + seek_num_frames[prev_i]
                ]

                if segment_input_slice.shape[-1] < num_segment_frames:
                    # pad to 3000 if necessary
                    segment_input_slice = F.pad(
                        segment_input_slice, pad=(0, num_segment_frames - segment_input_slice.shape[-1])
                    )

                segment_input.append(segment_input_slice)

            segment_input = torch.cat(segment_input, dim=0)

            # 6.6 Batch generate current chunk
            seek_outputs = super().generate(
                segment_input,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

            if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                num_frames = getattr(generation_config, "num_frames", None)
                seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                    seek_outputs, generation_config.alignment_heads, num_frames=num_frames
                )

            if return_dict_in_generate:
                seek_sequences = seek_outputs["sequences"]
                seek_outputs = [
                    {k: v[i] for k, v in seek_outputs.items()}
                    for i in range(next(iter(seek_outputs.values())).size(0))
                ]
            else:
                seek_sequences = seek_outputs

            # 6.7 Loop over each decoded audio individually as each decoding can be of a different length
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = cur_to_prev_index_map[i]

                # make sure we cut a predicted EOS token if we are not finished with the generation yet
                is_not_final = (seek[prev_i] + num_segment_frames) < max_frames[prev_i]
                if is_not_final and seek_sequence[-1] == self.generation_config.eos_token_id:
                    seek_sequence = seek_sequence[:-1]

                # remove all padding tokens
                if seek_sequence[-1] == self.generation_config.pad_token_id:
                    num_paddings = (seek_sequence == self.generation_config.pad_token_id).sum()
                    seek_sequence = seek_sequence[:-num_paddings]

                segments, segment_offset = self._retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    cur_bsz=cur_bsz,
                    time_precision=time_precision,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                )

                current_segments[prev_i] += segments
                seek[prev_i] += segment_offset
        
        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        sequences = []
        max_total_length = 0
        for current_segment_list in current_segments:
            sequences.append(torch.cat([d["tokens"] for d in current_segment_list], dim=-1))
            max_total_length = max(max_total_length, len(sequences[-1]))

        for i in range(batch_size):
            sequences[i] = F.pad(
                sequences[i], pad=(0, max_total_length - len(sequences[i])), value=self.generation_config.pad_token_id
            )

        sequences = torch.stack(sequences, dim=0)

        # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
        if return_segments:
            return {"sequences": sequences, "segments": current_segments}

        return sequences

    @staticmethod
    def _retrieve_segment(
        seek_sequence,
        seek_outputs,
        time_offset,
        timestamp_begin,
        seek_num_frames,
        cur_bsz,
        time_precision,
        input_stride,
        prev_idx,
        idx,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == cur_bsz * [[False, True]]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))

            last_slice = 0
            # Add each segment to list of all segments
            for current_slice in slices:
                sliced_tokens = seek_sequence[last_slice + 1 : current_slice + 1]
                start_timestamp_pos = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_pos = sliced_tokens[-1].item() - timestamp_begin
                segments.append(
                    {
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + end_timestamp_pos * time_precision,
                        "tokens": sliced_tokens,
                        "result": seek_outputs[idx],
                    }
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                segment_offset = seek_num_frames[prev_idx]
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # here we throw away all predictions after the last predicted "end of segment"
                # since we are cutting right in the middle of an audio
                last_timestamp_pos = seek_sequence[last_slice].item() - timestamp_begin
                segment_offset = last_timestamp_pos * input_stride
        else:
            # If whisper does not predict any "end of segment" token, then
            # the whole decoding is considered a segment and we add it to the list of segments
            timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
            last_timestamp_pos = seek_num_frames[prev_idx]
            if timestamps.numel() > 0 and timestamps[-1].item() != timestamp_begin:
                # no consecutive timestamps but it has a timestamp; use the last one.
                last_timestamp_pos = timestamps[-1].item() - timestamp_begin

            segments = [
                {
                    "start": time_offset[prev_idx],
                    "end": time_offset[prev_idx] + last_timestamp_pos * time_precision,
                    "tokens": seek_sequence,
                    "result": seek_outputs[idx],
                }
            ]
            segment_offset = seek_num_frames[prev_idx]

        return segments, segment_offset

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": None,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def _extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02):
        """
        Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
        map each output token to a position in the input audio.

        Returns:
            tensor containing the timestamps in seconds for each predicted token
        """
        # Create a list with `decoder_layers` elements, each a tensor of shape
        # (batch size, attention_heads, output length, input length).
        cross_attentions = []
        for i in range(self.config.decoder_layers):
            cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))

        # Select specific cross-attention layers and heads. This is a tensor
        # of shape (batch size, num selected, output length, input length).
        weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
        weights = weights.permute([1, 0, 2, 3])

        # Normalize and smoothen the weights.
        std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
        weights = (weights - mean) / std
        weights = _median_filter(weights, self.config.median_filter_width)

        # Average the different cross-attention heads.
        matrix = weights.mean(dim=1)

        timestamps = torch.zeros_like(generate_outputs.sequences, dtype=torch.float32)

        # Perform dynamic time warping on each element of the batch.
        for batch_idx in range(timestamps.shape[0]):
            text_indices, time_indices = _dynamic_time_warping(-matrix[batch_idx].double().cpu().numpy())
            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
            jump_times = time_indices[jumps] * time_precision
            timestamps[batch_idx, 1:] = torch.tensor(jump_times)

        return timestamps


@add_start_docstrings(
    """
    Whisper Encoder Model with a sequence classification head on top (a linear layer over the pooled output) for tasks
    like SUPERB Keyword Spotting.
    """,
    WHISPER_ENCODER_INPUTS_DOCSTRING,
)
class WhisperForAudioClassification(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training. Only the projection layers and classification head will be updated.
        """
        self.encoder._freeze_parameters()

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.encoder.set_input_embeddings(value)

    @add_start_docstrings_to_model_forward(WHISPER_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, WhisperForAudioClassification
        >>> from datasets import load_dataset

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        >>> model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

        >>> ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
        >>> sample = next(iter(ds))

        >>> inputs = feature_extractor(
        ...     sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
        ... )
        >>> input_features = inputs.input_features

        >>> with torch.no_grad():
        ...     logits = model(input_features).logits

        >>> predicted_class_ids = torch.argmax(logits).item()
        >>> predicted_label = model.config.id2label[predicted_class_ids]
        >>> predicted_label
        'Afrikaans'
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.config.use_weighted_layer_sum:
            hidden_states = torch.stack(encoder_outputs, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs[0]

        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
