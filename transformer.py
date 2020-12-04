import torch
from torch import nn
from torch.nn.functional import relu
import torch.nn.functional as F
from math import sqrt
from torch.nn import LayerNorm as FusedLayerNorm
import math

NUM_SEQUENCES = 4
MAX_SEQUENCE_LENGTH = 10

class KroneckerModel(nn.Module):
    def __init__(self, num_tokens, input_embedding_dim, vector_dim, num_heads, num_layers, hidden_multiplier=4, dropout_prob=0.1):
        super().__init__()
        self.sequence_embedding = nn.Embedding(NUM_SEQUENCES, input_embedding_dim)
        self.term_embedding = nn.Embedding(MAX_SEQUENCE_LENGTH, input_embedding_dim)
        self.input_embedding = nn.Embedding(num_tokens, input_embedding_dim)
        self.num_layers = num_layers
        self.input_linear = nn.Linear(input_embedding_dim, vector_dim)
        self.transformer_layers = nn.ModuleList([EncoderLayer(vector_dim, hidden_multiplier,
                                                                  num_heads, dropout_prob)])
        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(vector_dim, 1)

    def forward(self, raw_input, sequence_input, term_input, attention_mask):
        if attention_mask is not None:
            attention_mask = torch.abs(attention_mask - 1)  # Invert the mask
            attention_mask[attention_mask == 1] = -math.inf
            attention_mask = torch.unsqueeze(torch.unsqueeze(attention_mask, 1), 1)  # Insert two dummy dimensions
        tensor = self.input_embedding(raw_input) + self.sequence_embedding(sequence_input) + self.term_embedding(term_input)
        tensor = gelu(self.input_linear(tensor))
        for _ in range(self.num_layers):
            tensor = self.transformer_layers[0](tensor, attention_mask)
        out = self.out(tensor)
        return out

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def initialize_layers(layers, weight_std=0.02):
    # The stdev comes from the BERT repo - they use truncated normal, but I can't imagine it matters that much
    for layer in layers:
        if hasattr(layer, 'weight'):
            nn.init.normal_(layer.weight, mean=0, std=weight_std)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, max_decode_length=None):
        super().__init__()
        self.query_linear = nn.Linear(dim, dim, bias=False)
        self.key_linear = nn.Linear(dim, dim, bias=False)
        self.value_linear = nn.Linear(dim, dim, bias=False)
        self.output_linear = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([self.query_linear, self.key_linear,
                                     self.value_linear, self.output_linear, self.dropout])
        initialize_layers(self.layers)
        self.dim = dim
        self.num_heads = num_heads

    def _multi_head_split(self, queries, keys, values):
        num_heads = self.num_heads
        batch_size, query_timesteps, dim = queries.shape
        key_value_timesteps = keys.shape[1]
        queries = queries.view(batch_size, query_timesteps, num_heads, dim // num_heads).permute([0, 2, 1, 3])
        keys = keys.view(batch_size, key_value_timesteps, num_heads, dim // num_heads).permute([0, 2, 3, 1])
        values = values.view(batch_size, key_value_timesteps, num_heads, dim // num_heads).permute([0, 2, 1, 3])
        return queries, keys, values

    def forward(self, tensor_in, attention_mask):
        batch_size, timesteps, dim = tensor_in.size()
        queries = self.query_linear(tensor_in)
        keys = self.key_linear(tensor_in)
        values = self.value_linear(tensor_in)
        queries, keys, values = self._multi_head_split(queries, keys, values)
        weights = torch.matmul(queries, keys)
        if attention_mask is not None:
            weights += attention_mask  
        weights = self.dropout(F.softmax(weights / sqrt(dim), dim=-1))
        context = torch.matmul(weights, values)
        reformed_context = context.permute([0, 2, 1, 3]).contiguous().view(batch_size, timesteps, dim)
        output = self.output_linear(reformed_context)
        return output


class StepwiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_multiplier, dropout_prob):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        hidden_dim = input_dim * hidden_multiplier
        self.in_to_hid = nn.Linear(input_dim, hidden_dim)
        self.hid_to_out = nn.Linear(hidden_dim, input_dim)
        self.layers = nn.ModuleList([self.in_to_hid, self.hid_to_out, self.dropout])
        initialize_layers(self.layers)

    def forward(self, tensor_in):
        hidden = self.dropout(gelu(self.in_to_hid(tensor_in)))
        return self.hid_to_out(hidden)


class EncoderLayer(nn.Module):
    def __init__(self, vector_dim, hidden_multiplier, num_heads, dropout_prob):
        super().__init__()
        self.attention = Attention(vector_dim, num_heads)
        self.layer_norm_1 = FusedLayerNorm(vector_dim)
        self.feed_forward = StepwiseFeedForward(vector_dim, hidden_multiplier, dropout_prob)
        self.layer_norm_2 = FusedLayerNorm(vector_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.layers = nn.ModuleList([self.attention, self.feed_forward, self.dropout])

    def forward(self, tensor_in, attention_mask):
        attention_out = self.dropout(self.attention(tensor_in, attention_mask))
        feedforward_in = self.layer_norm_1(attention_out + tensor_in)
        feedforward_out = self.dropout(self.feed_forward(feedforward_in))
        return self.layer_norm_2(feedforward_out + feedforward_in)

