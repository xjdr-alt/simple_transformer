from typing import List, NamedTuple

import jax
import jax.numpy as jnp


class LayerWeights(NamedTuple):
  attn_norm: jax.Array
  ffn_norm: jax.Array
  w_q_dhk: jax.Array
  w_k_dhk: jax.Array
  w_v_dhk: jax.Array
  w_o_hkd: jax.Array
  w1: jax.Array
  w2: jax.Array
  w3: jax.Array


class XfmrWeights(NamedTuple):
  tok_embeddings: jax.Array
  layer_weights: List[LayerWeights]
  norm: jax.Array
  output: jax.Array


def norm(x, w, eps: float = 1e-6):
    return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))

def attention(input_bld, params):
    """
    B: batch size
    L: sequence length
    M: memory length 
    D: model dimension
    H: number of attention heads in a layer
    K: size of each attention key or value
    """
    normalized_bld = norm(input_bld, params.attn_norm)
    query_blhk = jnp.einsum('bld,dhk->blhk', normalized_bld, params.w_q_dhk)
    key_blhk = jnp.einsum('bld,dhk->blhk', normalized_bld, params.w_k_dhk)
    value_blhk = jnp.einsum('bld,dhk->blhk', normalized_bld, params.w_v_dhk)
    logits_bhlm = jnp.einsum('blhk,bmhk->bhlm', query_blhk, key_blhk)
    _, l, h, k = query_blhk.shape
    logits_bhlm = logits_bhlm / jnp.sqrt(k)
    mask = jnp.triu(jnp.ones((l, l)), k=1).astype(input_bld.dtype)
    logits_bhlm = logits_bhlm - jnp.inf * mask[None, None, :, :]
    weights_bhlm = jax.nn.softmax(logits_bhlm, axis=-1)
    wtd_values_blhk = jnp.einsum('blhk,bhlm->blhk', value_blhk, weights_bhlm)
    out_bld = jnp.einsum('blhk,hkd->bld', wtd_values_blhk, params.w_o_hkd)
    return out_bld

def ffn(x: jax.Array, w1: jax.Array, w2: jax.Array, w3: jax.Array) -> jax.Array:
  return jnp.dot(jax.nn.silu(jnp.dot(x, w1)) * jnp.dot(x, w3), w2)

def transformer(tokens: jax.Array, params: jax.Array) -> jax.Array:
  x = params.tok_embeddings[tokens]
  def scan_fn(h, layer_weights):
    h += attention(h, layer_weights)
    h += ffn(norm(h, layer_weights.ffn_norm), layer_weights.w1, layer_weights.w2, layer_weights.w3)
    return h, None
  h, _ = jax.lax.scan(scan_fn, x, params.layer_weights)
  h = norm(h, params.norm)
  logits = jnp.dot(h, params.output.T)
  return logits

if __name__ == '__main__':
  vocab_size = 32000
  dim = 4096
  hidden_dim = 14336
  n_layers = 1
  n_heads = 32
  head_dim = dim // n_heads

  layer_weights = LayerWeights(
      attn_norm=jnp.ones((n_layers, dim,)),
      ffn_norm=jnp.ones((n_layers, dim,)),
      w_q_dhk=jnp.zeros((n_layers, dim, n_heads, head_dim)),
      w_k_dhk=jnp.zeros((n_layers, dim, n_heads, head_dim)),
      w_v_dhk=jnp.zeros((n_layers, dim, n_heads, head_dim)),
      w_o_hkd=jnp.zeros((n_layers, n_heads, head_dim, dim)),
      w1=jnp.zeros((n_layers, dim, hidden_dim)),
      w2=jnp.zeros((n_layers, hidden_dim, dim)),
      w3=jnp.zeros((n_layers, dim, hidden_dim))
    )
  params = XfmrWeights(tok_embeddings=jnp.ones((vocab_size, dim)), layer_weights=layer_weights, norm=jnp.ones((dim,)), output=jnp.ones((vocab_size, dim)))
  tokens = jnp.array([[123,234,234,345,446]])
  out = transformer(tokens, params)
  print(f'{out.shape=}')