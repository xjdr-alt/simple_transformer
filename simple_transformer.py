import jax
import jax.numpy as jnp

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
  normalized_bld = norm(input_bld, params.layernorm_params)
  query_blhk = jnp.einsum('bld, dhk -> blhk', normalized_bld, params.w_q_dhk)
  key_bmhk = jnp.einsum('bld, dhk -> blhk', normalized_bld, params.w_k_dhk)
  value_bmhk = jnp.einsum('bld, dhk -> blhk', normalized_bld, params.w_k_dhk)
  logits_bhlm = jnp.einsum('blhk, bmhk -> bhlm', query_blhk, key_bmhk)
  b, l, h, k = query_blhk.shape
  logits_bhlm = logits_bhlm / jnp.sqrt(k)
  mask = jnp.triu(np.ones((l, l)), k=1).astype(input_bld.dtype)
  logits_bhlm = logits_bhlm - jnp.inf * (1.0 - mask)
  weights_bhlm = jax.nn.softmax(logits_bhlm)
  wtd_values_blhk = jnp.einsum('bmhk, bhlm -> blhk', value_bmhk, weights_bhlm)
  out_bld = jnp.einsum('blhk, hkd -> bld', wtd_values_blhk, params.w_o_hkd)
  return out_bld

def ffn(x: jax.Array, w1: jax.Array, w2: jax.Array, w3: jax.Array) -> jax.Array:
  return jnp.dot(jax.nn.silu(jnp.dot(x, w1.T)) * jnp.dot(x, w3.T), w2.T)

def transformer(tokens: jax.Array, params: jax.Array) -> jax.Array:
  x = params.embedding[tokens]
  def scan_fn(h, layer_weights):
    h += attention(h, layer_weights)
    h += ffn(norm(h, layer_weights.ffn_norm), layer_weights.w1, layer_weights.w2, layer_weights.w3)
    return h
  h = jax.lax.scan(scan_fn, x, params.layer_weights)
  h = norm(h, params.norm)
  logits = jnp.dot(h, params.embedding.T)
  return logits

