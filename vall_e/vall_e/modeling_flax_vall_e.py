from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from vall_e.modeling_flax_utils import ACT2FN
from .configuration_vall_e import VALLEConfig


def create_sinusoidal_positions(
    max_len: int, dim: int, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=dtype) / dim))
    sinusoid_inp = jnp.einsum("i,j->ij", jnp.arange(max_len, dtype=dtype), inv_freq)
    sin, cos = jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)
    pos_emb = jnp.concatenate([sin, cos], axis=-1)
    return pos_emb


class FlaxVALLEMLP(nn.Module):
    config: VALLEConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.fc_in = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, self.dtype
            ),
        )
        self.fc_out = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, self.dtype
            ),
        )

        self.act = ACT2FN[self.config.hidden_act]
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class FlaxVALLEAttention(nn.Module):
    config: VALLEConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.query_key_value = nn.Dense(
            self.config.hidden_size * 3,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, self.dtype
            ),
        )
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, self.dtype
            ),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(
            hidden_states.shape[:2] + (self.config.hidden_size,)
        )

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):

        qkv = self.query_key_value(hidden_states)
        query, key, value = jnp.split(qkv, 3, axis=-1)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        if attention_mask is not None:
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
                    self.dtype
                ),
            )
        else:
            attention_bias = None

        attn_weights = nn.dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout_prob,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.dense(attn_output)
        attn_output = self.dropout(attn_output, deterministic=deterministic)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxVALLELayer(nn.Module):
    config: VALLEConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.attention = FlaxVALLEAttention(self.config, dtype=self.dtype)
        self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.mlp = FlaxVALLEMLP(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic=True,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        atten_outputs = self.attention(hidden_states, deterministic=deterministic)
        atten_output = atten_outputs[0]
        hidden_states = residual + atten_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(
            hidden_states, deterministic=deterministic
        )
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (atten_outputs[1],)
        return outputs


class FlaxVALLELayerCollection(nn.Module):
    config: VALLEConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxVALLELayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class MultiEmbedding(nn.Module):
    config: VALLEConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embeddings = [
            nn.Embed(
                self.config.codec_size,
                self.config.hidden_size,
                embedding_init=jax.nn.initializers.normal(
                    self.config.initializer_range, self.dtype
                ),
                dtype=self.dtype,
            )
            for _ in range(self.config.num_embed_levels)
        ]

    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        embeddings = []
        for i in range(input_ids.shape[1]):
            embeddings.append(self.embeddings[i](input_ids[:, i, :]))
        embeddings = jnp.stack(embeddings, axis=1)
        return jnp.sum(embeddings, axis=1)


class FlaxVALLEEmbeddings(nn.Module):
    config: VALLEConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.phoneme_embeddings = nn.Embed(
            self.config.phoneme_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                self.config.initializer_range, self.dtype
            ),
            dtype=self.dtype,
        )
        if self.config.num_embed_levels > 1:
            self.acoustic_embeddings = MultiEmbedding(self.config, dtype=self.dtype)
        else:
            self.acoustic_embeddings = nn.Embed(
                self.config.codec_size,
                self.config.hidden_size,
                embedding_init=jax.nn.initializers.normal(
                    self.config.initializer_range, self.dtype
                ),
                dtype=self.dtype,
            )

        self.separator = self.param(
            "separator",
            jax.nn.initializers.normal(self.config.initializer_range),
            (1, 1, self.config.hidden_size),
            self.dtype,
        )

    def __call__(
        self,
        phoneme_ids: jnp.ndarray,
        prompt_ids: jnp.ndarray = None,
        speech_ids: jnp.ndarray = None,
        phoneme_attention_mask: jnp.ndarray = None,
        prompt_attention_mask: jnp.ndarray = None,
        speech_attention_mask: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        bs = phoneme_ids.shape[0]
        phoneme_embeddings = self.phoneme_embeddings(phoneme_ids)
        phoneme_embeddings += create_sinusoidal_positions(
            phoneme_embeddings.shape[1], self.config.hidden_size, dtype=self.dtype
        )
        if phoneme_attention_mask is None:
            phoneme_attention_mask = jnp.ones_like(phoneme_ids)

        embeddings = phoneme_embeddings
        attention_mask = phoneme_attention_mask
        if prompt_ids is not None:
            prompt_embeddings = self.acoustic_embeddings(prompt_ids)
            prompt_embeddings += create_sinusoidal_positions(
                prompt_embeddings.shape[1], self.config.hidden_size, dtype=self.dtype
            )

            if prompt_ids.ndim == 3:
                prompt_attention_mask = jnp.ones_like(prompt_ids[:, 0, :])
            else:
                prompt_attention_mask = jnp.ones_like(prompt_ids)

            embeddings = jnp.concatenate(
                [
                    embeddings,
                    jnp.repeat(self.separator, bs, axis=0),
                    prompt_embeddings,
                ],
                axis=1,
            )
            attention_mask = jnp.concatenate(
                [
                    attention_mask,
                    jnp.ones((bs, 1), dtype=self.dtype),
                    prompt_attention_mask,
                ],
                axis=1,
            )

        if speech_ids is not None:
            speech_embeddings = self.acoustic_embeddings(speech_ids)
            speech_embeddings += create_sinusoidal_positions(
                speech_embeddings.shape[1], self.config.hidden_size, dtype=self.dtype
            )

            if speech_ids.ndim == 3:
                speech_attention_mask = jnp.ones_like(speech_ids[:, 0, :])
            else:
                speech_attention_mask = jnp.ones_like(speech_ids)

            embeddings = jnp.concatenate(
                [
                    embeddings,
                    jnp.repeat(self.separator, bs, axis=0),
                    speech_embeddings,
                ],
                axis=1,
            )
            attention_mask = jnp.concatenate(
                [
                    attention_mask,
                    jnp.ones((bs, 1), dtype=self.dtype),
                    speech_attention_mask,
                ],
                axis=1,
            )

        return embeddings, attention_mask


class VALLEModule(nn.Module):
    config: VALLEConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embeddings = FlaxVALLEEmbeddings(self.config, self.dtype)
        self.encoder = FlaxVALLELayerCollection(self.config, dtype=self.dtype)

        self.lm_head = nn.Dense(
            self.config.codec_size,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range, self.dtype
            ),
            dtype=self.dtype,
        )

    def __call__(
        self,
        phoneme_ids,
        prompt_ids=None,
        speech_ids=None,
        phoneme_attention_mask=None,
        prompt_attention_mask=None,
        speech_attention_mask=None,
        deterministic=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        hidden_states, attention_mask = self.embeddings(
            phoneme_ids,
            prompt_ids,
            speech_ids,
            phoneme_attention_mask,
            prompt_attention_mask,
            speech_attention_mask,
        )

        outputs = self.encoder(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxVALLEPreTrainedModel(FlaxPreTrainedModel):
    config_class = VALLEConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: VALLEConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None
    ) -> FrozenDict:
        phoneme_ids = jnp.zeros(input_shape, dtype="i4")
        if self.config.num_embed_levels > 1:
            prompt_ids = jnp.zeros(
                (input_shape[0], self.config.num_embed_levels, input_shape[1]),
                dtype="i4",
            )
            speech_ids = jnp.zeros(
                (input_shape[0], self.config.num_embed_levels, input_shape[1]),
                dtype="i4",
            )

        else:
            prompt_ids = jnp.zeros(input_shape, dtype="i4")
            speech_ids = jnp.zeros(input_shape, dtype="i4")

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(
            rngs,
            phoneme_ids,
            prompt_ids,
            speech_ids,
            return_dict=False,
        )

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        phoneme_ids,
        prompt_ids=None,
        speech_ids=None,
        phoneme_attention_mask=None,
        prompt_attention_mask=None,
        speech_attention_mask=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            phoneme_ids=phoneme_ids,
            prompt_ids=prompt_ids,
            speech_ids=speech_ids,
            phoneme_attention_mask=phoneme_attention_mask,
            prompt_attention_mask=prompt_attention_mask,
            speech_attention_mask=speech_attention_mask,
            deterministic=not train,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
        )


class FlaxVALLE(FlaxVALLEPreTrainedModel):
    module_class = VALLEModule
