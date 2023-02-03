from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class VALLEConfig(PretrainedConfig):
    def __init__(
        self,
        phoneme_size: int = 84,
        codec_size: int = 1025,
        hidden_size: int = 1024,
        num_embed_levels: int = 1,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.phoneme_size = phoneme_size
        self.codec_size = codec_size
        self.hidden_size = hidden_size
        self.num_embed_levels = num_embed_levels
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
