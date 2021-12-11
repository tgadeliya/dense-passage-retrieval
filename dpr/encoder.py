import typing as T

import torch
from torch.nn import Module
from transformers import PreTrainedTokenizer, PreTrainedModel, PretrainedConfig


class Encoder(Module):
    """
    Simple wrapper, combining HF model and corresponding tokenizer
    """
    REPRESENTATION_TYPES: T.List[str] = ["CLS", "MEAN", "BASE"]

    def __init__(
        self,
        base_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        representation_type: str,
    ):
        super().__init__()
        assert (
            representation_type in self.REPRESENTATION_TYPES
        ), f"Choose one of the representation_type from {self.REPRESENTATION_TYPES}"
        self.base_encoder: PreTrainedModel = base_encoder
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.config: PretrainedConfig = self.base_encoder.config
        self.representation_type: str = representation_type

    def forward(self, input_text: T.List[str]):
        encoded_input = self.encode(input_text)
        output = self.base_encoder(**encoded_input)
        if self.representation_type == "BASE":
            return output
        last_hidden_state, cls_token = output.last_hidden_state, output.pooler_output
        if self.representation_type == "CLS":
            return output.pooler_output
        elif self.representation_type == "MEAN":
            return torch.mean(last_hidden_state, dim=1)

    def encode(self, text: T.Union[T.List[str], str]):
        # TODO: Add dynamic padding
        return self.tokenizer(
            text,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            # TODO: Add all params
        )

    def decode(self, encoded_text):
        pass
