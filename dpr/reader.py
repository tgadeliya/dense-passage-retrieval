import torch
from dpr import Encoder

from torch.nn import Module


class DPRReader(Module):
    def __init__(
        self,
        span_prediction_model: Encoder,
        num_sampled_passages: int = 24,
    ):
        super().__init__()
        self.num_sampled_passages = num_sampled_passages
        self.encoder: Encoder = span_prediction_model
        # output start and end token
        self.span_classifier = torch.nn.Linear(
            self.encoder.config.hidden_size, 2
        )
        # output selected passage
        self.selected_classifier = torch.nn.Linear(
            self.encoder.config.hidden_size, 1
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_passages):
        encoded = self.encode(input_passages)
        return self.predict_span(encoded)

    def predict_span(self, input_passages):
        # start, end token
        start_end_token = self.span_classifier(input_passages)
        start_token = self.softmax(start_end_token[:, 0])
        end_token = self.softmax(start_end_token[:, 1])

        # selected passage token from CLS token
        selected_passage_token = self.softmax(self.selected_classifier(input[:, 0, :]))

        return {
            "start_token": start_token,
            "end_token": end_token,
            "selected_passage_token": selected_passage_token,
        }
