import typing as T
import logging

import pytorch_lightning as pl
from transformers import PreTrainedModel, AutoModel, AutoTokenizer

logger = logging.getLogger()

from dpr import Encoder, NeuralRetriever, DPRReader


# class DensePassageRetrieval(PreTrainedModel):
#     def __init__(
#         self,
#         passages,
#     ):
#         model_alias = "allegro/herbert-base-cased"
#         herbert = AutoModel.from_pretrained(model_alias)
#         self.retriever = NeuralRetriever(
#             question_encoder=Encoder(), passage_encoder=Encoder(), top_k=10
#         )
#         self.reader = DPRReader()
#         self.indexed_passages = self.build_index_given_model_passages(
#             encoder=self.retriever.question_encoder,
#             passages=passages,
#         )
#
#     def forward(self, batch):
#         retriever_out = self.retriever(batch)
#         out = self.reader(retriever_out)
#         return out


class DensePassageRetrieval(pl.LightningModule):
    def __init__(self):
        super(DensePassageRetrieval, self).__init__()
        model_alias = "allegro/herbert-base-cased"
        herbert = AutoModel.from_pretrained(model_alias)
        encoder = Encoder(
            base_encoder=herbert,
            tokenizer=AutoTokenizer.from_pretrained(model_alias),
            representation_type="CLS"
        )
        self.retriever = NeuralRetriever(
            encoder, encoder
        )
        self.reader = DPRReader(
            encoder
        )

    def training_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def validation_epoch_end(self, outputs: T.List[T.Any]) -> None:
        pass
