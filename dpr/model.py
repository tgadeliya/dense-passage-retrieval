import logging

from transformers import PreTrainedModel, AutoModel, AutoTokenizer

logger = logging.getLogger()

from dpr import Encoder, NeuralRetriever, DPRReader

class DensePassageRetrieval(PreTrainedModel):
    def __init__(
        self,
        passages,
    ):
        model_alias = "allegro/herbert-base-cased"
        herbert = AutoModel.from_pretrained(model_alias)
        self.retriever = NeuralRetriever(
            question_encoder=Encoder(), passage_encoder=Encoder(), top_k=10
        )
        self.reader = DPRReader()
        self.indexed_passages = self.build_index_given_model_passages(
            encoder=self.retriever.question_encoder,
            passages=passages,
        )

    def forward(self, batch):
        retriever_out = self.retriever(batch)
        out = self.reader(retriever_out)
        return out



