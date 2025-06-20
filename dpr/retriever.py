import typing as T
from dataclasses import dataclass
from math import ceil
import logging

import faiss
from faiss import IndexFlatL2
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import Module

from dpr import Encoder
from dpr.utils import cp_from_gcs_to_local

logger = logging.getLogger()


@dataclass
class RetrieverInput:
    questions: T.List[str]
    answers: T.List[str]
    # TODO: Add len() assert


class NeuralRetriever(Module):
    def __init__(
        self,
        question_encoder: Encoder,
        answer_encoder: Encoder,
        top_k: int = 20,
    ):
        super().__init__()
        self.source_passage_index: T.Optional[faiss.IndexFlatL2] = None

        self.question_encoder = question_encoder
        self.answer_encoder = answer_encoder

    def forward(self, batch: RetrieverInput):
        questions = self.question_encoder(batch.questions)  # [bs, hidden_size]
        # TODO: return retrieved passages
        return 1


    def add_source_passage_for_inference(
        self, source_passages: T.Union[T.List[str], str]
    ):
        # TODO: Add tests
        if type(source_passages) is str:
            logger.info(
                "get source_passages as type str. Assuming, this is the link to already builded index."
            )
            source_passages = (
                cp_from_gcs_to_local(source_passages)
                if source_passages.startswith("gs://")
                else source_passages
            )
            builded_index = faiss.read_index(source_passages)
        elif isinstance(source_passages, IndexFlatL2):
            logger.info("Using already prepared index object")
            builded_index = source_passages
        else:
            logger.info("Building source index from given passages")
            builded_index = self.build_faiss_index_given_encoder_and_passages(
                source_passages,
                self.answer_encoder,
            )
        self.source_passage_index = builded_index

    @staticmethod
    def build_faiss_index_given_encoder_and_passages(
        passages, encoder, dim=768, index_type=faiss.IndexFlatL2
    ):
        assert (
            encoder.config.hidden_size == dim
        ), f"Encoder representation dimensions {encoder.config.hidden_size} and dim={dim} are not equal"
        bs = 512
        num_passages = len(passages)

        encoder.base_encoder.eval()
        encoded_passages_matrix = np.empty(
            shape=(num_passages, encoder.config.hidden_size)
        ).astype("float32")
        for i in tqdm(range(ceil(num_passages / bs)), desc="Encoding passages"):
            lidx, ridx = i * bs, (i + 1) * bs
            batch = passages[lidx:ridx]
            encoded_passages_matrix[lidx:ridx] = (
                encoder(batch).detach().numpy()
            )  # TODO: Check output == float32
        index = index_type(dim)
        index.add(encoded_passages_matrix)
        return index
