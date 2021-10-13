import typing as T
from math import ceil
from dataclasses import dataclass
import logging
import os

from tqdm import tqdm
import numpy as np
import torch
import faiss
from transformers import PreTrainedModel, PreTrainedTokenizer

import gcsfs

def cp_from_gcs_to_local(gcs_path, output_dir = "."):
    local_path = os.path.join(output_dir, gcs_path.split("/")[-1])
    fs = gcsfs.GCSFileSystem()
    fs.get_file(rpath=gcs_path, lpath=local_path)
    return local_path

logger = logging.getLogger()

REPRESENTATION_TYPES = ["CLS", "MEAN", "BASE"]


@dataclass
class DPRRetrieverInput:
    questions: T.List[str]
    answers: T.List[str]



# class DensePassageRetrieval(PreTrainedModel):
#     def __init__(
#         self,
#         passages,
#     ):
#         model_alias = "allegro/herbert-base-cased"
#         herbert = AutoModel.from_pretrained(model_alias)
#         self.retriever = DPRRetriever(
#             question_encoder=Encoder(), passage_encoder=Encoder(), top_k=10
#         )
#         self.reader = DPRReader()
#         self.indexed_passages = self.build_index_given_model_passages(
#             encoder=self.retriever.question_encoder,
#             passages=passages,
#         )
#
#     def forward(self, batch: DPRInput):
#         retriever_out = self.retriever(batch)
#         out = self.reader(retriever_out)
#         return out

class Encoder(torch.nn.Module):
    def __init__(
        self,
        base_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        representation_type: str,
    ):
        super().__init__()
        assert (
            representation_type in REPRESENTATION_TYPES
        ), f"Choose one of the representation_type from {REPRESENTATION_TYPES}"
        self.base_encoder = base_encoder
        self.tokenizer = tokenizer
        self.config = self.base_encoder.config
        self.representation_type = representation_type

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

    def encode(self, text: T.List):
        return self.tokenizer(
            text,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
            # TODO: Add all params
        )


class DPRRetriever(PreTrainedModel):
    def __init__(
        self,
        question_encoder: Encoder,
        answer_encoder: Encoder,
        top_k: int = 20,
    ):
        self.source_passage_index: T.Optional[faiss.IndexFlatL2] = None

        self.question_encoder = question_encoder
        self.answer_encoder = answer_encoder
        self.top_k = top_k

    def forward(self, batch: DPRRetrieverInput):
        questions = self.question_encoder(batch.questions)  # [bs, hidden_size]
        answers = self.answer_encoder(batch.answers)  # [bs, hidden_size]
        loss = self.calculate_loss(questions, answers)
        return {
            "loss": loss
        }

    @staticmethod
    def calculate_loss(question_matrix, answer_matrix):
        """
        Loss for in-batch negatives
        """
        sim_matrix = torch.matmul(question_matrix, answer_matrix.T)
        sim_matrix = np.exp(sim_matrix)
        nll_loss = - torch.log(sim_matrix.diag() / sim_matrix.sum(dim=1))
        return nll_loss

    def add_source_passage_for_inference(self, source_passages: T.Union[T.List[str], str]):
        # TODO: Add tests
        if type(source_passages) is str:
            logger.info("get source_passages as type str. Assuming, this is the link to already build index.")
            source_passages = cp_from_gcs_to_local(source_passages) if source_passages.startswith("gs://") else source_passages
            builded_index = faiss.read_index(source_passages)
        else:
            logger.info("Building source index from given passages")
            prepared_source_index = self.build_faiss_index_given_encoder_and_passages(
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
        encoded_passages_matrix = np.empty(shape=(num_passages, encoder.config.hidden_size)).astype("float32")
        for i in tqdm(range(ceil(num_passages / bs)), desc="Encoding passages"):
            lidx, ridx = i * bs, (i + 1) * bs
            batch = passages[lidx:ridx]
            encoded_passages_matrix[lidx:ridx] = encoder(batch).detach().numpy()  # TODO: Check output == float32
        index = index_type(dim)
        index.add(encoded_passages_matrix)
        return index


class DPRReader:
    def __init__(
        self,
        span_prediction_model: Encoder,
        num_sampled_passages: int = 24,
    ):
        self.num_sampled_passages = num_sampled_passages
        self.encoder: Encoder = span_prediction_model
        # output start and end token
        self.span_classifier = torch.nn.Linear(
            self.span_prediction.config.hidden_size, 2)
        # output selected passage
        self.selected_classifier = torch.nn.Linear(
            self.span_prediction.config.hidden_size, 1
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
        selected_passage_token = self.softmax(
            self.selected_classifier(input[:, 0, :])
        )

        return {
            "start_token": start_token,
            "end_token": end_token,
            "selected_passage_token": selected_passage_token
        }
