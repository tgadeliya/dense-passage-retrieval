import typing as T
from dataclasses import dataclass

import torch
from torch.nn import Module


@dataclass
class DPRRetrieverInput:
    questions: T.List[str]
    answers: T.List[str]


class DPRRetriever(Module):
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
        # TODO: Add returning subclass of transformers ModelOutput
        return {"loss": loss}

    @staticmethod
    def calculate_loss(question_matrix, answer_matrix):
        """
        Loss for in-batch negatives
        """
        sim_matrix = torch.matmul(question_matrix, answer_matrix.T)
        sim_matrix = np.exp(sim_matrix)
        nll_loss = -torch.log(sim_matrix.diag() / sim_matrix.sum(dim=1))
        return nll_loss

    def add_source_passage_for_inference(
        self, source_passages: T.Union[T.List[str], str]
    ):
        # TODO: Add tests
        if type(source_passages) is str:
            logger.info(
                "get source_passages as type str. Assuming, this is the link to already build index."
            )
            source_passages = (
                cp_from_gcs_to_local(source_passages)
                if source_passages.startswith("gs://")
                else source_passages
            )
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
