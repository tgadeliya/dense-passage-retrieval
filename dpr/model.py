import typing as T
import logging

from faiss import IndexFlatL2
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import pytorch_lightning as pl
from transformers import PreTrainedModel, AutoModel, AutoTokenizer
from dpr import Encoder, NeuralRetriever

logger = logging.getLogger()


class DensePassageRetrieval(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hparams: pl.utilities.parsing.AttributeDict
        self.save_hyperparameters()
        model_alias = "allegro/herbert-base-cased"  # TODO: Change to English encoder
        herbert = AutoModel.from_pretrained(model_alias)
        encoder = Encoder(
            base_encoder=herbert,
            tokenizer=AutoTokenizer.from_pretrained(model_alias),
            representation_type="CLS"
        )
        self.retriever = NeuralRetriever(
            encoder, encoder
        )
        self.passage_index: IndexFlatL2 = self.retriever.source_passage_index

        # TODO: Add dropout

    def training_step(self, batch: T.Dict[str, T.Any], batch_idx):
        questions_encoded = self.retriever.question_encoder(
            batch["question"]
        )
        answers_encoded = self.retriever.answer_encoder(
            batch["answer"]
        )
        print(questions_encoded, answers_encoded)
        # print(questions_encoded.size(), answers_encoded.size())
        loss = self.calculate_nll_loss(questions_encoded, answers_encoded)
        # return {"loss": loss}

    @staticmethod
    def calculate_nll_loss(question_matrix, answer_matrix):
        """
        Loss for in-batch negatives
        """
        sim_matrix = torch.matmul(question_matrix, answer_matrix.T)
        sim_matrix = torch.exp(sim_matrix)
        nll_loss = -torch.log(sim_matrix.diag() / sim_matrix.sum(dim=1))
        return nll_loss.mean()

    def configure_optimizers(self):
        total_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        optimizer = AdamW(self.retriever.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=400,
            num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "interval": "step"
        }

    def validation_step(self, batch, batch_idx):
        questions, answers = batch["question"], batch["answer"]
        q_enc = self.retriever.question_encoder(questions)
        a_enc = self.retriever.answer_encoder(answers)

        # calculate nll loss
        nll_val_loss = self.calculate_nll_loss(q_enc, a_enc)

        # calculate rank accuracy
        k_a_enc = self.encode_top_k_answers(a_enc, batch["answer_negatives"])
        rank_acc = self.calculate_rank_accuracy(q_enc, k_a_enc)
        return {"nll_val_loss": nll_val_loss, **rank_acc}

    def encode_top_k_answers(self, a_enc, a_neg: T.List[T.List[str]]):
        bsz, d = a_enc.size()
        k = self.hparams.top_k
        a_neg_enc = torch.empty((bsz, d, k), dtype=a_enc.dtype, requires_grad=False)  # TODO: Choose proper dtype
        a_neg_enc[:, :, 0] = a_enc  # golden answers

        for i in range(len(a_neg)):  # adding negative answers to measure rank accurac
            i_neg_enc = self.retriever.answer_encoder(a_neg[i])
            a_neg_enc[i, :, 1:] = i_neg_enc.permute([1, 0])  # TODO: Find out why we need to do permute
        return a_neg_enc

    @staticmethod
    def calculate_rank_accuracy(q_enc, a_neg_enc) -> T.Dict[str, torch.Tensor]:
        # Every golden passage has index 0
        q_enc = torch.unsqueeze(q_enc, 1)  # (N x d) -> (N x 1 x d)
        scores = torch.matmul(q_enc, a_neg_enc)  # (N x 1 x d)  X (N x d x k) = (N x 1 x k)
        scores = torch.squeeze(scores, 1)

        ranks = torch.argsort(scores, dim=1)
        pos_ans_rank = ranks[:, 0]  # 1 x N

        max_k = a_neg_enc.size()[-1]
        top_k_list = [1, 5, 20, 100]

        rank_acc = {}
        for k in filter(lambda x: x <= max_k, top_k_list + [max_k]):
            top_k_preds = (pos_ans_rank <= k - 1).to(torch.float32)  # TODO: Whether flatten is a need
            rank_acc[f"top_{k}_accuracy"] = top_k_preds
        return rank_acc

    def validation_epoch_end(self, outputs: T.List[T.Any]) -> None:
        print(outputs)
        # TODO: Add result merging from outputs to calculate proper mean
