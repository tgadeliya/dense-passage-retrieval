from dataclasses import dataclass
import typing as T
from collections import namedtuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QAPair = namedtuple("QAPair", ["question", "answer"])
QAPairBatch = namedtuple("QAPair", ["question_list", "pair_list"])


class PolEvalQA(Dataset):
    def __init__(self):
        answer_path = (
            "/Users/tsimur.hadeliya/code/repos/DRP/data/PolEvalQA/dev-0/expected.tsv"
        )
        question_path = (
            "/Users/tsimur.hadeliya/code/repos/DRP/data/PolEvalQA/dev-0/in.tsv"
        )
        self.question = self.read_files(question_path)
        self.answer = self.read_files(answer_path)
        assert len(self.question) == len(self.answer)
        self.num_pairs = len(self.question)

    @staticmethod
    def read_files(path: str):
        return open(path).read().splitlines()

    def __getitem__(self, idx: int):
        return {
            "question": self.question[idx],
            "answer": self.answer[idx]
        }

    def __len__(self):
        return self.num_pairs


class ValidationDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, top_k) -> None:
        self.base_dataset: PolEvalQA = dataset
        self.top_k = top_k # Finally sample with top_k -1
        self.base_dataset_len = len(self.base_dataset)

    def __len__(self):
        return self.base_dataset_len

    def __getitem__(self, idx:int):
        qa = self.base_dataset.__getitem__(idx)
        sampled_negative_answers = self.sample_negative_answers(idx)
        return {
            "question": qa["question"],
            "answer": qa["answer"],
            "answer_negatives": sampled_negative_answers  # Return top_k - 1 negatives (one for positive)
        }

    def sample_negative_answers(self, pos_ans_idx):
        sampled_idx = self.sample_negative_answers_idx(pos_ans_idx)
        sampled_neg_answers = []
        for idx in sampled_idx:
            sampled_answer = self.base_dataset.answer[idx]
            sampled_neg_answers.append(sampled_answer)

        return sampled_neg_answers

    def sample_negative_answers_idx(self, pos_ans_idx):
        sampled_idx = np.random.choice(self.base_dataset_len, self.top_k, replace=False)
        reserve_idx = sampled_idx[-1]
        sampled_idx = sampled_idx[:-1]
        pos_idx_in_sampled = np.flatnonzero(sampled_idx == pos_ans_idx)
        if len(pos_idx_in_sampled) > 0:  # == 1
            sampled_idx[pos_idx_in_sampled[0]] = reserve_idx

        return sampled_idx

    @staticmethod
    def collate(l: T.List):
        n_d = {}
        for k,v in l[0].items():
            n_d[k] = [o[k] for o in l]
        return n_d

class QADataModule(pl.LightningDataModule):
    def __init__(self):
        self.top_k = 6
        super().__init__()
        self.dataset = PolEvalQA()
        self.train_dataset: Dataset

    def prepare_data(self, *args, **kwargs):
        # TODO: Add neccesary preparation
        pass

    def setup(self, stage: T.Optional[str] = None):
        if stage == "fit":
            self.train_dataset = self.dataset

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(dataset=PolEvalQA(), batch_size=4, num_workers=0)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(dataset=ValidationDatasetWrapper(self.dataset, self.top_k), batch_size=4, num_workers=0, collate_fn=ValidationDatasetWrapper.collate)
