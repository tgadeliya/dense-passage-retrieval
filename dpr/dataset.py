from dataclasses import dataclass
import typing as T
from collections import namedtuple

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
        return QAPair(question=self.question[idx], answer=self.answer[idx])

    def __len__(self):
        return self.num_pairs


class PolEvalQADataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = PolEvalQA()

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: T.Optional[str] = None):
        if stage == "fit":
            self.train_dataset = self.dataset

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(dataset=PolEvalQA())

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(dataset=PolEvalQA())
