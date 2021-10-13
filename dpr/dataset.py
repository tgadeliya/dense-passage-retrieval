from dataclasses import dataclass
import typing as T
from collections import namedtuple

import torch
from torch.utils.data import Dataset

QAPair = namedtuple("QAPair", ["question", "pair"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DPRInput:
    questions: T.List
    passages: T.List


@dataclass
class DPRRetrieverInput:
    questions: T.List[str]
    answers: T.List[str]


@dataclass
class DPRReaderInput:
    pass


@dataclass
class DPRReaderInput:
    pass

class PolEvalQA(Dataset):
    def __init__(self):
        answer_path = "/Users/tsimur.hadeliya/code/repos/DRP/data/PolEvalQA/dev-0/expected.tsv"
        question_path = "/Users/tsimur.hadeliya/code/repos/DRP/data/PolEvalQA/dev-0/in.tsv"
        self.question = self.read_files(question_path)
        self.answer = self.read_files(answer_path)
        assert len(self.question) == len(self.answer)
        self.len = len(self.question)

    @staticmethod
    def read_files(path):
        return open(path).read().splitlines()

    def __getitem__(self, idx):
        return QAPair(
            question=self.question[idx],
            answer=self.answer[idx]
        )