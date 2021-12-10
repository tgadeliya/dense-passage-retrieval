from dataset import PolEvalQA, QAPair

from torch.utils.data import DataLoader
import pytest


def test_polevalqa_output():
    dataset = PolEvalQA()
    assert type(dataset[0]) == QAPair


@pytest.mark.parametrize("batch_size", [1, 2, 10])
def test_polevalqa_automatic_batching(batch_size):
    dataset = PolEvalQA()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    batch = next(iter(dataloader))
    # if "passed", assumed that automatic collate_fn works properly
    assert len(batch.question) == len(batch.answer) == batch_size
