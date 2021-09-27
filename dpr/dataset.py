from dataclasses import dataclass
import typing as T
import faiss
import torch
import tqdm
import torch
from transformers import PreTrainedTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DPRInput:
    questions: T.List
    passages: T.List


def build_faiss_index_given_encedor_and_passages(
    passages, encoder, dim=768, index_type=faiss.IndexFlatL2
):
    assert (
        encoder.config.hidden_size == dim
    ), f"Encoder representation dimensions and dim={dim}"
    index = index_type()


def convert_passages_into_representation_matrix(
    passages, encoder, tokenizer: PreTrainedTokenizer
):
    num_pasages, repr_dim = len(passages), encoder.config.hidden_size
    padding_len = tokenizer.model_max_length

    representation_matrix = torch.empty(
        (num_pasages, repr_dim), dtype=torch.float32, device=DEVICE
    )
    tokenized_input_ids = torch.empty((num_pasages, padding_len))

    for i in tqdm(range(len(passages)), desc="Tokenize and perform forward pass for every passage"):
        tokenized_input_ids[i, :] = tokenizer(
            passages[i], padding=True, return_length=512, return_tensors="pt"
        )["input_ids"]
        representation_matrix[i, :] = encoder(input_ids=tokenized_input_ids[i])

    return