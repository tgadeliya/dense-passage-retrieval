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


def cp_from_gcs_to_local(gcs_path, output_dir="."):
    local_path = os.path.join(output_dir, gcs_path.split("/")[-1])
    fs = gcsfs.GCSFileSystem()
    fs.get_file(rpath=gcs_path, lpath=local_path)
    return local_path


logger = logging.getLogger()

REPRESENTATION_TYPES = ["CLS", "MEAN", "BASE"]



class DensePassageRetrieval(PreTrainedModel):
    def __init__(
        self,
        passages,
    ):
        model_alias = "allegro/herbert-base-cased"
        herbert = AutoModel.from_pretrained(model_alias)
        self.retriever = DPRRetriever(
            question_encoder=Encoder(), passage_encoder=Encoder(), top_k=10
        )
        self.reader = DPRReader()
        self.indexed_passages = self.build_index_given_model_passages(
            encoder=self.retriever.question_encoder,
            passages=passages,
        )

    def forward(self, batch: DPRInput):
        retriever_out = self.retriever(batch)
        out = self.reader(retriever_out)
        return out


