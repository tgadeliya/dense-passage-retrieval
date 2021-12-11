import pytest
from pytest import fixture

from transformers import AutoModel, AutoTokenizer

from dpr import Encoder, NeuralRetriever


def test_building_faiss_index():
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    encoder = Encoder(
        base_encoder=AutoModel.from_pretrained("allegro/herbert-base-cased"),
        tokenizer=AutoTokenizer.from_pretrained("allegro/herbert-base-cased"),
        representation_type="CLS",
    )
    text = ["asdasdlask;d", "asdasd", "asdpqokpqwd", "asdpqpwpqpwe21"]
    index = NeuralRetriever.build_faiss_index_given_encoder_and_passages(
        passages=text,
        encoder=encoder,
        dim=encoder.config.hidden_size,
    )
    assert (index.ntotal, index.d) == (len(text), encoder.config.hidden_size)