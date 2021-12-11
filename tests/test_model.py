from pathlib import Path
import pytest


import transformers.modeling_outputs
from transformers import AutoModel, AutoTokenizer

from dpr.model import Encoder, cp_from_gcs_to_local, DPRRetriever


def test_pytest():
    assert 1


# TODO: Learn setUp and tearDown in pytest
@pytest.mark.skip("Skipping wrong repr type")
@pytest.mark.parametrize("representation_type", ["CLSS", "MAEN", "OTHER"])
def test_encoder_error_representation_type(representation_type):
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    with pytest.raises(AssertionError):
        reader = Encoder(
            base_encoder=model,
            tokenizer=tokenizer,
            representation_type=representation_type,
        )


@pytest.mark.skip("Skipping proper repr type")
@pytest.mark.parametrize("representation_type", ["CLS"])
def test_encoder_valid_representation_type(representation_type):
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    reader = Encoder(
        base_encoder=model, tokenizer=tokenizer, representation_type=representation_type
    )


@pytest.mark.skip("Skipping proper repr type")
def test_encoder_forward_cls():
    text = ["Stolica Polski?", "Największe miasto Polski?"]
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    reader = Encoder(base_encoder=model, tokenizer=tokenizer, representation_type="CLS")
    out = reader(text)
    assert out.size() == (2, reader.model_config.hidden_size)


@pytest.mark.skip("Skipping proper repr type")
def test_encoder_forward_mean():
    text = ["Stolica Polski?", "Największe miasto Polski?"]
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    reader = Encoder(
        base_encoder=model, tokenizer=tokenizer, representation_type="MEAN"
    )
    out = reader(text)
    assert out.size() == (2, reader.model_config.hidden_size)


@pytest.mark.skip("Skipping proper repr type")
def test_encoder_forward_base():
    text = ["Stolica Polski?", "Największe miasto Polski?"]
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    reader = Encoder(
        base_encoder=model, tokenizer=tokenizer, representation_type="BASE"
    )
    out = reader(text)
    assert (
        type(out)
        == transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
    )


@pytest.mark.skip("Skipping proper repr type")
def test_cp_from_gcs():
    gcs_path = (
        "gs://kraken_cex_explorer/models/"
        "botanist_mlm_pretraining/botanist_mlm_pretraining_0329_101203_bs32_gloo_a100_ep40_half/"
        "config.json"
    )

    test_local_path = Path("./" + gcs_path.split("/")[-1])
    assert not test_local_path.exists(), "File from GCS"
    local_path = cp_from_gcs_to_local(gcs_path)
    local_path = Path(local_path)
    assert local_path.is_file()
    local_path.unlink()


def test_building_faiss_index():
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    encoder = Encoder(
        base_encoder=AutoModel.from_pretrained("allegro/herbert-base-cased"),
        tokenizer=AutoTokenizer.from_pretrained("allegro/herbert-base-cased"),
        representation_type="CLS",
    )
    text = ["asdasdlask;d", "asdasd", "asdpqokpqwd", "asdpqpwpqpwe21"]
    index = DPRRetriever.build_faiss_index_given_encoder_and_passages(
        passages=text,
        encoder=encoder,
        dim=encoder.config.hidden_size,
    )
    assert (index.ntotal, index.d) == (len(text), encoder.config.hidden_size)
