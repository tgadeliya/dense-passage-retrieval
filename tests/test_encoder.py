import pytest
from pytest import fixture

from transformers import AutoModel, AutoTokenizer

from dpr import Encoder


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
@pytest.mark.parametrize("representation_type", ["CLS"])
def test_encoder_valid_representation_type(representation_type):
    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    reader = Encoder(
        base_encoder=model, tokenizer=tokenizer, representation_type=representation_type
    )




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
