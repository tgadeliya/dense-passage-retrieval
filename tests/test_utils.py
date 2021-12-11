from pathlib import Path
import pytest

from dpr.utils import  cp_from_gcs_to_local


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
