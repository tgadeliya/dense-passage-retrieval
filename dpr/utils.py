import os

import gcsfs


def cp_from_gcs_to_local(gcs_path: str, output_dir: str = ".") -> str:
    """
    Copy file from Google Cloud Storage.
    """
    local_path = os.path.join(output_dir, gcs_path.split("/")[-1])
    fs = gcsfs.GCSFileSystem()
    fs.get_file(rpath=gcs_path, lpath=local_path)
    return local_path
