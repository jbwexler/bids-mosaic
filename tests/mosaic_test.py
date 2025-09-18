import pytest
import os
import bidsmosaic.mosaic as mosaic


@pytest.fixture
def dataset():
    return os.getenv('TEST_DATASET')

def test_run(dataset):
    metadata = '{"Dataset ID":"ds000000", "Dataset Name": "Test Dataset"}'

    mosaic.create_mosaic_pdf(
        dataset,
        "mosaic_test.pdf",
        anat=True,
        png_out_dir=None,
        downsample=2,
        freesurfer=None,
        metadata=metadata
    )
