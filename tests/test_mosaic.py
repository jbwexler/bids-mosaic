import pytest
import os
import PIL.Image
from reportlab.lib.styles import getSampleStyleSheet
import bidsmosaic.mosaic as mosaic


@pytest.fixture
def dataset():
    return os.getenv("TEST_DATASET")


def make_png(tmp_path, width, height):
    path = str(tmp_path / f"{width}x{height}.png")
    PIL.Image.new("L", (width, height)).save(path)
    return path


def test_create_sized_img_fits(tmp_path):
    path = make_png(tmp_path, 40, 40)
    img = mosaic.create_sized_img(path)
    assert img._width == 40
    assert img._height == 40


def test_create_sized_img_exact_boundary(tmp_path):
    path = make_png(tmp_path, mosaic.MAX_IMG_WIDTH, mosaic.MAX_IMG_HEIGHT)
    img = mosaic.create_sized_img(path)
    assert img._width == mosaic.MAX_IMG_WIDTH
    assert img._height == mosaic.MAX_IMG_HEIGHT


def test_create_sized_img_height_constrained(tmp_path):
    path = make_png(tmp_path, 40, 160)
    img = mosaic.create_sized_img(path)
    assert img._height == mosaic.MAX_IMG_HEIGHT
    assert img._width == pytest.approx(mosaic.MAX_IMG_HEIGHT / 160 * 40)


def test_create_sized_img_width_constrained(tmp_path):
    path = make_png(tmp_path, 160, 40)
    img = mosaic.create_sized_img(path)
    assert img._width == mosaic.MAX_IMG_WIDTH
    assert img._height == pytest.approx(mosaic.MAX_IMG_WIDTH / 160 * 40)


def test_create_filename_caption_normal():
    assert mosaic.create_filename_caption("sub-01_T1w.nii.gz.png") == "sub-01_T1w.nii.gz"


def test_create_filename_caption_colon_encoded():
    result = mosaic.create_filename_caption("sub-01:anat:sub-01_T1w.nii.gz.png")
    assert result == "sub-01/anat/sub-01_T1w.nii.gz"


def test_create_filename_caption_2d():
    result = mosaic.create_filename_caption("sub-01_T1w.nii.gz_2D.png")
    assert result == "sub-01_T1w.nii.gz (2D)"


def test_create_mosaic_table_empty_dir(tmp_path):
    styles = getSampleStyleSheet()
    with pytest.raises(SystemExit):
        mosaic.create_mosaic_table(str(tmp_path), 576, styles)


def test_run(dataset):
    assert dataset is not None, "TEST_DATASET environment variable must be set"

    metadata = '{"Dataset ID":"ds000000", "Dataset Name": "Test Dataset"}'
    mosaic.create_mosaic_pdf(
        dataset,
        "mosaic_test.pdf",
        anat=True,
        png_out_dir=None,
        downsample=2,
        freesurfer=None,
        metadata=metadata,
    )
