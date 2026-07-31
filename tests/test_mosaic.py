import pytest
import os
import sys
import json
import numpy as np
import nibabel as nb
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


def make_nifti(dir_path, name, shape=(8, 8, 8)):
    """Writes a nifti with a gradient, so its slice isn't cropped away."""
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    path = str(dir_path / name)
    nb.Nifti1Image(data, np.eye(4)).to_filename(path)
    return path


def make_out_dir(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return out_dir


def write_json(tmp_path, contents, name="images.json"):
    path = tmp_path / name
    if isinstance(contents, str):
        path.write_text(contents)
    else:
        path.write_text(json.dumps(contents))
    return str(path)


def run_main(monkeypatch, tmp_path, *args):
    """Runs main() as if from the command line, with tmp_path as the cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["bids-mosaic", *args])
    mosaic.main()


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


def test_unique_path_untaken(tmp_path):
    path = str(tmp_path / "img.png")
    assert mosaic.unique_path(path) == path


def test_unique_path_taken(tmp_path):
    (tmp_path / "img.png").touch()
    assert mosaic.unique_path(str(tmp_path / "img.png")) == str(tmp_path / "img_1.png")


def test_unique_path_multiple_taken(tmp_path):
    for name in ("img.png", "img_1.png"):
        (tmp_path / name).touch()
    assert mosaic.unique_path(str(tmp_path / "img.png")) == str(tmp_path / "img_2.png")


def test_create_slice_img_same_basename(tmp_path):
    """Images that share a basename shouldn't overwrite each other."""
    out_dir = make_out_dir(tmp_path)

    for sub in ("sub-01", "sub-02"):
        img_dir = tmp_path / sub
        img_dir.mkdir()
        mosaic.create_slice_img(make_nifti(img_dir, "T1w.nii.gz"), str(out_dir))

    assert sorted(p.name for p in out_dir.iterdir()) == [
        "T1w.nii.gz.png",
        "T1w.nii.gz_1.png",
    ]


def test_create_slice_img_ds_path(tmp_path):
    """With ds_path, pngs are named after the path relative to it."""
    img_dir = tmp_path / "ds" / "sub-01" / "anat"
    img_dir.mkdir(parents=True)
    out_dir = make_out_dir(tmp_path)

    mosaic.create_slice_img(
        make_nifti(img_dir, "T1w.nii.gz"), str(out_dir), ds_path=str(tmp_path / "ds")
    )

    assert [p.name for p in out_dir.iterdir()] == ["sub-01:anat:T1w.nii.gz.png"]


def test_create_slice_img_2d(tmp_path):
    out_dir = make_out_dir(tmp_path)

    mosaic.create_slice_img(
        make_nifti(tmp_path, "slice.nii.gz", shape=(8, 8)), str(out_dir)
    )

    assert [p.name for p in out_dir.iterdir()] == ["slice.nii.gz_2D.png"]


def test_create_slice_img_skips_4d(tmp_path):
    out_dir = make_out_dir(tmp_path)

    mosaic.create_slice_img(
        make_nifti(tmp_path, "bold.nii.gz", shape=(8, 8, 8, 2)), str(out_dir)
    )

    assert list(out_dir.iterdir()) == []


def test_main_json_input(monkeypatch, tmp_path):
    """The pdf is named after the json file when no dataset is given."""
    json_path = write_json(
        tmp_path, {"Anatomical": [make_nifti(tmp_path, "T1w.nii.gz")]}
    )

    run_main(monkeypatch, tmp_path, "--json-input", json_path)

    assert (tmp_path / "images_mosaic.pdf").exists()


def test_main_json_input_malformed(monkeypatch, tmp_path):
    json_path = write_json(tmp_path, "{not json")

    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--json-input", json_path)


def test_main_json_input_not_a_mapping(monkeypatch, tmp_path):
    json_path = write_json(tmp_path, ["T1w.nii.gz"])

    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--json-input", json_path)


def test_main_json_input_values_not_lists(monkeypatch, tmp_path):
    json_path = write_json(tmp_path, {"Anatomical": "T1w.nii.gz"})

    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--json-input", json_path)


def test_main_json_input_empty(monkeypatch, tmp_path):
    json_path = write_json(tmp_path, {})

    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--json-input", json_path)


def test_main_requires_dataset_or_json_input(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path)


def test_main_missing_images_reports_error(monkeypatch, tmp_path):
    """A json of paths that don't exist is an error, not an empty pdf."""
    json_path = write_json(tmp_path, {"Anatomical": [str(tmp_path / "nope.nii.gz")]})

    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--json-input", json_path)


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
