import pytest
import os
import io
import sys
import json
import gzip
import logging
import asyncio
import contextlib
import numpy as np
import nibabel as nb
import PIL.Image
from reportlab.lib.styles import getSampleStyleSheet
import bidsmosaic.mosaic as mosaic
import bidsmosaic.cli as cli


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


def image_bytes(image_class=nb.Nifti1Image, shape=(8, 8, 8), gzipped=True):
    """Returns the on-disk bytes of an image, as a file or stream would hold."""
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    raw = image_class(data, np.eye(4)).to_bytes()
    return gzip.compress(raw) if gzipped else raw


def make_nifti_stream(name="T1w.nii.gz", shape=(8, 8, 8)):
    """Returns a (filename, stream) tuple, as create_slice_img takes in place of
    a path. Gzipped when the name says so, like a real file would be."""
    return name, io.BytesIO(image_bytes(shape=shape, gzipped=name.endswith(".gz")))


def make_mgh_stream(name="001.mgz", shape=(8, 8, 8)):
    """Returns a (filename, stream) tuple holding a freesurfer image. .mgz is
    gzipped on disk, .mgh isn't."""
    return name, io.BytesIO(
        image_bytes(nb.MGHImage, shape, gzipped=name.endswith(".mgz"))
    )


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
    cli.main()


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
    assert (
        mosaic.create_filename_caption("sub-01_T1w.nii.gz.png") == "sub-01_T1w.nii.gz"
    )


def test_create_filename_caption_colon_encoded():
    result = mosaic.create_filename_caption("sub-01:anat:sub-01_T1w.nii.gz.png")
    assert result == "sub-01/anat/sub-01_T1w.nii.gz"


def test_create_filename_caption_2d():
    result = mosaic.create_filename_caption("sub-01_T1w.nii.gz_2D.png")
    assert result == "sub-01_T1w.nii.gz (2D)"


def test_create_mosaic_table_empty_dir(tmp_path):
    styles = getSampleStyleSheet()
    with pytest.raises(mosaic.MosaicError):
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
        make_nifti(tmp_path, "bold.nii.gz", shape=(8, 8, 8, 2)),
        str(out_dir),
        on_error="skip",
    )

    assert list(out_dir.iterdir()) == []


@pytest.mark.parametrize("image_class", [nb.Nifti1Image, nb.Nifti2Image])
@pytest.mark.parametrize("name", ["sub-01_T1w.nii.gz", "sub-01_T1w.nii"])
def test_create_slice_img_from_stream(tmp_path, image_class, name):
    """Nifti1 and Nifti2 share the .nii extension, so both have to stream,
    gzipped or not."""
    out_dir = make_out_dir(tmp_path)
    raw = image_bytes(image_class, gzipped=name.endswith(".gz"))

    mosaic.create_slice_img((name, io.BytesIO(raw)), str(out_dir))

    assert [p.name for p in out_dir.iterdir()] == [f"{name}.png"]


@pytest.mark.parametrize("image_class", [nb.Nifti1Image, nb.Nifti2Image])
def test_load_stream_img_picks_nifti_version(image_class):
    """The nifti version comes from the header, not the filename."""
    raw = image_bytes(image_class, gzipped=False)

    img = mosaic.load_stream_img(io.BytesIO(raw), "sub-01_T1w.nii")

    assert isinstance(img, image_class)


def test_load_stream_img_big_endian_nifti2():
    """Version detection is byte-swap aware, so big-endian files load too."""
    data = np.arange(512, dtype=">f4").reshape(8, 8, 8)
    raw = nb.Nifti2Image(data, np.eye(4)).to_bytes()

    img = mosaic.load_stream_img(io.BytesIO(raw), "sub-01_T1w.nii")

    assert isinstance(img, nb.Nifti2Image)
    assert np.array_equal(np.asanyarray(img.dataobj), data)


def test_load_stream_img_corrupt_nifti1_reports_its_own_error():
    """A broken nifti1 has to report the nifti1 problem. Deciding the version by
    trying nifti1 and falling back would report the nifti2 attempt's error
    instead, naming the wrong format and field."""
    raw = bytearray(image_bytes(gzipped=False))
    raw[70:72] = b"\xff\xff"  # datatype field -> -1, an invalid code

    with pytest.raises(nb.spatialimages.HeaderDataError, match="-1"):
        mosaic.load_stream_img(io.BytesIO(bytes(raw)), "sub-01_T1w.nii")


def test_create_slice_img_corrupt_nifti1_strict(tmp_path):
    """That error reaches the caller, rather than a misleading one."""
    out_dir = make_out_dir(tmp_path)
    raw = bytearray(image_bytes(gzipped=False))
    raw[70:72] = b"\xff\xff"

    with pytest.raises(mosaic.MosaicError, match="couldn't be read.*-1"):
        mosaic.create_slice_img(
            ("sub-01_T1w.nii", io.BytesIO(bytes(raw))), str(out_dir), on_error="strict"
        )


def test_create_slice_img_from_stream_ds_path(tmp_path):
    """Names come from the tuple, so ds_path works the same as for files."""
    out_dir = make_out_dir(tmp_path)

    mosaic.create_slice_img(
        make_nifti_stream("ds/sub-01/anat/T1w.nii.gz"),
        str(out_dir),
        ds_path="ds",
    )

    assert [p.name for p in out_dir.iterdir()] == ["sub-01:anat:T1w.nii.gz.png"]


@pytest.mark.parametrize("name", ["001.mgz", "001.mgh"])
def test_create_slice_img_from_mgh_stream(tmp_path, name):
    """Freesurfer images stream too, gzipped (.mgz) or not (.mgh)."""
    out_dir = make_out_dir(tmp_path)

    mosaic.create_slice_img(make_mgh_stream(name), str(out_dir))

    assert [p.name for p in out_dir.iterdir()] == [f"{name}.png"]


def test_create_slice_img_from_stream_skips_4d(tmp_path):
    out_dir = make_out_dir(tmp_path)

    mosaic.create_slice_img(
        make_nifti_stream("bold.nii.gz", shape=(8, 8, 8, 2)),
        str(out_dir),
        on_error="skip",
    )

    assert list(out_dir.iterdir()) == []


@pytest.mark.parametrize(
    "name, data",
    [
        ("garbage.nii", b"nope" * 200),
        ("empty.nii", b""),
        ("not_gzipped.nii.gz", b"nope" * 200),
    ],
)
def test_create_slice_img_from_unreadable_stream(tmp_path, name, data):
    """An unreadable stream is skipped, like a missing file is."""
    out_dir = make_out_dir(tmp_path)

    mosaic.create_slice_img((name, io.BytesIO(data)), str(out_dir), on_error="skip")

    assert list(out_dir.iterdir()) == []


@pytest.mark.parametrize(
    "img_path, message",
    [
        (lambda tmp_path: str(tmp_path / "nope.nii.gz"), "FileNotFoundError"),
        (
            lambda tmp_path: ("garbage.nii", io.BytesIO(b"nope" * 200)),
            "couldn't be read",
        ),
        (
            lambda tmp_path: make_nifti(tmp_path, "bold.nii.gz", shape=(8, 8, 8, 2)),
            "is 4D",
        ),
        # Data is read lazily, so a truncated file only fails once it's plotted.
        (
            lambda tmp_path: (
                "trunc.nii",
                io.BytesIO(image_bytes(gzipped=False)[:600]),
            ),
            "couldn't be plotted: OSError",
        ),
    ],
)
def test_create_slice_img_placeholder(tmp_path, img_path, message):
    """By default a bad image leaves a .error file holding the reason, in place
    of its png."""
    out_dir = make_out_dir(tmp_path)

    mosaic.create_slice_img(img_path(tmp_path), str(out_dir))

    [error_file] = out_dir.iterdir()
    assert error_file.suffix == ".error"
    assert message in error_file.read_text()


def test_create_slice_img_late_failure_removes_png(monkeypatch, tmp_path):
    """A failure after the png is saved doesn't leave it behind next to the
    placeholder. A plain MemoryError has no message, so its type names the
    problem."""
    out_dir = make_out_dir(tmp_path)

    def fail(img):
        raise MemoryError()

    monkeypatch.setattr(mosaic, "enhance_brightness", fail)
    mosaic.create_slice_img(make_nifti(tmp_path, "T1w.nii.gz"), str(out_dir))

    [error_file] = out_dir.iterdir()
    assert error_file.name == "T1w.nii.gz.error"
    assert error_file.read_text().endswith("couldn't be plotted: MemoryError\n")


@pytest.mark.parametrize("on_error", ["placeholder", "strict"])
def test_create_slice_img_logs_traceback(monkeypatch, tmp_path, caplog, on_error):
    """The traceback behind a bad image is logged at debug level, so a bug that
    shows up as an Error cell can be found with --debug, without cluttering
    the warnings."""

    def changed_signature(*args, **kwargs):
        raise TypeError("plot_img() got an unexpected keyword argument")

    monkeypatch.setattr(mosaic, "plot_img", changed_signature)
    caplog.set_level(logging.DEBUG, logger="bidsmosaic")
    path = make_nifti(tmp_path, "T1w.nii.gz")
    if on_error == "strict":
        with pytest.raises(mosaic.MosaicError):
            mosaic.create_slice_img(path, str(tmp_path), on_error=on_error)
    else:
        mosaic.create_slice_img(path, str(tmp_path), on_error=on_error)
        [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "couldn't be plotted" in warning.message and not warning.exc_info

    [traceback] = [r for r in caplog.records if r.exc_info]
    assert traceback.levelno == logging.DEBUG
    assert traceback.exc_info[0] is TypeError


def test_create_pdf_all_failed_across_datatypes(tmp_path):
    """The check is across the whole pdf: one datatype that's all placeholders
    is fine as long as another has images."""
    for dtype, name in [("Anatomical", "a.nii.gz.error"), ("Freesurfer", "b.mgz")]:
        (tmp_path / "pngs" / dtype).mkdir(parents=True)
    (tmp_path / "pngs" / "Anatomical" / "a.nii.gz.error").write_text("a is 4D.\n")
    PIL.Image.new("L", (40, 40)).save(tmp_path / "pngs" / "Freesurfer" / "b.mgz.png")

    mosaic.create_pdf(str(tmp_path / "pngs"), str(tmp_path / "out.pdf"))

    (tmp_path / "pngs" / "Freesurfer" / "b.mgz.png").unlink()
    (tmp_path / "pngs" / "Freesurfer" / "b.mgz.error").write_text("b is 4D.\n")
    with pytest.raises(mosaic.MosaicError, match="All 2 images failed.*a is 4D"):
        mosaic.create_pdf(str(tmp_path / "pngs"), str(tmp_path / "out2.pdf"))


needs_permissions = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason="file permissions aren't enforced for root or on Windows",
)


@contextlib.contextmanager
def chmod(path, mode):
    """Sets path's permissions for the duration, restoring them after so
    tmp_path can still be cleaned up."""
    old_mode = os.stat(path).st_mode
    os.chmod(path, mode)
    try:
        yield
    finally:
        os.chmod(path, old_mode)


@needs_permissions
def test_create_slice_img_unwritable_out_dir(tmp_path):
    """An out_dir that can't be written to fails the run, as an output problem
    rather than a bad image, even when skipping bad images."""
    out_dir = make_out_dir(tmp_path)
    path = make_nifti(tmp_path, "T1w.nii.gz")

    with chmod(out_dir, 0o555):
        with pytest.raises(PermissionError, match="T1w.nii.gz.png"):
            mosaic.create_slice_img(path, str(out_dir), on_error="skip")

    assert list(out_dir.iterdir()) == []


@needs_permissions
def test_main_unwritable_out_file_exits(monkeypatch, tmp_path, caplog):
    """main reports an OSError, here from --png-in-dir's pdf write, as an error
    rather than a traceback."""
    png_dir = tmp_path / "pngs" / "Anatomical"
    png_dir.mkdir(parents=True)
    PIL.Image.new("L", (40, 40)).save(png_dir / "a.nii.gz.png")
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()

    with chmod(locked_dir, 0o555):
        with pytest.raises(SystemExit):
            run_main(
                monkeypatch,
                tmp_path,
                "unused",
                "--png-in-dir",
                str(tmp_path / "pngs"),
                "-o",
                str(locked_dir / "out.pdf"),
            )

    assert "Permission denied" in caplog.text


def test_create_slice_img_placeholder_same_basename(tmp_path):
    """Placeholders that share a name don't overwrite each other either."""
    out_dir = make_out_dir(tmp_path)

    for sub in ("sub-01", "sub-02"):
        mosaic.create_slice_img(str(tmp_path / sub / "T1w.nii.gz"), str(out_dir))

    assert sorted(p.name for p in out_dir.iterdir()) == [
        "T1w.nii.gz.error",
        "T1w.nii.gz_1.error",
    ]


def test_create_mosaic_pdf_bad_on_error(monkeypatch, tmp_path):
    """A bad on_error fails before the dataset is indexed."""

    def fail(*args, **kwargs):
        raise AssertionError("BIDSLayout shouldn't be built")

    monkeypatch.setattr(mosaic, "BIDSLayout", fail)

    with pytest.raises(ValueError, match="on_error"):
        mosaic.create_mosaic_pdf(
            str(tmp_path), str(tmp_path / "out.pdf"), on_error="ignore"
        )


def test_create_mosaic_pdf_async_bad_on_error(tmp_path):
    """A bad on_error fails before any stream is opened."""
    opened = []

    with pytest.raises(ValueError, match="on_error"):
        asyncio.run(
            mosaic.create_mosaic_pdf_async(
                str(tmp_path / "out.pdf"),
                {"Anatomical": [("a.nii.gz", make_opener(image_bytes(), opened))]},
                on_error="ignore",
            )
        )

    assert opened == []


def test_create_filename_caption_error():
    result = mosaic.create_filename_caption("sub-01_T1w.nii.gz.error")
    assert result == "sub-01_T1w.nii.gz"


def test_create_mosaic_table_placeholder(tmp_path):
    """A .error file becomes an "Error" cell captioned with the file name."""
    PIL.Image.new("L", (40, 40)).save(tmp_path / "a.nii.gz.png")
    (tmp_path / "b.nii.gz.error").write_text("b.nii.gz is 4D.\n")

    table = mosaic.create_mosaic_table(str(tmp_path), 576, getSampleStyleSheet())

    image_cell, error_cell = table._cellvalues[0]
    assert isinstance(image_cell[0], mosaic.Image)
    assert error_cell[0].getPlainText() == "Error"
    assert error_cell[1].getPlainText() == "b.nii.gz"


def test_create_mosaic_table_only_placeholders(tmp_path):
    """With no images to size columns by, the max image width is used."""
    (tmp_path / "a.nii.gz.error").write_text("a.nii.gz is 4D.\n")

    table = mosaic.create_mosaic_table(str(tmp_path), 576, getSampleStyleSheet())

    assert table._colWidths[0] == int(576 / int(576 / mosaic.MAX_IMG_WIDTH))


def test_create_slice_img_strict_missing_file(tmp_path):
    out_dir = make_out_dir(tmp_path)

    with pytest.raises(mosaic.MosaicError, match="FileNotFoundError"):
        mosaic.create_slice_img(
            str(tmp_path / "nope.nii.gz"), str(out_dir), on_error="strict"
        )


def test_create_slice_img_strict_unreadable(tmp_path):
    out_dir = make_out_dir(tmp_path)

    with pytest.raises(mosaic.MosaicError, match="couldn't be read"):
        mosaic.create_slice_img(
            ("garbage.nii", io.BytesIO(b"nope" * 200)),
            str(out_dir),
            on_error="strict",
        )


def test_create_slice_img_strict_4d(tmp_path):
    out_dir = make_out_dir(tmp_path)

    with pytest.raises(mosaic.MosaicError, match="is 4D"):
        mosaic.create_slice_img(
            make_nifti(tmp_path, "bold.nii.gz", shape=(8, 8, 8, 2)),
            str(out_dir),
            on_error="strict",
        )


def test_create_slice_img_strict_ok(tmp_path):
    """A readable image is unaffected by on_error."""
    out_dir = make_out_dir(tmp_path)

    mosaic.create_slice_img(
        make_nifti(tmp_path, "T1w.nii.gz"), str(out_dir), on_error="strict"
    )

    assert [p.name for p in out_dir.iterdir()] == ["T1w.nii.gz.png"]


def test_create_mosaic_pdf_strict(tmp_path):
    """One bad image fails the whole pdf under strict."""
    with pytest.raises(mosaic.MosaicError):
        mosaic.create_mosaic_pdf(
            None,
            str(tmp_path / "out.pdf"),
            files_dict={
                "Anatomical": [
                    make_nifti(tmp_path, "T1w.nii.gz"),
                    str(tmp_path / "nope.nii.gz"),
                ]
            },
            on_error="strict",
        )


def test_main_strict_bad_image_exits(monkeypatch, tmp_path):
    """By default the bad image gets a placeholder; with strict, main exits."""
    json_path = write_json(
        tmp_path,
        {
            "Anatomical": [
                make_nifti(tmp_path, "T1w.nii.gz"),
                str(tmp_path / "nope.nii.gz"),
            ]
        },
    )

    run_main(monkeypatch, tmp_path, "--json-input", json_path)
    assert (tmp_path / "images_mosaic.pdf").exists()

    with pytest.raises(SystemExit):
        run_main(
            monkeypatch, tmp_path, "--json-input", json_path, "--on-error", "strict"
        )


def test_create_mosaic_pdf_from_streams(tmp_path):
    """Streams go through create_mosaic_pdf the same way paths do."""
    out_pdf = tmp_path / "out.pdf"

    mosaic.create_mosaic_pdf(
        None,
        str(out_pdf),
        files_dict={
            "Anatomical": [
                make_nifti_stream("sub-01_T1w.nii.gz"),
                make_nifti_stream("sub-02_T1w.nii.gz"),
            ]
        },
    )

    assert out_pdf.exists()


def make_opener(raw, opened=None, name=None):
    """Builds an opener of the shape create_mosaic_pdf_async expects: a zero-arg
    async callable resolving to an async byte-stream. Appends to `opened` when
    called, so tests can tell a file isn't opened before its turn."""

    async def stream():
        for i in range(0, len(raw), 4096):
            yield raw[i : i + 4096]

    async def opener():
        if opened is not None:
            opened.append(name)
        return stream()

    return opener


def test_create_mosaic_pdf_async(tmp_path):
    """The streaming path builds a pdf without the images ever hitting disk.
    Streams name their own compression, gzipped (.nii.gz) or not (.nii)."""
    out_pdf = tmp_path / "out.pdf"

    asyncio.run(
        mosaic.create_mosaic_pdf_async(
            str(out_pdf),
            {
                "Anatomical": [
                    ("sub-01_T1w.nii.gz", make_opener(image_bytes())),
                    ("sub-02_T1w.nii", make_opener(image_bytes(gzipped=False))),
                ]
            },
        )
    )

    assert out_pdf.exists()


def test_create_mosaic_pdf_async_opens_one_at_a_time(monkeypatch, tmp_path):
    """Each file is opened only when its turn comes, so one image is in memory
    at a time. Opens and plots have to alternate; opening every stream up front
    would give the same order but hold them all open at once."""
    events = []
    raw = image_bytes()
    names = [f"sub-0{i}_T1w.nii.gz" for i in (1, 2, 3)]
    files = [
        (name, make_opener(raw, opened=events, name=("open", name))) for name in names
    ]

    real_create_slice_img = mosaic.create_slice_img

    def spy(img_path, out_dir, **kwargs):
        events.append(("plot", img_path[0]))
        return real_create_slice_img(img_path, out_dir, **kwargs)

    monkeypatch.setattr(mosaic, "create_slice_img", spy)

    asyncio.run(
        mosaic.create_mosaic_pdf_async(str(tmp_path / "out.pdf"), {"Anatomical": files})
    )

    assert events == [
        step for name in names for step in (("open", name), ("plot", name))
    ]


def test_create_mosaic_pdf_async_multiple_datatypes(tmp_path):
    """Each datatype gets its own directory, so the pdf has a section per type."""
    out_pdf = tmp_path / "out.pdf"
    raw = image_bytes()

    asyncio.run(
        mosaic.create_mosaic_pdf_async(
            str(out_pdf),
            {
                "Anatomical": [("sub-01_T1w.nii.gz", make_opener(raw))],
                "Freesurfer": [("sub-01_001.mgz", make_opener(image_bytes(nb.MGHImage)))],
            },
        )
    )

    assert out_pdf.exists()


def test_create_mosaic_pdf_async_strict(tmp_path):
    """on_error reaches create_slice_img through the streaming path too."""
    with pytest.raises(mosaic.MosaicError, match="couldn't be read"):
        asyncio.run(
            mosaic.create_mosaic_pdf_async(
                str(tmp_path / "out.pdf"),
                {"Anatomical": [("garbage.nii", make_opener(b"nope" * 200))]},
                on_error="strict",
            )
        )


@pytest.mark.parametrize("on_error", ["placeholder", "skip"])
def test_create_mosaic_pdf_async_bad_image(tmp_path, on_error):
    """Without strict, the pdf still builds around an unreadable stream."""
    out_pdf = tmp_path / "out.pdf"

    asyncio.run(
        mosaic.create_mosaic_pdf_async(
            str(out_pdf),
            {
                "Anatomical": [
                    ("garbage.nii", make_opener(b"nope" * 200)),
                    ("sub-01_T1w.nii.gz", make_opener(image_bytes())),
                ]
            },
            on_error=on_error,
        )
    )

    assert out_pdf.exists()


async def failing_opener():
    raise ConnectionError("network down")


async def failing_midstream_opener():
    async def stream():
        yield b"partial"
        raise RuntimeError("dropped")  # eg an HTTP library's own error

    return stream()


@pytest.mark.parametrize("opener", [failing_opener, failing_midstream_opener])
def test_create_mosaic_pdf_async_download_failure(tmp_path, opener):
    """A download that fails, before or partway through, and with any kind of
    error, is handled per file like any other bad image, so the rest of the pdf
    still gets made."""
    files = {
        "Anatomical": [
            ("sub-01_T1w.nii.gz", opener),
            ("sub-02_T1w.nii.gz", make_opener(image_bytes())),
        ]
    }
    out_pdf = tmp_path / "out.pdf"

    asyncio.run(mosaic.create_mosaic_pdf_async(str(out_pdf), files))
    assert out_pdf.exists()

    with pytest.raises(mosaic.MosaicError, match="couldn't be downloaded"):
        asyncio.run(
            mosaic.create_mosaic_pdf_async(
                str(tmp_path / "strict.pdf"), files, on_error="strict"
            )
        )


def test_create_mosaic_pdf_async_no_usable_images(tmp_path):
    """When skipping, a datatype whose images all fail leaves an empty dir,
    which is an error."""
    with pytest.raises(mosaic.MosaicError, match="No images found"):
        asyncio.run(
            mosaic.create_mosaic_pdf_async(
                str(tmp_path / "out.pdf"),
                {"Anatomical": [("garbage.nii", make_opener(b"nope" * 200))]},
                on_error="skip",
            )
        )


def test_create_mosaic_pdf_async_only_placeholders(tmp_path):
    """With placeholders, a run whose images all fail is an error too."""
    with pytest.raises(mosaic.MosaicError, match="All 1 images failed"):
        asyncio.run(
            mosaic.create_mosaic_pdf_async(
                str(tmp_path / "out.pdf"),
                {"Anatomical": [("garbage.nii", make_opener(b"nope" * 200))]},
            )
        )


def test_create_mosaic_pdf_png_out_dir_not_empty(tmp_path):
    png_dir = tmp_path / "pngs"
    png_dir.mkdir()
    make_png(png_dir, 10, 10)

    with pytest.raises(mosaic.MosaicError, match="not empty"):
        mosaic.create_mosaic_pdf(
            None,
            str(tmp_path / "out.pdf"),
            png_out_dir=str(png_dir),
            files_dict={"Anatomical": [make_nifti(tmp_path, "T1w.nii.gz")]},
        )


def test_create_mosaic_pdf_png_out_dir_hidden_file(tmp_path):
    """Hidden files like .DS_Store don't count as contents."""
    png_dir = tmp_path / "pngs"
    png_dir.mkdir()
    (png_dir / ".DS_Store").touch()
    out_pdf = tmp_path / "out.pdf"

    mosaic.create_mosaic_pdf(
        None,
        str(out_pdf),
        png_out_dir=str(png_dir),
        files_dict={"Anatomical": [make_nifti(tmp_path, "T1w.nii.gz")]},
    )

    assert out_pdf.exists()


def test_create_pdf_no_image_dirs(tmp_path):
    with pytest.raises(mosaic.MosaicError, match="No image directories"):
        mosaic.create_pdf(str(tmp_path), str(tmp_path / "out.pdf"))


def test_create_pdf_ignores_loose_files(tmp_path):
    """Images have to be in a datatype subdirectory, not loose in the dir."""
    make_png(tmp_path, 10, 10)

    with pytest.raises(mosaic.MosaicError, match="No image directories"):
        mosaic.create_pdf(str(tmp_path), str(tmp_path / "out.pdf"))


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


def test_main_max_img_size(monkeypatch, tmp_path):
    """The flags have to land on mosaic's globals, since create_sized_img reads
    those, not the cli's."""
    json_path = write_json(
        tmp_path, {"Anatomical": [make_nifti(tmp_path, "T1w.nii.gz")]}
    )
    # Same values, so monkeypatch puts the defaults back after the test.
    monkeypatch.setattr(mosaic, "MAX_IMG_HEIGHT", mosaic.MAX_IMG_HEIGHT)
    monkeypatch.setattr(mosaic, "MAX_IMG_WIDTH", mosaic.MAX_IMG_WIDTH)

    run_main(
        monkeypatch,
        tmp_path,
        "--json-input",
        json_path,
        "--max-img-height",
        "40",
        "--max-img-width",
        "30",
    )

    assert (mosaic.MAX_IMG_HEIGHT, mosaic.MAX_IMG_WIDTH) == (40, 30)


def test_main_debug_enables_mosaic_logging(monkeypatch, tmp_path):
    """--debug has to reach mosaic's logger, not only the cli's."""
    json_path = write_json(
        tmp_path, {"Anatomical": [make_nifti(tmp_path, "T1w.nii.gz")]}
    )
    package_logger = logging.getLogger("bidsmosaic")
    previous_level = package_logger.level

    try:
        run_main(monkeypatch, tmp_path, "--json-input", json_path, "--debug")

        assert logging.getLogger(mosaic.__name__).isEnabledFor(logging.DEBUG)
    finally:
        package_logger.setLevel(previous_level)


def test_main_requires_dataset_or_json_input(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path)


def test_main_missing_images_reports_error(monkeypatch, tmp_path):
    """When skipping, a json of paths that don't exist is an error, not an
    empty pdf."""
    json_path = write_json(tmp_path, {"Anatomical": [str(tmp_path / "nope.nii.gz")]})

    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, "--json-input", json_path, "--on-error", "skip")


def test_run(dataset, tmp_path):
    assert dataset is not None, "TEST_DATASET environment variable must be set"

    metadata = '{"Dataset ID":"ds000000", "Dataset Name": "Test Dataset"}'
    out_file = tmp_path / "mosaic_test.pdf"
    mosaic.create_mosaic_pdf(
        dataset,
        str(out_file),
        anat=True,
        png_out_dir=None,
        downsample=2,
        freesurfer=None,
        metadata=metadata,
    )

    assert out_file.exists()
