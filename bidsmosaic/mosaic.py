import numpy as np
import os.path
import glob
import gzip
import io
import tempfile
import json
import logging
import PIL.Image
from bids import BIDSLayout
from nilearn.plotting import plot_img
import matplotlib.pyplot as plt
import nibabel as nb
from reportlab.platypus import (
    Paragraph,
    Image,
    Table,
    SimpleDocTemplate,
    PageBreak,
    TableStyle,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


logger = logging.getLogger(__name__)

MAX_IMG_HEIGHT = 80
MAX_IMG_WIDTH = 80


class MosaicError(Exception):
    """Raised when a mosaic can't be created. The command line interface
    reports these as errors; callers using bidsmosaic as a library are left to
    handle them."""


ON_ERROR_MODES = ("placeholder", "skip", "strict")


def check_on_error(on_error: str) -> None:
    """Raises if on_error isn't one of ON_ERROR_MODES. Called up front by each
    entry point, so a typo fails before any work is done rather than at the
    first image."""
    if on_error not in ON_ERROR_MODES:
        raise ValueError(
            "on_error must be one of %s, not %r" % (", ".join(ON_ERROR_MODES), on_error)
        )


def handle_bad_image(
    message: str, on_error: str, placeholder_path: str, exc=None
) -> None:
    """Reports an image that can't be turned into a slice. on_error picks how:
    "placeholder" warns and writes a .error file holding the message, which the
    pdf shows as a captioned "Error" cell; "skip" warns and leaves it out;
    "strict" raises so one bad image fails the whole run. exc is the exception
    behind it, if any. Its type and message are added to the message, the type
    since some messages, like MemoryError's, are empty, and its traceback is
    logged at debug level so a bug can be found without reproducing it."""
    if exc is not None:
        message += ": " + type(exc).__name__ + (": %s" % exc if str(exc) else "")
        logger.debug("Traceback for: %s" % message, exc_info=exc)
    if on_error == "strict":
        raise MosaicError(message)
    if on_error == "placeholder":
        logger.warning("%s Including a placeholder." % message)
        with open(unique_path(placeholder_path), "w") as f:
            f.write(message + "\n")
    else:
        logger.warning("%s Skipping." % message)


def enhance_brightness(
    img: PIL.Image, target_brightness=100, threshold=10
) -> PIL.Image:
    """Attempts to change the brightness of an image to the target brightness."""
    arr = np.array(img, dtype="float32")
    mask = arr > threshold
    rms = np.sqrt(np.mean(np.square(arr[mask])))
    brightness_factor = target_brightness / rms
    arr[mask] *= brightness_factor
    return PIL.Image.fromarray(arr).convert("L")


def unique_path(path: str) -> str:
    """Returns path, or path with _1, _2, ... appended to the filename if it is
    already taken, so images that share a name don't overwrite each other."""
    if not os.path.exists(path):
        return path

    stem, ext = os.path.splitext(path)
    count = 1
    while os.path.exists(f"{stem}_{count}{ext}"):
        count += 1
    return f"{stem}_{count}{ext}"


def load_stream_img(stream, filename: str):
    """Loads a nifti or freesurfer image from an open stream, gunzipping it if
    the filename says to. Image data is read lazily, so the stream must stay 
    open until the image is used."""
    if filename.endswith((".gz", ".mgz")):
        stream = gzip.open(stream)
    if filename.endswith((".mgz", ".mgh")):
        return nb.MGHImage.from_stream(stream)

    header = stream.read(nb.Nifti2Header.sizeof_hdr)
    stream.seek(0)
    if nb.Nifti2Header.may_contain_header(header):
        return nb.Nifti2Image.from_stream(stream)
    return nb.Nifti1Image.from_stream(stream)


def write_slice_png(
    img, out_path: str, display_mode, cut_coords, colorbar, downsample
) -> None:
    """Plots a 2D or 3D image to out_path, then crops, downsamples and
    brightens the png."""
    if len(img.shape) == 3:
        plot_img(
            img,
            display_mode=display_mode,
            cut_coords=cut_coords,
            colorbar=colorbar,
            annotate=False,
        )
        plt.savefig(out_path, transparent=True)
    else:
        img_data = img.get_fdata()
        img_data = np.flipud(img_data.T)

        plt.imsave(out_path, img_data, cmap="gray")
    plt.close()

    # Remove transparent margins
    png = PIL.Image.open(out_path)
    new_png = png.crop(png.getbbox()).convert("L")

    if downsample:
        height, width = new_png.size
        new_size = (round(height / downsample), round(width / downsample))
        new_png = new_png.resize(new_size)

    new_png = enhance_brightness(new_png)

    new_png.save(out_path)


def create_slice_img(
    img_path: str | tuple,
    out_dir: str,
    display_mode="x",
    cut_coords=np.array([0]),
    colorbar=False,
    ds_path=None,
    downsample=None,
    on_error="placeholder",
) -> None:
    """Creates a png of a slice(s) of a nifti. Defaults to a single midline
    sagittal slice. img_path is a path to read from disk, or a
    (filename, stream) tuple to read from an already open stream, in which case
    the filename is only used to name the png."""
    stream = None
    if isinstance(img_path, tuple):
        img_path, stream = img_path

    if ds_path:
        relpath = os.path.relpath(img_path, ds_path)
        out_file = relpath.replace("/", ":")
    else:
        out_file = os.path.basename(img_path)

    error_path = os.path.join(out_dir, out_file + ".error")

    logger.debug(f"Creating png from {img_path}")
    try:
        if stream is not None:
            img = load_stream_img(stream, img_path)
        else:
            img = nb.load(img_path)
    except Exception as e:
        handle_bad_image("%s couldn't be read" % img_path, on_error, error_path, e)
        return

    if len(img.shape) not in (2, 3):
        handle_bad_image(
            "%s is %dD." % (img_path, len(img.shape)), on_error, error_path
        )
        return

    if len(img.shape) == 2:
        logger.warning("%s is a 2D image." % img_path)
        out_file += "_2D"

    out_path = unique_path(os.path.join(out_dir, out_file + ".png"))

    try:
        write_slice_png(img, out_path, display_mode, cut_coords, colorbar, downsample)
    except Exception as e:
        plt.close()
        if os.path.exists(out_path):
            os.remove(out_path)
        if isinstance(e, OSError) and e.filename == out_path:
            raise
        handle_bad_image("%s couldn't be plotted" % img_path, on_error, error_path, e)


def create_sized_img(img_path: str) -> Image:
    """Creates a reportlab Image from a .png, scaled to fit within
    MAX_IMG_HEIGHT x MAX_IMG_WIDTH while maintaining aspect ratio."""
    img = PIL.Image.open(img_path)
    width, height = img.size

    h_ratio = MAX_IMG_HEIGHT / height
    w_ratio = MAX_IMG_WIDTH / width

    if h_ratio >= 1 and w_ratio >= 1:
        return Image(img_path, height=height, width=width)
    elif h_ratio <= w_ratio:
        new_height = MAX_IMG_HEIGHT
        new_width = h_ratio * width
    else:
        new_width = MAX_IMG_WIDTH
        new_height = w_ratio * height
    return Image(img_path, height=new_height, width=new_width)


def create_filename_caption(img_path: str) -> str:
    """Returns the filename of the image, formatted for use as a caption."""
    caption_text = os.path.basename(img_path)
    caption_text = caption_text.replace(":", "/")
    caption_text = os.path.relpath(caption_text)
    caption_text = caption_text.removesuffix(".png").removesuffix(".error")
    if caption_text.endswith("_2D"):
        caption_text = caption_text.removesuffix("_2D")
        caption_text += " (2D)"

    return caption_text


def create_mosaic_table(img_dir_path: str, page_width: int, styles) -> Table:
    """Creates reportlab Table of slice images with image-name captions."""
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=6,
        leading=6,
        textColor=colors.black,
        alignment="CENTER",
        leftIndent=0,
        rightIndent=0,
        spaceAfter=0,
        spaceBefore=0,
    )
    error_style = ParagraphStyle(
        "Error",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.red,
    )

    image_path_list = sorted(glob.glob(img_dir_path + "/*"))

    if not image_path_list:
        raise MosaicError(f"No images found in {img_dir_path}")

    table_data = [
        [
            (
                Paragraph("<para align=center>Error</para>", error_style)
                if img_path.endswith(".error")
                else create_sized_img(img_path)
            ),
            Paragraph(
                f"<para align=center spaceb=3>{create_filename_caption(img_path)}</para>",
                caption_style,
            ),
        ]
        for img_path in image_path_list
    ]

    img_widths = [row[0]._width for row in table_data if isinstance(row[0], Image)]
    largest_img_width = max(img_widths, default=MAX_IMG_WIDTH)
    num_col = int(page_width / largest_img_width)
    col_width = int(page_width / num_col)

    table_data_rows = [
        table_data[i : i + num_col] for i in range(0, len(image_path_list), num_col)
    ]

    table_style = TableStyle(
        [
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
        ]
    )
    table = Table(table_data_rows, colWidths=col_width)
    table.setStyle(table_style)

    return table


def create_metadata_table(metadata: str) -> Table:
    """Creates a reportlab Table containing user-inputted metadata."""
    metadata_dict = json.loads(metadata)
    metadata_list = list(metadata_dict.items())

    table_style = TableStyle(
        [
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
        ]
    )

    table = Table(metadata_list)
    table.setStyle(table_style)

    return table


def create_pdf(img_dir_path: str, out_path: str, metadata=None) -> None:
    """Creates a pdf containing images aligned in a grid"""
    styles = getSampleStyleSheet()
    pdf = SimpleDocTemplate(
        out_path,
        pagesize=(612, 792),
        leftMargin=18,
        rightMargin=18,
        topMargin=36,
        bottomMargin=36,
    )
    page_width = int(pdf.width)

    flowables = []

    img_dirs = sorted(
        d for d in glob.glob(os.path.join(img_dir_path, "*")) if os.path.isdir(d)
    )

    if not img_dirs:
        raise MosaicError(
            "No image directories found in %s. Images must be in a "
            "subdirectory named after their datatype, eg %s."
            % (img_dir_path, os.path.join(img_dir_path, "Anatomical"))
        )

    error_paths = sorted(glob.glob(os.path.join(img_dir_path, "*", "*.error")))
    if error_paths and not glob.glob(os.path.join(img_dir_path, "*", "*.png")):
        with open(error_paths[0]) as f:
            first_error = f.read().strip()
        raise MosaicError(
            "All %d images failed, which suggests a bug or a problem with the "
            "environment rather than bad data. The first error was: %s"
            % (len(error_paths), first_error)
        )

    for d in img_dirs:
        title_text = os.path.basename(d) + " Images"
        title = Paragraph(title_text, styles["Title"])
        flowables.append(title)
        flowables.append(Spacer(0, 15))

        mosaic_table = create_mosaic_table(d, page_width, styles)
        flowables.append(mosaic_table)

        flowables.append(PageBreak())

    if metadata:
        title = Paragraph("Metadata", styles["Title"])
        flowables.append(title)
        flowables.append(Spacer(0, 15))

        metadata_table = create_metadata_table(metadata)
        flowables.append(metadata_table)

    pdf.build(flowables)

    logger.info("Successfully created pdf")


def create_anat_images(layout: BIDSLayout, png_dir: str, **slice_kwargs) -> None:
    """Creates anatomical mosaic .png files. Extra keyword arguments are passed
    on to create_slice_img."""
    anat_layout_kwargs = {
        "datatype": "anat",
        "extension": ["nii", "nii.gz"],
    }

    files = layout.get(**anat_layout_kwargs)
    anat_png_dir = os.path.join(png_dir, "Anatomical")
    os.makedirs(anat_png_dir, exist_ok=True)

    for file in files:
        create_slice_img(file.path, anat_png_dir, **slice_kwargs)


def create_fs_images(fs_dir: str, png_dir: str, **slice_kwargs) -> None:
    """Creates freesurfer mosaic .png files. Extra keyword arguments are passed
    on to create_slice_img."""
    fs_png_dir = os.path.join(png_dir, "Freesurfer")
    os.makedirs(fs_png_dir, exist_ok=True)

    for file_path in glob.glob(os.path.join(fs_dir, "sub-*/mri/orig/*")):
        create_slice_img(file_path, fs_png_dir, ds_path=fs_dir, **slice_kwargs)


def create_dict_images(files_dict: dict, png_dir: str, **slice_kwargs) -> None:
    """Create mosaic .png files according to dictionary. Keys should be strings
    of name of datatype (eg "Anatomical") and values should be a list of paths to
    nifti image files. Extra keyword arguments are passed on to create_slice_img."""

    for dtype, file_list in files_dict.items():
        dtype_png_dir = os.path.join(png_dir, dtype)
        os.makedirs(dtype_png_dir, exist_ok=True)

        for file in file_list:
            create_slice_img(file, dtype_png_dir, **slice_kwargs)


def create_mosaic_pdf(
    dataset: str,
    out_file: str,
    anat=True,
    png_out_dir=None,
    downsample=None,
    freesurfer=None,
    metadata=None,
    files_dict=None,
    on_error="placeholder",
) -> None:
    """Creates a mosaic pdf."""
    check_on_error(on_error)

    if png_out_dir:
        png_dir = png_out_dir
        if os.path.isdir(png_dir) and any(
            not f.startswith(".") for f in os.listdir(png_dir)
        ):
            raise MosaicError("png-out-dir %s is not empty." % png_dir)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory()
        png_dir = temp_dir_obj.name

    slice_kwargs = {"downsample": downsample, "on_error": on_error}

    if files_dict is not None:
        logger.info(f"Creating images from files_dict in {png_dir}")
        create_dict_images(files_dict, png_dir, **slice_kwargs)
    else:
        layout = BIDSLayout(dataset, validate=False)

        if anat:
            logger.info(f"Creating anat images in {png_dir}")
            create_anat_images(layout, png_dir, **slice_kwargs)
        if freesurfer:
            logger.info(f"Creating freesurfer images in {png_dir}")
            create_fs_images(freesurfer, png_dir, **slice_kwargs)

    logger.info(f"Creating pdf at {out_file}")
    create_pdf(png_dir, out_file, metadata)

    if not png_out_dir:
        temp_dir_obj.cleanup()


async def create_mosaic_pdf_async(
    out_file: str,
    files_dict: dict,
    downsample=None,
    metadata=None,
    on_error="placeholder",
) -> None:
    """Streaming counterpart of create_mosaic_pdf.

    files_dict maps a datatype name (eg "Anatomical") to a list of
    (filename, opener) pairs, where opener is a zero-arg async callable
    returning an async byte-stream. Nothing is opened until that file's turn
    comes, so only one image is ever held in memory, and each stream is created
    on the same event loop that reads it."""
    check_on_error(on_error)

    slice_kwargs = {"downsample": downsample, "on_error": on_error}

    with tempfile.TemporaryDirectory() as png_dir:
        for dtype, file_list in files_dict.items():
            dtype_png_dir = os.path.join(png_dir, dtype)
            os.makedirs(dtype_png_dir)
            for filename, opener in file_list:
                logger.info(f"Streaming {filename} into {dtype_png_dir}")
                try:
                    stream = await opener()
                    data = b"".join([chunk async for chunk in stream])
                except Exception as e:
                    handle_bad_image(
                        "%s couldn't be downloaded" % filename,
                        on_error,
                        os.path.join(
                            dtype_png_dir, os.path.basename(filename) + ".error"
                        ),
                        e,
                    )
                    continue
                with io.BytesIO(data) as buf:
                    create_slice_img((filename, buf), dtype_png_dir, **slice_kwargs)

        logger.info(f"Creating pdf at {out_file}")
        create_pdf(png_dir, out_file, metadata=metadata)
