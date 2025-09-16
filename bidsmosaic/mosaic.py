import numpy as np
import argparse
import os.path
import glob
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


def add_margin_below(pil_img: Image, margin_size: int):
    """Adds white margin below a pillow image"""
    width, height = pil_img.size
    new_height = height + margin_size
    color = (255, 255, 255)
    result = Image.new(pil_img.mode, (width, new_height), color)
    result.paste(pil_img, (0, 0))
    return result


def create_slice_img(
    img_path: str,
    out_dir: str,
    ds_path: str,
    display_mode="x",
    cut_coords=np.array([0]),
    colorbar=False,
    ds_root=None,
    downsample=None,
) -> None:
    """Creates a png of a slice(s) of a nifti. Defaults to a single midline
    sagittal slice."""
    try:
        img = nb.load(img_path)
    except FileNotFoundError:
        logging.error("%s was not found." % img_path)
        return

    if ds_root:
        relpath = os.path.relpath(img_path, ds_path)
        out_file = relpath.replace("/", ":") + ".png"
    else:
        out_file = os.path.basename(img_path) + ".png"

    out_path = os.path.join(out_dir, out_file)

    plot_img(
        img,
        display_mode=display_mode,
        cut_coords=cut_coords,
        colorbar=colorbar,
    )
    plt.savefig(out_path, transparent=True)

    # Remove transparent margins
    png = PIL.Image.open(out_path)
    new_png = png.crop(png.getbbox())

    if downsample:
        height, width = new_png.size
        new_size = (round(height / downsample), round(width / downsample))
        new_png = new_png.resize(new_size)

    new_png.save(out_path)


def create_sized_img(img_path: str, new_height: int) -> Image:
    """Creates a reportlab Image from a .png. Resizes using new_height
    and calculating new_width to maintain aspect ratio."""
    img = PIL.Image.open(img_path)
    width, height = img.size
    new_width = (new_height / height) * width
    return Image(img_path, height=new_height, width=new_width)


def create_filename_caption(img_path: str) -> Paragraph:
    """Creates a reportlab Paragraph containing the filename of the image."""
    caption_text = os.path.basename(img_path)
    caption_text = caption_text.replace(":", "/")
    caption_text = os.path.relpath(caption_text)
    caption_text = caption_text.removesuffix(".png")
    return caption_text


def create_mosaic_table(img_dir_path: str, page_width: int, styles) -> Table:
    """Adds mosaic table to pdf."""
    page_width
    img_height = 80

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

    image_path_list = sorted(glob.glob(img_dir_path + "/*"))
    image_tup_list = [
        [
            create_sized_img(img_path, img_height),
            Paragraph(
                f"<para align=center spaceb=3>{create_filename_caption(img_path)}</para>",
                caption_style,
            ),
        ]
        for img_path in image_path_list
    ]
    img_width = image_tup_list[0][0]._width
    num_col = int(page_width / img_width)
    col_width = int(page_width / num_col)

    image_table_list = [
        image_tup_list[i : i + num_col] for i in range(0, len(image_path_list), num_col)
    ]

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
    table = Table(image_table_list, colWidths=col_width)
    table.setStyle(table_style)

    return table


def create_metadata_table(metadata: str) -> Table:
    """Creates a table containing user-inputted metadata."""
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
        leftMargin=18,
        rightMargin=18,
        topMargin=36,
        bottomMargin=36,
    )
    page_width = int(pdf.width)

    flowables = []

    for d in glob.glob(os.path.join(img_dir_path, "*")):
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


def create_anat_images(layout: BIDSLayout, png_dir: str, downsample=None) -> None:
    """Creates anatomical mosaic .png files."""
    anat_layout_kwargs = {
        "datatype": "anat",
        "extension": ["nii", "nii.gz"],
    }

    files = layout.get(**anat_layout_kwargs)
    anat_png_dir = os.path.join(png_dir, "Anatomical")
    os.makedirs(anat_png_dir, exist_ok=True)

    for file in files:
        create_slice_img(file.path, anat_png_dir, layout.root, downsample=downsample)


def create_fs_images(fs_dir: str, png_dir: str, downsample=None) -> None:
    """Creates freesurfer mosaic .png files."""
    fs_png_dir = os.path.join(png_dir, "Freesurfer")
    os.makedirs(fs_png_dir, exist_ok=True)

    for file_path in glob.glob(os.path.join(fs_dir, "sub-*/mri/orig/*")):
        create_slice_img(
            file_path, fs_png_dir, fs_dir, ds_root=fs_dir, downsample=downsample
        )


def create_mosaic_pdf(
    dataset: str,
    out_file,
    anat=True,
    png_out_dir=None,
    downsample=None,
    freesurfer=None,
    metadata=None,
) -> None:
    """Creates a mosaic pdf."""
    if png_out_dir:
        png_dir = png_out_dir
    else:
        temp_dir_obj = tempfile.TemporaryDirectory()
        png_dir = temp_dir_obj.name

    layout = BIDSLayout(dataset, validate=False)

    if anat:
        create_anat_images(layout, png_dir, downsample=downsample)
    if freesurfer:
        create_fs_images(freesurfer, png_dir, downsample=downsample)

    create_pdf(png_dir, out_file, metadata)

    if not png_out_dir:
        temp_dir_obj.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Path to dataset")
    parser.add_argument(
        "-o",
        "--out-file",
        type=str,
        help="Path to output pdf. Defaults to <input dir name>_mosaics.pdf in working directory.",
    )
    parser.add_argument(
        "--png-in-dir",
        type=str,
        help="Path to existing directory of .png files, bypassing creation of those from .nii files.",
    )
    parser.add_argument(
        "--png-out-dir",
        type=str,
        help="Path to directory to output .png slice images too, instead of creating a temp directory.",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        help="JSON string to include as metadata at the end of the output file.",
    )
    parser.add_argument(
        "--no-anat",
        action="store_false",
        dest="anat",
        help="Include mosaic of all anatomical images.",
    )
    parser.add_argument(
        "--freesurfer",
        type=str,
        help="Path to freesurfer data.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        help="Factor by which to downsample images.",
    )

    args = parser.parse_args()

    if args.out_file:
        out_file = args.out_file
    else:
        in_abs = os.path.abspath(args.dataset)
        out_file = os.path.basename(in_abs) + "_mosaic.pdf"

    if not args.png_in_dir:
        create_mosaic_pdf(
            args.dataset,
            out_file,
            anat=args.anat,
            png_out_dir=args.png_out_dir,
            downsample=args.downsample,
            freesurfer=args.freesurfer,
            metadata=args.metadata,
        )
    else:
        create_pdf(args.png_in_dir, out_file, args.metadata)


if __name__ == "__main__":
    main()
