import numpy as np
import argparse
import os.path
import glob
import tempfile
import json
import logging
from bids import BIDSLayout
from bids.layout.models import BIDSImageFile
from nilearn.plotting import plot_img
import nibabel as nb
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont


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
    display_mode="x",
    cut_coords=np.array([0]),
    colorbar=False,
) -> None:
    """Creates a png of a slice(s) of a nifti. Defaults to a single midline
    sagittal slice."""
    try:
        img = nb.load(img_path)
    except FileNotFoundError:
        logging.error("%s was not found." % img_path)
        return

    out_file = img_path.replace("/", ":") + ".png"
    out_path = os.path.join(out_dir, out_file)

    plot_img(
        img,
        output_file=out_path,
        display_mode=display_mode,
        cut_coords=cut_coords,
        colorbar=colorbar,
    )

    return out_path


def wrap_text(text: str, font: ImageFont, max_width: int, draw: ImageDraw):
    """Wraps text to fit within the max_width."""
    chars = list(text)
    lines = []
    current_line = []

    for char in chars:
        test_line = "".join(current_line + [char])
        width = draw.textlength(test_line, font=font)
        if width <= max_width:
            current_line.append(char)
        else:
            lines.append("".join(current_line))
            current_line = [char]

    if current_line:
        lines.append("".join(current_line))

    return lines


def add_image_text(img_path: str, text: str) -> None:
    """Adds text below an image."""
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    font_size = 18
    padding = 4
    line_height = font_size + padding
    font = ImageFont.load_default(size=font_size)
    width, height = img.size

    text_lines = wrap_text(text, font, width, draw)
    for line in text_lines:
        img = add_margin_below(img, line_height)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        draw.text(
            (width / 2, height - line_height),
            line,
            font=font,
            anchor="ma",
            fill="black",
        )

    img = add_margin_below(img, 4)
    img.save(img_path, "PNG")


def add_new_section(pdf: FPDF, title: str) -> None:
    """Creates a new pdf page with a title."""
    pdf.add_page()
    pdf.set_font("Helvetica", size=32, style="B")
    pdf.cell(0, 10, title, align="C")
    pdf.ln(20)

def add_mosaic_table(pdf: FPDF, img_dir_path: str) -> None:
    """Adds mosaic table to pdf."""
    num_col = 6
    image_list = sorted(glob.glob(img_dir_path + "/*"))
    image_table_list = [
        image_list[i : i + num_col] for i in range(0, len(image_list), num_col)
    ]

    with pdf.table() as table:
        for i, img_row in enumerate(image_table_list):
            row = table.row()
            for j, img in enumerate(img_row):
                row.cell(img=img)

    

def create_pdf(img_dir_path: str, out_path: str, metadata=None) -> None:
    """Creates a pdf containing images aligned in a grid"""
    pdf = FPDF()

    for dir in glob.glob(os.path.join(img_dir_path, "*")):
        title = os.path.basename(dir) + " Images"
        add_new_section(pdf, title)
        add_mosaic_table(pdf, dir)

    if metadata:
        add_new_section(pdf, "Metadata")

        pdf.set_font("Helvetica", size=12)
        meta_dict = json.loads(metadata)

        with pdf.table(first_row_as_headings=False) as table:
            for k, v in meta_dict.items():
                row = table.row()
                row.cell(text=k)
                row.cell(text=v)

    pdf.output(out_path)

def create_anat_images(layout: BIDSLayout, temp_dir: str) -> None:
    """Creates anatomical mosaic .png files."""
    anat_layout_kwargs = {
        "datatype": "anat",
        "extension": ["nii", "nii.gz"],
    }

    files = layout.get(**anat_layout_kwargs)
    anat_temp_dir = os.path.join(temp_dir, "Anatomical")

    for file in files:
        png_path = create_slice_img(file.path, anat_temp_dir)
        if png_path:
            add_image_text(png_path, file.filename)


def create_fs_images(fs_dir: str, temp_dir: str) -> None:
    """Creates freesurfer mosaic .png files."""
    fs_temp_dir = os.path.join(temp_dir, "Freesurfer")

    for file_path in glob.glob(os.path.join(fs_dir, "sub-*/mri/orig/*")):
        png_path = create_slice_img(file_path, fs_temp_dir)
        if png_path:
            add_image_text(png_path, os.path.relpath(file_path, fs_dir))


def create_mosaic(args: argparse.Namespace) -> None:
    """Creates a mosaic pdf."""
    if args.out_file:
        out_file = args.out_file
    else:
        in_abs = os.path.abspath(args.dataset)
        out_file = os.path.basename(in_abs) + "_mosaic.pdf"

    if not args.png_dir:
        temp_dir_obj = tempfile.TemporaryDirectory()
        layout = BIDSLayout(args.dataset, validate=False)
        
        if args.anat:
            create_anat_images(layout, temp_dir_obj.name)
        if args.freesurfer:
            create_fs_images(args.freesurfer, temp_dir_obj.name)

        create_pdf(temp_dir_obj.name, out_file, args.metadata)

        temp_dir_obj.cleanup()
    else:
        create_pdf(args.png_dir, out_file, args.metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="Path to dataset")
    parser.add_argument(
        "-o",
        "--out-file",
        type=str,
        help="Path to output pdf. Defaults to <input dir name>_mosaics.pdf in working directory.",
    )
    parser.add_argument(
        "--png-dir",
        type=str,
        help="Path to existing directory of .png files, bypassing creation of those from .nii files.",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        help="JSON string to include as metadata at the end of the output file.",
    )
    parser.add_argument(
        "--anat",
        action="store_true",
        help="Include mosaic of all anatomical images.",
    )
    parser.add_argument(
        "--freesurfer",
        type=str,
        help="Path to freesurfer data.",
    )

    args = parser.parse_args()
    create_mosaic(args)


if __name__ == "__main__":
    main()
