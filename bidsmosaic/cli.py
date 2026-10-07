import argparse
import os.path
import sys
import json
import logging

from . import mosaic
from .mosaic import MosaicError, create_mosaic_pdf, create_pdf


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, nargs="?", help="Path to dataset")
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
        help="Path to directory to output .png slice images to, instead of creating a temp directory.",
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
        help="Do not include anatomical images.",
    )
    parser.add_argument(
        "--freesurfer",
        type=str,
        help="Path to freesurfer data.",
    )
    parser.add_argument(
        "--json-input",
        type=str,
        help="Path to json file containing keys of datatypes (eg Anatomical) and values"
        "of lists of paths to image files.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        help="Factor by which to downsample images.",
    )
    parser.add_argument(
        "--max-img-height",
        type=int,
        help="Max height of images.",
    )
    parser.add_argument(
        "--max-img-width",
        type=int,
        help="Max width of images.",
    )
    parser.add_argument(
        "--on-error",
        choices=mosaic.ON_ERROR_MODES,
        default="placeholder",
        help="What to do with an image that can't be read or plotted: include a "
        "captioned placeholder in its place (placeholder, the default), leave it "
        "out (skip), or exit with an error (strict).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set logging level to DEBUG.",
    )

    args = parser.parse_args()

    if args.debug:
        # The whole package, not just this module, so mosaic's debug lines show.
        logging.getLogger(__package__).setLevel(logging.DEBUG)

    if args.json_input:
        try:
            with open(args.json_input, "r") as file:
                files_dict = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            parser.error(f"could not read --json-input: {e}")

        if not isinstance(files_dict, dict) or not all(
            isinstance(file_list, list) for file_list in files_dict.values()
        ):
            parser.error("--json-input must map datatype names to lists of image paths")
        if not files_dict:
            parser.error(f"{args.json_input} contains no datatypes")
    elif not args.dataset:
        parser.error("dataset required unless --json-input present")
    else:
        files_dict = None

    if args.max_img_height:
        mosaic.MAX_IMG_HEIGHT = args.max_img_height
    if args.max_img_width:
        mosaic.MAX_IMG_WIDTH = args.max_img_width

    if args.out_file:
        out_file = args.out_file
    else:
        if args.dataset:
            in_abs = os.path.abspath(args.dataset)
        else:
            in_abs = os.path.abspath(args.json_input)
            in_abs = os.path.splitext(in_abs)[0]
        out_file = os.path.basename(in_abs) + "_mosaic.pdf"

    try:
        if not args.png_in_dir:
            create_mosaic_pdf(
                args.dataset,
                out_file,
                anat=args.anat,
                png_out_dir=args.png_out_dir,
                downsample=args.downsample,
                freesurfer=args.freesurfer,
                metadata=args.metadata,
                files_dict=files_dict,
                on_error=args.on_error,
            )
        else:
            logger.info(f"Creating pdf at {out_file}")
            create_pdf(args.png_in_dir, out_file, args.metadata)
    except MosaicError as e:
        logger.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
