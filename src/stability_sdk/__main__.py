#!/bin/which python3

# fmt: off

import pathlib
import sys
import os
import uuid
import random
import io
import logging
import time
import mimetypes

import grpc
from argparse import ArgumentParser, Namespace
from typing import Dict, Generator, List, Optional, Union, Any, Sequence, Tuple
from google.protobuf.json_format import MessageToJson
from PIL import Image

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    pass
else:
    load_dotenv()

thisPath = pathlib.Path(__file__).parent.resolve()
genPath = thisPath / "interfaces/gooseai/generation"
sys.path.append(str(genPath))

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

from stability_sdk.client import (
    StabilityInference,
)
from stability_sdk.utils import (
    SAMPLERS,
    MAX_FILENAME_SZ,
    truncate_fit,
    get_sampler_from_str,
    open_images,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

# Set up logging for output to console.
fh = logging.StreamHandler()
fh_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(filename)s(%(process)d) - %(message)s"
)
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

STABILITY_HOST = os.getenv("STABILITY_HOST", "grpc.stability.ai:443")
STABILITY_KEY = os.getenv("STABILITY_KEY", "")

if not STABILITY_HOST:
    logger.warning("STABILITY_HOST environment variable needs to be set.")
    sys.exit(1)

if not STABILITY_KEY:
    logger.warning(
        "STABILITY_KEY environment variable needs to be set. You may"
        " need to login to the Stability website to obtain the"
        " API key."
    )
    sys.exit(1)
    
    
    
# CLI parsing
parser = ArgumentParser()
parser.add_argument(
    "--height", "-H", type=int, default=512, help="[512] height of image"
)
parser.add_argument(
    "--width", "-W", type=int, default=512, help="[512] width of image"
)
parser.add_argument(
    "--start_schedule",
    type=float,
    default=0.5,
    help="[0.5] start schedule for init image (must be greater than 0, 1 is full strength text prompt, no trace of image)",
)
parser.add_argument(
    "--end_schedule",
    type=float,
    default=0.01,
    help="[0.01] end schedule for init image",
)
parser.add_argument(
    "--cfg_scale", "-C", type=float, default=7.0, help="[7.0] CFG scale factor"
)
parser.add_argument(
    "--sampler",
    "-A",
    type=str,
    default="k_lms",
    help="[k_lms] (" + ", ".join(SAMPLERS.keys()) + ")",
)
parser.add_argument(
    "--steps", "-s", type=int, default=50, help="[50] number of steps"
)
parser.add_argument("--seed", "-S", type=int, default=0, help="random seed to use")
parser.add_argument(
    "--prefix",
    "-p",
    type=str,
    default="generation_",
    help="output prefixes for artifacts",
)
parser.add_argument(
    "--no-store", action="store_true", help="do not write out artifacts"
)
parser.add_argument(
    "--num_samples", "-n", type=int, default=1, help="number of samples to generate"
)
parser.add_argument("--show", action="store_true", help="open artifacts using PIL")
parser.add_argument(
    "--engine",
    "-e",
    type=str,
    help="engine to use for inference",
    default="stable-diffusion-v1-5",
)
parser.add_argument(
    "--init_image",
    "-i",
    type=str,
    help="Init image",
)
parser.add_argument(
    "--mask_image",
    "-m",
    type=str,
    help="Mask image",
)
parser.add_argument("prompt", nargs="*")

args = parser.parse_args()
if not args.prompt and not args.init_image:
    logger.warning("prompt or init image must be provided")
    parser.print_help()
    sys.exit(1)
else:
    args.prompt = " ".join(args.prompt)

if args.init_image:
    args.init_image = Image.open(args.init_image)

if args.mask_image:
    args.mask_image = Image.open(args.mask_image)

request =  {
    "height": cli_args.height,
    "width": cli_args.width,
    "start_schedule": cli_args.start_schedule,
    "end_schedule": cli_args.end_schedule,
    "cfg_scale": cli_args.cfg_scale,
    "sampler": get_sampler_from_str(cli_args.sampler),
    "steps": cli_args.steps,
    "seed": cli_args.seed,
    "samples": cli_args.num_samples,
    "init_image": cli_args.init_image,
    "mask_image": cli_args.mask_image,
}


stability_api = StabilityInference(
    STABILITY_HOST, STABILITY_KEY, engine=args.engine, verbose=True
)

answers = stability_api.generate(args.prompt, **request)
artifacts = process_artifacts_from_answers(
    args.prefix, args.prompt, answers, write=not args.no_store, verbose=True
)
if args.show:
    for artifact in open_images(artifacts, verbose=True):
        pass
else:
    for artifact in artifacts:
        pass
