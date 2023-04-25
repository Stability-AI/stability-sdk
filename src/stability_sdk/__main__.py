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
    process_artifacts_from_answers,
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
subparsers = parser.add_subparsers(dest='command')

parser_upscale = subparsers.add_parser('upscale')
parser_upscale.add_argument(
    "--init_image",
    "-i",
    type=str,
    help="Init image",
    required=True
)
parser_upscale.add_argument(
    "--height", "-H", type=int, default=None, help="height of upscaled image"
)
parser_upscale.add_argument(
    "--width", "-W", type=int, default=None, help="width of upscaled image"
)
parser_upscale.add_argument(
    "--cfg_scale", "-C", type=float, default=7.0, help="[7.0] CFG scale factor"
)
parser_upscale.add_argument(
    "--steps", "-s", type=int, default=None, help="[auto] number of steps"
)
parser_upscale.add_argument(
    "--seed", "-S", type=int, default=0, help="random seed to use"
)
parser_upscale.add_argument(
    "--prefix",
    "-p",
    type=str,
    default="upscale_",
    help="output prefixes for artifacts",
)
parser_upscale.add_argument(
    "--artifact_types",
    "-t",
    action='append',
    type=str,
    help="filter artifacts by type (ARTIFACT_IMAGE, ARTIFACT_TEXT, ARTIFACT_CLASSIFICATIONS, etc)"
)
parser_upscale.add_argument(
    "--no-store", action="store_true", help="do not write out artifacts"
)
parser_upscale.add_argument(
    "--show", action="store_true", help="open artifacts using PIL"
)
parser_upscale.add_argument(
    "--engine",
    "-e",
    type=str,
    help="engine to use for upscale",
    default="esrgan-v1-x2plus",
)
parser_upscale.add_argument(
    "prompt", nargs="*"
)


parser_generate = subparsers.add_parser('generate')
parser_generate.add_argument(
    "--height", "-H", type=int, default=512, help="[512] height of image"
)
parser_generate.add_argument(
    "--width", "-W", type=int, default=512, help="[512] width of image"
)
parser_generate.add_argument(
    "--start_schedule",
    type=float,
    default=0.5,
    help="[0.5] start schedule for init image (must be greater than 0, 1 is full strength text prompt, no trace of image)",
)
parser_generate.add_argument(
    "--end_schedule",
    type=float,
    default=0.01,
    help="[0.01] end schedule for init image",
)
parser_generate.add_argument(
    "--cfg_scale", "-C", type=float, default=7.0, help="[7.0] CFG scale factor"
)
parser_generate.add_argument(
    "--sampler",
    "-A",
    type=str,
    help="[auto-select] (" + ", ".join(SAMPLERS.keys()) + ")",
)
parser_generate.add_argument(
    "--steps", "-s", type=int, default=None, help="[auto] number of steps"
)
parser_generate.add_argument(
    "--seed", "-S", type=int, default=0, help="random seed to use"
)
parser_generate.add_argument(
    "--prefix",
    "-p",
    type=str,
    default="generation_",
    help="output prefixes for artifacts",
)
parser_generate.add_argument(
    "--artifact_types",
    "-t",
    action='append',
    type=str,
    help="filter artifacts by type (ARTIFACT_IMAGE, ARTIFACT_TEXT, ARTIFACT_CLASSIFICATIONS, etc)"
)
parser_generate.add_argument(
    "--no-store", action="store_true", help="do not write out artifacts"
)
parser_generate.add_argument(
    "--num_samples", "-n", type=int, default=1, help="number of samples to generate"
)
parser_generate.add_argument(
    "--show", action="store_true", help="open artifacts using PIL"
)
parser_generate.add_argument(
    "--engine",
    "-e",
    type=str,
    help="engine to use for inference",
    default="stable-diffusion-xl-beta-v2-2-2",
)
parser_generate.add_argument(
    "--init_image",
    "-i",
    type=str,
    help="Init image",
)
parser_generate.add_argument(
    "--mask_image",
    "-m",
    type=str,
    help="Mask image",
)
parser_generate.add_argument("prompt", nargs="*")

# handle backwards compatibility, default command to generate
input_args = sys.argv[1:]
command = None
if len(input_args)>0:
    command = input_args[0]
if command not in subparsers.choices.keys() and command != '-h' and command != '--help':
    logger.warning(f"command {command} not recognized, defaulting to 'generate'")
    logger.warning(
    "[Deprecation Warning] The method you have used to invoke the sdk will be deprecated shortly."
    "[Deprecation Warning] Please modify your code to call the sdk with the following syntax:"
    "[Deprecation Warning] python -m stability_sdk <command> <args>"
    "[Deprecation Warning] Where <command> is one of: upscale, generate"
    )
    input_args = ['generate'] + input_args
    
args = parser.parse_args(input_args)

if args.command == "upscale":
    args.init_image = Image.open(args.init_image)
    if not args.prompt:
        args.prompt = [""]
    args.prompt = " ".join(args.prompt)

    request =  {
        "height": args.height,
        "width": args.width,
        "init_image": args.init_image,
        "steps": args.steps,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "prompt": args.prompt,
        }
    stability_api = StabilityInference(
        STABILITY_HOST, STABILITY_KEY, upscale_engine=args.engine, verbose=True
    )
    answers = stability_api.upscale(**request)
    artifacts = process_artifacts_from_answers(
        args.prefix, args.prompt, answers, write=not args.no_store, verbose=True,
        filter_types=args.artifact_types,
    )
elif args.command == "generate":
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
        "height": args.height,
        "width": args.width,
        "start_schedule": args.start_schedule,
        "end_schedule": args.end_schedule,
        "cfg_scale": args.cfg_scale,
        "samples": args.num_samples,
        "init_image": args.init_image,
        "mask_image": args.mask_image,
    }

    if args.sampler:
        request["sampler"] = get_sampler_from_str(args.sampler)
    if args.seed and args.seed > 0:
        request["seed"] = args.seed

    stability_api = StabilityInference(
        STABILITY_HOST, STABILITY_KEY, engine=args.engine, verbose=True
    )

    answers = stability_api.generate(args.prompt, **request)
    artifacts = process_artifacts_from_answers(
        args.prefix, args.prompt, answers, write=not args.no_store, verbose=True,
        filter_types=args.artifact_types,
    )

if args.show:
    for artifact in open_images(artifacts, verbose=True):
        pass
else:
    for artifact in artifacts:
        pass
