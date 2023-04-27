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

from stability_sdk.client import (
    process_cli
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

process_cli(
    logger=logger,
    warn_client_call_deprecated=False
)