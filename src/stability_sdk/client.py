#!/bin/which python3
import pathlib
import sys
import os
import uuid
import random
import io
import logging
import time

import grpc
from argparse import ArgumentParser, Namespace
from typing import Dict, Generator, List, Union, Any, Sequence, Tuple
from dotenv import load_dotenv

load_dotenv()

thisPath = pathlib.Path(__file__).parent.resolve()
genPath = thisPath / "interfaces/gooseai/generation"
sys.path.append(str(genPath))

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

algorithms: Dict[str, int] = {
    "ddim": generation.SAMPLER_DDIM,
    "plms": generation.SAMPLER_DDPM,
    "k_euler": generation.SAMPLER_K_EULER,
    "k_euler_ancestral": generation.SAMPLER_K_EULER_ANCESTRAL,
    "k_heun": generation.SAMPLER_K_HEUN,
    "k_dpm_2": generation.SAMPLER_K_DPM_2,
    "k_dpm_2_ancestral": generation.SAMPLER_K_DPM_2_ANCESTRAL,
    "k_lms": generation.SAMPLER_K_LMS,
}


def get_sampler_from_str(s: str) -> generation.DiffusionSampler:
    """
    Convert a string to a DiffusionSampler enum.

    :param s: The string to convert.
    :return: The DiffusionSampler enum.
    """
    algorithm_key = s.lower().strip()
    algorithm = algorithms.get(algorithm_key, None)
    if algorithm is None:
        raise ValueError(f"unknown sampler {s}")
    return algorithm


def process_artifacts_from_answers(
    prefix: str,
    answers: Union[
        Generator[generation.Answer, None, None], Sequence[generation.Answer]
    ],
    write: bool = True,
    verbose: bool = False,
) -> Generator[Tuple[str, generation.Artifact], None, None]:
    """
    Process the Artifacts from the Answers.

    :param prefix: The prefix for the artifact filenames.
    :param answers: The Answers to process.
    :param write: Whether to write the artifacts to disk.
    :param verbose: Whether to print the artifact filenames.
    :return: A Generator of tuples of artifact filenames and Artifacts, intended
        for passthrough.
    """
    idx = 0
    for resp in answers:
        for artifact in resp.artifacts:
            img_p = f"{prefix}-{resp.request_id}-{resp.answer_id}-{idx}.png"
            if write:
                open(img_p, "wb").write(artifact.binary)
                if verbose:
                    logger.info(f"wrote {img_p}")
            yield [img_p, artifact]
            idx += 1


def open_images(
    images: Union[
        Sequence[Tuple[str, generation.Artifact]],
        Generator[Tuple[str, generation.Artifact], None, None],
    ]
) -> Generator[Tuple[str, generation.Artifact], None, None]:
    """
    Open the images from the filenames and Artifacts.

    :param images: The filenames and Artifacts to open.
    :return:  A Generator of tuples of image filenames and Artifacts, intended
     for passthrough.
    """
    from PIL import Image

    for path, artifact in images:
        img = Image.open(io.BytesIO(artifact.binary))
        img.show()
        yield [path, artifact]


class StabilityInference:
    def __init__(
        self,
        host: str = "grpc.stability.ai:443",
        key: str = "",
        engine: str = "stable-diffusion-v1",
        verbose=False,
    ):
        """
        Initialize the client.

        :param host: Host to connect to.
        :param key: Key to use for authentication.
        :param engine: Engine to use.
        :param verbose: Whether to print debug messages.
        """
        self.verbose = verbose
        self.engine = engine

        self.args = {"wait_for_ready": True}

        if verbose:
            logger.info(f"Opening channel to {host}")

        call_credentials = []

        if host.endswith("443"):
            if key:
                call_credentials.append(grpc.access_token_call_credentials(f"{key}"))
            else:
                raise ValueError(f"key is required for {host}")
            channel_credentials = grpc.composite_channel_credentials(
                grpc.ssl_channel_credentials(), *call_credentials
            )
            channel = grpc.secure_channel(host, channel_credentials)
        else:
            if key:
                logger.warning(
                    "Not using authentication token due to non-secure transport"
                )
            channel = grpc.insecure_channel(host)

        if verbose:
            logger.info(f"Channel opened to {host}")
        self.stub = generation_grpc.GenerationServiceStub(channel)

    def generate(
        self,
        prompt: Union[List[str], str],
        height: int = 512,
        width: int = 512,
        cfg_scale: float = 7.0,
        sampler: generation.DiffusionSampler = generation.SAMPLER_K_LMS,
        steps: int = 50,
        seed: Union[Sequence[int], int] = 0,
        samples: int = 1,
    ) -> Generator[generation.Answer, None, None]:
        """
        Generate images from a prompt.

        :param prompt: Prompt to generate images from.
        :param height: Height of the generated images.
        :param width: Width of the generated images.
        :param cfg_scale: Scale of the configuration.
        :param sampler: Sampler to use.
        :param steps: Number of steps to take.
        :param seed: Seed for the random number generator.
        :param samples: Number of samples to generate.
        :return: Generator of Answer objects.
        """
        if not prompt:
            raise ValueError("prompt must be provided")

        request_id = str(uuid.uuid4())

        if not seed:
            seed = [random.randrange(0, 4294967295)]

        if isinstance(prompt, str):
            prompt = [generation.Prompt(text=prompt)]
        else:
            prompt = [generation.Prompt(text=p) for p in prompt]

        rq = generation.Request(
            engine_id=self.engine,
            request_id=request_id,
            prompt=prompt,
            image=generation.ImageParameters(
                transform=generation.TransformType(diffusion=sampler),
                height=height,
                width=width,
                seed=seed,
                steps=steps,
                samples=samples,
                parameters=[
                    generation.StepParameter(
                        scaled_step=0,
                        sampler=generation.SamplerParameters(cfg_scale=cfg_scale),
                    )
                ],
            ),
        )

        if self.verbose:
            logger.info("Sending request.")

        start = time.time()
        for answer in self.stub.Generate(rq, **self.args):
            duration = time.time() - start
            if self.verbose:
                logger.info(f"Got {answer.answer_id} in {duration:0.2f}s")

            yield answer
            start = time.time()


def build_request_dict(cli_args: Namespace) -> Dict[str, Any]:
    """
    Build a Request arguments dictionary from the CLI arguments.
    """
    return {
        "height": cli_args.height,
        "width": cli_args.width,
        "cfg_scale": cli_args.cfg_scale,
        "sampler": get_sampler_from_str(cli_args.sampler),
        "steps": cli_args.steps,
        "seed": cli_args.seed,
        "samples": cli_args.num_samples,
    }


if __name__ == "__main__":
    # Set up logging for output to console.
    fh = logging.StreamHandler()
    fh_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s(%(process)d) - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    INFERENCE_HOST = os.getenv("STABILITY_HOST", "grpc.stability.ai:443")
    API_KEY = os.getenv("API_KEY", "")

    if not INFERENCE_HOST:
        logger.warning("STABILITY_HOST environment variable needs to be set.")
        sys.exit(1)

    if not API_KEY:
        logger.warning(
            "API_KEY environment variable needs to be set. You may"
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
        "--cfg_scale", "-C", type=float, default=7.0, help="[7.0] CFG scale factor"
    )
    parser.add_argument(
        "--sampler",
        "-A",
        type=str,
        default="k_lms",
        help="[k_lms] (" + ", ".join(algorithms.keys()) + ")",
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=50, help="[50] number of steps"
    )
    parser.add_argument("--seed", "-S", type=int, default=0, help="random seed to use")
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="output prefixes for artifacts",
        default="generation",
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
        default="stable-diffusion-v1",
    )
    parser.add_argument("prompt", nargs="+")

    args = parser.parse_args()
    if not args.prompt:
        logger.warning("prompt must be provided")
        parser.print_help()
        sys.exit(1)
    else:
        args.prompt = " ".join(args.prompt)

    request = build_request_dict(args)

    stability_api = StabilityInference(
        INFERENCE_HOST, API_KEY, engine=args.engine, verbose=True
    )

    answers = stability_api.generate(args.prompt, **request)
    artifacts = process_artifacts_from_answers(
        args.prefix, answers, write=not args.no_store, verbose=True
    )
    if args.show:
        open_images(artifacts)
    else:
        for image in artifacts:
            pass
