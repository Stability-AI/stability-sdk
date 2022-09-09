#!/bin/which python3
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
from typing import Dict, Generator, List, Union, Any, Sequence, Tuple
from dotenv import load_dotenv
from google.protobuf.json_format import MessageToJson
from PIL import Image

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

def image_to_prompt(im, init: bool = False, mask: bool = False) -> Tuple[str, generation.Prompt]:
    if init and mask:
        raise ValueError("init and mask cannot both be True")
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    buf.seek(0)
    if mask:
        return generation.Prompt(
            artifact=generation.Artifact(
                type=generation.ARTIFACT_MASK,
                binary=buf.getvalue()
            )
        )
    return generation.Prompt(
        artifact=generation.Artifact(
            type=generation.ARTIFACT_IMAGE,
            binary=buf.getvalue()
        ),
        parameters=generation.PromptParameters(
            init=init
        ),
    )

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
            artifact_p = f"{prefix}-{resp.request_id}-{resp.answer_id}-{idx}"
            if artifact.type == generation.ARTIFACT_IMAGE:
                ext = mimetypes.guess_extension(artifact.mime)
                contents = artifact.binary
            elif artifact.type == generation.ARTIFACT_CLASSIFICATIONS:
                ext = ".pb.json"
                contents = MessageToJson(artifact.classifier).encode("utf-8")
            elif artifact.type == generation.ARTIFACT_TEXT:
                ext = ".pb.json"
                contents = MessageToJson(artifact).encode("utf-8")
            else:
                ext = ".pb"
                contents = artifact.SerializeToString()
            out_p = f"{artifact_p}{ext}"
            if write:
                with open(out_p, "wb") as f:
                    f.write(bytes(contents))
                    if verbose:
                        artifact_t = generation.ArtifactType.Name(artifact.type)
                        logger.info(f"wrote {artifact_t} to {out_p}")

            yield [out_p, artifact]
            idx += 1


def open_images(
    images: Union[
        Sequence[Tuple[str, generation.Artifact]],
        Generator[Tuple[str, generation.Artifact], None, None],
    ],
    verbose: bool = False,
) -> Generator[Tuple[str, generation.Artifact], None, None]:
    """
    Open the images from the filenames and Artifacts tuples.

    :param images: The tuples of Artifacts and associated images to open.
    :return:  A Generator of tuples of image filenames and Artifacts, intended
     for passthrough.
    """
    from PIL import Image

    for path, artifact in images:
        if artifact.type == generation.ARTIFACT_IMAGE:
            if verbose:
                logger.info(f"opening {path}")
            img = Image.open(io.BytesIO(artifact.binary))
            img.show()
        yield [path, artifact]


class StabilityInference:
    def __init__(
        self,
        host: str = "grpc.stability.ai:443",
        key: str = "",
        engine: str = "stable-diffusion-v1-5",
        verbose: bool = False,
        wait_for_ready: bool = True,
    ):
        """
        Initialize the client.

        :param host: Host to connect to.
        :param key: Key to use for authentication.
        :param engine: Engine to use.
        :param verbose: Whether to print debug messages.
        :param wait_for_ready: Whether to wait for the server to be ready, or
            to fail immediately.
        """
        self.verbose = verbose
        self.engine = engine

        self.grpc_args = {"wait_for_ready": wait_for_ready}

        if verbose:
            logger.info(f"Opening channel to {host}")

        call_credentials = []

        if host.endswith("443"):
            if key:
                call_credentials.append(
                    grpc.access_token_call_credentials(f"{key}"))
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
        init_image: Image.Image = None,
        mask_image: Image.Image = None,
        height: int = 512,
        width: int = 512,
        start_schedule: float = 1.0,
        end_schedule: float = 0.01,
        cfg_scale: float = 7.0,
        sampler: generation.DiffusionSampler = generation.SAMPLER_K_LMS,
        steps: int = 50,
        seed: Union[Sequence[int], int] = 0,
        samples: int = 1,
        safety: bool = True,
        classifiers: generation.ClassifierParameters = None,
    ) -> Generator[generation.Answer, None, None]:
        """
        Generate images from a prompt.

        :param prompt: Prompt to generate images from.
        :param init_image: Init image.
        :param mask_image: Mask image
        :param height: Height of the generated images.
        :param width: Width of the generated images.
        :param start_schedule: Start schedule for init image.
        :param end_schedule: End schedule for init image.
        :param cfg_scale: Scale of the configuration.
        :param sampler: Sampler to use.
        :param steps: Number of steps to take.
        :param seed: Seed for the random number generator.
        :param samples: Number of samples to generate.
        :param safety: Whether to use safety mode.
        :param classifiers: Classifier parameters to use.
        :return: Generator of Answer objects.
        """
        if safety and classifiers is None:
            classifiers = generation.ClassifierParameters()

        if (prompt is None) and (init_image is None):
            raise ValueError("prompt and/or init_image must be provided")

        if (mask_image is not None) and (init_image is None):
            raise ValueError("If mask_image is provided, init_image must also be provided")

        request_id = str(uuid.uuid4())

        if not seed:
            seed = [random.randrange(0, 4294967295)]
        else:
            seed = [seed]

        if isinstance(prompt, str):
            prompt = [generation.Prompt(text=prompt)]
        elif isinstance(prompt, Sequence):
            prompt = [generation.Prompt(text=p) for p in prompt]
        else:
            raise TypeError("prompt must be a string or a sequence")

        if (init_image is not None):
            prompt += [image_to_prompt(init_image, init=True)]
            parameters = generation.StepParameter(
                    scaled_step=0,
                    sampler=generation.SamplerParameters(
                        cfg_scale=cfg_scale,
                    ),
                    schedule=generation.ScheduleParameters(
                        start=start_schedule,
                        end=end_schedule,
                    )
                ),
            if (mask_image is not None):
                prompt += [image_to_prompt(mask_image, mask=True)]
        else:
            parameters = generation.StepParameter(
                    scaled_step=0,
                    sampler=generation.SamplerParameters(
                        cfg_scale=cfg_scale),
                ),

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
                parameters=parameters,
            ),
            classifier=classifiers,
        )

        if self.verbose:
            logger.info("Sending request.")

        start = time.time()
        for answer in self.stub.Generate(rq, **self.grpc_args):
            duration = time.time() - start
            if self.verbose:
                if len(answer.artifacts) > 0:
                    artifact_ts = [
                        generation.ArtifactType.Name(artifact.type)
                        for artifact in answer.artifacts
                    ]
                    logger.info(
                        f"Got {answer.answer_id} with {artifact_ts} in "
                        f"{duration:0.2f}s"
                    )
                else:
                    logger.info(
                        f"Got keepalive {answer.answer_id} in "
                        f"{duration:0.2f}s"
                    )

            yield answer
            start = time.time()


def build_request_dict(cli_args: Namespace) -> Dict[str, Any]:
    """
    Build a Request arguments dictionary from the CLI arguments.
    """
    return {
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


if __name__ == "__main__":
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
        type=float, default=0.5, help="[0.5] start schedule for init image (must be greater than 0, 1 is full strength text prompt, no trace of image)"
    )
    parser.add_argument(
        "--end_schedule",
        type=float, default=0.01, help="[0.01] end schedule for init image"
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
        default="generation",
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
        "--init_image", "-i",
        type=str,
        help="Init image",
    )
    parser.add_argument(
        "--mask_image", "-m",
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

    request = build_request_dict(args)

    stability_api = StabilityInference(
        STABILITY_HOST, STABILITY_KEY, engine=args.engine, verbose=True
    )

    answers = stability_api.generate(args.prompt, **request)
    artifacts = process_artifacts_from_answers(
        args.prefix, answers, write=not args.no_store, verbose=True
    )
    if args.show:
        for artifact in open_images(artifacts, verbose=True):
            pass
    else:
        for artifact in artifacts:
            pass
