#!/bin/which python3

# fmt: off

import getpass
import grpc
import logging
import mimetypes
import os
import random
import sys
import time
import uuid

from argparse import ArgumentParser, Namespace
from google.protobuf.json_format import MessageToJson
from google.protobuf.struct_pb2 import Struct
from PIL import Image
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

from .api import open_channel
from .utils import (
    SAMPLERS,
    MAX_FILENAME_SZ,
    artifact_type_to_string,
    image_to_prompt,
    open_images,
    sampler_from_string,
    truncate_fit,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def process_artifacts_from_answers(
    prefix: str,
    prompt: str,
    answers: Union[
        Generator[generation.Answer, None, None], Sequence[generation.Answer]
    ],
    write: bool = True,
    verbose: bool = False,
    filter_types: Optional[List[str]] = None,
) -> Generator[Tuple[str, generation.Artifact], None, None]:
    """
    Process the Artifacts from the Answers.

    :param prefix: The prefix for the artifact filenames.
    :param prompt: The prompt to add to the artifact filenames.
    :param answers: The Answers to process.
    :param write: Whether to write the artifacts to disk.
    :param verbose: Whether to print the artifact filenames.
    :return: A Generator of tuples of artifact filenames and Artifacts, intended
        for passthrough.
    """
    idx = 0
    for resp in answers:
        for artifact in resp.artifacts:
            artifact_start = time.time()
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
            out_p = truncate_fit(prefix, prompt, ext, int(artifact_start), idx, MAX_FILENAME_SZ)
            is_allowed_type = filter_types is None or artifact_type_to_string(artifact.type) in filter_types
            if write:
                if is_allowed_type:
                    with open(out_p, "wb") as f:
                        f.write(bytes(contents))
                        if verbose:
                            logger.info(f"wrote {artifact_type_to_string(artifact.type)} to {out_p}")
                else:
                    if verbose:
                        logger.info(
                            f"skipping {artifact_type_to_string(artifact.type)} due to artifact type filter")

            yield (out_p, artifact)
            idx += 1


class StabilityInference:
    def __init__(
        self,
        host: str = "grpc.stability.ai:443",
        key: str = "",
        engine: str = "stable-diffusion-xl-1024-v1-0",
        upscale_engine: str = "esrgan-v1-x2plus",
        verbose: bool = False,
        wait_for_ready: bool = True,
    ):
        """
        Initialize the client.

        :param host: Host to connect to.
        :param key: Key to use for authentication.
        :param engine: Engine to use.
        :param upscale_engine: Upscale engine to use.
        :param verbose: Whether to print debug messages.
        :param wait_for_ready: Whether to wait for the server to be ready, or
            to fail immediately.
        """
        self.verbose = verbose
        self.engine = engine
        self.upscale_engine = upscale_engine

        self.grpc_args = {"wait_for_ready": wait_for_ready}
        if verbose:
            logger.info(f"Opening channel to {host}")

        call_credentials = []

        # Increase the max message size to 10MB to allow for larger images.
        max_message_size: int = os.getenv("MAX_MESSAGE_SIZE")
        if max_message_size is None:
            max_message_size = 10 * 1024 * 1024 # 10MB
        options = [
            ("grpc.max_send_message_length", int(max_message_size)),
            ("grpc.max_receive_message_length",int(max_message_size)),
        ]

        if host.endswith("443"):
            if key:
                call_credentials.append(grpc.access_token_call_credentials(f"{key}"))
            else:
                raise ValueError(f"key is required for {host}")
            channel_credentials = grpc.composite_channel_credentials(
                grpc.ssl_channel_credentials(), *call_credentials
            )
            channel = grpc.secure_channel(host, channel_credentials, options=options)
        else:
            if key:
                logger.warning(
                    "Not using authentication token due to non-secure transport"
                )
            channel = grpc.insecure_channel(host, options=options)

        if verbose:
            logger.info(f"Channel opened to {host}")
        self.stub = generation_grpc.GenerationServiceStub(channel)

    def generate(
        self,
        prompt: Union[str, List[str], generation.Prompt, List[generation.Prompt]],
        init_image: Optional[Image.Image] = None,
        mask_image: Optional[Image.Image] = None,
        height: int = 1024,
        width: int = 1024,
        start_schedule: float = 1.0,
        end_schedule: float = 0.01,
        cfg_scale: float = 7.0,
        sampler: generation.DiffusionSampler = None,
        steps: Optional[int] = None,
        seed: Union[Sequence[int], int] = 0,
        samples: int = 1,
        safety: bool = True,
        classifiers: Optional[generation.ClassifierParameters] = None,
        guidance_preset: generation.GuidancePreset = generation.GUIDANCE_PRESET_NONE,
        guidance_cuts: int = 0,
        guidance_strength: Optional[float] = None,
        guidance_prompt: Union[str, generation.Prompt] = None,
        guidance_models: List[str] = None,
        adapter_type: generation.T2IAdapter = None,
        adapter_strength: float = 0.4,
        adapter_init_type: generation.T2IAdapterInit = generation.T2IADAPTERINIT_IMAGE,
        style_preset: Optional[str] = None
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
        :param safety: DEPRECATED/UNUSED - Cannot be disabled.
        :param classifiers: DEPRECATED/UNUSED - Has no effect on image generation.
        :param guidance_preset: Guidance preset to use. See generation.GuidancePreset for supported values.
        :param guidance_cuts: Number of cuts to use for guidance.
        :param guidance_strength: Strength of the guidance. We recommend values in range [0.0,1.0]. A good default is 0.25
        :param guidance_prompt: Prompt to use for guidance, defaults to `prompt` argument (above) if not specified.
        :param guidance_models: Models to use for guidance.
        :param adapter_type: T2I adapter type, if any.
        :param adapter_strength: Float between 0, 1 representing the proportion of unet passes into which we inject adapter weights
        :param adapter_init_type: If T2IADAPTERINIT_IMAGE then init_image is converted into an initialising image corresponding to the adapter_type. i.e.
        a sketch/depthmap/canny edge. If T2IADAPTERINIT_ADAPTER_IMAGE, then the init_image is treated as already a a sketch/depthmap/canny edge.
        :param style_preset: Style preset name to use (see https://platform.stability.ai/docs/api-reference#tag/v1generation)
        :return: Generator of Answer objects.
        """
        if (prompt is None) and (init_image is None):
            raise ValueError("prompt and/or init_image must be provided")

        if (mask_image is not None) and (init_image is None):
            raise ValueError(
                "If mask_image is provided, init_image must also be provided"
            )

        if not seed:
            seed = [random.randrange(0, 4294967295)]
        elif isinstance(seed, int):
            seed = [seed]
        else:
            seed = list(seed)

        prompts: List[generation.Prompt] = []
        if any(isinstance(prompt, t) for t in (str, generation.Prompt)):
            prompt = [prompt]
        for p in prompt:
            if isinstance(p, str):
                p = generation.Prompt(text=p)
            elif not isinstance(p, generation.Prompt):
                raise TypeError("prompt must be a string or generation.Prompt object")
            prompts.append(p)

        step_parameters = dict(
            scaled_step=0,
            sampler=generation.SamplerParameters(cfg_scale=cfg_scale),
        )

        # NB: Specifying schedule when there's no init image causes washed out results
        if init_image is not None:
            step_parameters['schedule'] = generation.ScheduleParameters(
                start=start_schedule,
                end=end_schedule,
            )
            prompts += [image_to_prompt(init_image)]

            if mask_image is not None:
                prompts += [image_to_prompt(mask_image, type=generation.ARTIFACT_MASK)]


        if guidance_prompt:
            if isinstance(guidance_prompt, str):
                guidance_prompt = generation.Prompt(text=guidance_prompt)
            elif not isinstance(guidance_prompt, generation.Prompt):
                raise ValueError("guidance_prompt must be a string or Prompt object")
        if guidance_strength == 0.0:
            guidance_strength = None
        
        # Build our CLIP parameters
        if guidance_preset is not generation.GUIDANCE_PRESET_NONE:
            # to do: make it so user can override this
            step_parameters['sampler']=None

            if guidance_models:
                guiders = [generation.Model(alias=model) for model in guidance_models]
            else:
                guiders = None

            if guidance_cuts:
                cutouts = generation.CutoutParameters(count=guidance_cuts)
            else:
                cutouts = None

            step_parameters["guidance"] = generation.GuidanceParameters(
                guidance_preset=guidance_preset,
                instances=[
                    generation.GuidanceInstanceParameters(
                        guidance_strength=guidance_strength,
                        models=guiders,
                        cutouts=cutouts,
                        prompt=guidance_prompt,
                    )
                ],
            )

        adapter_parameters = generation.T2IAdapterParameter(
            adapter_type = adapter_type,
            adapter_strength = adapter_strength,
            adapter_init_type = adapter_init_type,
        )

        transform=None
        if sampler:
            transform=generation.TransformType(diffusion=sampler)

        image_parameters=generation.ImageParameters(
            transform=transform,
            height=height,
            width=width,
            seed=seed,
            steps=steps,
            samples=samples,
            adapter=adapter_parameters,
            parameters=[generation.StepParameter(**step_parameters)],
        )

        if style_preset and style_preset.lower() != 'none':
            extras = Struct()
            extras.update({ '$IPC': { "preset": style_preset } })
        else:
            extras = None

        return self.emit_request(prompt=prompts, image_parameters=image_parameters, extra_parameters=extras)
    
    def upscale(
        self,
        init_image: Image.Image,
        height: int = None,
        width: int = None,
        prompt: Union[str, generation.Prompt] = None,
        steps: Optional[int] = 20,
        cfg_scale: float = 7.0,
        seed: int = 0
    ) -> Generator[generation.Answer, None, None]:
        """
        Upscale an image.

        :param init_image: Image to upscale.

        Optional parameters for upscale method:

        :param height: Height of the output images.
        :param width: Width of the output images.
        :param prompt: Prompt used in text conditioned models
        :param steps: Number of diffusion steps
        :param cfg_scale: Intensity of the prompt, when a prompt is used
        :param seed: Seed for the random number generator.

        Some variables are not used for specific engines, but are included for consistency.

        Variables ignored in ESRGAN engines: prompt, steps, cfg_scale, seed

        :return: Tuple of (prompts, image_parameters)
        """

        step_parameters = dict(
            sampler=generation.SamplerParameters(cfg_scale=cfg_scale)
        )

        image_parameters=generation.ImageParameters(
            height=height,
            width=width,
            seed=[seed],
            steps=steps,
            parameters=[generation.StepParameter(**step_parameters)],
        )

        prompts = [image_to_prompt(init_image)]

        if prompt:
            if isinstance(prompt, str):
                prompt = generation.Prompt(text=prompt)
            elif not isinstance(prompt, generation.Prompt):
                raise ValueError("prompt must be a string or Prompt object")
            prompts.append(prompt)

        return self.emit_request(prompt=prompts, image_parameters=image_parameters, engine_id=self.upscale_engine)
    

    # The motivation here is to facilitate constructing requests by passing protobuf objects directly.
    def emit_request(
        self,
        prompt: generation.Prompt,
        image_parameters: generation.ImageParameters,
        extra_parameters: Optional[Struct] = None,
        engine_id: str = None,
        request_id: str = None,
    ):
        if not request_id:
            request_id = str(uuid.uuid4())
        if not engine_id:
            engine_id = self.engine

        rq = generation.Request(
            engine_id=engine_id,
            request_id=request_id,
            prompt=prompt,
            image=image_parameters,
            extras=extra_parameters
        )

        if self.verbose:
            logger.info("Sending request.")

        start = time.time()
        for answer in self.stub.Generate(rq, **self.grpc_args):
            duration = time.time() - start
            if self.verbose:
                if len(answer.artifacts) > 0:
                    artifact_ts = [
                        artifact_type_to_string(artifact.type)
                        for artifact in answer.artifacts
                    ]
                    logger.info(
                        f"Got answer {answer.answer_id} with artifact types {artifact_ts} in "
                        f"{duration:0.2f}s"
                    )
                else:
                    logger.info(
                        f"Got keepalive {answer.answer_id} in " f"{duration:0.2f}s"
                    )

            yield answer
            start = time.time()

def process_cli(
    logger: logging.Logger = None,
    warn_client_call_deprecated: bool = True,
):
    if not logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)

        # Set up logging for output to console.
        fh = logging.StreamHandler()
        fh_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(filename)s(%(process)d) - %(message)s"
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)
    
    if warn_client_call_deprecated:
        logger.warning(
            "[Deprecation Warning] The method you have used to invoke the sdk will be deprecated shortly."
            "[Deprecation Warning] Please modify your code to call the sdk without invoking the 'client' module instead."
            "[Deprecation Warning] rather than:"
            "[Deprecation Warning]    $ python -m stability_sdk.client ...  "
            "[Deprecation Warning] instead do this:"
            "[Deprecation Warning]    $ python -m stability_sdk ...  "
        )

    STABILITY_HOST = os.getenv("STABILITY_HOST", "grpc.stability.ai:443")
    STABILITY_KEY = os.getenv("STABILITY_KEY", "")

    if not STABILITY_KEY:
        print(
            "Please enter your API key from dreamstudio.ai or set the "
            "STABILITY_KEY environment variable to skip this prompt."
        )
        STABILITY_KEY = getpass.getpass("Enter your Stability API key: ")

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
        "--cfg_scale", "-C", type=float, default=7.0, help="[7.0] CFG scale factor (ignored in esrgan engines)"
    )
    parser_upscale.add_argument(
        "--steps", "-s", type=int, default=None, help="[20] number of steps (ignored in esrgan engines)"
    )
    parser_upscale.add_argument(
        "--seed", "-S", type=int, default=0, help="random seed to use (ignored in esrgan engines)"
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


    parser_animate = subparsers.add_parser('animate')
    parser_animate.add_argument("--gui", action="store_true", help="serve Gradio UI")
    parser_animate.add_argument("--share", action="store_true", help="create shareable UI link")
    parser_animate.add_argument("--output", "-o", type=str, default=".", help="root output folder")    
    

    parser_generate = subparsers.add_parser('generate')
    parser_generate.add_argument(
        "--height", "-H", type=int, default=1024, help="[1024] height of image"
    )
    parser_generate.add_argument(
        "--width", "-W", type=int, default=1024, help="[1024] width of image"
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
        "--seed", "-S", type=int, default=0, help="random seed to use")
    parser_generate.add_argument("--style_preset", type=str, help="style preset name")
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
    parser_generate.add_argument("--show", action="store_true", help="open artifacts using PIL")
    parser_generate.add_argument(
        "--engine",
        "-e",
        type=str,
        help="engine to use for inference",
        default="stable-diffusion-xl-1024-v1-0",
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
            "seed": args.seed,
            "samples": args.num_samples,
            "init_image": args.init_image,
            "mask_image": args.mask_image,
            "style_preset": args.style_preset,
        }

        if args.sampler:
            request["sampler"] = sampler_from_string(args.sampler)

        if args.steps:
            request["steps"] = args.steps

        stability_api = StabilityInference(
            STABILITY_HOST, STABILITY_KEY, engine=args.engine, verbose=True
        )
        answers = stability_api.generate(args.prompt, **request)
        artifacts = process_artifacts_from_answers(
            args.prefix, args.prompt, answers, write=not args.no_store, verbose=True,
            filter_types=args.artifact_types,
        )
    elif args.command == "animate":
        if args.gui:
            from .animation_ui import create_ui
            from .api import Context
            ui = create_ui(Context(STABILITY_HOST, STABILITY_KEY), args.output)
            ui.queue(concurrency_count=2, max_size=2)
            ui.launch(show_api=False, debug=True, height=768, share=args.share, show_error=True)
            sys.exit(0)
        else:
            logger.warning("animate must be invoked with --gui")
            sys.exit(1)

    
    if args.show:
        for artifact in open_images(artifacts, verbose=True):
            pass
    else:
        for artifact in artifacts:
            pass


if __name__ == "__main__":
    process_cli(logger)
