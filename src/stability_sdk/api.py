import grpc
import io
import logging
import random
import time

from google.protobuf.struct_pb2 import Struct
from PIL import Image
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import stability_sdk.interfaces.gooseai.dashboard.dashboard_pb2 as dashboard
import stability_sdk.interfaces.gooseai.dashboard.dashboard_pb2_grpc as dashboard_grpc
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

from .utils import (
    image_mix,
    image_to_prompt,
    tensor_to_prompt,
)


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def open_channel(host: str, api_key: str = None, max_message_len: int = 20*1024*1024) -> grpc.Channel:
    options=[
        ('grpc.max_send_message_length', max_message_len),
        ('grpc.max_receive_message_length', max_message_len),
    ]    
    if host.endswith(":443"):
        call_credentials = [grpc.access_token_call_credentials(api_key)]
        channel_credentials = grpc.composite_channel_credentials(
            grpc.ssl_channel_credentials(), *call_credentials
        )
        channel = grpc.secure_channel(host, channel_credentials, options=options)
    else:
        channel = grpc.insecure_channel(host, options=options)
    return channel


class ClassifierException(Exception):
    """Raised when server classifies generated content as inappropriate.

    Attributes:
        classifier_result: Categories the result image exceeded the threshold for
        prompt: The prompt that was classified as inappropriate
    """    
    def __init__(self, classifier_result: Optional[generation.ClassifierParameters]=None, prompt: Optional[str]=None):
        self.classifier_result = classifier_result
        self.prompt = prompt

class OutOfCreditsException(Exception):
    """Raised when account doesn't have enough credits to perform a request."""
    def __init__(self, details: str):
        self.details = details


class Endpoint:
    def __init__(self, stub, engine_id):
        self.stub = stub
        self.engine_id = engine_id


class Context:
    def __init__(
            self, 
            host: str="", 
            api_key: str=None, 
            stub: generation_grpc.GenerationServiceStub=None,
            generate_engine_id: str="stable-diffusion-xl-1024-v1-0",
            inpaint_engine_id: str="stable-inpainting-512-v2-0",
            interpolate_engine_id: str="interpolation-server-v1",
            transform_engine_id: str="transform-server-v1",
            upscale_engine_id: str="esrgan-v1-x2plus",
        ):
        if not host and stub is None:
            raise Exception("Must provide either GRPC host or stub to Api")

        channel = open_channel(host, api_key) if host else None
        if not stub:
            stub = generation_grpc.GenerationServiceStub(channel)

        self._dashboard_stub = dashboard_grpc.DashboardServiceStub(channel) if channel else None

        self._generate = Endpoint(stub, generate_engine_id)
        self._inpaint = Endpoint(stub, inpaint_engine_id)
        self._interpolate = Endpoint(stub, interpolate_engine_id)
        self._transform = Endpoint(stub, transform_engine_id)
        self._upscale = Endpoint(stub, upscale_engine_id)

        self._debug_no_chains = False
        self._max_retries = 5             # retry request on RPC error
        self._request_timeout = 30.0      # timeout in seconds for each request
        self._retry_delay = 1.0           # base delay in seconds between retries, each attempt will double
        self._retry_obfuscation = False   # retry request with different seed on classifier obfuscation
        self._retry_schedule_offset = 0.1 # increase schedule start by this amount on each retry after the first

        self._user_organization_id: Optional[str] = None
        self._user_profile_picture: str = ''

    def generate(
        self,
        prompts: List[str], 
        weights: List[float], 
        width: int = 1024, 
        height: int = 1024, 
        steps: Optional[int] = None,
        seed: Union[Sequence[int], int] = 0,
        samples: int = 1,
        cfg_scale: float = 7.0, 
        sampler: generation.DiffusionSampler = None,
        init_image: Optional[Image.Image] = None,
        init_strength: float = 0.0,
        init_noise_scale: Optional[float] = None,
        init_depth: Optional[Image.Image] = None,
        mask: Optional[Image.Image] = None,
        masked_area_init: generation.MaskedAreaInit = generation.MASKED_AREA_INIT_ORIGINAL,
        guidance_preset: generation.GuidancePreset = generation.GUIDANCE_PRESET_NONE,
        guidance_cuts: int = 0,
        guidance_strength: float = 0.0,
        preset: Optional[str] = None,
        return_request: bool = False,
    ) -> Dict[int, List[Any]]:
        """
        Generate an image from a set of weighted prompts.

        :param prompts: List of text prompts
        :param weights: List of prompt weights
        :param width: Width of the generated image
        :param height: Height of the generated image
        :param steps: Number of steps to run the diffusion process
        :param seed: Random seed for the starting noise
        :param samples: Number of samples to generate
        :param cfg_scale: Classifier free guidance scale
        :param sampler: Sampler to use for the diffusion process
        :param init_image: Initial image to use
        :param init_strength: Strength of the initial image
        :param init_noise_scale: Scale of the initial noise
        :param mask: Mask to use (0 for pixels to change, 255 for pixels to keep)
        :param masked_area_init: How to initialize the masked area
        :param guidance_preset: Preset to use for CLIP guidance
        :param guidance_cuts: Number of cuts to use with CLIP guidance
        :param guidance_strength: Strength of CLIP guidance
        :param preset: Style preset to use
        :param return_request: Whether to return the request instead of running it
        :return: dict mapping artifact type to data
        """
        if not prompts and init_image is None:
            raise ValueError("prompt and/or init_image must be provided")

        if (mask is not None) and (init_image is None) and not return_request:
            raise ValueError("If mask_image is provided, init_image must also be provided")

        p = [generation.Prompt(text=prompt, parameters=generation.PromptParameters(weight=weight)) for prompt,weight in zip(prompts, weights)]
        if init_image is not None:
            p.append(image_to_prompt(init_image))
        if mask is not None:
            p.append(image_to_prompt(mask, type=generation.ARTIFACT_MASK))
        if init_depth is not None:
            p.append(image_to_prompt(init_depth, type=generation.ARTIFACT_DEPTH))

        start_schedule = 1.0 - init_strength
        image_params = self._build_image_params(width, height, sampler, steps, seed, samples, cfg_scale, 
                                                start_schedule, init_noise_scale, masked_area_init, 
                                                guidance_preset, guidance_cuts, guidance_strength)

        extras = Struct()
        if preset and preset.lower() != 'none':
            extras.update({ '$IPC': { "preset": preset } })

        request = generation.Request(engine_id=self._generate.engine_id, prompt=p, image=image_params, extras=extras)
        if return_request:
            return request

        results = self._run_request(self._generate, request)

        return results

    def get_user_info(self) -> Tuple[float, str]:
        """Get the number of credits the user has remaining and their profile picture."""
        if not self._user_organization_id:
            user = self._dashboard_stub.GetMe(dashboard.EmptyRequest())
            self._user_profile_picture = user.profile_picture
            self._user_organization_id = user.organizations[0].organization.id
        organization = self._dashboard_stub.GetOrganization(dashboard.GetOrganizationRequest(id=self._user_organization_id))
        return organization.payment_info.balance * 100, self._user_profile_picture

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompts: List[str], 
        weights: List[float], 
        steps: Optional[int] = None, 
        seed: Union[Sequence[int], int] = 0,
        samples: int = 1,
        cfg_scale: float = 7.0, 
        sampler: generation.DiffusionSampler = None,
        init_strength: float = 0.0,
        init_noise_scale: Optional[float] = None,
        masked_area_init: generation.MaskedAreaInit = generation.MASKED_AREA_INIT_ZERO,
        guidance_preset: generation.GuidancePreset = generation.GUIDANCE_PRESET_NONE,
        guidance_cuts: int = 0,
        guidance_strength: float = 0.0,
        preset: Optional[str] = None,
    ) -> Dict[int, List[Any]]:
        """
        Apply inpainting to an image.
        
        :param image: Source image
        :param mask: Mask image with 0 for pixels to change and 255 for pixels to keep
        :param prompts: List of text prompts
        :param weights: List of prompt weights
        :param steps: Number of steps to run
        :param seed: Random seed
        :param samples: Number of samples to generate
        :param cfg_scale: Classifier free guidance scale        
        :param sampler: Sampler to use for the diffusion process
        :param init_strength: Strength of the initial image
        :param init_noise_scale: Scale of the initial noise
        :param masked_area_init: How to initialize the masked area
        :param guidance_preset: Preset to use for CLIP guidance
        :param guidance_cuts: Number of cuts to use with CLIP guidance
        :param guidance_strength: Strength of CLIP guidance
        :param preset: Style preset to use
        :return: dict mapping artifact type to data
        """
        p = [generation.Prompt(text=prompt, parameters=generation.PromptParameters(weight=weight)) for prompt,weight in zip(prompts, weights)]
        p.append(image_to_prompt(image))
        p.append(image_to_prompt(mask, type=generation.ARTIFACT_MASK))

        width, height = image.size
        start_schedule = 1.0-init_strength
        image_params = self._build_image_params(width, height, sampler, steps, seed, samples, cfg_scale, 
                                                start_schedule, init_noise_scale, masked_area_init, 
                                                guidance_preset, guidance_cuts, guidance_strength)

        extras = Struct()
        if preset and preset.lower() != 'none':
            extras.update({ '$IPC': { "preset": preset } })

        request = generation.Request(engine_id=self._inpaint.engine_id, prompt=p, image=image_params, extras=extras)        
        results = self._run_request(self._inpaint, request)

        return results

    def interpolate(
        self,
        images: Sequence[Image.Image], 
        ratios: List[float],
        mode: generation.InterpolateMode = generation.INTERPOLATE_LINEAR,
    ) -> List[Image.Image]:
        """
        Interpolate between two images

        :param images: Two images with matching resolution
        :param ratios: In-between ratios to interpolate at
        :param mode: Interpolation mode
        :return: One image for each ratio
        """
        assert len(images) == 2
        assert len(ratios) >= 1

        if len(ratios) == 1:
            if ratios[0] == 0.0:
                return [images[0]]
            elif ratios[0] == 1.0:
                return [images[1]]
            elif mode == generation.INTERPOLATE_LINEAR:
               return [image_mix(images[0], images[1], ratios[0])]

        p = [image_to_prompt(image) for image in images]
        request = generation.Request(
            engine_id=self._interpolate.engine_id,
            prompt=p,
            interpolate=generation.InterpolateParameters(ratios=ratios, mode=mode)
        )

        results = self._run_request(self._interpolate, request)
        return results[generation.ARTIFACT_IMAGE]

    def transform_and_generate(
        self,
        image: Optional[Image.Image],
        params: List[generation.TransformParameters],
        generate_request: generation.Request,
        extras: Optional[Dict] = None,
    ) -> Image.Image:
        extras_struct = None
        if extras is not None:
            extras_struct = Struct()
            extras_struct.update(extras)

        if not params:
            results = self._run_request(self._generate, generate_request)
            return results[generation.ARTIFACT_IMAGE][0]

        assert image is not None
        requests = [
            generation.Request(
                engine_id=self._transform.engine_id,
                requested_type=generation.ARTIFACT_TENSOR,
                prompt=[image_to_prompt(image)],
                transform=param,
                extras=extras_struct,
            ) for param in params
        ]

        if self._debug_no_chains:
            prev_result = None
            for rq in requests:
                if prev_result is not None:
                    rq.prompt.pop()
                    rq.prompt.append(tensor_to_prompt(prev_result))
                prev_result = self._run_request(self._transform, rq)[generation.ARTIFACT_TENSOR][0]
            generate_request.prompt.append(tensor_to_prompt(prev_result))
            results = self._run_request(self._generate, generate_request)
        else:
            stages = []
            for idx, rq in enumerate(requests):
                stages.append(generation.Stage(
                    id=str(idx),
                    request=rq, 
                    on_status=[generation.OnStatus(
                        action=[generation.STAGE_ACTION_PASS], 
                        target=str(idx+1)
                    )]
                ))
            stages.append(generation.Stage(
                id=str(len(params)),
                request=generate_request,
                on_status=[generation.OnStatus(
                    action=[generation.STAGE_ACTION_RETURN],
                    target=None
                )]
            ))
            chain_rq = generation.ChainRequest(request_id="xform_gen_chain", stage=stages)
            results = self._run_request(self._transform, chain_rq)

        return results[generation.ARTIFACT_IMAGE][0]

    def transform(
        self,
        images: Sequence[Image.Image],
        params: Union[generation.TransformParameters, List[generation.TransformParameters]],
        extras: Optional[Dict] = None
    ) -> Tuple[List[Image.Image], Optional[List[Image.Image]]]:
        """
        Transform images

        :param images: One or more images to transform
        :param params: Transform operations to apply to each image
        :return: One image artifact for each image and one transform dependent mask
        """
        assert len(images)
        assert isinstance(images[0], Image.Image)

        extras_struct = None
        if extras is not None:
            extras_struct = Struct()
            extras_struct.update(extras)

        if isinstance(params, List) and len(params) > 1:
            if self._debug_no_chains:
                for param in params:
                    images, mask = self.transform(images, param, extras)
                return images, mask

            assert extras is None
            stages = []
            for idx, param in enumerate(params):
                final = idx == len(params) - 1
                rq = generation.Request(
                    engine_id=self._transform.engine_id,
                    prompt=[image_to_prompt(image) for image in images] if idx == 0 else None,
                    transform=param,
                    extras_struct=extras_struct
                )
                stages.append(generation.Stage(
                    id=str(idx),
                    request=rq, 
                    on_status=[generation.OnStatus(
                        action=[generation.STAGE_ACTION_PASS if not final else generation.STAGE_ACTION_RETURN], 
                        target=str(idx+1) if not final else None
                    )]
                ))
            chain_rq = generation.ChainRequest(request_id="xform_chain", stage=stages)
            results = self._run_request(self._transform, chain_rq)
        else:
            request = generation.Request(
                engine_id=self._transform.engine_id,
                prompt=[image_to_prompt(image) for image in images],
                transform=params[0] if isinstance(params, List) else params,
                extras=extras_struct
            )
            results = self._run_request(self._transform, request)

        images = results.get(generation.ARTIFACT_IMAGE, []) + results.get(generation.ARTIFACT_DEPTH, [])
        masks = results.get(generation.ARTIFACT_MASK, None)
        return images, masks

    def transform_3d(
        self, 
        images: Sequence[Image.Image], 
        depth_calc: generation.TransformParameters,
        transform: generation.TransformParameters,
        extras: Optional[Dict] = None
    ) -> Tuple[List[Image.Image], Optional[List[Image.Image]]]:
        assert len(images)
        assert isinstance(images[0], Image.Image)

        image_prompts = [image_to_prompt(image) for image in images]
        warped_images = []
        warp_mask = None
        op_id = "resample" if transform.HasField("resample") else "camera_pose"

        extras_struct = Struct()
        if extras is not None:
            extras_struct.update(extras)

        rq_depth = generation.Request(
            engine_id=self._transform.engine_id,
            requested_type=generation.ARTIFACT_TENSOR,
            prompt=[image_prompts[0]],
            transform=depth_calc,
        )
        rq_transform = generation.Request(
            engine_id=self._transform.engine_id,
            prompt=image_prompts,
            transform=transform,
            extras=extras_struct
        )

        if self._debug_no_chains:
            results = self._run_request(self._transform, rq_depth)
            rq_transform.prompt.append(
                generation.Prompt(
                    artifact=generation.Artifact(
                        type=generation.ARTIFACT_TENSOR,
                        tensor=results[generation.ARTIFACT_TENSOR][0]
                    )
                )
            )
            results = self._run_request(self._transform, rq_transform)
        else:
            chain_rq = generation.ChainRequest(
                request_id=f"{op_id}_3d_chain",
                stage=[
                    generation.Stage(
                        id="depth_calc",
                        request=rq_depth,
                        on_status=[generation.OnStatus(action=[generation.STAGE_ACTION_PASS], target=op_id)]
                    ),
                    generation.Stage(
                        id=op_id,
                        request=rq_transform,
                        on_status=[generation.OnStatus(action=[generation.STAGE_ACTION_RETURN])]
                    ) 
                ])
            results = self._run_request(self._transform, chain_rq)

        warped_images = results[generation.ARTIFACT_IMAGE]
        warp_mask = results.get(generation.ARTIFACT_MASK, None)

        return warped_images, warp_mask
    
    def upscale(
        self,
        init_image: Image.Image,
        width: Optional[int] = None,
        height: Optional[int] = None,
        prompt: Union[str, generation.Prompt] = None,
        steps: Optional[int] = 20,
        cfg_scale: Optional[float] = 7.0,
        seed: int = 0
    ) -> Image.Image:
        """
        Upscale an image.

        :param init_image: Image to upscale.

        Optional parameters for upscale method:

        :param width: Width of the output images.
        :param height: Height of the output images.
        :param prompt: Prompt used in text conditioned models
        :param steps: Number of diffusion steps
        :param cfg_scale: Intensity of the prompt, when a prompt is used
        :param seed: Seed for the random number generator.

        Some variables are not used for specific engines, but are included for consistency.

        Variables ignored in ESRGAN engines: prompt, steps, cfg_scale, seed

        :return: Tuple of (prompts, image_parameters)
        """

        prompts = [image_to_prompt(init_image)]
        if prompt:
            if isinstance(prompt, str):
                prompt = generation.Prompt(text=prompt)
            elif not isinstance(prompt, generation.Prompt):
                raise ValueError("prompt must be a string or Prompt object")
            prompts.append(prompt)

        request = generation.Request(
            engine_id=self._upscale.engine_id,
            prompt=prompts, 
            image=generation.ImageParameters(
                width=width,
                height=height,
                seed=[seed],
                steps=steps,
                parameters=[generation.StepParameter(
                    sampler=generation.SamplerParameters(cfg_scale=cfg_scale)
                )],
            )
        )
        results = self._run_request(self._upscale, request)
        return results[generation.ARTIFACT_IMAGE][0]

    def _adjust_request_engine(self, request: generation.Request):
        if request.engine_id == self._transform.engine_id:
            assert request.HasField("transform")
            if request.transform.HasField("color_adjust") or \
                (request.transform.HasField("resample") and len(request.transform.resample.transform.data) == 9):
                request.engine_id = self._transform.engine_id + "-cpu"

    def _adjust_request_for_retry(self, request: generation.Request, attempt: int):
        logger.warning(f"  adjusting request, will retry {self._max_retries-attempt} more times")
        request.image.seed[:] = [random.randrange(0, 4294967295) for _ in request.image.seed]
        if attempt > 0 and request.image.parameters and request.image.parameters[0].HasField("schedule"):
            schedule = request.image.parameters[0].schedule
            if schedule.HasField("start"):
                schedule.start = max(0.0, min(1.0, schedule.start + self._retry_schedule_offset))

    def _build_image_params(self, width, height, sampler, steps, seed, samples, cfg_scale, 
                            schedule_start, init_noise_scale, masked_area_init, 
                            guidance_preset, guidance_cuts, guidance_strength):

        if not seed:
            seed = [random.randrange(0, 4294967295)]
        elif isinstance(seed, int):
            seed = [seed]
        else:
            seed = list(seed)

        step_parameters = {
            "scaled_step": 0,
            "sampler": generation.SamplerParameters(cfg_scale=cfg_scale, init_noise_scale=init_noise_scale),
        }
        if schedule_start != 1.0:
            step_parameters["schedule"] = generation.ScheduleParameters(start=schedule_start)

        if guidance_preset is not generation.GUIDANCE_PRESET_NONE:
            cutouts = generation.CutoutParameters(count=guidance_cuts) if guidance_cuts else None
            if guidance_strength == 0.0:
                guidance_strength = None
            step_parameters["guidance"] = generation.GuidanceParameters(
                guidance_preset=guidance_preset,
                instances=[
                    generation.GuidanceInstanceParameters(
                        cutouts=cutouts,
                        guidance_strength=guidance_strength,
                        models=None, prompt=None
                    )
                ]
            )

        return generation.ImageParameters(
            transform=None if sampler is None else generation.TransformType(diffusion=sampler),
            height=height,
            width=width,
            seed=seed,
            steps=steps,
            samples=samples,
            masked_area_init=masked_area_init,
            parameters=[generation.StepParameter(**step_parameters)],
        )

    def _process_response(self, response) -> Dict[int, List[Any]]:
        results: Dict[int, List[Any]] = {}
        for resp in response:
            for artifact in resp.artifacts:
                # check for classifier rejecting a text prompt
                if artifact.finish_reason == generation.FILTER and artifact.type == generation.ARTIFACT_TEXT:
                    raise ClassifierException(prompt=artifact.text)

                if artifact.type not in results:
                    results[artifact.type] = []

                if artifact.type == generation.ARTIFACT_CLASSIFICATIONS:
                    results[artifact.type].append(artifact.classifier)
                elif artifact.type in (generation.ARTIFACT_DEPTH, generation.ARTIFACT_IMAGE, generation.ARTIFACT_MASK):
                    image = Image.open(io.BytesIO(artifact.binary))
                    results[artifact.type].append(image)
                elif artifact.type == generation.ARTIFACT_TENSOR:
                    results[artifact.type].append(artifact.tensor)
                elif artifact.type == generation.ARTIFACT_TEXT:
                    results[artifact.type].append(artifact.text)

        return results

    def _run_request(
        self, 
        endpoint: Endpoint, 
        request: Union[generation.ChainRequest, generation.Request]
    ) -> Dict[int, List[Any]]:        
        if isinstance(request, generation.Request):
            self._adjust_request_engine(request)
        elif isinstance(request, generation.ChainRequest):
            for stage in request.stage:
                self._adjust_request_engine(stage.request)

        for attempt in range(self._max_retries+1):
            try:
                if isinstance(request, generation.Request):
                    response = endpoint.stub.Generate(request, timeout=self._request_timeout)
                else:
                    response = endpoint.stub.ChainGenerate(request, timeout=self._request_timeout)

                results = self._process_response(response)

                # check for classifier obfuscation
                if generation.ARTIFACT_CLASSIFICATIONS in results:
                    for classifier in results[generation.ARTIFACT_CLASSIFICATIONS]:
                        if classifier.realized_action == generation.ACTION_OBFUSCATE:
                            raise ClassifierException(classifier)

                break
            except ClassifierException as ce:
                if attempt == self._max_retries or not self._retry_obfuscation or ce.prompt is not None:
                    raise ce
                
                for exceed in ce.classifier_result.exceeds:
                    logger.warning(f"Received classifier obfuscation. Exceeded {exceed.name} threshold")
                
                if isinstance(request, generation.Request) and request.HasField("image"):
                    self._adjust_request_for_retry(request, attempt)
                elif isinstance(request, generation.ChainRequest):
                    for stage in request.stage:
                        if stage.request.HasField("image"):
                            self._adjust_request_for_retry(stage.request, attempt)
                else:
                    raise ce
            except grpc.RpcError as rpc_error:
                if hasattr(rpc_error, "code"):
                    if rpc_error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                        if "message larger than max" in rpc_error.details():
                            raise rpc_error
                        raise OutOfCreditsException(rpc_error.details())
                    elif rpc_error.code() == grpc.StatusCode.UNAUTHENTICATED:
                        raise rpc_error

                if attempt == self._max_retries:
                    raise rpc_error

                logger.warning(f"Received RpcError: {rpc_error} will retry {self._max_retries-attempt} more times")
                time.sleep(self._retry_delay * 2**attempt)
        return results
