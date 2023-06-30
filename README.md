# stability-sdk

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stability-ai/stability-sdk/blob/main/nbs/demo_colab.ipynb)

Client implementations that interact with the Stability API. 

## Getting an API key

Follow the [instructions](https://platform.stability.ai/docs/getting-started/authentication) on [Platform](https://platform.stability.ai) to obtain an API key.

## PyPI Package Installation

Install the [PyPI](https://pypi.org/project/stability-sdk/) package via:

`pip install stability-sdk`

## Python Client

`client.py` is both a command line client and an API class that wraps the gRPC based API. To try the client:

- Use Python venv: `python3 -m venv pyenv`
- Set up in venv dependencies: `pyenv/bin/pip3 install -e .`
- `pyenv/bin/activate` to use the venv.
- Set the `STABILITY_HOST` environment variable. This is by default set to the production endpoint `grpc.stability.ai:443`.
- Set the `STABILITY_KEY` environment variable.

Then to invoke:

`python3 -m stability_sdk generate -W 1024 -H 1024 "A stunning house."`

It will generate and put PNGs in your current directory.

To upscale:
`python3 -m stability_sdk upscale -i "/path/to/image.png"`

## Animation UI

Install with 
`pip install stability-sdk[anim_ui]`

Then run with 
`python3 -m stability_sdk animate --gui`

## SDK Usage

Be sure to check out [Platform](https://platform.stability.ai) for comprehensive documentation on how to interact with our API.

## Command line usage

```
usage: python -m stability_sdk generate [-h] [--height HEIGHT] [--width WIDTH] 
                [--start_schedule START_SCHEDULE] [--end_schedule END_SCHEDULE] 
                [--cfg_scale CFG_SCALE] [--sampler SAMPLER] [--steps STEPS] 
                [--style_preset STYLE_PRESET] [--seed SEED] [--prefix PREFIX] [--engine ENGINE]
                [--num_samples NUM_SAMPLES] [--artifact_types ARTIFACT_TYPES]
                [--no-store] [--show] [--init_image INIT_IMAGE] [--mask_image MASK_IMAGE]
                [prompt ...]

positional arguments:
  prompt

options:
  -h, --help            show this help message and exit
  --height HEIGHT, -H HEIGHT
                        [1024] height of image
  --width WIDTH, -W WIDTH
                        [1024] width of image
  --start_schedule START_SCHEDULE
                        [0.5] start schedule for init image (must be greater than 0; 1 is full strength
                        text prompt, no trace of image)
  --end_schedule END_SCHEDULE
                        [0.01] end schedule for init image
  --cfg_scale CFG_SCALE, -C CFG_SCALE
                        [7.0] CFG scale factor
  --sampler SAMPLER, -A SAMPLER
                        [auto-select] (ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2,
                        k_dpm_2_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_2s_ancestral)
  --steps STEPS, -s STEPS
                        [auto] number of steps
  --style_preset STYLE_PRESET
                        [none] (3d-model, analog-film, anime, cinematic, comic-book, digital-art, enhance, 
                        fantasy-art, isometric, line-art, low-poly, modeling-compound, neon-punk, origami, 
                        photographic, pixel-art, tile-texture)
  --seed SEED, -S SEED  random seed to use
  --prefix PREFIX, -p PREFIX
                        output prefixes for artifacts
  --artifact_types ARTIFACT_TYPES, -t ARTIFACT_TYPES
                        filter artifacts by type (ARTIFACT_IMAGE, ARTIFACT_TEXT, ARTIFACT_CLASSIFICATIONS, etc)
  --no-store            do not write out artifacts
  --num_samples NUM_SAMPLES, -n NUM_SAMPLES
                        number of samples to generate
  --show                open artifacts using PIL
  --engine ENGINE, -e ENGINE
                        engine to use for inference
  --init_image INIT_IMAGE, -i INIT_IMAGE
                        Init image
  --mask_image MASK_IMAGE, -m MASK_IMAGE
                        Mask image
```
For upscale:
```
usage: client.py upscale
       [-h]
       --init_image INIT_IMAGE
       [--height HEIGHT] [--width WIDTH] [--prefix PREFIX] [--artifact_types ARTIFACT_TYPES]
       [--no-store] [--show] [--engine ENGINE]

positional arguments:
  prompt (ignored in esrgan engines)

options:
  -h, --help            show this help message and exit
  --init_image INIT_IMAGE, -i INIT_IMAGE
                        Init image
  --height HEIGHT, -H HEIGHT
                        height of upscaled image in pixels
  --width WIDTH, -W WIDTH
                        width of upscaled image in pixels
  --steps STEPS, -s STEPS
                        [auto] number of steps (ignored in esrgan engines)
  --seed SEED, -S SEED  random seed to use (ignored in esrgan engines)
  --cfg_scale CFG_SCALE, -C CFG_SCALE
                        [7.0] CFG scale factor (ignored in esrgan engines)
  --prefix PREFIX, -p PREFIX
                        output prefixes for artifacts
  --artifact_types ARTIFACT_TYPES, -t ARTIFACT_TYPES
                        filter artifacts by type (ARTIFACT_IMAGE, ARTIFACT_TEXT, ARTIFACT_CLASSIFICATIONS, etc)
  --no-store            do not write out artifacts
  --show                open artifacts using PIL
  --engine ENGINE, -e ENGINE
                        engine to use for upscale
  
```


## Connecting to the API using languages other than Python

If a language you would like to connect to the API with is not currently documented on [Platform](https://platform.stability.ai) you can use the following
protobuf definition to compile stubs for your language:

- [protobuf spec](https://github.com/Stability-AI/api-interfaces/blob/main/src/proto/)

## Community-contributed clients

* Typescript client: https://github.com/jakiestfu/stability-ts
* Guide to building for Ruby: https://github.com/kmcphillips/stability-sdk/blob/main/src/ruby/README.md
* C# client: https://github.com/Katarzyna-Kadziolka/StabilityClient.Net

## Stability API TOS

Usage of the Stability API falls under the [STABILITY AI API Terms of Service.
](https://platform.stability.ai/docs/terms-of-service)
