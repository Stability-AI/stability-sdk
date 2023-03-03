# stability-sdk

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stability-ai/stability-sdk/blob/main/nbs/demo_colab.ipynb)

Client implementations that interact with the Stability Generator API

# Installation

Install the PyPI package via:

`pip install stability-sdk`

# Getting an API key
You can manage API keys in your dreamstudio account [here](https://beta.dreamstudio.ai/membership?tab=apiKeys)

# Python client

`client.py` is both a command line client and an API class that wraps the gRPC based API. To try the client:

- Use Python venv: `python3 -m venv pyenv`
- Set up in venv dependencies: `pyenv/bin/pip3 install -e .`
- `pyenv/bin/activate` to use the venv.
- Set the `STABILITY_HOST` environment variable. This is by default set to the production endpoint `grpc.stability.ai:443`.
- Set the `STABILITY_KEY` environment variable.

Then to invoke:

`python3 -m stability_sdk generate -W 512 -H 512 "A stunning house."`

It will generate and put PNGs in your current directory.

To upscale:
`python3 -m stability_sdk upscale -i "/path/to/image.png"`

## SDK Usage

See usage demo notebooks in ./nbs

## Command line usage

```
usage: python -m stability_sdk generate [-h] [--height HEIGHT] [--width WIDTH] [--start_schedule START_SCHEDULE]
                 [--end_schedule END_SCHEDULE] [--cfg_scale CFG_SCALE] [--sampler SAMPLER]
                 [--steps STEPS] [--seed SEED] [--prefix PREFIX] [--engine ENGINE]
                 [--num_samples NUM_SAMPLES] [--artifact_types ARTIFACT_TYPES]
                 [--no-store] [--show] [--init_image INIT_IMAGE] [--mask_image MASK_IMAGE]
                 [prompt ...]

positional arguments:
  prompt

options:
  -h, --help            show this help message and exit
  --height HEIGHT, -H HEIGHT
                        [512] height of image
  --width WIDTH, -W WIDTH
                        [512] width of image
  --start_schedule START_SCHEDULE
                        [0.5] start schedule for init image (must be greater than 0, 1 is full strength
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

options:
  -h, --help            show this help message and exit
  --init_image INIT_IMAGE, -i INIT_IMAGE
                        Init image
  --height HEIGHT, -H HEIGHT
                        height of upscaled image in pixels
  --width WIDTH, -W WIDTH
                        width of upscaled image in pixels
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

The `src` subdirectory contains pre-compiled gRPC stubs for the following languages:

- [Javascript/Typescript](https://github.com/Stability-AI/stability-sdk/tree/main/src/js)

If a language you would like to connect to the API with is not listed above, you can use the following
protobuf definition to compile stubs for your language:

- [protobuf spec](https://github.com/Stability-AI/api-interfaces/blob/main/src/proto/)

## Community-contributed clients

* Typescript client: https://github.com/jakiestfu/stability-ts
* Guide to building for Ruby: https://github.com/kmcphillips/stability-sdk/blob/main/src/ruby/README.md

## DreamStudio API TOS

Unless otherwise specified, usage of the dreamstudio API falls under the same usage terms as the dreamstudio web interface: 

* https://beta.dreamstudio.ai/terms-of-service
