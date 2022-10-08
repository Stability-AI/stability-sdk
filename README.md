# stability-clients

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

`python3 -m stability_sdk.client -W 512 -H 512 "A stunning house."`

It will generate and put PNGs in your current directory.

## SDK Usage

See usage demo notebooks in ./nbs

## Command line usage

```
usage: python -m stability_sdk.client [-h] [--height HEIGHT] [--width WIDTH]
	  								  [--cfg_scale CFG_SCALE] [--sampler SAMPLER] [--steps STEPS]
									  [--seed SEED] [--prefix PREFIX] [--no-store]
									  [--num_samples NUM_SAMPLES] [--show]
									  prompt [prompt ...]

positional arguments:
  prompt

optional arguments:
  -h, --help            show this help message and exit
  --height HEIGHT, -H HEIGHT
                        [512] height of image
  --width WIDTH, -W WIDTH
                        [512] width of image
  --cfg_scale CFG_SCALE, -C CFG_SCALE
                        [7.0] CFG scale factor
  --sampler SAMPLER, -A SAMPLER
                        [k_lms] (ddim, plms, k_euler, k_euler_ancestral,
                        k_heun, k_dpm_2, k_dpm_2_ancestral, k_lms)
  --steps STEPS, -s STEPS
                        [50] number of steps
  --seed SEED, -S SEED  random seed to use
  --prefix PREFIX, -p PREFIX
                        output prefixes for artifacts
  --no-store            do not write out artifacts
  --num_samples NUM_SAMPLES, -n NUM_SAMPLES
                        number of samples to generate
  --show                open artifacts using PIL
  --engine, -e          engine to use for inference
```


## Connecting to the API in using languages other than python

The `src` subdirectory contains pre-compiled gRPC stubs for the following languages:

- [Javascript/Typescript](https://github.com/Stability-AI/stability-sdk/tree/main/src/js)

And guides for the following languages:

* [Ruby](https://github.com/Stability-AI/stability-sdk/tree/main/src/ruby/README.md)

If a language you would like to connect to the API with is not listed above, you can use the following
protobuf definition to compile stubs for your language:

- [protobuf spec](https://github.com/Stability-AI/api-interfaces/blob/main/src/proto/)

## Community-contributed clients

* Typescript client: https://github.com/jakiestfu/stability-ts
* Guide to building for Ruby: https://github.com/kmcphillips/stability-sdk/blob/main/src/ruby/README.md

## DreamStudio API TOS

Unless otherwise specified, usage of the dreamstudio API falls under the same usage terms as the dreamstudio web interface: 

* https://beta.dreamstudio.ai/terms-of-service
