# stability-clients
Client implementations that interact with the Stability Generator API

# Python client

`client.py` is both a command line client and an API class that wraps the gRPC based API. To try the client:

* Use Python venv: `python3 -m venv pyenv`
* Set up in venv dependencies: `pyenv/bin/pip3 install -r requirements.txt`
* `pyenv/bin/enable` to use the venv.
* Set the `STABILITY_HOST` environment variable. This is by default set to the production endpoint `grpc.stability.ai:443`.
* Set the `API_KEY` environment variable.

Then to invoke:
* `python3 client.py -W 512 -H 512 "A stunning house."`

It will generate and put PNGs in your current directory.

## Usage
```
usage: client.py [-h] [--height HEIGHT] [--width WIDTH]
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
```
