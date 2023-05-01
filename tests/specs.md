# stability-sdk

This document contains usage expectations not contained in the main README.md

# Generation

These examples generate and put PNGs in your current directory.

Command line:

`python3 -m stability_sdk generate -W 512 -H 512 "A stunning house."`

SDK Usage:

See usage demo notebooks in ./nbs

# Upscale 

## Engine selection
The upscale engine can be optionally chosen when initializing the client:

```
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], # API Key reference.
    upscale_engine="upscale_engine_name", # The name of the upscaling model we want to use.
)
```

Command line example:

`python3 -m stability_sdk upscale -e "upscale_engine_name" -i "/path/to/img.png"`

Default upscale_engine_name is "esrgan-v1-x2plus"

## Inputs
**Required inputs:**
```
init_image  
```

**Optional inputs:**
```
height  
width 
```

**These inputs are only used in the diffusion models (they are ignored for esrgan):**
```
prompt :     Prompt used in text conditioned models  (default = '')
steps :      Number of diffusion steps  (default = 20)
cfg_scale :  Intensity of the prompt (only when a prompt is used)  (default 7.0)
seed :       Seed for the random number generator  (default = 0 -> random)
```

## Additional requirements:
Max steps = 50

Max input size = 1048576 pixels (ie. the total pixels in a 1024 x 1024 image)  
Max output size = 4194304 pixels (ie. the total pixels in a 2048 x 2048 image)

The default output size is set by the specific endpoint.  
For example, upscale_engine == "esrgan-v1-x2plus" will upscale to 2x the input size


If height or width is provided, the original aspect ratio will be maintained.

Specifying both height and width will throw an error. This is so original aspect ratio is maintained.

For example:
```
# This is fine
answers = stability_api.upscale(
    init_image=img
) # results in a 2x image if using default upscale_engine

# This is fine
answers = stability_api.upscale(
    width=1000,
    init_image=img
)

# !! This will throw an error !!
answers = stability_api.upscale(
    width=1000,
    height=1000,
    init_image=img
)
```

## Example calls

Command line:

`python3 -m stability_sdk upscale -i "/path/to/image.png"`

`python3 -m stability_sdk upscale --engine "esrgan-v1-x2plus" -i "/path/to/image.png"`

`python3 -m stability_sdk upscale -H 1200 -i "/path/to/image.png"`

`python3 -m stability_sdk upscale -W 1200 -i "/path/to/image.png"`

SDK Usage:

See usage demo notebooks in ./nbs

