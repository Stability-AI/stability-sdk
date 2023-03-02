# stability-sdk

This document contains usage expectations not contained in the main README.md

# Generation

These examples generate and put PNGs in your current directory.

Command line:
`python3 -m stability_sdk generate -W 512 -H 512 "A stunning house."`

SDK Usage:

See usage demo notebooks in ./nbs

# Upscale 

Required inputs:
init_image

Optional inputs:
height
width

Additional requirements:
Max input size = 1048576 pixels (ie. the total pixels in a 1024 x 1024 image)
Max output size = 4194304 pixels (ie. the total pixels in a 2048 x 2048 image)

The default output size is set by the specific endpoint.
For example, upscale_engine == "esrgan-v1-x2plus" will upscale to 2x the input size

If height or width is provided, the original aspect ratio will be maintained.
Specifying both height and width will throw an error. This is to guarantee that the original aspect ratio is maintained.

Examples:

Command line:
`python3 -m stability_sdk upscale -i "/path/to/image.png"`
`python3 -m stability_sdk upscale -H 1200 -i "/path/to/image.png"`

SDK Usage:

See usage demo notebooks in ./nbs

