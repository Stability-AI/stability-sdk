import pathlib, sys

# this is necessary because of how the auto-generated code constructs its imports
# should be a way to move this upstream
this_path = pathlib.Path(__file__).parent.resolve()
sys.path.extend([
    str(this_path / "interfaces/gooseai/dashboard"), 
    str(this_path / "interfaces/gooseai/generation"), 
    str(this_path / "interfaces/gooseai/project"), 
    str(this_path / "interfaces/src/tensorizer/tensors")
])