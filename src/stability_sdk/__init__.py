import pathlib, sys

# this is necessary because of how the auto-generated code constructs its imports
# should be a way to move this upstream
thisPath = pathlib.Path(__file__).parent.resolve()
genPath = thisPath / "interfaces/gooseai/generation"
tensPath = thisPath / "interfaces/src/tensorizer/tensors"
#sys.path.append(str(genPath))
sys.path.extend([str(genPath), str(tensPath)])