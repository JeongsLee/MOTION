from netCDF4 import Dataset
import os
f="/code-vol/data/poseidon/Wave-Layer/solution_0.nc"
with Dataset(f) as nc:
    print("dims:", {k:v.size for k,v in nc.dimensions.items()})
    for vn,v in nc.variables.items():
        if v.ndim>=3: print(f"var '{vn}': shape={v.shape} dims={v.dimensions}")
