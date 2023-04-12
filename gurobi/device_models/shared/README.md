# Structure
There are three directories for the three devices:
cpu, fpga, netro

# Dependencies
python3 scipy palettable numpy matplotlib pandas
Example commands to install dependencies using conda:
```
conda create -yn heterosketch python=3 scipy palettable numpy matplotlib pandas
conda activate heterosketch
```

# Generate Plots
Following commands generate plots and the device profiles based on micro-benchmarks:
```
cd cpu && python cpu-general.py beluga14-prefetch && cd ..
cd netro && python netro.py all-in-sandbox && cd ..
cd fpga && python fpga.py u280 && cd ..
```

Plots will be generated in the `plots` subdirectory within the benchmarking directories:
```
ls cpu/beluga14-prefetch/plots
ls netro/all-in-sandbox/plots
ls fpga/u280/plots
```

# Device Profiles
You can interpret the device profiles from the `model` function
in `cpu/cpu-general.py`, `netro/netro.py`, and `fpga/fpga.py`.
```
cat cpu/cpu-general.py | grep "def model"
cat netro/netro.py | grep "def model"
cat fpga/fpga.py | grep "def model"
```

FYI, I have shared code without UnivMon profiling for simplicity.
