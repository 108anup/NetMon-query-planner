## Reference code for Heterosketch (NSDI 22)

Anup Agarwal, Zaoxing Liu, and Srinivasan Seshan. 
"HeteroSketch: Coordinating network-wide monitoring in heterogeneous and dynamic networks."
19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22). 2022.


### Disclaimer

Please note that the code is a research prototype, has dead/old code snippets, and is not properly documented. Please reach out to me if you have any questions. My contact information can be found at [my webpage](https://www.cs.cmu.edu/~anupa/).

### Citation
If you find our work helpful, please cite using:
```
@inproceedings {276972,
author = {Anup Agarwal and Zaoxing Liu and Srinivasan Seshan},
title = {{HeteroSketch}: Coordinating Network-wide Monitoring in Heterogeneous and Dynamic Networks},
booktitle = {19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22)},
year = {2022},
isbn = {978-1-939133-27-4},
address = {Renton, WA},
pages = {719--741},
url = {https://www.usenix.org/conference/nsdi22/presentation/agarwal},
publisher = {USENIX Association},
month = apr
}
```


## Information

The optimizer code resides in the `gurobi` directory (exp is an old bruteforce implementation attempt at the optimizer). 
The optimizer produces the placement of sketches across devices given monitoring requirements. 
Our main focus was on heterogeneity (variation between different devices) instead of dynamics (variation over time). 
For dynamics we speed up the optimizer (using clustering) so that it can be rerun. We also updated the optimizer to only recompute placement for a subset of devices (`app-dynamics` branch has this feature).

`device_models` directory has the code that constructs the device profiles given measurement data. This repository does not include the code to run the measurement experiments. Please contact me if you need this.
If you are just interested in the clustering part, there is also the following work which is very similar to ours: https://www.usenix.org/system/files/nsdi21-abuzaid.pdf. Their code is available at: https://github.com/netcontract/ncflow.
Please let me know if you have any questions.

## Dependencies
I used [conda](https://docs.conda.io/en/latest/) to manage python packages.
```
conda install networkx numpy matplotlib scipy pytest
conda install -c conda-forge orderedset hdbscan scikit-learn scikit-learn-extra
conda install gurobi -c https://conda.anaconda.org/gurobi
```
You will also need to obtain a [license](https://www.gurobi.com/solutions/licensing/) to run gurobi. I was able to obtain a free [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Running the code and reproducing evaluation results (section 7.2)

I used the following commands to run the optimizer. In the `gurobi` directory:

### First time setup:
```
# Install dependencies (see above). There may be some dependencies I skipped, please refer to requirements.txt to install others.
./download.sh  # This downloads topology-zoo
mkdir -p pickle_objs
```

### Optimizer run:
```
python main.py -i 45 -v -s Netmon -v --parallel  # This performs a single run of the optimizer.
python main.py -i 1 -v -s Netmon -v --parallel
# -i points to an input number in the input_generator.py file which has example inputs. The overlay flag in the input specifies the clustering scheme.
# -s selects the placement scheme. For legacy reasons HeteroSketch in the paper is called Netmon in the code. The Baseline from the paper is called Univmon in the code, and Greedy is called UnivmonGreedy and UnivmonGreedyRows.
# config.py configures various parameters used by the optimizer including what objective to optimize for.
```

### Figure 7
```
# IIRC, I used commands similar to following to get the data for Figure7 (a) and (b) respectively:
pytest -s tests/test_topology.py::test_vary_topology --pdb
pytest -s tests/test_scale.py::test_scale_clos --pdb
```

## Reproducing dynamics evaluation (section 7.3)
For the dynamics part (recompute placement for a subset of devices), I used a separate branch of code (`app-dynamics`).
In the `gurobi` directory:

Basic setup:
```
mkdir -p pickle_objs
```

### Figure 9:
```
pytest -s tests/test_clustering.py::test_flow_dynamics --pdb
pytest -s tests/test_clustering.py::test_full_rerun_flow_dynamics --pdb
```

### Figure 10:
For this, I changed the `pytest` parameters in above 2 tests to vary the number of pods, and keep the number of change events to 1.
