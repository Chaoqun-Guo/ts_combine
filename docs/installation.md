## How to install ts_combine

step-1: clone ts_combine.
```
git clone https://github.com/Chaoqun-Guo/ts_combine.git
cd ts_combine
```
step-2: update submodules for source codes, you can also ignore this step and jump to step-3 for just using the python package insteade of installing these packages through source codes.
```
git submodule update --init repos/darts
git submodule update --init repos/Merlion
git submodule update --init repos/Prophet
git submodule update --init repos/sktime
git submodule update --init repos/tsfresh
git submodule update --init repos/tslearn
```
step-3: using conda to create a new python environment and install dependencises for ts_combine.
```
conda create -n ts_combine
conda activate ts_combine 
pip install -r requirements.txt
```