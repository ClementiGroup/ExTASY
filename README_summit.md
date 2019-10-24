
# Environment Setup on OLCF Summit

## Conda env & modules

```
conda create -n vampextasy python=2.7.15
conda activate vampextasy
module load cuda/9.1.85
module load cmake
module load gcc/7.4.0
```

## Required Packages

```
pip install radical.entk
conda install -y swig
conda install -y numpy
conda install -y cython
conda install -y scipy
conda install -y h5py
```

## OpenMM

- Set destination to install (Change to preferred location)

```
export openmm_install_path=$HOME/.conda/envs/vampextasy
```

```bash
git clone https://github.com/pandegroup/openmm.git
cd openmm
mkdir -p build_openmm
cd build_openmm
cmake .. -DCUDA_HOST_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/gcc \
         -DCUDA_SDK_ROOT_DIR=/sw/summit/cuda/9.1.85/samples \
         -DCUDA_TOOLKIT_ROOT_DIR=/sw/summit/cuda/9.1.85/ \
         -DCMAKE_CXX_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/g++ \
         -DCMAKE_C_COMPILER=/sw/summitdev/gcc/7.1.1-20170802/bin/gcc \
         -DCMAKE_INSTALL_PREFIX=${openmm_install_path}

make -j 40      
make install     
make PythonInstall         
```

### Verification

```
python -m simtk.testInstallation
```

Sample output
```
OpenMM Version: 7.4
Git Revision: 493a228775ba7a325db8ee3a676946978db3a7d2

There are 3 Platforms available:

1 Reference - Successfully computed forces
2 CPU - Successfully computed forces
3 CUDA - Successfully computed forces

Median difference in forces between platforms:

Reference vs. CPU: 1.9434e-06
Reference vs. CUDA: 6.72374e-06
CPU vs. CUDA: 6.2079e-06

All differences are within tolerance.
```

## TensorFlow-GPU

```
conda install -y tensorflow-gpu
```

## MDAnalysis

- (Optional) if numpy version < 1.16

```
pip install --upgrade numpy
```

```
pip install MDAnalysis MDAnalysisTests
```

### Verification

```
python -c 'import MDAnalysis as mda; print (mda.__version__)'
```

Sample output
```
0.19.2
```

## MSMTools

```
pip install msmtools
```

## verify entk
```
radical-stack
```
