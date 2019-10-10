
## Conda Environment Setup on XSEDE Bridges

### Python2 remote RCT installation on Bridges

```
module load anaconda2
source /opt/packages/anaconda/anaconda2-5.2.0/etc/profile.d/conda.sh
conda create -n entk python=2.7.15
conda activate entk
pip install radical.entk
```


### Python3 remote installation on Bridges
```
module load cuda/10.1
module load anaconda3
source /opt/packages/anaconda/anaconda3-5.2.0/etc/profile.d/conda.sh
conda create -n vampextasy
conda activate vampextasy
conda install -y -c omnia openmm
conda install -y -c omnia mdtraj
conda install -y -c conda-forge pyemma
conda install -y -c anaconda scikit-learn
conda install -y -c pip
pip install tensorflow-gpu
```

## check installation of Python2 on Bridges
```
module load anaconda2
source /opt/packages/anaconda/anaconda2-5.2.0/etc/profile.d/conda.sh
conda create -n entk python=2.7.15
conda activate entk
radical-stack
```

## check installation of Python3 on Bridges
interactive on GPU node
```
module load cuda/10.1
module load anaconda3
source /opt/packages/anaconda/anaconda3-5.2.0/etc/profile.d/conda.sh
conda activate vampextasy
python -c 'import tensorflow as tf; print(tf.__version__)'
python -m simtk.testInstallation
```

## Configuration - RabbitMQ, MongoDB

```
export RMQ_HOSTNAME=two.radical-project.org
export RMQ_PORT=33239
(optional) export RADICAL_PILOT_DBURL='mongodb://'
```


## Run

```
module load anaconda2
source /opt/packages/anaconda/anaconda2-5.2.0/etc/profile.d/conda.sh
conda activate entk
python extasy_tica_bridges.py --Kconfig settings_extasy_tica3_chignolin_long_bridges.wcfg 2>&1 | tee log_extasy_tica_chignolin_long_bridges.log
```

Other instructions are omitted as described in the `README.md`

