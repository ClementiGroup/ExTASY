
## Conda Environment Setup on XSEDE Bridges

### Python2 remote RCT installation on Bridges <a href="rct_install"></a>

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


### Run on Bridges login node (Client Inside)

Extasy can be located in Bridges which will be one of login nodes e.g. `[user@login018 ~]`. This will require `ssh` to connect to Bridges using PSC credentials like `gsissh` to connect from outside. This needs to exercise the ssh key registration at PSC described below.

- Setting up a PSC password if not done before, this is not XSEDE password. (https://apr.psc.edu/autopwdreset/autopwdreset.html)
- If your PSC account is not active or not created, contact to bridges@psc.edu.
- Create a ssh key pair or use existing one to submit a key (public key, i.e. `.pub` file extension) to PSC (https://grants.psc.edu/cgi-bin/ssh/listKeys.pl)
   - Note that the key has to be passwordless
- Administrator reviews/authorizes a key registration in 1-2 days

Once a key registration is complete, setup conda or virtualenv on Bridges login nodes: go back to the [installation](#rct_install)
- place the private key (passphraseless) in `~/.ssh/` with a name `id_rsa`
    - use a vim/nano/emacs editor for a quick copy and paste of key strings
- change the permission to `read/write for owner only` by
    - ```chmod 600 ~/.ssh/id_rsa```
- Verify ssh passwordless connection by
    - ```ssh bridges```

Last step is to specify `ssh` as a protocol instead of `gsissh` as we run Extasy inside of Bridges.
- Change the script i.e. `extasy_tica_bridges.py`
- `access_schema` has to be `ssh` in the `res_dict` e.g. line 270 and 283 in the script, `gsissh` only works with myproxy credentials
