# ExTASY (Adaptive Sampling)


## Local system python installation

Make a new conda python2.7 installation:
```
conda install -c conda-forge rabbitmq-server tmux pip git python=2.7.14
# Installing release - most cases
pip install radical.entk

# Installing specific branch of a repo
git clone radical.utils, saga-python, radical.pilot radical.entk, radical.analytics
cd <repo name>
git checkout <branch name>
git pull
pip install .
```

Check ```radical-stack``` to make sure radical packages are installed
Alternative Docker installation [here](https://radicalentk.readthedocs.io/en/latest/install.html#installing-rabbitmq-using-docker) provides two methods  how to install RabbitMQ.

## Setting up access to HPCs
Currently, all packages and permissions are setup for Blue Waters.
An password less connection to the HPC system is required. For Bluewaters GSISSH access has to be set up, described [here](https://bluewaters.ncsa.illinois.edu/user-guide). Instructions to setup gsissh access for Ubuntu can be 
found [here](https://github.com/vivek-bala/docs/blob/master/misc/gsissh_setup_stampede_ubuntu_xenial.sh/).
Please note that this has been tested only for xenial and trusty (for trusty, 
simple replace 'xenial' with 'trusty' in all commands). 
Test with this command, it should work without typing any password:
```
gsissh username@bw.ncsa.illinois.edu
```

## Installation on HPC

* no need to install anything on bluewaters, the python environment selected is publicly accesible 
* alternatively install on HPC an environment with pip and condaa an envirnment which runs all MD and analysis packages.

## Setup MongoDB
Radical Pilot requires a MongoDB, for example at [mlab](https://mlab.com/)

## Running instructions

Locally clone the current repository

```
git clone https://github.com/ClementiGroup/ExTASY.git
cd ExTASY
```

Next, you need to set a few environment variables, you can replace the RADICAL_PILOT_DBURL with your own 
```
export RADICAL_ENTK_VERBOSE=info
export GLOBUS_LOCATION='/usr/' #assuming gsissh is at /usr/bin/gsissh
export RADICAL_ENTK_PROFILE=True
export RADICAL_PILOT_PROFILE=True
export SAGA_PTY_SSH_TIMEOUT=300
export RADICAL_PILOT_DBURL='mongodb://...' # MongoDB you installed
```

Start the rabbitmq server

```
rabbitmq-server &
```

The behavior of the RabbitMQ server is visible under http://localhost:5672/#/ with login guest and password guest. If you need to restart the rabbitmq server type:
```
rabbitmqctl stop
rabbitmq-server &
```
## Notes 
The ```extasy_tica.py``` script contains the information about the application
execution workflow and the associated data movement. Please take a look at all
the comments to understand the various sections. 

## Executing the script

Change the walltime, allocation and cores you require in the called wcfg files. If you want to start a new adaptive sampling set start_iter to 0, if you want to extend the last adaptive sampling with more iterations set the parameter start_iter to the next iteration to run.  For longer simulations run inside tmux on an machine which can run undisturbed for long times: local system.


Execution command for Chignolin protein on GPUs on bluewaters:
```
python extasy_tica.py --Kconfig settings_extasy_tica3_chignolin_long.wcfg 2>&1 | tee log_extasy_tica_chignolin_long.log
```



## Results
* The output directory as specified in settings_*.wcfg files will have some output for each iteration.
* The full output on bluewaters on at "remote_output_directory" as set in settings_*.wcfg.


## Adapt to your own protein

* Copy the directory files-chignolin, replace the pdb structures and the xml files.
* Copy settings_extasy_tica3_chignolin_long.wcfg and change md_dir, md_input_file, md_reference.

## Adapt to your own MD engine 

* Based on helper_scrips/run-openmm-xml3.py generate a script for your MD engine.
* Change in wcfg file parameter md_run_file from run-openmm-xml3.py to your md script.
* Copy extasy_tica.py and change sim_task parameters, the sim_task.arguments has to call your script with required files and paths.  

## Change the analysis step
* Copy helper_scrips/run-tica-msm5.py and change the analysis steps as desired.
* Change in wcfg file parameter script_ana from run-tica-msm5.py to your analysis script.


