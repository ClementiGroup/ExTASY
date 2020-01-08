# ExTASY

The goal of this package is to execute Adaptive sampling on HPCs in a scalable, efficient way.

## New release v2
Now with python3 and asynchronous execution supported.

## Setup and installation on Local system, not necessary for Summit
Make a new conda python (python2 or python3) installation:
```
conda install -c conda-forge rabbitmq-server tmux pip git python
pip install radical.entk
```
Check ```radical-stack``` to make sure radical packages are installed.
## Setup and installation on remote system
#### For Summit
Load publicly accessible python environment:
```
export RADICAL_LOG_LVL="DEBUG"
export RADICAL_LOG_TGT="radical.log"
export RMQ_HOSTNAME=two.radical-project.org
export RMQ_PORT=33239
module load cuda/9.1.85
module load cmake
module load gcc/7.4.0
module load python/3.6.6-anaconda3-5.3.0
conda activate extasy
cd $MEMBERWORK/bipxxx
````
## Download this repository
```
git clone https://github.com/ClementiGroup/extasy.git
cd extasy
```
## Executing the example
The settings file for the example is ```settings/settings_extasy_chignolin_example.wcfg```. It will run ```cmicro``` adaptive strategy for Chignolin protein with 5 replicas for couple iterations.
Change the ```remote_output_directory``` in the settings file, so that the directory is empty and accessible to you. ExTASY will append to the previous results in the directory. For longer simulations run inside tmux on a machine which can run undisturbed for long times.
Execution command for example:
```
python extasy.py --Kconfig settings/settings_extasy_chignolin_example.wcfg
```
The expected run time is 10 minutes in addition to the queue time.

## Results
* The output directory as specified in settings files will have some output for each iteration.
* The subdirectories md_logs, results, restart and alltrajs will have additional information.
* To debug look into the ```re.session.xxx``` directory as specified in the output of the execution. Additional files to debug are in ```radical.pilot.sandbox```


## Adapt to a protein of your choice
* Copy the directory files-chignolin, replace the pdb structures and the xml files with files of your protein.
* Copy settings/settings_extasy_example_chignolin.wcfg and change md_dir, md_input_file, md_reference.
* If necessary change in the settings file the walltime, NODESIZE and num_replicas.

## Change molecular dynamics engine to your choice 
* Copy extasy_tica.py and change sim_task parameters, the sim_task.arguments has to call your script with required files and paths.  
* copy helper_scrips/run-openmm7.py and change the script to your MD engine.
* Change in settings file parameter md_run_file from run-openmm7.py to your md script.
* change in extasy.py the molecular dynamics tasks parameters, the sim_task.arguments has to call your script with required files and paths.  

## Adapt analysis step to your choice 
* Copy helper_scrips/run-vamp3.py and change the analysis steps as desired.
* Change in settings file parameter script_ana from run-vamp3.py to your analysis script.
* change in extasy.py the analysis tasks parameters, the ana_task.arguments has to call your script with required files and paths.  

## Notes 
The ```extasy.py``` script contains information about the application
execution workflow and the associated data movement. The directory ```helper_scripts``` contains script performing molecular dynamics and analysis.


