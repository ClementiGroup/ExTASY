#!/usr/bin/env python

__author__ = 'Vivek <vivek.balasubramanian@rutgers.edu> Eugen <eh22@rice.edu>'
__copyright__ = 'Copyright 2019, http://radical.rutgers.edu,  http://clementiresearch.rice.edu'
__license__ = 'MIT'
__use_case_name__ = 'Asynch Adaptive simulation-analysis using EnTK'


from radical.entk import Pipeline, Stage, Task, AppManager
import argparse
import os
import glob
import sys
import imp
import json
import traceback
import time
import socket
#import numpy as np
import math

# CONDA_PREFIX reads - $HOME/.conda/envs/[CONDA ENV NAME]
PYTHON_PATH="{}/bin/python".format(os.environ['CONDA_PREFIX'])
print(socket.gethostname())

def create_workflow(Kconfig,args):


    wf = Pipeline()

    # ------------------------------------------------------------------------------------------------------------------
    cur_iter = int(Kconfig.start_iter)#0
    #assumed of iteration non zero that files are in combined_path
    if str(socket.gethostname())=='giotto.rice.edu':
      combined_path=str(Kconfig.remote_output_directory)+'-giotto' 
    else:
      combined_path=str(Kconfig.remote_output_directory)  #'/u/sciteam/hruska/scratch/extasy-tica'
    num_parallel=int(Kconfig.NODESIZE)*int(Kconfig.GPUs_per_NODE)
    num_replicas=int(Kconfig.num_replicas)
    script_ana=str(Kconfig.script_ana)#run-tica-msm4.py
    try:
      systemxml=str(Kconfig.systemxml)
    except:
      systemxml='system-5.xml'
    try:
      integratorxml=str(Kconfig.integratorxml)
    except:
      integratorxml='integrator-5.xml'
    md_settings=Kconfig.md_env
    if Kconfig.env_ana_same =='True':
      ana_settings=md_settings
    else:
      ana_settings=Kconfig.ana_env
    print("set", num_parallel,md_settings)
    iter_found=0
    while len(glob.glob('%s/iter%s_input*.pdb' % (combined_path, iter_found)))>=num_replicas:
      iter_found+=1
    cur_iter=max(0,iter_found-1)
    print("cur_iter",cur_iter)
    if cur_iter==0:
      #pre_proc_stage = Stage()
      #pre_proc_task = Task()
      #pre_proc_task.pre_exec = ['export tasks=pre_proc_task','export iter=%s' % cur_iter, 'export OMP_NUM_THREADS=1']
      #pre_proc_task.executable = ['mv']
      #pre_proc_task.arguments = [ combined_path, combined_path + time.strftime("%Y-%m-%d-%H-%M") ]
      #pre_proc_task_ref = '$Pipeline_%s_Stage_%s_Task_%s' % (wf.uid, pre_proc_stage.uid, pre_proc_task.uid)
      #pre_proc_stage.add_tasks(pre_proc_task)
      #wf.add_stages(pre_proc_stage)
      pre_proc_stage2 = Stage()
      pre_proc_task2 = Task()
      pre_proc_task2.pre_exec = ['export tasks=pre_proc_task','export iter=%s' % cur_iter, 'export OMP_NUM_THREADS=1']
      pre_proc_task2.executable = ['ls']
      pre_proc_task2.arguments = ['-l']
      pre_proc_task2.copy_input_data = ['$SHARED/%s > %s/%s' % (args.Kconfig,combined_path, args.Kconfig),
                                     '$SHARED/%s > %s/%s' % (script_ana,combined_path,script_ana),
                                     '$SHARED/%s > %s/%s' % (Kconfig.md_run_file,combined_path,Kconfig.md_run_file),
                                       '$SHARED/%s > %s/%s' % (Kconfig.md_reference, combined_path, Kconfig.md_reference)]# '$SHARED/%s > %s/%s' % ('analyze3.py', combined_path, 'analyze3.py') ] 
      pre_proc_task_ref2 = '$Pipeline_%s_Stage_%s_Task_%s' % (wf.uid, pre_proc_stage2.uid, pre_proc_task2.uid)
      pre_proc_stage2.add_tasks(pre_proc_task2)
      wf.add_stages(pre_proc_stage2)
 
      # ------------------------------------------------------------------------------------------------------------------
    start_iter=cur_iter 
    while(cur_iter <  int(Kconfig.num_iterations) and cur_iter<start_iter+1):

        # --------------------------------------------------------------------------------------------------------------
        # sim_stage:
        #     Purpose:  In iter=1, use the input files from pre_loop, else use the outputs of the analysis stage in the
        #               previous iteration. Run gromacs on each of the smaller files. Parameter files and executables
        #                are input from pre_loop. There arei 'numCUs' number of instances of gromacs per iteration.
        #     Arguments :
        #           grompp    = gromacs parameters filename
        #           topol     = topology filename

        sim_stage = Stage()
        sim_task_ref = list()
        num_allocated_rep=0
        num_used_parallel=0
        #num_used_threads=0
        #print(def_rep_per_thread)
        while(num_allocated_rep<num_replicas):
          #if (num_used_threads>=num_parallel):
          #   print("ALLERT tried use more gpus than allocated")
          def_rep_per_thread=int(math.ceil(float(num_replicas-num_allocated_rep)/float(num_parallel-num_used_parallel)))
          use_replicas=min(def_rep_per_thread, num_replicas-num_allocated_rep)
          #if ((num_replicas-num_allocated_rep)>def_rep_per_thread):  # check if use all threads
          #   use_replicas=def_rep_per_thread
          #else:  #use pnly part of threads
          #   use_replicas=(num_replicas-num_allocated_rep)
          print("u", cur_iter, use_replicas, num_replicas, num_parallel, def_rep_per_thread, num_allocated_rep,num_used_parallel)
          sim_task = Task()
          sim_task.executable = ['python']
          
          pre_exec_arr = md_settings + ['export tasks=md','export iter=%s' % cur_iter ]
          #if cur_iter==0 and num_allocated_rep==0:
          #  pre_exec_arr = pre_exec_arr + [ 'mv %s']
          sim_task.pre_exec = pre_exec_arr
          sim_task.gpu_reqs = { 'processes': 1,
                                    'process_type': None,
                                    'threads_per_process': 1,
                                    'thread_type': 'CUDA'
                                }
          sim_task.cpu_reqs = { 'processes': 1, 
                                    'process_type': None, 
                                    'threads_per_process': 20, 
                                    'thread_type': 'OpenMP'
                                  }
          sim_task.arguments = ['run_openmm.py','--Kconfig', str(args.Kconfig), '--idxstart',str(num_allocated_rep), '--idxend',str(num_allocated_rep+use_replicas),
                                  '--path',combined_path,'>', 'md.log']
                                  #'--trajstride', str(Kconfig.trajstride),'--Kconfig', str(args.Kconfig), 
                                  #'--idxstart',str(num_allocated_rep), '--idxend',str((num_allocated_rep+use_replicas)),
                                  #'--path',combined_path,'--iter',str(cur_iter),
                                  #'--md_steps',str(Kconfig.md_steps), '--save_traj', 'True','>', 'md.log']
          if Kconfig.md_use_xml=='yes':
            link_arr=['$SHARED/%s > run_openmm.py' % (os.path.basename(Kconfig.md_run_file)),
                      '$SHARED/%s > %s' % (systemxml, systemxml),
                      '$SHARED/%s > %s' % (integratorxml, integratorxml), '$SHARED/%s > %s'%(args.Kconfig,args.Kconfig)]            
          else:
            link_arr=['$SHARED/%s > run_openmm.py' % (os.path.basename(Kconfig.md_run_file)), '$SHARED/%s > %s'%(args.Kconfig,args.Kconfig)]
          copy_arr=[]
          if cur_iter==0:
            for idx in range(num_allocated_rep, num_allocated_rep+use_replicas):
              copy_arr=copy_arr+['$SHARED/%s > %s/iter0_input%s.pdb' % (Kconfig.md_input_file, combined_path, idx)]           
            #if num_allocated_rep==0:
            #  copy_arr=copy_arr + ['$SHARED/%s > %s/%s' % (args.Kconfig,combined_path, args.Kconfig),
            #                         '$SHARED/run-tica-msm.py > %s/run-tica-msm.py' % combined_path,
            #                         '$SHARED/%s > %s/%s' % (Kconfig.md_run_file,combined_path,Kconfig.md_run_file)
            #                           ]

    
          #if cur_iter==0 and num_allocated_rep==0:
          #   copy_arr = copy_arr +['$SHARED/%s > %s/%s' % (args.Kconfig, combined_path, args.Kconfig)]
          sim_task.link_input_data = link_arr #+ copy_arr
          sim_task.copy_input_data = copy_arr
          copy_out=[]
          #if str(Kconfig.strategy)=='extend':
          #  for idx in range(num_allocated_rep, num_allocated_rep+use_replicas):
              #copy_arr=copy_arr+['$SHARED/%s > iter0_input%s.pdb' % (Kconfig.md_input_file, idx)]
          #    copy_out=copy_out+['%s/iter%s_out%s.pdb > %s/iter%s_input%s.pdb' % (combined_path, cur_iter, idx, combined_path, (cur_iter+1), idx)]
          
         #for idx in range(num_allocated_rep, num_allocated_rep+use_replicas):
         ##     #copy_arr=copy_arr+['$SHARED/%s > iter0_input%s.pdb' % (Kconfig.md_input_file, idx)]
         #     copy_out=copy_out+['md.log > %s/md_logs/iter%s_md%s.log' % (combined_path, cur_iter, idx)] 
          
          sim_task.copy_output_data = copy_out  
            #if Kconfig.ndx_file is not None:
            #    sim_task.link_input_data.append('$SHARED/{0}'.format(os.path.basename(Kconfig.ndx_file)))
            
          num_allocated_rep=num_allocated_rep+use_replicas
          num_used_parallel= num_used_parallel+1
          sim_task_ref.append('$Pipeline_%s_Stage_%s_Task_%s' % (wf.uid, sim_stage.uid, sim_task.uid))
          sim_stage.add_tasks(sim_task)
        if str(Kconfig.strategy)!='extend':
         for anatask in range(1):
          print("analysis task", anatask)
          ana_task = Task()
          ana_task.executable = ['python']
          pre_exec_arr = ana_settings
          ana_task.pre_exec = pre_exec_arr
          ana_task.link_input_data = ['$SHARED/%s > %s'%(script_ana, script_ana), '$SHARED/%s > %s'%(args.Kconfig,args.Kconfig)]
          ana_task.gpu_reqs = { 'processes': 1,
                                    'process_type': None,
                                    'threads_per_process': 1,
                                    'thread_type': 'CUDA'
                                }
          ana_task.cpu_reqs = { 'processes': 1, 
                                    'process_type': None, 
                                    'threads_per_process': 20, 
                                    'thread_type': 'OpenMP'
                                  }
          ana_task.arguments = [script_ana,'--Kconfig', str(args.Kconfig), '>', "analysis.log"]
          ana_task.copy_output_data = ['analysis.log > %s/analysis_iter%s_r%s.log' % (combined_path, cur_iter, anatask)]
          ana_task_ref = '$Pipeline_%s_Stage_%s_Task_%s'%(wf.uid, sim_stage.uid, ana_task.uid)
          sim_stage.add_tasks(ana_task) 
        wf.add_stages(sim_stage)
        cur_iter += 1
        Kconfig.start_iter=str(cur_iter)

    return wf


# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    try:

        parser = argparse.ArgumentParser()
        parser.add_argument('--Kconfig', help='link to Kernel configurations file')
        args = parser.parse_args()
        
        if args.Kconfig is None:
            parser.error('Please enter a Kernel configuration file')
            sys.exit(0)

        Kconfig = imp.load_source('Kconfig', args.Kconfig)
        combined_path=str(Kconfig.remote_output_directory)
        wf = create_workflow(Kconfig, args)

        # Create a dictionary describe four mandatory keys:
        # resource, walltime, cores and project
        if Kconfig.use_gpus=='False':
          res_dict = {
            'resource': Kconfig.REMOTE_HOST,
            'walltime': Kconfig.WALLTIME,
            'cores': Kconfig.PILOTSIZE,
            'project': Kconfig.ALLOCATION,
            'queue': Kconfig.QUEUE,
            'access_schema': 'gsissh'
          }
        elif Kconfig.use_gpus=='True':
          print("using gpus")
          res_dict = {
            'resource': Kconfig.REMOTE_HOST,
            'walltime': Kconfig.WALLTIME,
            #'cores': Kconfig.PILOTSIZE,
            'cpus': Kconfig.NODESIZE*Kconfig.CPUs_per_NODE,
            #'cpu_processes': Kconfig.num_CUs_per_MD_replica,#PILOTSIZE,
            'gpus': Kconfig.NODESIZE*Kconfig.GPUs_per_NODE,
            'project': Kconfig.ALLOCATION,
            'queue': Kconfig.QUEUE,
            'schema': Kconfig.schema   
            #'gsissh'
          }	  
        else:
          print("use_gpus not recognized")
          
        print (res_dict)
        # Create Resource Manager object with the above resource description
        #rman = ResourceManager(res_dict)
        # Data common to multiple tasks -- transferred only once to common staging area
        shared_data_all = [args.Kconfig
                           ]
        script_ana=str(Kconfig.script_ana)
        try:
          systemxml=str(Kconfig.systemxml)
        except:
           systemxml='system-5.xml'
        try:
          integratorxml=str(Kconfig.integratorxml)
        except:
          integratorxml='integrator-5.xml'

        if Kconfig.md_use_xml=='yes':
          shared_data_all=shared_data_all+['%s/%s' % (Kconfig.md_dir, systemxml),
                                           '%s/%s' % (Kconfig.md_dir, integratorxml),
                                           Kconfig.md_dir+Kconfig.md_reference,
                                           Kconfig.md_run_dir+Kconfig.md_run_file,
                                           Kconfig.md_dir+Kconfig.md_input_file,
                                          '%s/%s' %(Kconfig.helper_scripts, script_ana)  ]
        else:
          shared_data_all=shared_data_all+[Kconfig.md_dir+Kconfig.md_input_file,
                                           Kconfig.md_dir+Kconfig.md_reference,
                                           Kconfig.md_run_dir+Kconfig.md_run_file,
                                          '%s/%s' %(Kconfig.helper_scripts, script_ana)]
        print ("shared_data_all", shared_data_all )
       #if Kconfig.ndx_file is not None:
        #    rman.shared_data.append(Kconfig.ndx_file)

        # Create Application Manager, only one extasy script on one rabbit-mq server now
        port = int(os.environ.get('RMQ_PORT', 5672))
        hostname = os.environ.get('RMQ_HOSTNAME', 'localhost')
        appman = AppManager(hostname='two.radical-project.org', port=port)#port=args.port)
        #appman = AppManager(hostname='localhost', port=5672)
        # appman = AppManager(port=) # if using docker, specify port here.
        appman.resource_desc = res_dict
        appman.shared_data = shared_data_all

        # Assign resource manager to the Application Manager
        #appman.resource_manager = rman

        # Assign the workflow as a set of Pipelines to the Application Manager
        appman.workflow = set([wf])

        # Run the Application Manager
        appman.run()

    except Exception as ex:

        print ('Error: {0}'.format(str(ex)))
        print (traceback.format_exc())
