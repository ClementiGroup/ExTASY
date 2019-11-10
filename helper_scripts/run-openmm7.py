
from __future__ import absolute_import, print_function

import os, sys
import argparse
from sys import stdout, exit
import socket
import numpy as np
import imp
import time, random
import shutil
import simtk.unit as u
from simtk.openmm import Platform, XmlSerializer
from simtk.openmm.app import PDBFile, Simulation, DCDReporter, StateDataReporter
from simtk.openmm.app import CheckpointReporter
from datetime import datetime
import argparse
import glob
import shutil
from itertools import combinations
from time import sleep
import mdtraj 
import linecache

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


def get_xml(xml_file):
    # TODO file access control
    attempt = 0
    retries = 500
    if not xml_file.endswith('.xml'):
        raise IOError("{} must end in '.xml' for reading as XML file".format(xml_file))
    while True:
        try:
            with open(xml_file) as f:
                xml = f.read()
                cereal = XmlSerializer.deserialize(xml)
            return xml, cereal
        except ValueError as e:
            if attempt < retries:
                attempt += 1
                time.sleep(5*random.random())
            else:
                raise e


def get_platform(platform_name):
    if platform_name == 'fastest':
        platform = None
    else:
        # TODO file access control
        attempt = 0
        retries = 500
        while True:
            try:
                platform = Platform.getPlatformByName(platform_name)
                return platform
            except IndexError as e:
                if attempt < retries:
                    attempt += 1
                    time.sleep(5*random.random())
                else:
                    raise e


def get_pdbfile(topology_pdb):
    # TODO file access control
    attempt = 0
    retries = 500
    if not topology_pdb.endswith('.pdb'):
        raise IOError("{} must end in '.pdb' for reading as PDB file".format(topology_pdb))
    while True:
        try:
            pdb = PDBFile(topology_pdb)
            return pdb
        except IndexError as e:
            if attempt < retries:
                attempt += 1
                time.sleep(5*random.random())
            else:
                raise e


def read_input(platform, pdbfile, system, integrator):
    return_order = ['get_platform', 'get_pdbfile',
                    'get_system', 'get_integrator']
    funcs = {'get_platform':   [get_platform, platform],
             'get_pdbfile':    [get_pdbfile, pdbfile],
             'get_system':     [get_xml, system],
             'get_integrator': [get_xml, integrator]}
    kfuncs = list(funcs.keys())
    random.shuffle(kfuncs)
    returns = dict()
    while kfuncs:
        op_name = kfuncs.pop(0)
        func, arg = funcs[op_name]
        returns.update({op_name: func(arg)})
    return [returns[nxt] for nxt in return_order]



platformNames = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
print(platformNames)


parser = argparse.ArgumentParser()
#parser.add_argument('--save_traj',dest='save_traj',required=True,type=str)
#parser.add_argument('--md_steps',dest='md_steps',required=True,type=int)
#parser.add_argument('--trajstride',dest='trajstride', required=False, type=int, default=10)
parser.add_argument('--idxstart',dest='idxstart',required=True,type=int)
parser.add_argument('--idxend',dest='idxend',required=True,type=int)
#parser.add_argument('--iter',dest='iter',required=True,type=int)
parser.add_argument('--path',dest='path',required=True,type=str)
parser.add_argument("--Kconfig", type=str,dest="Kconfig",required=True)
#parser.add_argument("--extend", type=str, dest="extend", default='True')
args = parser.parse_args()
Kconfig = imp.load_source('Kconfig', args.Kconfig)
#grofile_name='start2.gro

time_start_all=time.time()
#pdb=mdtraj.load(grofile_name)
#save_traj=False
#save_traj='True'
os.makedirs(args.path+'/md_logs', exist_ok=True)
try:
  saveall=str(Kconfig.save_alltraj)
except:
  saveall="False"
try:
  strategy=str(Kconfig.strategy)
except:
  strategy='extend'

os.makedirs(args.path+'/tmp', exist_ok=True)
if saveall=='True':
  os.makedirs(args.path+'/alltrajs', exist_ok=True)
if strategy!='extend':
  os.makedirs(args.path+'/restart', exist_ok=True)

#print("num of structures:",str(args.idxend-args.idxstart))
#print("found num:", str(len(glob.glob(args.path+'/iter'+str(args.iter)+'_input*.pdb')))) 
while True:
  try:#for rep in range(3):
    for i in range(args.idxstart,args.idxend):
      iter_found=0
      while os.path.isfile('%s/iter%s_out%s.pdb' % (args.path, iter_found, i)):
        iter_found+=1
      #iter_found=max(0,iter_found)
      
      mdlog=args.path+'/md_logs/iter'+str(iter_found)+'_md'+str(i)+'.log'
      mdlog2=args.path+'/md_logs/iter'+str(iter_found)+'_md'+str(i)+'_2.log'
      sys.stdout = open(mdlog2,'a')
      print("iter", iter_found)
      print("pdb "+str(i)+" from "+ str(args.idxend))
      trajstride=Kconfig.trajstride
      todosteps=Kconfig.md_steps
      fileoutpdb=args.path+'/iter'+str(iter_found)+'_out'+str(i)+'.pdb'
      oldoutpdb=args.path+'/iter'+str(iter_found-1)+'_out'+str(i)+'.pdb'
      restartpdb=args.path+'/restart/restart'+str(i)+'.pdb'
      fileextend=args.path+'/iter'+str(iter_found)+'_input'+str(i)+'.pdb'
      argsrestart=args.path+'/iter'+str(iter_found)+'_restart'+str(i)+'.npz'
      savedcdfile=args.path+'/iter'+str(iter_found)+'_traj'+str(i)+'.dcd'
      savedcdfileextend=args.path+'/iter'+str(iter_found)+'_traj'+str(i)+'extend.dcd'
      savedcdfileall=args.path+'/alltrajs/alltrajiter'+str(iter_found)+'_traj'+str(i)+'.dcd'
      savedcdfileextendall=args.path+'/alltrajs/alltrajiter'+str(iter_found)+'_traj'+str(i)+'extend.dcd'
      tmpfile=args.path+'/tmp/iter'+str(iter_found)+'_tmp'+str(i)+'.dcd'
      tmpfileall=args.path+'tmp/iter'+str(iter_found)+'_tmp'+str(i)+'.dcd'
      a_topology_pdb = args.path+'/iter'+str(iter_found)+'_input'+str(i)+'.pdb' 
      a_platform = 'fastest'
      properties = None
      a_system_xml = 'system-5.xml'
      a_integrator_xml = 'integrator-5.xml'
      print("a_topology_pdb", a_topology_pdb)      
 
      if strategy=='extend':
        if not os.path.isfile(fileextend):
          shutil.copy2(oldoutpdb,fileextend)
          print("extended", oldoutpdb, fileextend)
    
      if strategy!='extend':
        if not os.path.isfile(fileextend):
          shutil.copy2(restartpdb,fileextend)
          print("restart", restartpdb, fileextend)
    
      platform, pdb, (system_xml, system), (integrator_xml, integrator) \
       = read_input(a_platform, a_topology_pdb, a_system_xml, a_integrator_xml)
      simulation = Simulation( pdb.topology,  system,integrator, platform, properties)
      if i == args.idxstart:
        pdbtop=mdtraj.load(a_topology_pdb).topology
        print(pdbtop)
        prot_Select=pdbtop.select("protein")
        print("prot_Select", prot_Select)
        print('# platform used:', simulation.context.getPlatform().getName())
        print('# platforms available')
        for no_platform in range(Platform.getNumPlatforms()):
            # noinspection PyCallByClass,PyTypeChecker
            print('(%d) %s' % (no_platform, Platform.getPlatform(no_platform).getName()))
        #print(os.environ)
        print(Platform.getPluginLoadFailures())
        print(Platform.getDefaultPluginsDirectory())
      sys.stdout.flush()
      simulation.context.setPositions(pdb.positions)
      pbv = system.getDefaultPeriodicBoxVectors()
      simulation.context.setPeriodicBoxVectors(*pbv)
      # set velocities to temperature in integrator
      temperature = integrator.getTemperature()
      dt = integrator.getStepSize()
      simulation.context.setVelocitiesToTemperature(temperature)
      simulation.reporters.append(StateDataReporter(mdlog, trajstride, step=True,
      potentialEnergy=True, temperature=True)) 
      state = simulation.context.getState(getPositions=True, getVelocities=True)
      pbv = state.getPeriodicBoxVectors(asNumpy=True)
      allpdb=mdtraj.load(a_topology_pdb)
      protpdb=allpdb.atom_slice(prot_Select)
      if not os.path.isfile(args.path+'/iter0_prot0.pdb'):
        protpdb.save(args.path+'/iter0_prot0.pdb')
      if os.path.isfile(savedcdfileextend):
        print('combine previous extend with dcd')  
        dcd1=mdtraj.load(savedcdfile,top=protpdb)
        dcd2=mdtraj.load(savedcdfileextend,top=protpdb)
        dcd3=mdtraj.join([dcd1,dcd2])
        dcd3.save(tmpfile)
        os.remove(savedcdfileextend)
        shutil.move(tmpfile, savedcdfile)
        if saveall=="True":
          dcd1=mdtraj.load(savedcdfileall,top=allpdb)
          dcd2=mdtraj.load(savedcdfileextendall,top=allpdb)
          dcd3=mdtraj.join([dcd1,dcd2])
          dcd3.save(tmpfileall)
          os.remove(savedcdfileextendall)
          shutil.move(tmpfileall, savedcdfileall)
    
      if os.path.isfile(argsrestart):
        arr = np.load(argsrestart)
        simulation.context.setPositions(arr['positions'] * u.nanometers)
        simulation.context.setVelocities(arr['velocities'] * u.nanometers/u.picosecond)
        simulation.context.setPeriodicBoxVectors(*arr['box_vectors'] * u.nanometers)
        remainingsteps=arr['remainingsteps']
        print('restart remaining', remainingsteps)
        reporter=mdtraj.reporters.DCDReporter(savedcdfileextend, trajstride, atomSubset=prot_Select)
        if saveall=='True':
          reporter2=mdtraj.reporters.DCDReporter(savedcdfileextendall, trajstride)
      else:
        remainingsteps=todosteps
        reporter=mdtraj.reporters.DCDReporter(savedcdfile, trajstride, atomSubset=prot_Select)
        # first frame adding
        reporter.report(simulation, state)
        if saveall=='True':
          reporter2=mdtraj.reporters.DCDReporter(savedcdfileall, trajstride)
          reporter2.report(simulation, state)
        print("no restart")
      torunsteps=remainingsteps
      simulation.reporters.append(reporter)
      if saveall=='True':
        simulation.reporters.append(reporter2)
      sys.stdout.flush() 
      start=datetime.now()
      while remainingsteps>0:
        #print(remainingsteps)
        executesteps=min(trajstride, remainingsteps)
        simulation.step(executesteps)
        vel = state.getVelocities(asNumpy=True)
        pos = state.getPositions(asNumpy=True)
        remainingsteps=remainingsteps-executesteps
        np.savez(argsrestart, positions=pos, box_vectors=pbv, velocities=vel, remainingsteps=remainingsteps)
      
      end = datetime.now()
      elapsed = end -start
      time_el=elapsed.seconds + elapsed.microseconds*1e-6
      print('Integrated %d steps in %g seconds' % (torunsteps, time_el))
      print('%g ns/day' % (dt*torunsteps*86400/time_el).value_in_unit(u.nanoseconds))
      
      #combine final extend to dcd traj
      if os.path.isfile(savedcdfileextend):
        print('final combine extend with dcd')
        dcd1=mdtraj.load(savedcdfile,top=protpdb)
        dcd2=mdtraj.load(savedcdfileextend,top=protpdb)
        dcd3=mdtraj.join([dcd1,dcd2])
        dcd3.save(tmpfile)
        os.remove(savedcdfileextend)
        shutil.move(tmpfile, savedcdfile)
        if saveall=='True':
          print('final combine extend with all dcd')
          dcd1=mdtraj.load(savedcdfileall,top=allpdb)
          dcd2=mdtraj.load(savedcdfileextendall,top=allpdb)
          dcd3=mdtraj.join([dcd1,dcd2])
          dcd3.save(tmpfileall)
          os.remove(savedcdfileextendall)
          shutil.move(tmpfileall, savedcdfileall)
    
      state = simulation.context.getState(getPositions=True, getVelocities=True,getEnergy=True)
      pbv = state.getPeriodicBoxVectors(asNumpy=True)
      vel = state.getVelocities(asNumpy=True)
      pos = state.getPositions(asNumpy=True)
      print(state.getPotentialEnergy(), state.getKineticEnergy())
      PDBFile.writeFile(simulation.topology, pos, open(fileoutpdb, 'w'))
      print("saved", fileoutpdb)
      del simulation, integrator, system
      #if args.extend=='True':
      #  shutil.copy2(fileoutpdb,fileextend)
      #  print("copied to", fileextend)
      sys.stdout.flush()
  except Exception as e:
    print(type(e).__name__) # returns the name of the exception
    print(e.__doc__)
    print(e)
    PrintException()
    sleep(10)
    

time_end_all=time.time()
print('time all:', time_end_all-time_start_all, 's')
