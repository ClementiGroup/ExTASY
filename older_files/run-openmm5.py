
from __future__ import absolute_import, print_function

import os
import argparse
from sys import stdout, exit
import socket
import numpy as np
import imp
import time, random

import simtk.unit as u
from simtk.openmm import Platform, XmlSerializer
from simtk.openmm.app import PDBFile, Simulation, DCDReporter, StateDataReporter
from simtk.openmm.app import CheckpointReporter
from datetime import datetime
import argparse
import glob
import shutil
from itertools import combinations

import mdtraj 

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
parser.add_argument('--save_traj',dest='save_traj',required=True,type=str)
parser.add_argument('--md_steps',dest='md_steps',required=True,type=int)
parser.add_argument('--trajstride',dest='trajstride', required=False, type=int, default=10)
parser.add_argument('--idxstart',dest='idxstart',required=True,type=int)
parser.add_argument('--idxend',dest='idxend',required=True,type=int)
parser.add_argument('--iter',dest='iter',required=True,type=int)
parser.add_argument('--path',dest='path',required=True,type=str)
parser.add_argument("--Kconfig", type=str,dest="Kconfig",required=True)

args = parser.parse_args()
Kconfig = imp.load_source('Kconfig', args.Kconfig)
#grofile_name='start2.gro

time_start_all=time.time()
#pdb=mdtraj.load(grofile_name)
#save_traj=False
#save_traj='True'
print("num of structures:",str(args.idxend-args.idxstart))
print("found num:", str(len(glob.glob(args.path+'/iter'+str(args.iter)+'_input*.pdb')))) 
for i in range(args.idxstart,args.idxend):
  iter_found=0
  while os.path.isfile('%s/iter%s_input%s.pdb' % (args.path, iter_found, i)):
      iter_found+=1
  iter_found=max(0,iter_found-1)
  print("iter", iter_found)
  print("pdb "+str(i)+" from "+ str(args.idxend))
  trajstride=args.trajstride
  todosteps=args.md_steps
  fileoutpdb=args.path+'/iter'+str(iter_found)+'_out'+str(i)+'.pdb'
  argsrestart=args.path+'/iter'+str(iter_found)+'_restart'+str(i)+'.npz'
  savedcdfile=args.path+'/iter'+str(iter_found)+'_traj'+str(i)+'.dcd'
  savedcdfileextend=args.path+'/iter'+str(iter_found)+'_traj'+str(i)+'extend.dcd'
  a_topology_pdb = args.path+'/iter'+str(iter_found)+'_input'+str(i)+'.pdb' 
  a_platform = 'fastest'
  properties = None
  a_system_xml = 'system-5.xml'
  a_integrator_xml = 'integrator-5.xml'
  
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
  simulation.context.setPositions(pdb.positions)
  pbv = system.getDefaultPeriodicBoxVectors()
  simulation.context.setPeriodicBoxVectors(*pbv)
  # set velocities to temperature in integrator
  temperature = integrator.getTemperature()
  dt = integrator.getStepSize()
  simulation.context.setVelocitiesToTemperature(temperature)
  simulation.reporters.append(StateDataReporter(stdout, trajstride, step=True,
  potentialEnergy=True, temperature=True)) 
  state = simulation.context.getState(getPositions=True, getVelocities=True)
  pbv = state.getPeriodicBoxVectors(asNumpy=True)
  allpdb=mdtraj.load(a_topology_pdb)
  protpdb=allpdb.atom_slice(prot_Select)
  if os.path.isfile(savedcdfileextend):
    print('combine previous extend with dcd')  
    dcd1=mdtraj.load(savedcdfile,top=protpdb)
    dcd2=mdtraj.load(savedcdfileextend,top=protpdb)
    dcd3=mdtraj.join([dcd1,dcd2])
    dcd3.save(savedcdfile)
    os.remove(savedcdfileextend)

  if os.path.isfile(argsrestart):
    arr = np.load(argsrestart)
    simulation.context.setPositions(arr['positions'] * u.nanometers)
    simulation.context.setVelocities(arr['velocities'] * u.nanometers/u.picosecond)
    simulation.context.setPeriodicBoxVectors(*arr['box_vectors'] * u.nanometers)
    remainingsteps=arr['remainingsteps']
    print('restart remaining', remainingsteps)
    savedcdfile=savedcdfileextend
    reporter=mdtraj.reporters.DCDReporter(savedcdfile, trajstride, atomSubset=prot_Select)       
  else:
    remainingsteps=todosteps
    reporter=mdtraj.reporters.DCDReporter(savedcdfile, trajstride, atomSubset=prot_Select)
    # first frame adding
    reporter.report(simulation, state)

  simulation.reporters.append(reporter)
  
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
  print('Integrated %d steps in %g seconds' % (todosteps, time_el))
  print('%g ns/day' % (dt*todosteps*86400/time_el).value_in_unit(u.nanoseconds))
  
  #combine final extend to dcd traj
  if os.path.isfile(savedcdfileextend):
    print('final combine extend with dcd')
    dcd1=mdtraj.load(savedcdfile,top=protpdb)
    dcd2=mdtraj.load(savedcdfileextend,top=protpdb)
    dcd3=mdtraj.join([dcd1,dcd2])
    dcd3.save(savedcdfile)
    os.remove(savedcdfileextend)



  state = simulation.context.getState(getPositions=True, getVelocities=True,getEnergy=True)
  pbv = state.getPeriodicBoxVectors(asNumpy=True)
  vel = state.getVelocities(asNumpy=True)
  pos = state.getPositions(asNumpy=True)
  print(state.getPotentialEnergy(), state.getKineticEnergy())
  PDBFile.writeFile(simulation.topology, pos, open(fileoutpdb, 'w'))
  del simulation, integrator, system




time_end_all=time.time()
print('time all:', time_end_all-time_start_all, 's')
