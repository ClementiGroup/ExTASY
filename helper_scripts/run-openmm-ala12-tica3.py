from simtk.unit import *
from simtk.openmm import * #LangevinIntegrator, Simulation
from simtk.openmm.app import * 
from sys import stdout
import os
import mdtraj
import mdtraj.reporters
import pyemma
from datetime import datetime
import argparse
import glob
import numpy as np
import time

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
#parser.add_argument('--tica_lag',dest='tica_lag',required=False,type=int, default=50)
#parser.add_argument('--tica_dim',dest='tica_dim',required=False,type=int, default=2)
#parser.add_argument('--tica_stride',dest='tica_stride',required=False,type=int, default=1)

args = parser.parse_args()
#grofile_name='start2.gro
time_start_all=time.time()
#pdb=mdtraj.load(grofile_name)
#save_traj=False
#save_traj='True'
print("num of structures:",str(args.idxend-args.idxstart))
print("found num:", str(len(glob.glob(args.path+'/iter'+str(args.iter)+'_input*.pdb')))) 
for i in range(args.idxstart,args.idxend):
	print("pdb "+str(i)+" from "+ str(args.idxend))
	pdb=mdtraj.load(args.path+'/iter'+str(args.iter)+'_input'+str(i)+'.pdb') 
	topology = pdb.topology.to_openmm()
	#implicit forcefield
	forcefield = ForceField('amber99sbildn.xml', 'amber99_obc.xml')
	temp=300
	long_step=True
	#long_step=False
	if long_step:
	  system = forcefield.createSystem(topology,nonbondedMethod=CutoffNonPeriodic,constraints=AllBonds, hydrogenMass=4*amu)
	  dt=0.005*picoseconds
	  integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt)
	else:
	  system = forcefield.createSystem(topology, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds)
	  dt=0.002*picoseconds
	  integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, dt)
	simulation = Simulation(topology, system, integrator)
	simulation.context.setPositions(pdb.xyz[0])
	simulation.context.setVelocitiesToTemperature(temp*kelvin)
	#simulation.minimizeEnergy() 
	#simulation.reporters.append(PDBReporter('output.pdb', 1000)) 
	simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
	potentialEnergy=True, temperature=True)) 
	if args.save_traj=='True':
	  simulation.reporters.append(DCDReporter(args.path+'/iter'+str(args.iter)+'_traj'+str(i)+'.dcd', args.trajstride)) 
	steps=args.md_steps #1000=2sec each, 10000=20sec
	start=datetime.now()
	simulation.step(steps)
	end = datetime.now()
	elapsed = end -start
	time_el=elapsed.seconds + elapsed.microseconds*1e-6
	print('Integrated %d steps in %g seconds' % (steps, time_el))
	print('%g ns/day' % (dt*steps*86400/time_el).value_in_unit(nanoseconds))
	state = simulation.context.getState(getPositions=True, getVelocities=True,getEnergy=True)
	pbv = state.getPeriodicBoxVectors(asNumpy=True)
	vel = state.getVelocities(asNumpy=True)
	pos = state.getPositions(asNumpy=True)
	print(state.getPotentialEnergy(), state.getKineticEnergy())
	PDBFile.writeFile(simulation.topology, pos, open(args.path+'/iter'+str(args.iter)+'_out'+str(i)+'.pdb', 'a'))
	del simulation, integrator, system
        
print('STARTING npy production')
pdb_file=glob.glob(args.path+'/iter0_input0.pdb')[0]

topfile = mdtraj.load(pdb_file)
featurizer = pyemma.coordinates.featurizer(topfile)
featurizer.add_residue_mindist(residue_pairs='all', scheme='closest-heavy')
featurizer.add_backbone_torsions(cossin=True)
print(featurizer.dimension())
for i in range(args.idxstart,args.idxend):
        print("npy "+str(i)+" from "+ str(args.idxend))
        time_start=time.time()
        inp = pyemma.coordinates.source(args.path+'/iter'+str(args.iter)+'_traj'+str(i)+'.dcd', featurizer)
        get_out=inp.get_output()[0]
        np.save(args.path+'/iter'+str(args.iter)+'_traj'+str(i)+'.npy',get_out)
        time_end=time.time()
        print('time:', time_end-time_start, 's')

time_end_all=time.time()
print('time all:', time_end_all-time_start_all, 's')
          




