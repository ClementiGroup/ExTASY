import sys, os, pickle
print(os.path.dirname(sys.executable))
import time
import os
import argparse
import numpy as np
import pyemma
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import glob
import mdtraj
import imp
import collections
from pyemma import plots
matplotlib.rcParams.update({'font.size': 14})
print("pyemma version",pyemma.__version__)
import msmtools
import sklearn.preprocessing
from itertools import combinations
import matplotlib.gridspec as gridspec
import sklearn.preprocessing as pre
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#K.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))
from hde import HDE, analysis
print(tf.__version__)
from random import randint
from time import sleep
plt.rcParams.update({'font.size': 14})
plt.rc('text', usetex=False)
plt.rcParams.update({'figure.dpi': 300})
plt.rcParams['text.usetex'] = False
plt.rcParams["figure.figsize"] = [6.4, 4.8]

def select_restart_state(values, select_type, microstates, nparallel=1, parameters=None):
    if select_type == 'rand':
        #print(values,values.shape)
        len_v=values.shape[0]
        p = np.full(len_v,1.0/len_v)
        #print(microstates, p)
        return np.random.choice(microstates, p = p, size=nparallel)
    elif select_type == 'sto_inv_linear':
        inv_values = 1.0 / values
        p = inv_values / np.sum(inv_values)
        return np.random.choice(microstates, p = p, size=nparallel)
    elif select_type == 'sto_linear':
        if np.sum(values)==0:
          values=values+0.0001
        p = values / np.sum(values)
        return np.random.choice(microstates, p = p, size=nparallel)
    else:
        print('ERROR: selected select_type in select_restart_state does not exist')


def _reduce_box_vectors(vectors):
    """Make sure box vectors are in reduced form."""
    (bv1, bv2, bv3) = vectors
    bv3 -= bv2*round(bv3[1]/bv2[1]);
    bv3 -= bv1*round(bv3[0]/bv1[0]);
    bv2 -= bv1*round(bv2[0]/bv1[0]);
    return (bv1, bv2, bv3)

def _distance_mic(xyz, pairs, box_vectors, orthogonal):
    """Distance between pairs of points in each frame under the minimum image
    convention for periodic boundary conditions.
    The computation follows scheme B.9 in Tukerman, M. "Statistical
    Mechanics: Theory and Molecular Simulation", 2010.
    This is a slow pure python implementation, mostly for testing.
    """
    out = np.empty((xyz.shape[0], pairs.shape[0]), dtype=np.float32)
    for i in range(len(xyz)):
        bv1, bv2, bv3 = _reduce_box_vectors(box_vectors[i].T)
        for j, (a,b) in enumerate(pairs):
            r12 = xyz[i,b,:] - xyz[i,a,:]
            r12 -= bv3*round(r12[2]/bv3[2]);
            r12 -= bv2*round(r12[1]/bv2[1]);
            r12 -= bv1*round(r12[0]/bv1[0]);
            dist = np.linalg.norm(r12)
            if not orthogonal:
                for ii in range(-1, 2):
                    v1 = bv1*ii
                    for jj in range(-1, 2):
                        v12 = bv2*jj + v1
                        for kk in range(-1, 2):
                            new_r12 = r12 + v12 + bv3*kk
                            dist = min(dist, np.linalg.norm(new_r12))
            out[i, j] = dist
    return out

def _distance_box_fast(xyz,pairs, box_vectors):
  bv1, bv2, bv3 = _reduce_box_vectors(box_vectors[0].T)
  delta = np.diff(xyz[:, pairs], axis=2)[:, :, 0]
  delta -= (np.round(delta[:,:,2]/bv3[2])[:,:,np.newaxis]*bv3)
  delta -= (np.round(delta[:,:,1]/bv2[1])[:,:,np.newaxis]*bv2)
  delta -= (np.round(delta[:,:,0]/bv1[0])[:,:,np.newaxis]*bv1)         
  return (delta ** 2.).sum(-1) ** 0.5  

parser = argparse.ArgumentParser()
parser.add_argument("--Kconfig", type=str,dest="Kconfig",required=True)
args = parser.parse_args()
Kconfig = imp.load_source('Kconfig', args.Kconfig)

#Kconfig = imp.load_source('Kconfig', 'settings_extasy_vamp_chignolin.wcfg')

print("strategy", Kconfig.strategy)


trajprotdir=Kconfig.remote_output_directory
trajalldir=trajprotdir+'alltrajs/'
resultspath=trajprotdir+'results/'
os.makedirs(resultspath, exist_ok=True)
name_data='resvamp-'

reweight=Kconfig.reweight
select_n_macro_type=Kconfig.select_n_macro_type
num_macrostates=Kconfig.num_macrostates
vamp_lag=Kconfig.vamp_lag 
vamp_dim =Kconfig.vamp_dim
vamp_stride=Kconfig.vamp_stride
vamp_epochs=Kconfig.vamp_epochs
kmeans_stride =Kconfig.kmeans_stride
msm_states =Kconfig.msm_states
msm_lag=Kconfig.msm_lag
n_pick=Kconfig.num_replicas
try: 
  castride=Kconfig.castride
except:
  castride=1
try:
  select_micro_within_macro_type = Kconfig.select_micro_within_macro_type
except:
  select_micro_within_macro_type='sto_inv_linear'
try:
  macrostate_method=Kconfig.macrostate_method
except:
  macrostate_method='kmeans'
try:
  projection=Kconfig.projection
except:
  projection='vamp'
try:
  learning_rate=float(Kconfig.learning_rate)
except:
  learning_rate=1e-4
try:
  hidden_layer_depth=int(Kconfig.hidden_layer_depth)
except:
  hidden_layer_depth=6
try:
  hidden_size=int(Kconfig.hidden_size)
except:
  hidden_size=50
try:
  if Kconfig.inverse_dist=="True":
    inverse_dist=True
  else:
    inverse_dist=False
except:
  inverse_dist=False
try:
  activation=str(Kconfig.activation)
except:
  activation='tanh'
try:
  latent_space_noise=float(Kconfig.latent_space_noise)
except:
  latent_space_noise=0.1
try:
  dropout_rate=float(Kconfig.dropout_rate)
except:
  dropout_rate=0.1

project_tica='True'
refticapath='/gpfs/alpine/proj-shared/bip191/objchigtica1-tica.obj'

while True:
 try:
  iter_found=0
  while os.path.isfile('%s/iter%s_out0.pdb' % (trajprotdir, iter_found)):
        iter_found+=1
  #iter_found=max(0,iter_found-1)
  mdlog=trajprotdir+'md_logs/analysisiter'+str(iter_found)+'.log'
  sys.stdout = open(mdlog,'a')
  time_start=time.time()
  print("macrostate_method", macrostate_method)
  print("projection", projection)
  print("learning_rate", learning_rate)
  print("hidden_layer_depth", hidden_layer_depth)
  print("hidden_size", hidden_size)
  print("inverse_dist", inverse_dist)
  print("dropout_rate", dropout_rate)
  print("latent_space_noise", latent_space_noise)
  print("iter_found", iter_found)
  print("strategy", Kconfig.strategy)
  print("########start")
  topfile=trajprotdir+"iter0_prot0.pdb"
  topfileall=trajprotdir+"iter0_input0.pdb"
  traj_fns0 = sorted(glob.glob(trajprotdir+"*.dcd"))
  traj_fns=[f for f in traj_fns0]# if time_start-os.stat(f).st_mtime>100]
  print("num trajs", len(traj_fns))
  traj_tmp=mdtraj.load(traj_fns[0], top=topfile)
  length_single=traj_tmp.xyz.shape[0]
  print("length_single", length_single, "together", length_single*len(traj_fns))
  vamp_stride=1+int(length_single*len(traj_fns)/1e6)
  print("vamp_stride", vamp_stride)
  print(trajprotdir+'alltrajs/all'+traj_fns[0][len(trajprotdir):])
  #print(traj_fns)
  feat = pyemma.coordinates.featurizer(topfile)
  allCA = feat.select('name CA')[::castride]
  print('allCA', allCA)
  ca_pairs=feat.pairs(allCA, excluded_neighbors=0)
  data=[]
  traj_fns2=[]
  for idx in range(len(traj_fns)):
    try:
      if idx%10==0:
        print(idx, end=' ')
        sys.stdout.flush()
      traj=mdtraj.load(traj_fns[idx], top=topfile)
      box=traj.unitcell_vectors#[0]
      orthogonal = np.allclose(traj.unitcell_angles, 90)
      tmp_dist=_distance_box_fast(traj.xyz[::vamp_stride], ca_pairs,box)
      if inverse_dist:
         data.append(np.hstack((tmp_dist, 1./tmp_dist)))
      else:
         data.append(tmp_dist)
      traj_fns2.append(traj_fns[idx])
    except:
      print("failed load",traj_fns[idx])

  if projection=='vamp':
    print("datamax", data[0].max())
    scaler = pre.MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(np.concatenate(data))
    data2 = [scaler.transform(d) for d in data]
    print("max",data2[1].max())
    print("len data2", len(data2), data2[0].shape)
    nfeat = feat.dimension()
    print("feat dimensions", nfeat)
    lengthsmax=max([data22.shape[0] for data22 in data2])
    print("lengthsmax",lengthsmax)
    lengthsall=np.concatenate(data2)
    print("lengthall", lengthsall.shape)
    if lengthsmax<int(vamp_lag/vamp_stride):
      print("trajs not long enough")
      sys.stdout.flush()
      sleep(10)
      continue
    print("time0", time.time()-time_start)
    if inverse_dist:
      n_input=len(ca_pairs)*2
    else:
      n_input=len(ca_pairs)
    print("n_input", n_input)
    model = HDE(
      n_input, 
      n_components=vamp_dim, 
      n_epochs=vamp_epochs, 
      learning_rate=learning_rate,
      lag_time=int(vamp_lag/vamp_stride),
      batch_normalization=False,
      reversible=True,
      hidden_layer_depth=hidden_layer_depth,
      hidden_size=hidden_size,
      activation=activation,
      latent_space_noise=latent_space_noise,
      dropout_rate=dropout_rate,
      verbose=False
    )
    model.fit(data2)
    print("time1", time.time()-time_start)
    slow_modes = [model.transform(d) for d in data2]
    hde_timescales = model.timescales_
    del model
    print("hde_timescales", hde_timescales)
    print("slow modes", slow_modes[0][0,:10])
    yall_slow_modes = np.concatenate(slow_modes)
    yall=yall_slow_modes
    yall_slow_modes.shape
    figX, axX,mi = pyemma.plots.plot_free_energy(yall_slow_modes[:,0], yall_slow_modes[:,1],legacy=False)
    axX.set_xlabel('SRV 0')
    axX.set_ylabel('SRV 1')
    figX.savefig(resultspath+name_data+"vampfe_i"+str(iter_found)+".png")

  if projection=='tica':
    lengthsmax=max([data22.shape[0] for data22 in data])
    print("lengthsmax",lengthsmax)
    tica_obj_tmp = pyemma.coordinates.tica(data, lag=int(vamp_lag/vamp_stride), dim=vamp_dim, kinetic_map=True, stride=1, weights='empirical')
    print("time1", time.time()-time_start)
    slow_modes = tica_obj_tmp.get_output(stride=1)
    tica_timescales = tica_obj_tmp.timescales
    print("tica_timescales", tica_timescales)
    print("slow modes", slow_modes[0][0,:10])
    yall_slow_modes = np.concatenate(slow_modes)
    yall=yall_slow_modes
    yall_slow_modes.shape
    figX, axX = pyemma.plots.plot_free_energy(yall_slow_modes[:,0], yall_slow_modes[:,1])
    figX.savefig(resultspath+name_data+'ticafe_i'+str(iter_found)+'.png')

  if Kconfig.project_tica=='True':
    tica_obj = pyemma.load(refticapath)
    yticaproj=np.concatenate(tica_obj.transform(data))
    xlim=(-2.2,1.7)
    ylim=(-2.5,3.4)
    vmax_set=10
    step=0.1
    levels=np.arange(0,vmax_set+step,step)
    figX, axX,mi = pyemma.plots.plot_free_energy(yticaproj[:,0], yticaproj[:,1], levels=levels,legacy=False)
    axX.set_xlim(xlim)
    axX.set_ylim(ylim)
    axX.set_xlabel('TICA 0')
    axX.set_ylabel('TICA 1')
    figX.savefig(resultspath+name_data+"ticaprojreffe_i"+str(iter_found)+".png")
    if False:
      tica_obj = pyemma.coordinates.tica(data,lag=vamp_lag, dim=4, kinetic_map = True, commute_map=False)
      yticaproj=np.concatenate(tica_obj.transform(data)) 
      figX, axX,mi = pyemma.plots.plot_free_energy(yticaproj[:,0], yticaproj[:,1],legacy=False)
      axX.set_xlabel('TICA 0')
      axX.set_ylabel('TICA 1')
      figX.savefig(resultspath+name_data+"ticafe_i"+str(iter_found)+".png")
  print("time1.1", time.time()-time_start)
  try:
    km = pyemma.coordinates.cluster_kmeans(slow_modes, k = msm_states,max_iter=50,stride=kmeans_stride)
  except:
    #if not enough states
    km = pyemma.coordinates.cluster_kmeans(slow_modes, k = min(msm_states, 100),max_iter=50) 
  dtrajs = km.dtrajs
  if lengthsmax<int(msm_lag/vamp_stride):
    print("traj not long enough")
    sys.stdout.flush()
    sleep(10)
    continue
  msm=pyemma.msm.estimate_markov_model(dtrajs, lag=int(msm_lag/vamp_stride))
  print("msm timescales", msm.timescales(10))
  c=msm.count_matrix_full
  s =  np.sum(c+c.T, axis=1)
  q = 1.0 / s
  q[s<1]=1.0
  n_states=c.shape[0]
  print("s min",s.min(), "q min", q.min())
  frame_state_list = {n: [] for n in range(n_states)}
  for nn, dt in enumerate(dtrajs):
    for mm, state in enumerate(dt):
          frame_state_list[state].append((nn,mm))
  for k in range(n_states):
           if len(frame_state_list[k]) == 0:
              print('removing state '+str(k)+' no frames')
              q[k] = 0.0
                      # and normalize the remaining one
  q /= np.sum(q)
  print("time2", time.time()-time_start)
  if Kconfig.strategy=='cmicro':
    print("probpick top 10", np.sort(q)[::-1][:10], np.sort(q)[::-1][:10].sum(), np.argsort(q)[::-1][:10])
    state_picks = np.random.choice(np.arange(len(q)), size=n_pick, p=q)
    state_picks=sorted(state_picks.astype('int'))
    print("state_picks", state_picks)
    fig,ax = plt.subplots(nrows=1, figsize=(7,4.5), sharex=True)
    pyemma.plots.plot_free_energy(yall[:,0],yall[:,1], nbins=64, weights=np.concatenate(msm.trajectory_weights()), cmap='nipy_spectral', ax=ax,legacy=False)
    ax.set_xlabel("SRV 1")
    ax.set_ylabel("SRV 2")
    ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c= 'gray', s=2, lw=0.5, edgecolor="k", label=f"states")
    ax.scatter(km.cluster_centers_[state_picks,0], km.cluster_centers_[state_picks,1], c='red',s=25, lw=0.5, edgecolor="k", label=f"cmicro")
    ax.legend( fontsize=8)#loc=5
    fig.tight_layout()
    plt.savefig(resultspath+name_data+"cmicro_state_picks_i"+str(iter_found)+".png")
  if Kconfig.strategy=='cmacro':
    num_eigenvecs_to_compute = 10
    microstate_transitions_used=c
    #cache['too_small']='False'
    num_visited_microstates=c.shape[0]
    states_unique=np.arange(num_visited_microstates)
    visited_microstates=states_unique
    largest_visited_set=msmtools.estimation.largest_connected_set(microstate_transitions_used)
    C_largest0=microstate_transitions_used[largest_visited_set, :][:, largest_visited_set]
    rowsum = np.ravel(C_largest0.sum(axis=1))
    largest_visited_set2=largest_visited_set[rowsum>0]
    C_largest=microstate_transitions_used[largest_visited_set2, :][:, largest_visited_set2]
    rowsum = C_largest.sum(axis=1)
    if C_largest.shape[0]>10:
                if(np.min(rowsum) == 0.0):
                    print("failed because rowsum", rowsum, C_largest)
                    cache['small']='True'
                    #raise ValueError("matrix C contains rows with sum zero.")
                #try:
                #print("try")
                T_largest=msmtools.estimation.transition_matrix(C_largest, reversible=True)
                #print(T_largest.shape)
                states_largest=largest_visited_set2
                print("largest_connected_set", states_largest.shape[0])
                #print(states_largest, states_unique)
                MSM_largest=pyemma.msm.markov_model(T_largest)
                current_eigenvecs = MSM_largest.eigenvectors_right(num_eigenvecs_to_compute)
                current_timescales = np.real(MSM_largest.timescales())
                current_eigenvals = np.real(MSM_largest.eigenvalues())
                not_connect=np.where(np.in1d(states_unique, states_largest,invert=True))[0]
                all_connect=np.where(np.in1d(states_unique, states_largest))[0]
                print("worked timescales",current_timescales[:10])
                print("not_connected states",not_connect)
                print("s not connected", s[not_connect])
    projected_microstate_coords_scaled = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(current_eigenvecs[:,1:])
    projected_microstate_coords_scaled *= np.sqrt(current_timescales[:num_eigenvecs_to_compute-1] / current_timescales[0]).reshape(1, num_eigenvecs_to_compute-1)
    num_macrostates = min(num_macrostates,num_visited_microstates)
    if macrostate_method=='pcca':
      pcca=msm.pcca(num_macrostates)
      macrostate_assignments = { k:v for k,v in enumerate(msm.metastable_sets) }
      largest_assign = msm.metastable_assignments
      print("macrostate assignments", macrostate_assignments)
      print("mismatch", "largest_assign", largest_assign.shape, "num_visited_microstates", num_visited_microstates) 
      #all_assign=largest_assign
      all_assign=np.zeros(num_visited_microstates)
      all_assign[all_connect]=largest_assign
      all_assign[not_connect]=np.arange(not_connect.shape[0])+largest_assign.max()+1
      prob_pcca=[msm.stationary_distribution[pcca.metastable_assignment==i].sum() \
                                    for i in range(num_macrostates)]
      print("prob_pcca",prob_pcca )
    else:
      #kmeans
      kmeans_obj = pyemma.coordinates.cluster_kmeans(data=projected_microstate_coords_scaled, k=num_macrostates, max_iter=50)
      largest_assign=kmeans_obj.assign()[0]
      print('time macrostate kmeans finished', str(time.time()-time_start))
      all_assign=np.zeros(num_visited_microstates)
      all_assign[all_connect]=largest_assign
      all_assign[not_connect]=np.arange(not_connect.shape[0])+largest_assign.max()+1
    col=collections.Counter(all_assign)
    print("all assign counts", [col[k] for k in sorted(list(col))])
    print("reweight", np.concatenate(msm.trajectory_weights()).min(),np.concatenate(msm.trajectory_weights()).max())
    macrostate_assignment_of_visited_microstates=all_assign.astype('int')
    macrostate_counts = np.array([np.sum(s[states_unique][macrostate_assignment_of_visited_microstates == macrostate_label]) for macrostate_label in range(macrostate_assignment_of_visited_microstates.max()+1)])
    print("len not_connect", len(not_connect), 0.3*n_pick)
    if len(not_connect)<0.3*n_pick:
      selected_macrostate=np.concatenate((sorted(select_restart_state(macrostate_counts[:num_macrostates]+1, 'sto_inv_linear', np.arange(num_macrostates), nparallel=(n_pick-len(not_connect)))),np.arange(num_macrostates,len(macrostate_counts))))
    else:
      p=1./macrostate_counts
      p[macrostate_counts==0]=0
      p[num_macrostates:]=1./n_pick
      p[:num_macrostates]=p[:num_macrostates]/p[:num_macrostates].sum()
      p=p/p.sum()
      selected_macrostate=sorted(np.random.choice(np.arange(len(p)), p = p, size=n_pick))
    #macrostate_counts[num_macrostates:]+=macrostate_counts[:num_macrostates].min()+1
    #selected_macrostate = sorted(select_restart_state(macrostate_counts[macrostate_counts > 0], 'sto_inv_linear', np.arange(macrostate_counts.shape[0])[macrostate_counts > 0], nparallel=n_pick))
    print("macrostate_counts", macrostate_counts)
    print("inv macrostate_counts", 1./macrostate_counts)
    print("selected_macrostate", selected_macrostate)
    fig,ax = plt.subplots(nrows=1, figsize=(7,4.5), sharex=True)
    plt.scatter(projected_microstate_coords_scaled[:,0],projected_microstate_coords_scaled[:,1], c=largest_assign,cmap='nipy_spectral')
    plt.xlabel("MSM ev 1")
    plt.ylabel("MSM ev 2")
    fig.tight_layout()
    plt.savefig(resultspath+name_data+"msmevspace_i"+str(iter_found)+".png")
    fig,ax = plt.subplots(nrows=1, figsize=(7,4.5), sharex=True)
    pyemma.plots.plot_free_energy(yall[:,0],yall[:,1], nbins=64, weights=np.concatenate(msm.trajectory_weights()), cmap='nipy_spectral', ax=ax,legacy=False)
    ax.set_xlabel("SRV 1")
    ax.set_ylabel("SRV 2")
    for i in range(num_macrostates):
      ax.scatter(km.cluster_centers_[macrostate_assignment_of_visited_microstates==i,0], 
                      km.cluster_centers_[macrostate_assignment_of_visited_microstates==i,1], 
                      c=f"C{(1+i)}", s=25, lw=0.5, edgecolor="k", label=f"C{(1+i)}")
    for i in range(num_macrostates,len(macrostate_counts)):
      ax.scatter(km.cluster_centers_[macrostate_assignment_of_visited_microstates==i,0], 
                  km.cluster_centers_[macrostate_assignment_of_visited_microstates==i,1], 
                  c="black", s=70, lw=0.5, edgecolor="k")
    ax.legend( fontsize=8)
    fig.tight_layout()
    plt.savefig(resultspath+name_data+"macrostates_i"+str(iter_found)+".png")
    restart_state=np.empty((0))
    for i in range(n_pick):
        selected_macrostate_mask = (macrostate_assignment_of_visited_microstates == selected_macrostate[i])
        #print(selected_macrostate, microstate_transitions_used[visited_microstates], macrostate_counts, counts[states_unique][selected_macrostate])
        counts_in_selected_macrostate = s[states_unique][selected_macrostate_mask]+1
        #print(parameters['select_micro_within_macro_type'])
        if select_micro_within_macro_type == 'sto_inv_linear':
            # within a macrostate, select a microstate based on count
            add_microstate=select_restart_state(counts_in_selected_macrostate, 'sto_inv_linear', visited_microstates[selected_macrostate_mask], nparallel=1)
        elif select_micro_within_macro_type == 'rand': 
            add_microstate=select_restart_state(counts_in_selected_macrostate, 'rand', visited_microstates[selected_macrostate_mask], nparallel=1)
            #restart_state = [np.random.choice(visited_microstates[selected_macrostate_mask])] * nparallel
        print(i,selected_macrostate[i],add_microstate, counts_in_selected_macrostate,visited_microstates[selected_macrostate_mask])
        restart_state=np.append(restart_state,add_microstate)
    state_picks=sorted(restart_state.astype('int'))
    print("state_picks",state_picks)
    print("no exceptions")
    fig,ax = plt.subplots(nrows=1, figsize=(7,4.5), sharex=True)
    pyemma.plots.plot_free_energy(yall[:,0],yall[:,1], nbins=64, weights=np.concatenate(msm.trajectory_weights()), cmap='nipy_spectral', ax=ax,legacy=False)#,cax=ax)
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c= 'gray', s=2, lw=0.5, edgecolor="k", label=f"states")
    ax.scatter(km.cluster_centers_[state_picks,0], km.cluster_centers_[state_picks,1], c='red',s=25, lw=0.5, edgecolor="k", label=f"cmacro")
    #ax[1].set_xlabel("TIC 1")
    #ax[1].set_ylabel("TIC 3")
    ax.legend( fontsize=8)#loc=5
    fig.tight_layout()
    plt.savefig(resultspath+name_data+"cmacro_state_picks_i"+str(iter_found)+".png")
    fig,ax = plt.subplots(nrows=1, figsize=(7,4.5), sharex=True)
    pyemma.plots.plot_free_energy(yall[:,0],yall[:,2], nbins=64, weights=np.concatenate(msm.trajectory_weights()), cmap='nipy_spectral', ax=ax,legacy=False)#,cax=ax)
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 3")
    ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,2], c= 'gray', s=2, lw=0.5, edgecolor="k", label=f"states")
    ax.scatter(km.cluster_centers_[state_picks,0], km.cluster_centers_[state_picks,2], c='red',s=25, lw=0.5, edgecolor="k", label=f"cmacro")
    ax.legend( fontsize=8)#loc=5
    fig.tight_layout()
    plt.savefig(resultspath+name_data+"cmacro_state_picks2_i"+str(iter_found)+".png")
    fig,ax = plt.subplots(nrows=1, figsize=(7,4.5), sharex=True)
    pyemma.plots.plot_free_energy(yall[:,0],yall[:,1], nbins=64, weights=np.concatenate(msm.trajectory_weights()), cmap='nipy_spectral', ax=ax,legacy=False)
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    for i in np.unique(selected_macrostate):
        ax.scatter(km.cluster_centers_[macrostate_assignment_of_visited_microstates==i,0], 
                      km.cluster_centers_[macrostate_assignment_of_visited_microstates==i,1], 
                      c=f"C{(1+i)}", s=40, lw=0.5, edgecolor="k", label=f"C{(1+i)}")

    #ax.legend( fontsize=8)
    fig.tight_layout()
    plt.savefig(resultspath+name_data+"cmacro_state_picks3_i"+str(iter_found)+".png")
  picks = [
                frame_state_list[state][np.random.randint(0,
                len(frame_state_list[state]))]
                for state in state_picks
                ]
  print("picks",picks)
  traj_select = [traj_fns2[pick[0]] for pick in picks]
  traj_select_all=[trajprotdir+'alltrajs/alltraj'+t[len(trajprotdir):] for t in traj_select]
  frame_select = [pick[1]*vamp_stride for pick in picks]
  print("all",traj_select_all)
  print("time3", time.time()-time_start)
  idx=0
  print("writing")
  for idx in range(n_pick):
      try:
        sys.stdout.flush()
        tmp=mdtraj.load(topfileall)
        if os.path.isfile(traj_select_all[idx]):
          file = mdtraj.load(traj_select_all[idx], top=topfileall)
          tmp.xyz[0,:,:]=file.xyz[frame_select[idx],:,:]
          tmp.save_pdb(trajprotdir+'/restart/restart'+str(idx)+'.pdb')
          del tmp
          print(idx, end=' ')
        else:
          print(idx, "-missing", traj_select_all[idx])
      except:
        print(idx, "-fail write")
  print(" ")
  print("time4", time.time()-time_start)
  sys.stdout.flush()
  plt.close('all') 
  sleep(10)
 except Exception as e:
    print("fail", type(e).__name__) # returns the name of the exception
    print(e.__doc__)
    print(e)
    sleep(10)
