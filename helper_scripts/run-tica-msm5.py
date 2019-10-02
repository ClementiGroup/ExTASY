import sys, os
print(os.path.dirname(sys.executable))
import time
import os
import argparse
import numpy as np
import pyemma
import matplotlib
matplotlib.use('Agg')
import glob
import mdtraj as md
import imp
from matplotlib.pyplot import *
from pyemma import plots
matplotlib.rcParams.update({'font.size': 14})
print("pyemma version",pyemma.__version__)
import msmtools
import sklearn.preprocessing
from itertools import combinations


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
    else:
        print('ERROR: selected select_type in select_restart_state does not exist')




class Runticamsm(object):
    """
    A class used to perform the TICA and MSM
    """

    def create_arg_parser(self):

        parser = argparse.ArgumentParser(description="run TICA, MSM")

        # required options
        parser.add_argument("--path",
           type=str,
           dest="path",
           required=True,
           help="Full path of all files input, trajs, out")

        parser.add_argument("--cur_iter",
           type=int,
           dest="cur_iter",
           required=True)

        parser.add_argument("--n_select",
           type=int,
           dest="n_select",
           required=True)
        
        parser.add_argument("--Kconfig",
           type=str,
           dest="Kconfig",
           required=True)

        parser.add_argument("--ref",
           type=str,
           dest='ref',
           required=True)
        return parser

    def run(self):
        time_start=time.time()
        print("start")
        parser = self.create_arg_parser()
        args = parser.parse_args()
        
	
	#parser = argparse.ArgumentParser()
        #parser.add_argument('--Kconfig', help='link to Kernel configurations file')
        #parser.add_argument('--port', dest="port", help='port for RabbitMQ server', default=5672, type=int)
        #args = parser.parse_args()

        Kconfig = imp.load_source('Kconfig', args.Kconfig)


        pdb_file=glob.glob(args.path+'/iter*_input*.pdb')[0]
        #pdb_file=glob.glob('iter*_input*.pdb')[0]
        #traj_files=glob.glob(args.path+'/iter*_traj*.dcd')
        p_cont=True
        p_iter=0
        traj_files=[]
        traj_files_npy=[]
        iter_arr=[]
        while(p_cont):
           traj_files_tmp=glob.glob(args.path+'/iter'+str(p_iter)+'_traj*.dcd')
           traj_files_npy_tmp=glob.glob(args.path+'/iter'+str(p_iter)+'_traj*.npy')
           traj_files_tmp.sort()
           if len(traj_files_tmp)==0:
             p_cont=False
           else:
             print("iter", str(p_iter), " # files", str(len(traj_files_tmp)), "shape", np.load(traj_files_npy_tmp[0]).shape) 
             traj_files=traj_files+traj_files_tmp
             traj_files_npy=traj_files_npy+traj_files_npy_tmp
             iter_arr=iter_arr+[p_iter]*len(traj_files_tmp)
             p_iter=p_iter+1

        p_iter_max=p_iter-1
        iter_arr=np.array(iter_arr)
        #traj_files=glob.glob('iter*_traj*.dcd')
        #traj_files.sort()
        get_out_arr=[]
        for i, file in enumerate(traj_files_npy):
          get_out_arr=get_out_arr+[np.load(file)]
        
        print('npy shape', get_out_arr[0].shape)
        #topfile = md.load(pdb_file)
        #featurizer = pyemma.coordinates.featurizer(topfile)
        #featurizer.add_residue_mindist(residue_pairs='all', scheme='closest-heavy')
        #featurizer.add_backbone_torsions(cossin=True)
        #featurizer.dimension()

        #inp = pyemma.coordinates.source(traj_files, featurizer)
        #inp.get_output()
        #print("n atoms",topfile.n_atoms)
        #print("n frames total",inp.n_frames_total())
        #print("n trajs",inp.number_of_trajectories())
        #print(" traj lengths", inp.trajectory_lengths())
        #print(" input dimension",inp.dimension())

        
        tica_lag=Kconfig.tica_lag#1
        tica_dim=Kconfig.tica_dim
        tica_stride=Kconfig.tica_stride
        if Kconfig.koopman=='yes':
          try:
            tica_obj = pyemma.coordinates.tica(get_out_arr, lag=tica_lag, dim=tica_dim, kinetic_map=True, stride=tica_stride, weights='koopman')
            print("koopman works, koppman dim", tica_obj.timescales.shape[0])
            if tica_obj.timescales.shape[0]<int(tica_dim):
              print("koopman too few dimensions, using non-koopman tica")
              tica_obj = pyemma.coordinates.tica(get_out_arr, lag=tica_lag, dim=tica_dim, kinetic_map=True, stride=tica_stride, weights='empirical')
            else:
              tica_dim=tica_obj.timescales.shape[0]  
          except:
            tica_obj = pyemma.coordinates.tica(get_out_arr, lag=tica_lag, dim=tica_dim, kinetic_map=True, stride=tica_stride, weights='empirical') 
            print("koopman failed, using non-koopman tica")    
        else:
          tica_obj = pyemma.coordinates.tica(get_out_arr, lag=tica_lag, dim=tica_dim, kinetic_map=True, stride=tica_stride, weights='empirical')
        
        #  tica_weights='empirical', tica_weights='koopman'
        #tica_obj = pyemma.coordinates.tica(inp, lag=tica_lag, dim=tica_dim, kinetic_map=True, stride=tica_stride, weights=tica_weights)
        print("iter", args.cur_iter)
        print("TICA eigenvalues", tica_obj.eigenvalues)
        tica_timescales_arr=tica_obj.timescales
        print("TICA timescales",tica_timescales_arr.shape[0], tica_timescales_arr)

        y = tica_obj.get_output(stride=tica_stride)
        np.save(args.path+'/npy_iter'+str(args.cur_iter)+'_tica_y.npy',y)
        #y[0].shape
        print('time tica finished', str(time.time()-time_start))
        msm_states=Kconfig.msm_states
        msm_stride=Kconfig.msm_stride
        msm_lag=Kconfig.msm_lag
        cl = pyemma.coordinates.cluster_kmeans(data=y, k=msm_states, max_iter=10, stride=msm_stride)
        #np.save(args.path+'/npy_iter'+str(args.cur_iter)+'_tica_cl.npy',cl)
        np.save(args.path+'/npy_iter'+str(args.cur_iter)+'_tica_dtrajs.npy',cl.dtrajs)
        #cl = pyemma.coordinates.cluster_mini_batch_kmeans(data=y, k=msm_states, max_iter=10, n_jobs=None)
        print('time kmeans finished', str(time.time()-time_start)) 
        
        m = pyemma.msm.estimate_markov_model(cl.dtrajs, msm_lag)
        np.save(args.path+'/npy_iter'+str(args.cur_iter)+'_tica_m.npy',m)
        print('time msm finished', str(time.time()-time_start))



        ########################################
        #print(tica_obj.eigenvectors)

        print("MSM eigenvalues",m.eigenvalues(10))
        #print(m.eigenvectors_left(10))
        #print(m.eigenvectors_right(10))
        print("MSM P connected",m.P)  #only connected

        #print("MSM clustercenters",cl.clustercenters)
        
        print("MSM timescales", m.timescales(10))
        #print("MSM stat", m.stationary_distribution)
        print("MSM active set", m.active_set)
        print('fraction of states used = ', m.active_state_fraction)
        print('fraction of counts used = ', m.active_count_fraction)

        c = m.count_matrix_full
        s =  np.sum(c, axis=1)
        print("count matrix sums",s)
        #if 0 not in s:
        q = 1.0 / s
        q[s<1]=1.0
        n_states=c.shape[0]

        dtrajs = [ t for t in cl.dtrajs ]

        #print("msm dtrajs", dtrajs)

        #get frame_list for each msm state
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


        n_pick=int(args.n_select)#100

        if Kconfig.strategy=='cmicro':
          print("probpick top 10", np.sort(q)[::-1][:10], np.sort(q)[::-1][:10].sum(), np.argsort(q)[::-1][:10])
          state_picks = np.random.choice(np.arange(len(q)), size=n_pick, p=q)
          state_picks=state_picks.astype('int')
          print("state_picks",np.sort(state_picks))
          print("no exceptions")
        elif Kconfig.strategy=='cmacro':
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

          #print("C_largest", C_largest.shape[0])
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


          projected_microstate_coords_scaled = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(current_eigenvecs[:,1:])

          projected_microstate_coords_scaled *= np.sqrt(current_timescales[:num_eigenvecs_to_compute-1] / current_timescales[0]).reshape(1, num_eigenvecs_to_compute-1)

          select_n_macro_type=Kconfig.select_n_macro_type #'kin_content' #Kconfig.select_n_macro_type
          if select_n_macro_type == 'const': # 1_over_cmacro_estim
              par_num_macrostates=int(Kconfig.num_macrostates)#30    
              num_macrostates = min(par_num_macrostates,num_visited_microstates)
          elif select_n_macro_type == 'kin_var': # 1_over_cmacro_estim3
              frac_kin_var=0.5
              kin_var = np.cumsum(current_eigenvals**2)
              cut = kin_var[kin_var < kin_var.max()*frac_kin_var]
              num_macrostates = min(max(cut.shape[0],1),num_visited_microstates)
          elif select_n_macro_type == 'kin_content': # 1_over_cmacro_estim4
              frac_kin_content=0.5
              kin_cont = np.cumsum(-1./np.log(np.abs(current_eigenvals[1:])))/2.
              cut = kin_cont[kin_cont < kin_cont.max()*frac_kin_content]
              num_macrostates = min(max(cut.shape[0],1),num_visited_microstates)

          macrostate_method='pcca'
          #macrostate_method='kmeans'
          if macrostate_method=='pcca':
            m.pcca(num_macrostates)
            macrostate_assignments = { k:v for k,v in enumerate(m.metastable_sets) }
            largest_assign = m.metastable_assignments
            print("macrostate assignments", macrostate_assignments)
            print("mismatch", "largest_assign", largest_assign.shape, "num_visited_microstates", num_visited_microstates) 
            #all_assign=largest_assign
            all_assign=np.zeros(num_visited_microstates)
            all_assign[all_connect]=largest_assign
            all_assign[not_connect]=np.arange(not_connect.shape[0])+largest_assign.max()+1
            print('time macrostate pcca finished', str(time.time()-time_start))
          else:
            kmeans_obj = pyemma.coordinates.cluster_kmeans(data=projected_microstate_coords_scaled, k=num_macrostates, max_iter=10)
            largest_assign=kmeans_obj.assign()[0]
            print('time macrostate kmeans finished', str(time.time()-time_start))
            all_assign=np.zeros(num_visited_microstates)
            all_assign[all_connect]=largest_assign
            all_assign[not_connect]=np.arange(not_connect.shape[0])+largest_assign.max()+1
          
          macrostate_assignment_of_visited_microstates=all_assign.astype('int')
          np.save(args.path+'/npy_iter'+str(args.cur_iter)+'_msm_macrostates.npy',macrostate_assignment_of_visited_microstates)
          print("all_assign",all_assign)


          select_macro_type = 'sto_inv_linear'
          if select_macro_type=='dmdmd':
            macrostate_counts = np.array([np.sum(s[states_unique][macrostate_assignment_of_visited_microstates == macrostate_label]) for macrostate_label in range(macrostate_assignment_of_visited_microstates.max()+1)])
            selected_macrostate = select_restart_state(macrostate_counts[macrostate_counts > 0], 'rand', np.arange(macrostate_counts.shape[0])[macrostate_counts > 0], nparallel=nparallel)
            #print(macrostate_counts[macrostate_counts > 0], np.arange(num_macrostates)[macrostate_counts > 0], selected_macrostate)
          elif select_macro_type == 'sto_inv_linear': 
            macrostate_counts = np.array([np.sum(s[states_unique][macrostate_assignment_of_visited_microstates == macrostate_label]) for macrostate_label in range(macrostate_assignment_of_visited_microstates.max()+1)])
            
            selected_macrostate = select_restart_state(macrostate_counts[macrostate_counts > 0], 'sto_inv_linear', np.arange(macrostate_counts.shape[0])[macrostate_counts > 0], nparallel=n_pick)

          print("macrostate_counts", macrostate_counts)
          print("selected_macrostate", selected_macrostate)




          select_micro_within_macro_type='sto_inv_linear'
          restart_state=np.empty((0))
          for i in range(n_pick):
            selected_macrostate_mask = (macrostate_assignment_of_visited_microstates == selected_macrostate[i])
            #print(selected_macrostate, microstate_transitions_used[visited_microstates], macrostate_counts, counts[states_unique][selected_macrostate])
            counts_in_selected_macrostate = s[states_unique][selected_macrostate_mask]
            #print(parameters['select_micro_within_macro_type'])
            if select_micro_within_macro_type == 'sto_inv_linear':
                # within a macrostate, select a microstate based on count
                add_microstate=select_restart_state(counts_in_selected_macrostate, 'sto_inv_linear', visited_microstates[selected_macrostate_mask], nparallel=1)
            elif select_micro_within_macro_type == 'rand': 
                add_microstate=select_restart_state(counts_in_selected_macrostate, 'rand', visited_microstates[selected_macrostate_mask], nparallel=1)
                #restart_state = [np.random.choice(visited_microstates[selected_macrostate_mask])] * nparallel
            restart_state=np.append(restart_state,add_microstate)
            #print(i,selected_macrostate[i], add_microstate)

          state_picks=restart_state.astype('int')
          print("state_picks",np.sort(state_picks))
          print("no exceptions")
          #except:
          #state_picks = np.random.choice(np.arange(len(q)), size=n_pick, p=q)
          #print("state_picks",state_picks)
          #print("exception found")           
        elif Kconfig.strategy=='spec':
          num_eigenvecs_to_compute = 10
          microstate_transitions_used=c
          num_visited_microstates=c.shape[0]
          states_unique=np.arange(num_visited_microstates)
          visited_microstates=states_unique
            
          largest_visited_set=msmtools.estimation.largest_connected_set(microstate_transitions_used)
          C_largest0=microstate_transitions_used[largest_visited_set, :][:, largest_visited_set]
          rowsum = np.ravel(C_largest0.sum(axis=1))
          largest_visited_set2=largest_visited_set[rowsum>0]
          C_largest=microstate_transitions_used[largest_visited_set2, :][:, largest_visited_set2]
          rowsum = C_largest.sum(axis=1)

          print("min rowsum", np.min(rowsum), "max rowsum", np.max(rowsum))

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
          use_l=states_largest.shape[0]

          import scipy 
          sc_eg,sc_ev1, sc_ev2=scipy.linalg.eig(T_largest, left=True, right=True)
          d_arr=np.zeros((use_l,use_l))
          for idx in range(use_l):
            #print(idx)
            for idx2 in range(use_l):
              mat_d=np.zeros((use_l,use_l))
              mat_d[idx,idx2]=1.
              d_eg=np.matmul(sc_ev1[:,1],np.matmul(mat_d, sc_ev2[:,1]))
              d_arr[idx,idx2]=d_eg

          print("d_arr min", d_arr.min(), "d_arr max", d_arr.max())

          T_largest_tmp=C_largest / rowsum[:, np.newaxis]
          c_diff=np.zeros(use_l)
          delta_exp=np.zeros(use_l)
          for idx in range(use_l):
           factor=np.zeros(use_l)
           for idx2 in range(use_l):
            delta_c=np.zeros(use_l)
            delta_c[idx2]=1.
            c_diff=(C_largest[idx,:]+delta_c)/(rowsum[idx]+1)-T_largest_tmp[idx]
            factor[idx2]=np.abs(np.sum(c_diff*d_arr[idx,:]))  
           delta_exp[idx]=np.matmul(T_largest[idx,:],factor)

          print("delta_exp min", delta_exp.min(), "delta_exp max", delta_exp.max())


          p1=1./rowsum
          p1=p1/p1.sum()
          av_delta1=np.sum(delta_exp*p1)
          #print(av_delta1)
          p2=delta_exp**3
          p2=p2/p2.sum()
          av_delta2=np.sum(delta_exp*p2)
          #print(av_delta2)
          print("expected improvement","cmicro", av_delta1, "spec",  av_delta2, "ratio", av_delta2/av_delta1)

          prob_pick=np.zeros(n_states)
          prob_pick[all_connect]=delta_exp**3
          prob_pick=prob_pick/prob_pick.sum()

          print("probpick top 10", np.sort(prob_pick)[::-1][:10], np.sort(prob_pick)[::-1][:10].sum(), np.argsort(prob_pick)[::-1][:10])
          state_picks = np.random.choice(np.arange(len(q)), size=n_pick, p=prob_pick)
          state_picks=state_picks.astype('int')
          print("state_picks",np.sort(state_picks))

          clf()
          fig=figure()
          ax = fig.add_subplot(111)
          imshow(d_arr)
          ax.set_ylabel('Tmatrix axis')
          ax.set_xlabel('Tmatrix axis')
          colorbar()
          #ax.set_yscale('log')
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_d_arr.png', bbox_inches='tight', dpi=200)


          clf()
          fig=figure()
          ax = fig.add_subplot(111)
          ax.scatter(1./rowsum,delta_exp)
          ax.set_xlabel('inverse counts')
          ax.set_ylabel('spec strategy delta_exp')
          ylim([delta_exp.min(),delta_exp.max()])
          #ax.set_yscale('log')
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_correlation_spec_counts.png', bbox_inches='tight', dpi=200)

          clf()
          fig=figure()
          ax = fig.add_subplot(111)
          ax.scatter(sc_ev2[:,1],delta_exp)
          ax.set_xlabel('first right eigenvector')
          ax.set_ylabel('spec strategy delta_exp')
          #yscale('log')
          ylim([delta_exp.min(),delta_exp.max()])
          #ax.set_yscale('log')
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_correlation_spec_ev1.png', bbox_inches='tight', dpi=200)

          clf()
          fig=figure()
          ax = fig.add_subplot(111)
          ax.scatter(rowsum,delta_exp)
          ax.set_xlabel('counts')
          ax.set_ylabel('spec strategy delta_exp')
          #yscale('log')
          ylim([delta_exp.min(),delta_exp.max()])
          #ax.set_yscale('log')
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_correlation_counts_ev1.png', bbox_inches='tight', dpi=200)
          
          print("no exceptions")
        else:
          print("didn't recognize strategy")
        print("selected msm restarts", state_picks)        

        picks = [
            frame_state_list[state][np.random.randint(0,
            len(frame_state_list[state]))]
            for state in state_picks
            ]




        traj_select = [traj_files[pick[0]] for pick in picks]
        frame_select = [pick[1]*tica_stride*msm_stride for pick in picks]
        print('picks',picks)
        print('frame_select picks', frame_select)
        print('traj_select',traj_select)
        print('time frame selection finished', str(time.time()-time_start))
        text_file = open(args.path + "/traj_select.txt", "w")
        for idx in range(n_pick):
          text_file.write(traj_select[idx]+' to iter '+str(args.cur_iter)+' idx '+str(idx)+' \n')

        text_file.close()


        # write new input files from frames


        for idx in range(n_pick):
          print(idx, end='  ')
          tmp =md.load(args.path+'/iter0_input0.pdb')
          files = md.load(traj_select[idx], top=args.path+'/iter0_input0.pdb')
          tmp.xyz[0,:,:]=files.xyz[frame_select[idx],:,:]
          tmp.save_pdb(args.path+'/iter'+str(args.cur_iter+1)+'_input'+str(idx)+'.pdb')

        print('time writing new frames finished', str(time.time()-time_start))
        #rg rmsd
        save_old_rgrmsd=False
        if save_old_rgrmsd:
          out_files=glob.glob(args.path+'/iter*_feat*_rgrmsdq.npy')
          out_files.sort()
          print("out_files", out_files)
          rg_arr=[]
          rmsd_arr=[]
          q_arr=[]
          for file in out_files:
            vals=np.load(file)
            rg_arr.append(vals[0])
            rmsd_arr.append(vals[1])
            q_arr.append(vals[2])
          print("rgrmsdq shapes", len(rg_arr), len(rmsd_arr), len(q_arr))
          print('time rg rmsd q loaded', str(time.time()-time_start))
        
        ########################################
        #python2.7
        #colornames=[name for name, color in matplotlib.colors.cnames.iteritems()]
        #python3
        colornames=[name for name, color in matplotlib.colors.cnames.items()] 
        tica0=np.array([])
        tica1=np.array([])
        for i in range(len(y)):
          tica0=np.append(tica0,y[i][:,0])
          tica1=np.append(tica1,y[i][:,1])


        clf()
        fig=figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.arange(tica_obj.timescales.shape[0]),tica_obj.timescales)
        ax.set_ylabel('TICA Timescales (steps)')
        ax.set_xlabel('# TICA eigenvector')
        ax.set_yscale('log')
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_tica_timescales.png', bbox_inches='tight', dpi=200)

        cumvar = np.cumsum(tica_obj.timescales)
        cumvar /= cumvar[-1]
        clf()
        plot(cumvar, linewidth=2)
        for thres in [0.5,0.8,0.95]:
          threshold_index=np.argwhere(cumvar > thres)[0][0]
          print("tica thres, thres_idx", thres, threshold_index)
          vlines(threshold_index, 0.0, 1.0, linewidth=2)
          hlines(thres, 0, cumvar.shape[0], linewidth=2)

        xlabel('Eigenvalue Number', fontsize = 16)
        ylabel('cumulative kinetic content', fontsize = 16)
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_tica_cumulative_kinetic_content.png', bbox_inches='tight', dpi=200)

        msm_timescales=m.timescales(100)
        clf()
        fig=figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.arange(msm_timescales.shape[0]),msm_timescales*tica_stride)
        ax.set_ylabel('MSM Timescales (steps)')
        ax.set_xlabel('# MSM eigenvector')
        ax.set_yscale('log')
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_timescales.png', bbox_inches='tight', dpi=200)

        cumvar = np.cumsum(m.timescales(100))
        cumvar /= cumvar[-1]
        clf()
        plot(cumvar, linewidth=2)
        for thres in [0.5,0.8,0.95]:
          threshold_index=np.argwhere(cumvar > thres)[0][0]
          print("msm thres, thres_idx", thres, threshold_index)
          vlines(threshold_index, 0.0, 1.0, linewidth=2)
          hlines(thres, 0, cumvar.shape[0], linewidth=2)

        xlabel('Eigenvalue Number', fontsize = 16)
        ylabel('cumulative kinetic content', fontsize = 16)
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_cumulative_kinetic_content.png', bbox_inches='tight', dpi=200)





        clf()
        xlabel("TICA ev0")
        ylabel("TICA ev1")
        cp = scatter(tica0, tica1, s=10, c='blue', marker='o', linewidth=0.,cmap='jet', label='MSM states')
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_tica_evs.png', bbox_inches='tight', dpi=200)


        clf()
        fig, ax = plots.plot_free_energy(tica0, tica1,cmap='Spectral')
        xlabel("TICA ev0")
        ylabel("TICA ev1")
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_tica_evs2.png', bbox_inches='tight', dpi=200)

        clf()
        fig, ax = plots.plot_free_energy(tica0, tica1,cmap='Spectral')
        cp = scatter(cl.clustercenters[:,0], cl.clustercenters[:,1], s=10, c='blue', marker='o', linewidth=0.,cmap='jet', label='MSM state centers')
        xlabel("TICA ev0")
        ylabel("TICA ev1")
        legend()
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_tica_evs3_centers.png', bbox_inches='tight', dpi=200)

        #plot msm ev
        clf()
        xlabel("MSM ev1")
        ylabel("MSM ev2")
        cp = scatter(m.eigenvectors_right(10)[:,1], m.eigenvectors_right(10)[:,2], s=10, c='blue', marker='o', linewidth=0.,cmap='jet', label='MSM states')
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_evs.png', bbox_inches='tight', dpi=200)

        #plot msm ev
        clf()
        fig, ax = plots.plot_free_energy(m.eigenvectors_right(10)[:,1], m.eigenvectors_right(10)[:,2], cmap='Spectral', weights=m.stationary_distribution, nbins=30)
        xlabel("MSM ev1")
        ylabel("MSM ev2")
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_evs2.png', bbox_inches='tight', dpi=200)

        if save_old_rgrmsd:
          clf()
          xlabel("RMSD")
          ylabel("Rg")
          cp = scatter(rmsd_arr, rg_arr, s=10, c='blue', marker='o', linewidth=0.,cmap='jet', label='MSM states')
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_rgrmsd.png', bbox_inches='tight', dpi=200)

        if save_old_rgrmsd:
          clf()
          fig, ax = plots.plot_free_energy(rmsd_arr, rg_arr, cmap='Spectral', nbins=30)
          xlabel("RMSD")
          ylabel("Rg") 
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_rgrmsd2.png', bbox_inches='tight', dpi=200)

        if save_old_rgrmsd:
          clf()
          xlabel("Q")
          ylabel("Rg")
          cp = scatter(q_arr, rg_arr, s=10, c='blue', marker='o', linewidth=0.,cmap='jet', label='MSM states')
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_qrg.png', bbox_inches='tight', dpi=200)

        if save_old_rgrmsd:
          clf()
          fig, ax = plots.plot_free_energy(q_arr, rg_arr, cmap='Spectral', nbins=10)
          xlabel("Q")
          ylabel("Rg")
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_qrg_2.png', bbox_inches='tight', dpi=200)

        if save_old_rgrmsd:
          #Q 1d free energy
          clf()
          z, x = np.histogram(q_arr, bins=10)
          F = -np.log(z)
          F=F-F.min()
          plot(x[1:], F)
          scatter(x[1:], F)
          xlabel('Q', fontsize = 15)
          ylabel('Free Energy [kT]', fontsize =15)
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_free_energy_q.png', bbox_inches='tight', dpi=200)

        #MSM 1d free energy
        clf()
        n_step=int(m.P.shape[0]/10)
        bins=np.sort(m.eigenvectors_right(10)[:,1])[::n_step]
        bins=np.append(bins,np.sort(m.eigenvectors_right(10)[:,1])[-1])
        z, x = np.histogram(m.eigenvectors_right(10)[:,1], weights=m.stationary_distribution, density=True, bins=bins)
        F = -np.log(z)
        F=F-F.min()
        plot(x[1:], F)
        scatter(x[1:], F)
        xlabel('MSM ev1', fontsize = 15)
        ylabel('Free Energy [kT]', fontsize =15)
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_free_energy.png', bbox_inches='tight', dpi=200)


        #which tica frames seleted


        tica0_sel=np.array([])
        tica1_sel=np.array([])
        for i in range(n_pick):
          tica0_sel=np.append(tica0_sel,y[picks[i][0]][frame_select[i],0])
          tica1_sel=np.append(tica1_sel,y[picks[i][0]][frame_select[i],1])

        clf()
        xlabel("TICA ev0")
        ylabel("TICA ev1")
        cp = scatter(tica0, tica1, s=10, c='blue', marker='o', linewidth=0.,cmap='jet', label='all frames')
        cp = scatter(tica0_sel, tica1_sel, s=10, c='red', marker='o', linewidth=0.,cmap='jet', label='selected')
        legend()
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_tica_evs4_selected.png', bbox_inches='tight', dpi=200)


        print('time ck test starting', str(time.time()-time_start))


        #m.ck_test

        ck=m.cktest(3, mlags=None)

        clf()
        pyemma.plots.plot_cktest(ck, diag=True, figsize=(7,7), layout=(2,2), padding_top=0.1, y01=False, padding_between=0.3)#, dt=0.1, units='ns')
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_cktest.png')
        print('time ck test finished', str(time.time()-time_start))

        #lags = [1,2,5,10,20,50,100,200, 500,1000]
        #its = pyemma.msm.its(dtrajs, nits=10, lags=lags)
        #clf()
        #pyemma.plots.plot_implied_timescales(its, ylog=True, units='steps', linewidth=2)
        #xlim(0, 40); ylim(0, 120);
        #savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_its.png', bbox_inches='tight', dpi=200)

        plot_its=False
        if plot_its:
          its = pyemma.msm.its(dtrajs, errors='bayes', nits=10)
          clf()
          pyemma.plots.plot_implied_timescales(its, ylog=True, units='steps', linewidth=2)
          #xlim(0, 40); ylim(0, 120);
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_its2.png', bbox_inches='tight', dpi=200)
          print('time its finished', str(time.time()-time_start))
        #clf()
        #pyemma.plots.plot_implied_timescales(its, ylog=False, units='steps', linewidth=2, show_mle=False)
        ##xlim(0, 40); ylim(0, 120);
        #savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_its3.png', bbox_inches='tight', dpi=200)


        #which msm states selected
        #warning m only connected, c full -selected
        #m.active_set
        #state_picks
        #msm_states
        p_picks_active=[]
        for i in state_picks:
          if i in m.active_set:
            p_picks_active.append(np.argwhere(i==m.active_set)[0][0])

        p_picks_active=np.unique(np.array(p_picks_active)).astype(int)
          



        clf()
        xlabel("MSM ev1")
        ylabel("MSM ev2")
        s_dot=30
        cp = scatter(m.eigenvectors_right(10)[:,1], m.eigenvectors_right(10)[:,2], s=s_dot, c='blue', marker='o', linewidth=0.,cmap='jet', label='MSM states')
        cp = scatter(m.eigenvectors_right(10)[p_picks_active,1], m.eigenvectors_right(10)[p_picks_active,2], s=s_dot, c='red', marker='o', linewidth=0.,cmap='jet', label='selected')
        legend(loc='center left', bbox_to_anchor=(1, 0.5))
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_evs_4_select.png', bbox_inches='tight', dpi=200)




        p_states=np.array([])
        p_unique=[]
        for p_iter in range(p_iter_max+1):
            p_arr=np.argwhere(iter_arr==p_iter)
            for i in p_arr:
              #print(i[0])
              p_states=np.append(p_states,dtrajs[i[0]])
            p_states=np.unique(p_states).astype(int)
            p_unique.append(p_states.shape[0])

        p_unique=np.array(p_unique)

        np.save(args.path+'/npy_iter'+str(args.cur_iter)+'_p_unique.npy',p_unique)
        clf()
        fig=figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.arange(p_unique.shape[0]),p_unique)
        ax.set_ylabel('# of current msm states explored')
        ax.set_xlabel('iteration')
        #ax.set_yscale('log')
        ax.set_ylim(0, ax.get_ylim()[1]);
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_strategy.png', bbox_inches='tight', dpi=200)


        clf()
        xlabel("TICA ev0")
        ylabel("TICA ev1")

        for p_iter in range(p_iter_max,-1,-1):
            p_arr=np.argwhere(iter_arr==p_iter)
            tica0_tmp=np.array([])
            tica1_tmp=np.array([])
            for i in p_arr:
              #print(i[0])
              tica0_tmp=np.append(tica0_tmp,y[i[0]][:,0])
              tica1_tmp=np.append(tica1_tmp,y[i[0]][:,1])
            cp = scatter(tica0_tmp, tica1_tmp, s=10, marker='o', linewidth=0.,cmap='jet', c=colornames[p_iter], label='iter '+str(p_iter))

        #legend(loc='center left', bbox_to_anchor=(1, 0.5))
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_tica_evs5_iters.png', bbox_inches='tight', dpi=200)


        clf()
        xlabel("MSM ev1")
        ylabel("MSM ev2")
        for p_iter in range(p_iter_max,-1,-1):
            p_arr=np.argwhere(iter_arr==p_iter)
            p_states=np.array([])
            for i in p_arr:
              #print i[0]
              p_states=np.append(p_states,dtrajs[i[0]])
            p_states=np.unique(p_states).astype(int)
            p_states_active=[]
            for i in p_states:
              if i in m.active_set:
                p_states_active.append(np.argwhere(i==m.active_set)[0][0])
            p_states_active=np.unique(np.array(p_states_active)).astype(int)
            cp = scatter(m.eigenvectors_right(10)[p_states_active,1], m.eigenvectors_right(10)[p_states_active,2], s=s_dot,  marker='o', linewidth=0., cmap='spectral', c=colornames[p_iter], label='iter '+str(p_iter))

        if int(args.cur_iter)<15:
          legend(loc='center left', bbox_to_anchor=(1, 0.5))
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_msm_evs_3_iter.png', bbox_inches='tight', dpi=200)
        
        print('time plotting rgrmsd2 starting', str(time.time()-time_start))
        #out_files=glob.glob(args.path+'/iter*_feat*_rgrmsdq2.npy')
        #out_files.sort()
        out_files=[]
        for p_iter in range(p_iter_max+1):
           out_files_tmp=glob.glob(args.path+'/iter'+str(p_iter)+'_feat*_rgrmsdq2.npy')
           #traj_files_npy_tmp=glob.glob(args.path+'/iter'+str(p_iter)+'_traj*.npy')
           out_files_tmp.sort()
           out_files=out_files+out_files_tmp

        #out_files=np.array(out_files)
        print("has to match", traj_files[-1], out_files[-1])
        print("out_files", out_files) 
        rg_arr=np.array([])
        rmsd_arr=np.array([])
        q_arr=np.array([])

        for file in out_files:
          vals=np.load(file)
          rg_arr=np.append(rg_arr,vals[0])
          rmsd_arr =np.append(rmsd_arr,vals[1])
          q_arr=np.append(q_arr,vals[2])

        print("rgrmsdq shapes", rg_arr.shape, rmsd_arr.shape, q_arr.shape)
        system= Kconfig.system
        xlimmin=0; xlimmax=0.9; ylimmin=0.4; ylimmax=1.0
        if system=='CLN025':
          xlimmin=0; xlimmax=0.9; ylimmin=0.4; ylimmax=1.0
        if system=='2F4K':
          xlimmin=0; xlimmax=1.8; ylimmin=0.75; ylimmax=2.3
        if system=='1FME':
          xlimmin=0; xlimmax=1.7; ylimmin=0.75; ylimmax=2.4
        if system=='NuG2':
          xlimmin=0; xlimmax=2.7; ylimmin=0.95; ylimmax=2.8

        #clf()
        #xlabel("RMSD [nm]")
        #ylabel("Rg [nm]")
        #xlim(0, 1.0); ylim(0.4, 1.4);
        #cp = scatter(rmsd_arr, rg_arr, s=3, c='blue', marker='o', linewidth=0.,cmap='jet', label='MSM states')
        #savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_p2_rgrmsd_1.png', bbox_inches='tight', dpi=200)


        clf()
        xlabel("RMSD [nm]")
        ylabel("Rg [nm]")
        #xlim(0, 1.0); ylim(0.4, 1.4);
        xlim(xlimmin, xlimmax); ylim(ylimmin, ylimmax);
        cp = scatter(rmsd_arr, rg_arr, s=3, c='blue', marker='o', linewidth=0.,cmap='jet', label='MSM states')
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_p2_rgrmsd_2.png', bbox_inches='tight', dpi=200)


        clf()
        fig, ax = plots.plot_free_energy(rmsd_arr, rg_arr, cmap='Spectral', nbins=100)
        xlabel("RMSD [nm]")
        ylabel("Rg [nm]") 
        #xlim(0, 1.0); ylim(0.4, 1.4);
        xlim(xlimmin, xlimmax); ylim(ylimmin, ylimmax);
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_p2_rgrmsd_3.png', bbox_inches='tight', dpi=200)

        clf()
        fig, ax = plots.plot_free_energy(rmsd_arr, rg_arr, cmap='Spectral', nbins=200)
        xlabel("RMSD [nm]")
        ylabel("Rg [nm]") 
        xlim(xlimmin, xlimmax); ylim(ylimmin, ylimmax); #xlim(0, 1.0); ylim(0.4, 1.4);
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_p2_rgrmsd_4.png', bbox_inches='tight', dpi=200)


        clf()
        xlabel("Q")
        ylabel("Rg")
        cp = scatter(q_arr, rg_arr, s=3, c='blue', marker='o', linewidth=0.,cmap='jet', label='MSM states')
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_p2_qrg.png', bbox_inches='tight', dpi=200)

        clf()
        fig, ax = plots.plot_free_energy(q_arr, rg_arr, cmap='Spectral', nbins=100)
        xlabel("Q")
        ylabel("Rg")
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_p2_qrg_2.png', bbox_inches='tight', dpi=200)
        
        mstat=m.stationary_distribution
        active_set=m.active_set
        c = m.count_matrix_full
        s =  np.sum(c, axis=1)
        print("row_sums", s.min(), s.max(), s)
        n_states=c.shape[0]
        mstat_all=np.zeros(n_states)
        mstat_all[active_set]=mstat
        #ratio=mstat_all/(s/s.sum())
        #ratio=ratio/ratio.min()
        #ratio.max()
        #s.sum()

        dtrajs = [ t for t in cl.dtrajs ]
        su=0
        for idx in range(len(dtrajs)):
          su=su+len(dtrajs[0])

        frame_state_sum = np.zeros(n_states)
        for nn, dt in enumerate(dtrajs):
            for mm, state in enumerate(dt):
                    frame_state_sum[state]=frame_state_sum[state]+1

        print(frame_state_sum.sum())
        print(frame_state_sum[active_set].sum())
        ratio2=np.ones(n_states)

        ratio2[active_set]=mstat/(frame_state_sum[active_set]/frame_state_sum[active_set].sum())

        print("ratio2", ratio2.min(),ratio2.max(), ratio2.shape)
        #ratio2[dtrajs[0]]
        dtraj_weights = [ ratio2[t] for t in dtrajs ]
        
        np.save(args.path+'/npy_iter'+str(args.cur_iter)+'_tica_dtraj_weights.npy',dtraj_weights)

        dtraj_arr=np.array([])
        for t in dtrajs:
          dtraj_arr=np.append(dtraj_arr,t)

        dtraj_w_arr=np.array([])
        for t in dtraj_weights:
          dtraj_w_arr=np.append(dtraj_w_arr,t)

        print("shapes", dtrajs[0].shape, dtraj_weights[0].shape)
        #clf()
        #fig, ax = plots.plot_free_energy(rmsd_arr, rg_arr, weights=dtraj_w_arr, cmap='Spectral', nbins=100)
        #xlabel("RMSD [nm]")
        #ylabel("Rg [nm]") 
        #xlim(xlimmin, xlimmax); ylim(ylimmin, ylimmax);
        #savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_p2_rgrmsd_3_weights.png', bbox_inches='tight', dpi=200)

        try:
          clf()
          fig, ax = plots.plot_free_energy(rmsd_arr, rg_arr, weights=dtraj_w_arr, cmap='Spectral', nbins=200)
          xlabel("RMSD [nm]")
          ylabel("Rg [nm]") 
          xlim(xlimmin, xlimmax); ylim(ylimmin, ylimmax);
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_p2_rgrmsd_4_weights.png', bbox_inches='tight', dpi=200)
        except:
          print("failed "+'plot_iter'+str(args.cur_iter)+'_p2_rgrmsd_4_weights.png')

        try:
          clf()
          scatter(rmsd_arr, rg_arr, s=10, c=dtraj_w_arr, marker='o', linewidth=0.,cmap='jet', label='MSM states')
          #fig, ax = plots.plot_free_energy(rmsd_arr, rg_arr, weights=dtraj_w_arr, cmap='Spectral', nbins=200)
          xlabel("RMSD [nm]")
          xlim(xlimmin, xlimmax); ylim(ylimmin, ylimmax);
          colorbar()
          savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_p2_rgrmsd_4_weights2.png', bbox_inches='tight', dpi=200)
        except:
          print("failed "+'plot_iter'+str(args.cur_iter)+'_p2_rgrmsd_4_weights2.png')  

        clf()
        xlabel("TICA ev0")
        ylabel("TICA ev1")
        cp = scatter(tica0, tica1, s=10, c=dtraj_arr, marker='o', linewidth=0.,cmap='jet', label='MSM states')
        colorbar()
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_tica_evs_mstates.png', bbox_inches='tight', dpi=200)

        clf()
        z, x = np.histogram(q_arr, bins=10)
        F = -np.log(z)
        F=F-F.min()
        plot(x[1:], F)
        scatter(x[1:], F)
        xlabel('Q', fontsize = 15)
        ylabel('Free Energy [kT]', fontsize =15)
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_free_energy_q.png', bbox_inches='tight', dpi=200)

        clf()
        z, x = np.histogram(q_arr, bins=25)
        print("q min max", q_arr.min(), q_arr.max())
        F = -np.log(z)
        F=F-F.min()
        plot(x[1:], F)
        scatter(x[1:], F)
        xlabel('Q', fontsize = 15)
        ylabel('Free Energy [kT]', fontsize =15)
        savefig(args.path+'/plot_iter'+str(args.cur_iter)+'_free_energy_q_2.png', bbox_inches='tight', dpi=200)


        import resource
        print("memory",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000,"MB")
        print('time plotting finished', str(time.time()-time_start))


if __name__ == '__main__':
    Runticamsm().run()
