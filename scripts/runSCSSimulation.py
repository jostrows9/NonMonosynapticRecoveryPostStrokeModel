from itertools import product
import pickle
import sys
sys.path.append('../ModelingMechanismsSCSinStroke')
from models import MotoneuronSoma
from models import IntFireMN
from models import IntFireSimple
from models.IaAfferentSynapticPotential import get_ia_afferent_release_prob
from neuron import h 
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_simulation_results(data): 

    plt.subplots(5, 1, sharex=True)

    # plot MN firing
    plt.subplot(5,1,1)
    for i in range(data['num_mn']): plt.plot(data["mn_spikes"][i], i*np.ones(len(data["mn_spikes"][i])), ".", c='grey', markersize=2)
    plt.ylabel('Motoneuron\n Firing')

    # plot SCS pulses
    plt.subplot(5,1,2)
    for i in range(len(data["scs_pulse_times"])): plt.plot(data["scs_pulse_times"][i], i*np.ones(len(data["scs_pulse_times"][i])), ".", c='grey', markersize=1)
    plt.ylabel('Afferent Input')

    # plot supraspinal firing
    plt.subplot(5,1,3)
    for i in range(data['num_supraspinal']): plt.plot(data["supraspinal_spike_times"][i], i*np.ones(len(data["supraspinal_spike_times"][i])), ".", c='grey', markersize=1)
    plt.ylabel('Supraspinal\n Input')

    # plot EMG signal
    plt.subplot(5,1,4)
    plt.plot([t for t in range(len(data["emg"]))], data["emg"])
    plt.ylabel('EMG Signal\n (mV)')

    # plot normalized P2P amplitude
    plt.subplot(5,1,5)
    if (len(data["p2p_amp"]) > 0): 
        if (data["p2p_amp"][0] != 0):
            evoked_response_time = [int(pulseTime)+15 for pulseTime in data["scs_paradigm"]]
            plt.plot(evoked_response_time[:len(data["p2p_amp"])], data["p2p_amp"]/data["p2p_amp"][0])
    plt.ylabel('Normalized P2P\n Amplitude (%)')
    plt.xlabel('Time (ms)')

    plt.tight_layout()
    plt.show()


def ensure_dir(path):
    """Ensure directory exists; if not, make directory."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")


def estimateP2PAmp(emg, scsParadigm, delay_ms=10, dt=1): 
    p2p = []
    if len(scsParadigm) > 0:
        scsPulses = [int(pulseTime) for pulseTime in scsParadigm]
        for pulseInd in scsPulses: 
            start_response_ind = pulseInd + int(delay_ms/dt)
            end_response_ind = pulseInd + int(delay_ms+10/dt)
            if end_response_ind < len(emg):
                p2pPulse = np.max(emg[start_response_ind:end_response_ind])-np.min(emg[start_response_ind:end_response_ind])
                if p2pPulse > 0: 
                    p2p.append(p2pPulse)
                else: 
                    p2p.append(0)
    return p2p


def estimateEMG(firings, delay_ms=10):
    """
    Estimate the EMG activity given the cell firings. 

    Delay is the delay between the motoneuron action potential and the motor unit action potential (MUAP).
    """
    nCells = len(firings)
    nSamples = firings.shape[1]

    # MUAP duration between 5-10ms (Day et al 2001) -> 7.5 +-1 
    meanLenMUAP = 8
    stdLenMUAP = 1

    nS = [int(meanLenMUAP+np.random.normal(0,stdLenMUAP)) for _ in range(nCells)]
    amp = [abs(1+np.random.normal(0,0.2)) for _ in range(nCells)]
    EMG = np.zeros(nSamples + max(nS)+delay_ms)
    firingInd = []
    # create MUAP shape
    for i in range(nCells):
        n40perc = int(nS[i]*0.4)
        n60perc = nS[i]-n40perc
        amplitudeMod = (1-(np.linspace(0,1,nS[i])**2)) * np.concatenate((np.ones(n40perc),1/np.linspace(1,3,n60perc)))
        logBase = 1.05
        freqMod = np.log(np.linspace(1,logBase**(4*np.pi),nS[i]))/np.log(logBase)
        EMG_unit = amp[i]*amplitudeMod*np.sin(freqMod)
        for j in range(nSamples):
            if firings[i,j]==1:
                firingInd.append(j)
                EMG[j+delay_ms:j+delay_ms+nS[i]]=EMG[j+delay_ms:j+delay_ms+nS[i]]+EMG_unit
    
    EMG = EMG[:nSamples]
    
    return EMG


def run_scs_simulation_rdd(scs_freq, 
                           num_supra=20, 
                           supra_freq=60, 
                           num_aff=30,
                           Tb=50,
                           x0=0.5,
                           simulation_duration=200,
                           plot_results=True, 
                           save_data_folder='',
                           seed=672945):
    
    """
    Run simulation of spinal cord stimulation (SCS) in volitional control,
    with model of axonal & synaptic failure on afferent fibers. 

    Inputs: 
    - scs_freq: Hz, frequency of spinal cord stimulation 
    - num_supra: int, number of supraspinal inputs
    - supra_freq: Hz, frequency of supraspinal input firing
    - num_aff: int, number of scs inputs
    - Tb: int, rate of reuptake of neurotransmitter in the aff-MN synapse
    - x0: float, starting probability of axonal failure 
    - simulation_duration: ms (int), duration of simulation 
    - plot_results: bool, if True, will plot the outcome of the simulation 
    - save_data_folder: str, if not empty, will save .pickle file of data from simulation
    - seed: int, random seed

    Outputs: 
    None! See above for plotting and saving options.
    
    """
     # model parameters
    num_mns = 100
    synaptic_weight = 0.05
    synaptic_shape = 1.2

    # create MN pool
    np.random.seed(seed) # set the seed so the network is always the same

    mns = [IntFireSimple() for _ in range(num_mns)]
    
    # build and store afferent components
    afferent_inputs = []
    afferent_vecs = []
    afferent_neurons = []
    afferent_neuron_syns = []

    np.random.seed(seed) # set the seed so the network is always the same
    
    affW = np.random.gamma(synaptic_shape, scale=synaptic_weight/synaptic_shape, size=[num_aff, num_mns])

    for imn in range(num_mns): 
        afferent_inputs_mn = []
        afferent_vecs_mn = []
        afferent_neurons_mn = []
        afferent_neuron_syns_mn = []

        post_synaptic_release = get_ia_afferent_release_prob(
                                                    sim_dur=simulation_duration,
                                                    scs_freq=scs_freq,
                                                    n_axons=num_aff,
                                                    seed=seed,
                                                    x0=x0, 
                                                    Tb=Tb
                                                )
        
        for n_aff in range(num_aff): 
                stimTimes = [x for x in post_synaptic_release[n_aff]]
                vecStim = h.VecStim()
                vec = h.Vector(stimTimes)
                vecStim.play(vec)
                afferent_neurons_mn.append(vecStim)
                ncstim = h.NetCon(vecStim, mns[imn].cell)
                
                ncstim.weight[0] = affW[n_aff][imn]
                afferent_inputs_mn.append(ncstim)

                
        # store afferent components all
        afferent_inputs.append(afferent_inputs_mn)
        afferent_vecs.append(afferent_vecs_mn)
        afferent_neurons.append(afferent_neurons_mn)
        afferent_neuron_syns.append(afferent_neuron_syns_mn)

    # build and store supraspinal inputs
    
    supra_W = np.random.gamma(synaptic_shape, scale=synaptic_weight/synaptic_shape, size=[num_supra, num_mns])

    supra_neurons = []
    for supra_ind in range(num_supra):
        # create the supra point process stimulation to MN soma
        pre = h.NetStim()
        pre.interval = 1000/supra_freq
        pre.noise = 1
        pre.number = 1e999
        pre.start = 25 # ms
        supra_neurons.append(pre)

    supra_inputs = []
    supra_neurons_syns = []
    for mn_ind in range(num_mns):
        supra_inputs_mn = []
        supra_neurons_syns_mn = []
        for supra_ind in range(num_supra):
            nc = h.NetCon(supra_neurons[supra_ind], mns[mn_ind].cell)
            nc.weight[0] = supra_W[supra_ind][mn_ind]
            supra_inputs_mn.append(nc)
            
        supra_inputs.append(supra_inputs_mn)
        supra_neurons_syns.append(supra_neurons_syns_mn)

    # record SCS spikes
    ex_num_aff = len(afferent_neurons_mn)
    scs_times = [h.Vector() for i in range(ex_num_aff)]
    scs_detector =  [h.NetCon(afferent_neurons_mn[i], None) for i in range(ex_num_aff)]
    for i in range(ex_num_aff): scs_detector[i].record(scs_times[i])

    # record supraspinal spikes
    supra_times = [h.Vector() for i in range(num_supra)]
    supra_detector =  [h.NetCon(supra_neurons[i], None) for i in range(num_supra)]
    for i in range(num_supra): supra_detector[i].record(supra_times[i])

    # record MN spikes
    mn_times = [h.Vector() for i in range(num_mns)]
    mn_detector = []
    
    for i in range(num_mns):
        sp_detector = h.NetCon(mns[i].cell, None)
        mn_detector.append(sp_detector)
        mn_detector[i].record(mn_times[i])


    # run simulation
    h.load_file('stdrun.hoc')
    h.finitialize()
    h.tstop = simulation_duration
    h.run()
    
    supra_times = [np.array(supra_times[i]) if len(supra_times[i]) > 0 else [] for i in range(num_supra)]
    scs_times = [np.array(scs_times[i]) if len(scs_times[i]) > 0 else [] for i in range(len(scs_times))]
    mn_times = [np.array(mn_times[i]) if len(mn_times[i]) > 0 else [] for i in range(num_mns)]

    # reformat firing times, estimate EMG
    firings_int = [[int(spike) for spike in mn_firings] for mn_firings in mn_times]
    firings_mat = np.array([[1 if i in mn_firings else 0 for i in range(0, simulation_duration)] for mn_firings in firings_int])
    emg_signal = estimateEMG(firings_mat)

    # estimate P2P amp
    scs_pulses = [x*0.025+50 for x in range(int(simulation_duration/0.025)) if np.mod(x, int(1/scs_freq*1000/0.025)) == 0] # pulses from SCS
    p2p_amp = estimateP2PAmp(emg_signal, scs_pulses)
    
    # save data to pickle file
    data={}
    data["mn_spikes"] = mn_times
    data["supraspinal_spike_times"] = supra_times
    data["scs_pulse_times"] = scs_times
    data["scs_frequency"] = scs_freq
    data["num_scs_total"] = num_aff
    data["supraspinal_rate"] = supra_freq
    data["num_supraspinal"] = num_supra
    data["simulation_duration"] = simulation_duration
    data["num_mn"] = num_mns
    data["synaptic_weight_supra"] = synaptic_weight
    data["p2p_amp"] = p2p_amp
    data["emg"] = emg_signal
    data["scs_paradigm"] = scs_pulses
    data_filename = f"mnNum_{num_mns}_supraspinalNum_{num_supra}_supraspinalFR_{supra_freq}_SCSFreq_{scs_freq}_SCSTotal_{num_aff}_SynW_{synaptic_weight}_ta_{Tb}_seed_{seed}.pickle"

    if save_data_folder != '': 
        ensure_dir(save_data_folder)

        f=open(save_data_folder+data_filename,"wb")
        pickle.dump(data,f)
        f.close()

    if plot_results: 
        plot_simulation_results(data)
        
    return data_filename


if __name__ == '__main__':
    scs_freq = 80
    sim_duration = 100

    # run simulation in rest condition & plot results
    x0 = 0.3
    num_supra = 0
    run_scs_simulation_rdd(scs_freq, 
                           num_supra=num_supra, 
                           supra_freq=60, 
                           num_aff=30, 
                           Tb=100, 
                           x0=x0,
                           plot_results=True, 
                           simulation_duration=sim_duration, 
                           seed=1) 
    
    # run simulation in 25% MVC condition & plot results
    x0 = 0.8
    num_supra = 25
    run_scs_simulation_rdd(scs_freq, 
                           num_supra=num_supra, 
                           supra_freq=60, 
                           num_aff=30, 
                           Tb=100, 
                           x0=x0,
                           plot_results=True, 
                           simulation_duration=sim_duration, 
                           seed=1) 
    