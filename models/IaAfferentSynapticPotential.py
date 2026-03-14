import numpy as np


n_vesicle = 3

def get_ia_afferent_release_prob(sim_dur, 
                                scs_freq, 
                                n_axons, 
                                Tb=50, 
                                Ub=0.7, 
                                dt=0.025,
                                x0=0.5,
                                seed=672944): 
    """
    Function for modeling axonal and synaptic failure in response to stimulation. 

    SCS spikes constitute a "nascent spike", which may or may not travel to the MN itself, 
    either because of axonal or synaptic failure.

    The following function predicts the success or failure of spikes after axonal and synaptic failure.
    """
    np.random.seed(seed)

    # adjust time-based parameters to dt 
    Tb = Tb/dt # dt steps, mean of exponentially-distributed random variable for wait time before recovery, OG: 850
    
    # set up variables 
    t_vec_len = int(sim_dur/dt)
    n_all = n_vesicle*np.ones([n_axons, t_vec_len]) # docked vesicles in each synapse
    release = [[] for _ in range(n_axons)] # time of successful spikes
    scs_pulses = [1 if np.mod(x, int(1/scs_freq*1000/dt)) == 0 else 0 for x in range(t_vec_len)] # pulses from SCS
    scs_pulses_success = np.random.rand(n_axons, t_vec_len)

    # run simulation of axonal and synaptic failure, predict spike train
    for t in range(t_vec_len - 1): 
        for j in range(n_axons): 
            if (scs_pulses[t] == 1) & (scs_pulses_success[j][t] <= x0): # if SCS pulse succeeds   
                p = 1-(1-Ub)**n_all[j][t]
                if np.random.rand() <= p: # if synaptic success 
                    redock_time = np.random.exponential(Tb)
                    redock_time = int(np.round(np.min([t_vec_len-t, redock_time])))
                    n_all[j][t:] = np.max([0, n_all[j][t]-1])
                    n_all[j][t+redock_time:] += 1 
                    release[j].append(t*dt)

    return release
