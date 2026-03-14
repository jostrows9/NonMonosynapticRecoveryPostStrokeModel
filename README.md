# NonMonosynapticRecoveryPostStrokeModel
Repository for code to model the mechanisms underlying the recovery of volitional movement in spinal cord stimulation (SCS) in stroke, as used in Sorenson et al., 2026.

### Getting Started
All simulations are run in NEURON. Instructions for setting up NEURON can be found here. Once installed, NEURON must be configured by running the following from the home directory:

'''nrnivmodl mod_files'''

The rest of the required packages can be found in requirements.txt and can be installed using:

'''pip install -r requirements.txt'''

### Scripts
This repository includes a generic version of the simulation included in Sorenson et al., 2026. This can be run using: 

'''python scripts/runSCSSimulation.py'''


