from neuron import h
import numpy as np

class IntFireSimple():
	""" Integrate and Fire cell.

	This class implements an IntFire1 Neuron object.
	"""

	def __init__(self):
		""" Object initialization. """

		noisePerc = 0.1

		#Create IntFire1
		self.cell = h.IntFire1()
		self.cell.tau = np.random.normal(6,6*noisePerc)
		self.cell.refrac = np.random.poisson(15) # mean 66Hz