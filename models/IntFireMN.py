from neuron import h
import random as rnd

class IntFireMN():
	""" Integrate and Fire cell.

	This class implement and IntFire4 Neuron object.
	Taus are tuned in order to generate epsps lasting approximately 15 ms (Burke 1968)
	and IPSPs lasting 30 ms (McIntyre 2002)
	"""
	__eesWeight = 1.5 # Weight of a connection between an ees object and this cell

	def __init__(self):
		""" Object initialization. """

		noisePerc = 0.05

		#Create IntFire4
		self.cell = h.IntFireMn()
		self.cell.taue=0.25
		self.cell.taui1=2
		self.cell.taui2=4.5
		self.cell.taum= rnd.normalvariate(6,6*noisePerc)
		if self.cell.taum <= self.cell.taui2: self.cell.taum = self.cell.taui2 + 0.25
		self.cell.refrac=rnd.normalvariate(15,15*noisePerc) # mean 66Hz

	@classmethod
	def get_ees_weight(cls):
		""" Return the weight of a connection between an ees object and this cell. """
		return IntFireMN.__eesWeight