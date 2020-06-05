import matplotlib.pyplot as plt
import numpy as np

from potential import Finite_Temperature_Potential
from phase_transition import Phase_Transition

PT = Phase_Transition(mDelta = 400, LambdaDelta = 1e-2, kappa = 4)

#PT.plot_potential()
PT.plot_phase_transition()
#print(PT.phase_transition_critical())
