import matplotlib.pyplot as plt
import numpy as np

from potential import Finite_Temperature_Potential
from phase_transition import Phase_Transition

PT = Phase_Transition(mDelta = 700, LambdaDelta = 1e-2, kappa = 8)

PT.plot_potential()
#PT.plot_phase_transition()
