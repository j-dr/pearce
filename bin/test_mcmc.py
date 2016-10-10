#!/bin/bash
from pearce.emulator.emu import OriginalRecipe, ExtraCrispy 
from pearce.emulator.trainingData import parameter, PARAMS
import numpy as np
from os import path

training_dir = '/u/ki/swmclau2/des/PearceLHC/'

#emu = ExtraCrispy(training_dir)
or_params = PARAMS[:]
or_params.append(parameter('r', 0,1))
emu = OriginalRecipe(training_dir, or_params)

save_dir = '/u/ki/swmclau2/des/EmulatorMCMC/'
#true_rpoints = np.log10(np.loadtxt(path.join(save_dir, 'rpoints.npy')))
true_rpoints = np.loadtxt(path.join(save_dir, 'rpoints.npy'))
y = np.loadtxt(path.join(save_dir, 'xi.npy'))
cov = np.loadtxt(path.join(save_dir, 'cov.npy'))
true_cov = cov/(np.outer(y,y)*np.log(10)**2)
#T = np.diag(np.diag(T))
true_y = np.log10(y)

#removing lowest point because it's below interpolation range
#true_y = true_y[1:]
#true_rpoints = true_rpoints[1:]
#true_cov = true_cov[1:,:][:,1:]

chain = emu.run_mcmc(true_y, true_cov, true_rpoints, nwalkers = 2000, nsteps= 200,nburn = 50)

print chain.mean(axis=0)

np.savetxt(path.join(save_dir, 'chain.npy'), chain)
