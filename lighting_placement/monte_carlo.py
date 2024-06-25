from tqdm import tqdm
import sample_sim.environments.lighting.sdf_functions as sdf
import numpy as np
from scipy.stats import norm



def compute_mc_variance(lighting_placement, mc_iters,mc_variance,sensed_locations,sdf_fn,light_intensities,hardness):
    lightings = []
    #TODO We can probably march all these at once
    for i in tqdm(range(mc_iters), desc="McJittering"):
        jittered_lighting_placement = norm(lighting_placement.reshape(-1),mc_variance).rvs(lighting_placement.reshape(-1).size).reshape(lighting_placement.shape)
        current_lighting = sdf.light_at_locations(sensed_locations,sdf_fn,jittered_lighting_placement,light_intensities,hardness) 
        lightings.append(current_lighting)
    lightings = np.array(lightings)
    variance =  np.var(lightings,axis=0)
    return variance
