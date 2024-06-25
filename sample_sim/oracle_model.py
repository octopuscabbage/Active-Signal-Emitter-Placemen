from joblib import Memory

from sample_sim.data_model.data_model import TorchExactGPBackedDataModel

memory = Memory("cache",verbose=0)


@memory.cache
def fit_oracle_gp_model(sensed_locations,sensed_values):
    gp_for_hyper_parameters = TorchExactGPBackedDataModel(sensed_locations[::10, :], sensed_values[::10],
                                                          "default", force_cpu=True,
                                                          device=0)
    gp_for_hyper_parameters.update(sensed_locations[::10, :], sensed_values[::10])
    gp_for_hyper_parameters.fit(1000)
    print(f"Lengthscale found {gp_for_hyper_parameters.model.model.covar_module.base_kernel.lengthscale}")
    return gp_for_hyper_parameters

