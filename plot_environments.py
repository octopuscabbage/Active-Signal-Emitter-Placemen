target_environment = create_sdf_environment(
            specification["environment_seed"], generator_name=specification["target_sdf_environment"], 
            x_size=xmax, y_size=ymax, preset_light_intensities=desired_light_brightness,
            num_lights=specification["target_lights"])
ambient_light_environment = create_sdf_environment(
            specification["environment_seed"], generator_name=specification["ground_truth_sdf_environment"], 
            x_size=xmax, y_size=ymax, preset_light_intensities=ambient_light_brightness,
            num_lights=specification["ambient_lights"])
ambient_light_environment.lighting_computer.set_max_reflections( ground_truth_reflections)
gt_grid = generate_finite_grid(target_environment, tuple(state_space_dimensionality), sensor,
                                            number_of_edge_samples)

grid = generate_finite_grid(ambient_light_environment, tuple(state_space_dimensionality), sensor,
                                         number_of_edge_samples)
target_sensed_points = target_environment.sample(grid.get_sensed_locations())
