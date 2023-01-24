class MusCATPrivacy:

    def __init__(self, privacy_level=0): # 0: low, 1: medium, 2: high

        if privacy_level != 0 and privacy_level != 1 and privacy_level != 2:
            raise Exception("Undefined privacy level. Must be 0, 1, or 2.")

        # For differential privacy
        # (eps, delta) budget for each component
        self.disease_progression = (0.1, 1e-7)
        self.symptom_development = (0.1, 1e-7),
        self.exposure_load_population = (0.1, 1e-7),
        self.exposure_load_location = (0.1, 1e-7),
        self.model_training_sgd = (3, 1e-7),
        self.test_prediction = (5, 1e-7),

        # Thresholding and pruning parameters;
        # reduces the sensitivity of the quantities computed
        self.infection_duration_max = 10, # 10 days
        self.location_duration_max = 2*3600, # 2 hours
        self.contact_degrees_max = [5, 10, 15], # 1, 2, 3-hop neighbors
        self.contact_duration_max = 2*3600, # 2 hours
        self.sgd_grad_norm_max = 1
