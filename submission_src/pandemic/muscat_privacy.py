import numpy as np
from scipy.optimize import minimize

class MusCATPrivacy:

    def __init__(self, num_batches=20, num_epochs=2):

        # Thresholding and pruning parameters;
        # reduces the sensitivity of the quantities computed
        self.infection_duration_max = 10 # 10 days
        self.location_duration_max = 2*3600 # 2 hours
        self.contact_degrees_max = 5
        self.contact_duration_max = 2*3600 # 2 hours
        self.sgd_grad_norm_max = 1

        # For differential privacy
        # (eps, delta) budget for each component 
        self.disease_progression = [0.1, 1e-7]
        self.symptom_development = [0.1, 1e-7]
        self.exposure_load_population = [0.1, 1e-7]
        self.exposure_load_location = [5, 1e-7]
        self.feature_mean_stdev = [1, 1e-7]
        self.model_training_sgd = privacy_amplification_via_shuffling(1, 1e-7, N=1e6, 
            BSIZE=1e6/num_batches, N_EPOCHS=num_epochs, max_norm=self.sgd_grad_norm_max)
        self.test_prediction = [8, 0]

# Given epsilon delta for the entire training process (all epochs),
# compute the per batch noise level based on privacy amplification
# enabled by the fact we are shuffling the data points before creating batches;
# resorts to parallel composition when the amplification does not help
def privacy_amplification_via_shuffling(EPS_TOT, DELTA_TOT, N=1e6, BSIZE=5e4, N_EPOCHS=2, max_norm=5):
    target_delta = 1.0/np.power(N, 2)
    EPS = EPS_TOT/N_EPOCHS
    DELTA = DELTA_TOT/N_EPOCHS

    eps_0, delta_0 = [1, 1.0/np.power(N,2)]

    # convert per round (epsilon, delta) to shuffled epoch level (epsilon, delta)
    def per_round_privacy_to_total(eps_0, delta_0, n, bsize, target_delta=None, print_flag=False):
        num_batches = n/bsize
        if target_delta is None:
            target_delta = 1.0/np.power(num_batches, 2)
        if target_delta is not None and target_delta > 1: 
            raise ValueError("delta > 1")
        if eps_0 > np.log(num_batches/(16*np.log(2/target_delta))):
            raise ValueError('epsilon per round is too large for amplification by sampling')
    
        first_term = 8*np.exp(eps_0)/num_batches
        second_term = 8*np.sqrt(np.exp(eps_0)*np.log(4/target_delta))/np.sqrt(num_batches)
        final_epsilon = np.log(1 + ((np.exp(eps_0)-1)/(np.exp(eps_0)+ 1))*(first_term + second_term ))
        final_delta = target_delta + (np.exp(final_epsilon) + 1)*(1 + np.exp(-eps_0)/2)*num_batches*delta_0
        if print_flag:
            print(f'After {n/bsize} minibatches which each satisfy ({eps_0}, {delta_0})-DP, the epoch satisfies ({final_epsilon},{final_delta}-DP')
        return (final_epsilon, final_delta)

    # convert per round epsilon delta to noise level for gaussian noise
    # implement gaussian mechanism from differential privacy
    def gaussian_mech_noise(max_norm, epsilon, delta):
        return np.sqrt(2 * np.log(1.25 / delta) * np.power(max_norm, 2) / np.power(epsilon, 2))

    def error_per_round(priv_pars):
        eps_0 = priv_pars[0]
        delta_0 = priv_pars[1]
        fin_eps, fin_delta = per_round_privacy_to_total(eps_0, delta_0, N, BSIZE, target_delta)
        return np.abs(EPS-fin_eps) + np.max([0.0, delta_0-DELTA])

    def compute_per_round_noise_level(N_EPOCHS, BSIZE, N, max_norm=5):
        epsilon_ub = np.log((N/BSIZE)/(16*np.log(2/np.power(DELTA,2))))

        if epsilon_ub < 1e-4 or DELTA < 1.0/np.power(N,4):
            print("Privacy setting for mini-batch SGD: parallel composition")
            return EPS_TOT/N_EPOCHS, DELTA_TOT/N_EPOCHS

        print("Privacy setting for mini-batch SGD: amplification via shuffling")

        opt_pars = minimize(error_per_round,
            [eps_0, delta_0], method='Nelder-Mead',
            bounds= [(1e-4, epsilon_ub), (1.0/np.power(N,4), DELTA)],
        )
    
        new_eps_0, new_delta_0 = opt_pars.x

        l_2_sens = 1/BSIZE*max_norm
        per_batch_noise_sigma_2 = gaussian_mech_noise(l_2_sens, new_eps_0, new_delta_0)

        return new_eps_0, new_delta_0
    
    eps_0, delta_0 = compute_per_round_noise_level(N_EPOCHS, BSIZE, N, max_norm)

    return [eps_0, delta_0]