import json
import math
from pathlib import Path

import joblib
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from loguru import logger
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


class MusCATModel:

    def __init__(self, priv=None):
        self.priv = priv # Privacy parameters, if None, do not apply differential privacy

    def fit_disease_progression(self, Ytrain, agg_data=None):

        logger.info("Disease progression model...")

        if agg_data:
            count = agg_data["duration_cnt"]
        else:
            infected_days = (Ytrain == 1).sum(axis=1).ravel()
            recovered = (Ytrain[:,-1] == 2).ravel()
            _, count = np.unique(infected_days[(infected_days>0) & recovered], return_counts=True)

            logger.info(repr(count))
            count = count.ravel()
            logger.info(repr(count))

            # Consider up to 10 days
            new_vec = np.zeros(10)
            max_ind = min(len(count), len(new_vec))
            new_vec[:max_ind] = count[:max_ind]
            count = new_vec

            logger.info(repr(count))
            logger.info(repr(max_ind))

            # Differential privacy
            if self.priv:
                eps, delta = self.priv.disease_progression

                logger.info(f"Add noise for diff privacy: eps {eps} delta {delta}")

                count += GaussianMechNoise(eps=eps, delta=delta, l2_sens=1, shape=len(count))

        prob_delta = count / count.sum()
        prob_delta_cumul = prob_delta[::-1].cumsum()

        logger.info("Probability density for duration of infection:\n", prob_delta)

        self.prob_delta_cumul = prob_delta_cumul

        return count

    def fit_symptom_development(self, Ytrain, agg_data=None):

        logger.info("Symptom development model...")

        if agg_data:
            symptom_cnt = agg_data["symptom_cnt"]
            recov_cnt = agg_data["recov_cnt"]
        else:
            infection_observed = (Ytrain == 1).sum(axis=1) > 0
            recovered = Ytrain[:,-1] == 2
            susceptible_first_day = Ytrain[:,0] == 0
            sym, count = np.unique(infection_observed[susceptible_first_day & recovered], return_counts=True)
            symptom_cnt, recov_cnt = count[sym == True], count.sum()

            # Differential privacy
            if self.priv:
                eps, delta = self.priv.symptom_development

                logger.info(f"Add noise for diff privacy: eps {eps} delta {delta}")

                asymptom_cnt = recov_cnt - symptom_cnt

                noise = GaussianMechNoise(eps=eps, delta=delta, l2_sens=1, shape=2)
                symptom_cnt = float(symptom_cnt) + noise[0]
                asymptom_cnt = float(asymptom_cnt) + noise[1]
                
                recov_cnt = symptom_cnt + asymptom_cnt

        prob_alpha = float(symptom_cnt) / float(recov_cnt)

        self.prob_alpha = prob_alpha

        logger.info(f"Probability to develop symptoms given infection: {prob_alpha}")

        return np.array(symptom_cnt), np.array(recov_cnt)

    def compute_exposure_loads(self, beliefs, agg_data=None, day=None, id_map=None):
        eloads = dict()

        if agg_data:
            eloads["pop"] = np.ravel(agg_data["eloads_pop"])[day]
            eloads["loc-home"] = agg_data["eloads_res"][day]
            eloads["loc-act"] = agg_data["eloads_actloc"][day]
        else:
            eloads["pop"] = beliefs.sum()

            pids = self.le_res_act["pid"]
            if id_map:
                pids = np.array([id_map[v] for v in pids])

            val = beliefs[pids] * self.le_res_act["duration"]/3600
            I = np.zeros(len(self.le_res_act), dtype=int)
            J = self.le_res_act["lid"] - 1000000001
            eloads["loc-home"] = coo_matrix((val, (I, J))).toarray()[0]

            pids = self.le_other_act["pid"]
            if id_map:
                pids = np.array([id_map[v] for v in pids])

            val = beliefs[pids] * self.le_other_act["duration"]/3600
            I = np.zeros(len(self.le_other_act), dtype=int)
            J = self.le_other_act["lid"] - 1
            eloads["loc-act"] = coo_matrix((val, (I, J))).toarray()[0]

        return eloads

    def fit(self, Ytrain, num_days_for_pred=2, impute=True, neg_to_pos_ratio=3, batch_size=1000,
            num_epochs=15, use_adam=True, learn_rate=0.005, id_map=None):

        self.num_days_for_pred = num_days_for_pred
        self.impute = impute

        self.fit_disease_progression(Ytrain)

        self.fit_symptom_development(Ytrain)

        beliefs = (Ytrain == 1).transpose()

        self.train_feat, self.train_label = self.get_train_feat_all_days(beliefs, Ytrain,
            num_days_for_pred, impute, neg_to_pos_ratio, id_map=id_map)

        self.fit_disease_transmission(self.train_feat, self.train_label, batch_size, num_epochs,
            use_adam, learn_rate)


    def setup_features(self, person, actassign, popnet, id_map=None, cache_dir=None, is_deploy=False):
        if id_map:
            person = person.sort_values(by=["pid"])

        # Exposure loads ("eloads") from infection probabilities (beliefs)

        # Separate residence locations from other activity locations
        self.le_res_act = actassign.loc[actassign["lid"]>1000000000]
        self.le_other_act = actassign.loc[actassign["lid"]<1000000000]

        pe_age_bins = np.array([[16, 17],
                                [17, 18],
                                [18, 19],
                                [0, 10],
                                [10, 20],
                                [20, 30],
                                [30, 40],
                                [40, 50],
                                [50, 60],
                                [60, 70],
                                [70, 80],
                                [80, 200]])

        pe_age_feat = np.vstack([(ab[0] <= person["age"]) & (person["age"] < ab[1]) for ab in pe_age_bins])
        logger.info(f"Age features: {pe_age_feat.shape}")

        pe_sex_feat = np.vstack([person["sex"] == 1, person["sex"] == 2])
        logger.info(f"Sex features: {pe_sex_feat.shape}")

        pids = actassign["pid"]
        if id_map:
            pids = np.array([id_map[v] for v in pids])
        pe_act_feat = coo_matrix((actassign["duration"]/3600, (actassign["activity_type"]-1, pids)), shape=(7, len(person))).toarray()
        logger.info(f"Activity features: {pe_act_feat.shape}")

        pe_base_feat = np.vstack((pe_age_feat, pe_sex_feat, pe_act_feat))
        logger.info("-------------------")
        logger.info(f"Population exposure: {pe_base_feat.shape}")

        self.pe_base_feat = pe_base_feat

        if cache_dir:

            cacheFile = cache_dir / "contact_graphs.npz"

        if (not is_deploy) and cache_dir and cacheFile.exists():

            logger.info("Loading cached contact graphs")

            data = np.load(cacheFile)

            A_data = data["A_data"]
            A_data_act = data["A_data_act"]
            A_indptr = data["A_indptr"]
            A_indices = data["A_indices"]
            A_data_pruned = data["A_data_pruned"]
            A_data_act_pruned = data["A_data_act_pruned"]

        else:

            logger.info(f"Building contact graphs")

            # Ignore contacts shorter than 30 mins
            popnet_30m = popnet.loc[popnet["duration"] > 60*30]
            popnet_30m = popnet_30m.loc[popnet_30m["activity1"] > 0]
            popnet_30m = popnet_30m.loc[popnet_30m["activity2"] > 0]

            I = np.array(popnet_30m["pid1"])
            J = np.array(popnet_30m["pid2"])

            if id_map:
                I = np.array([id_map[v] for v in I])
                J = np.array([id_map[v] for v in J])

            nperson = max(I.max(), J.max()) + 1

            I2 = np.hstack([I, J])
            J2 = np.hstack([J, I])
            del I, J

            IJ = nperson * I2 + J2
            all_cols = np.unique(IJ)

            logger.info(f"Flatten indices")

            map_compact = {v:i for i,v in enumerate(all_cols)}
            IJ_compact = np.array([map_compact[i] for i in IJ])
            del map_compact, IJ
                
            D = np.array(popnet_30m["duration"], dtype=np.float32) / 3600
            A1 = np.array(popnet_30m["activity1"]) - 1
            A2 = np.array(popnet_30m["activity2"]) - 1
            del popnet_30m

            D = np.hstack([D, D])
            A1 = np.hstack([A1, A1])
            A2 = np.hstack([A2, A2])

            logger.info(f"Initialize CSR matrix")

            m0 = csr_matrix((np.zeros(len(all_cols)*7, dtype=np.float32), np.tile(all_cols, 7), np.arange(0, len(all_cols)*8, step=len(all_cols))), shape=(7, nperson**2))

            logger.info(f"Process activity1")

            m0.data[A1*len(all_cols) + IJ_compact] += D

            logger.info(f"Process activity2")

            m0.data[A2*len(all_cols) + IJ_compact] += D

            logger.info(f"Extract CSR matrices")

            A_data_act = m0.data.copy().reshape((7, -1)) / 4
            A_data = A_data_act.sum(axis=0) / 4
            del m0, IJ_compact, D, A1, A2

            A = csr_matrix((np.ones(len(I2), dtype=np.int8), (I2, J2)), shape=(nperson, nperson))
            A_indptr = A.indptr
            A_indices = A.indices
            del I2, J2, A

            logger.info(f"A_data: {A_data.shape}")
            logger.info(f"A_data_act: {A_data_act.shape}")
            logger.info(f"A_indptr: {A_indptr.shape}")
            logger.info(f"A_indices: {A_indices.shape}")

            if self.priv:
                logger.info(f"Prune graph (for differential privacy)")
                A_data_pruned, A_data_act_pruned = PruneGraph(A_data, A_data_act, A_indptr, A_indices, 
                                                              self.priv.contact_degrees_max)
            else:
                A_data_pruned = None
                A_data_act_pruned = None

            if cache_dir:
                logger.info("Caching contact graphs")
                np.savez(str(cacheFile), A_data=A_data, A_data_act=A_data_act, A_indptr=A_indptr, 
                         A_indices=A_indices, A_data_pruned=A_data_pruned, A_data_act_pruned=A_data_act_pruned)

        self.A_data = A_data
        self.A_data_act = A_data_act
        self.A_indptr = A_indptr
        self.A_indices = A_indices
        self.A_data_pruned = A_data_pruned
        self.A_data_act_pruned = A_data_act_pruned
        
    def eloads_to_pop_feat(self, eloads):
        pe = eloads["pop"] * self.pe_base_feat
        return pe

    def eloads_to_loc_feat(self, eloads, id_map=None):
        home_load = eloads["loc-home"]
        loc_load = eloads["loc-act"]

        I = np.zeros(len(self.le_res_act), dtype=int)
        J = self.le_res_act["pid"]
        if id_map:
            J = np.array([id_map[v] for v in J])

        val = home_load[self.le_res_act["lid"] - 1000000001] * self.le_res_act["duration"]/3600
        home_feat = coo_matrix((val, (I, J))).toarray()[0]

        I = self.le_other_act["activity_type"] - 2 # Ignore 1 (home)
        J = self.le_other_act["pid"]
        if id_map:
            J = np.array([id_map[v] for v in J])

        val = loc_load[self.le_other_act["lid"] - 1] * self.le_other_act["duration"]/3600
        loc_feat = coo_matrix((val, (I, J))).toarray()

        le_feat = np.vstack((home_feat, loc_feat))
        return le_feat

    def compute_contact_feat(self, beliefs, ndays=1, priv=None, is_training=False):
        out = beliefs.copy().ravel()

        if is_training and priv: # For DP-SGD, use pruned graphs
            A_data = self.A_data_pruned.copy()
            A_data_act = self.A_data_act_pruned.copy()
        else:
            A_data = self.A_data.copy()
            A_data_act = self.A_data_act.copy()

        graph_n = len(self.A_indptr) - 1
        
        if (not is_training) and priv: # For inference only, add noise to contact features

            logger.info("Adding noise to contact feat for inference")

            eps, delta = priv.test_prediction
            eps /= self.num_days_for_pred # Distribute over days used for prediction
            delta /= self.num_days_for_pred
            time_max = priv.contact_duration_max / 3600
            # deg_max = priv.contact_degrees_max
            # l2_sens = time_max * np.sqrt(deg_max * self.num_days_for_pred)

            logger.info("Randomized response")

            flip_prob = 1.0 / (1.0 + np.exp(eps))
            to_flip = np.random.uniform(size=out.shape) < flip_prob
            out[to_flip] = 1 - out[to_flip]

            logger.info("Clamp duration values")

            to_clip = A_data > time_max
            scaling = time_max / A_data[to_clip]
            A_data[to_clip] = time_max
            A_data_act[:,to_clip] *= scaling[np.newaxis,:]

            # logger.info("Norm clipping")
            # norm_max = priv.contact_norm_max                
            # NormClipGraphInPlace(A_clamped, norm_max)

        for i in range(ndays-1):

            A = csr_matrix((A_data, self.A_indices, self.A_indptr), shape=(graph_n, graph_n))
            out = A @ out
            del A

            # if i == 0 and (not is_training) and priv:
            #     out += GaussianMechNoise(eps=eps, delta=delta, l2_sens=l2_sens, shape=out.shape)

        res = [None] * A_data_act.shape[0]

        for atype in range(len(res)):
            A = csr_matrix((A_data_act[atype,:], self.A_indices, self.A_indptr), shape=(graph_n, graph_n))
            res[atype] = A @ out
            del A

        res = np.vstack(res)
            
        # if ndays == 1 and (not is_training) and priv: # if ndays is 1, need to add noise here instead
        #     res += GaussianMechNoise(eps=eps, delta=delta, l2_sens=l2_sens, shape=res.shape)

        return res

    def beliefs_to_all_features(self, beliefs, agg_data=None, day=None, id_map=None, priv=None, is_training=False):
        if len(beliefs.shape) == 1:
            beliefs = beliefs[np.newaxis,:]
        ndays = beliefs.shape[0]

        eloads = self.compute_exposure_loads(beliefs[-1], agg_data, day, id_map=id_map)

        pe_feat = self.eloads_to_pop_feat(eloads)
        le_feat = self.eloads_to_loc_feat(eloads, id_map)
        ce_feat = np.vstack([self.compute_contact_feat(beliefs[-1-d], d+1, priv=priv, is_training=is_training) for d in range(ndays)])
        
        return np.vstack((pe_feat,le_feat,ce_feat))

    def predict(self, Ytrain, id_map=None):

        beliefs = (Ytrain == 1).transpose()

        test_feat = self.get_test_feat(beliefs, Ytrain, self.num_days_for_pred, self.impute, id_map=id_map)

        test_feat = self.scaler.transform(test_feat)

        y_pred = self.model(torch.tensor(test_feat)).detach().numpy()

        return y_pred

    def get_test_feat(self, beliefs, Ytrain, ndays_for_feat=1, impute=False, agg_data=None, id_map=None, priv=None):

        beliefs_updated = beliefs[-ndays_for_feat:].copy().astype(np.float32)

        if impute and ndays_for_feat > 1:
            for d2 in range(beliefs_updated.shape[0]-1): # except last day
                d_idx = d2 + -ndays_for_feat # original day index
                asym = (Ytrain[:,d_idx] == 0) & (Ytrain[:,d_idx+1] == 2) # recovered on day d_idx+1
                beliefs_updated[:d2+1,asym.ravel()] = self.prob_delta_cumul[-d2-1:][:,np.newaxis]

        test_feat = self.beliefs_to_all_features(beliefs_updated, agg_data, 0, id_map, priv=priv, is_training=False).transpose().astype(np.float32)

        return test_feat

    # Given the beliefs for all training days
    # sample training instances and construct features
    def get_train_feat_all_days(self, beliefs, Ytrain, ndays_for_feat=1, impute=False, 
                                neg_to_pos_ratio=3, agg_data=None, id_map=None):
        days = Ytrain.shape[1]

        # Function to process each day d in parallel
        def process_day(d, agg_data):

            sus = Ytrain[:,d-1] == 0
            infected = Ytrain[:,d] == 1
            still_sus = Ytrain[:,d] == 0

            pos = sus & infected   # newly infected
            neg = sus & still_sus  # not infected

            npos = pos.sum()

            beliefs_updated = beliefs[d-ndays_for_feat:d].copy().astype(np.float32)

            if impute and ndays_for_feat > 1:

                for d2 in range(beliefs_updated.shape[0]-1): # except last day
                    d_idx = d2 + d-ndays_for_feat # original day index
                    asym = (Ytrain[:,d_idx] == 0) & (Ytrain[:,d_idx+1] == 2) # recovered on day d_idx+1
                    beliefs_updated[:d2+1,asym.ravel()] = self.prob_delta_cumul[-d2-1:][:,np.newaxis]

            # Construct features from previous beliefs
            all_feat = self.beliefs_to_all_features(beliefs_updated, agg_data, d-1, id_map, priv=self.priv, is_training=True)

            # Positive cases
            pos_feat = all_feat[:,np.where(pos)[0]]
            pos_label = np.ones(npos)

            # Sample negative cases
            nneg = npos * neg_to_pos_ratio
            negind = np.random.choice(np.where(neg)[0], size=nneg, replace=False)
            neg_feat = all_feat[:,negind]
            neg_label = np.zeros(len(negind))

            return np.hstack([pos_feat.squeeze(), neg_feat.squeeze()]), np.hstack([pos_label, neg_label])

        day_range = np.arange(ndays_for_feat, days)

        # Parallel for loop
        # results = thread_map(parallel_job, day_range, max_workers=4)
        results = [process_day(d, agg_data) for d in tqdm(day_range)]

        # Concatenate results
        train_feat_concat = np.hstack([t[0].squeeze() for t in results])
        train_feat_concat = train_feat_concat.transpose().astype(np.float32)

        train_label_concat = np.hstack([t[1] for t in results])

        return train_feat_concat, train_label_concat

    def fit_disease_transmission(self, train_feat, train_label, batch_size=1000, num_epochs=15, use_adam=True, learn_rate=0.005):

        model = PoissonExposureModel(train_feat.shape[1], 1)

        scaler = StandardScaler()
        scaler.fit(train_feat)
        train_feat = scaler.transform(train_feat)

        train_dataset = MatrixDataset(torch.tensor(train_feat), torch.tensor(train_label))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define the loss function and optimizer
        loss_fn = BinaryPoissonNLLLoss()
        if use_adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

        size = len(train_dataloader.dataset)
        for epoch in range(1, num_epochs + 1):
            logger.info(f"epoch: {epoch}")

            batch = 1
            for X, y in train_dataloader:
                # Compute prediction and loss
                pred = model(X)
                loss = loss_fn(pred, y[:,np.newaxis])

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 50 == 0:
                    loss, current = loss.item(), batch * len(X)
                    logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                batch += 1

        self.model = model
        self.scaler = scaler

    def initial_weights(self, num_features=None):
        # np.random.seed(0) # Deterministic seed so all clients have the same initial model
        # stdv = 1. / np.sqrt(num_features)
        # return np.random.uniform(low=-stdv, high=stdv, size=num_features+1)

        # Prior knowledge
        init_weights = np.array([ 0.42980912, -0.00924882,  0.00670033, -0.01610269, -0.03017593,
        0.09858311,  0.04946187,  0.05403953,  0.04034745,  0.0151224 ,
        0.01771312,  0.0054485 , -0.15104969, -0.01643871, -0.01045104,
       -0.17034666, -0.01693403, -0.00821398, -0.08441824, -0.18401848,
        0.12482014, -0.03288731,  0.00278582,  0.07604216,  0.0131966 ,
        0.02062159,  0.0053063 ,  0.04018513, -0.00103119,  0.11839021,
        0.44942504, -0.39994606,  0.2358387 ,  0.07346488,  0.22177164,
       -0.37389484,  0.52086437,  0.4199095 , -0.10497533,  0.12064194,
        0.04529913,  0.17248046, -0.02391332] + [0] * 100)
        return np.array(init_weights[:num_features+1])

    def save(self, model_dir: Path, is_fed=False):
        if is_fed:
            params = {"delta_cumul":self.prob_delta_cumul.tolist(),
                    "alpha":self.prob_alpha,
                    "num_days":self.num_days_for_pred,
                    "impute":self.impute,
                    "weights":self.weights.tolist(),
                    "center":self.center.tolist(),
                    "scale":self.scale.tolist(),
                    "adam_m":self.adam_m.tolist(),
                    "adam_v":self.adam_v.tolist(),
                    "adam_counter":self.adam_counter}
        else:
            torch.save(self.model, model_dir / "model.save")
            joblib.dump(self.scaler, model_dir / "scaler.save")
            params = {"delta_cumul":self.prob_delta_cumul.tolist(),
                    "alpha":self.prob_alpha,
                    "num_days":self.num_days_for_pred,
                    "impute":self.impute}

        with open(model_dir / "params.save", "w") as fp:
            json.dump(params, fp)

    def load(self, model_dir: Path, is_fed=False):
        with open(model_dir / "params.save", "r") as read_content:
                params = json.load(read_content)

        if is_fed:
            self.weights = np.array(params["weights"])
            self.center = np.array(params["center"])
            self.scale = np.array(params["scale"])
            if "adam_m" in params:
                self.adam_m = np.array(params["adam_m"])
                self.adam_v = np.array(params["adam_v"])
                self.adam_counter = params["adam_counter"]
        else:
            self.model = torch.load(model_dir / "model.save")
            self.scaler = joblib.load(model_dir / "scaler.save")

        self.prob_delta_cumul = np.array(params["delta_cumul"])
        self.prob_alpha = float(params["alpha"])
        self.num_days_for_pred = params["num_days"]
        self.impute = params["impute"]

    def compute_gradient_sum(self, feat, label):

        z = (self.weights[0] + feat @ self.weights[1:]).ravel()

        # Baseline rate of infection
        clamp_thres = -np.log(0.95) 
        clamp = z < clamp_thres
        z[clamp] = clamp_thres

        dloss_dz = 1 - (1 / (1 - np.exp(-z)))
        dloss_dz[label == 0] = 1
        dloss_dz[clamp] = 0

        grad = np.hstack([dloss_dz.ravel()[:,np.newaxis], (dloss_dz.ravel()[:,np.newaxis] * feat)])

        if self.priv:

            eps, delta = self.priv.model_training_sgd
            norm_thres = self.priv.sgd_grad_norm_max
            max_degree = self.priv.contact_degrees_max
            num_hops = self.num_days_for_pred
            group_size = max_degree ** num_hops + 1 # Max number of gradient terms
                                                    # affected by one individual

            logger.info(f"Add noise to gradients for diff privacy: eps {eps} delta {delta}")

            gradient_norm = np.sqrt((grad ** 2).sum(axis=1)).ravel()

            to_clip = gradient_norm > norm_thres
            scaling = norm_thres / gradient_norm[to_clip]
            grad[to_clip,:] *= scaling[:,np.newaxis]

            gradient_sum = grad.sum(axis=0)

            gradient_sum += GaussianMechNoise(eps=eps, delta=delta,
                l2_sens=norm_thres*np.sqrt(group_size), 
                shape=gradient_sum.shape)

        else:
            gradient_sum = grad.sum(axis=0)

        sample_count = len(z)

        return gradient_sum.ravel(), sample_count

class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        assert(len(data) == len(labels))
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def get_labels(self):
        return self.labels

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
       See [Maechler2012accurate]_ for details.
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )

def ClipModelOutput(x):
    max_neg_prob = torch.tensor(0.95)
    return torch.clip(x, min=-max_neg_prob.log())

class BinaryPoissonNLLLoss(nn.Module):
    def __init__(self, log_input=False, size_average=True, eps=1e-8):
        super(BinaryPoissonNLLLoss, self).__init__()
        self.log_input = log_input
        self.size_average = size_average
        self.eps = eps

    def forward(self, input, target):
        if self.log_input:
            input = input.log()

        input = ClipModelOutput(input)
        loss = (1 - target) * (input) + target * (- log1mexp(- input))

        if self.size_average:
            return loss.mean()
        return loss.sum()

class PoissonExposureModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PoissonExposureModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def GaussianMechNoise(eps=1, delta=1e-2, l2_sens=1, shape=1):
    sigma = np.sqrt(2*np.log(1.25/delta)) * l2_sens / eps
    return sp.stats.norm.rvs(loc=0, scale=sigma, size=shape)

def NormClipGraphInPlace(A_csr, max_norm):
    orig_data = A_csr.data.copy()
    A_csr.data = A_csr.data ** 2
    col_sqsum = A_csr.sum(axis=0)
    to_clip = col_sqsum > (max_norm ** 2)
    scaling = np.array(max_norm / np.sqrt(col_sqsum)).ravel()
    A_csr.data = orig_data * scaling[A_csr.indices]
    return A_csr

def PruneGraph(A_data, A_data_act, A_indptr, A_indices, degree_thres):
    A_data_pruned = A_data.copy()
    A_data_act_pruned = A_data_act.copy()

    n = len(A_indptr) - 1
    for cidx in tqdm(range(n), unit_scale=1, mininterval=6):
        st, en = A_indptr[cidx:cidx+2]
        if en - st > degree_thres: # Zero out edges except top degree_thres
            order = np.argsort(A_data_pruned[st:en])[::-1]
            A_data_pruned[st + order[degree_thres:]] = 0
            A_data_act_pruned[:,st + order[degree_thres:]] = 0    

    return A_data_pruned, A_data_act_pruned