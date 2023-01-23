from pathlib import Path
from loguru import logger

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm 
import joblib
import json
import math

import numpy as np

class MusCATModel:

    def fit_disease_progression(self, Ytrain, agg_data=None):

        logger.info("Disease progression model...")

        if agg_data:
            count = agg_data["duration_cnt"]
        else:
            infected_days = (Ytrain == 1).sum(axis=1).ravel()
            recovered = (Ytrain[:,-1] == 2).ravel()
            days, count = np.unique(infected_days[(infected_days>0) & recovered], return_counts=True)

        prob_delta = count / count.sum()
        prob_delta_cumul = prob_delta[::-1].cumsum()
        
        print("Probability density for duration of infection:\n", prob_delta)

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

        prob_alpha = (symptom_cnt / recov_cnt)[0]

        self.prob_alpha = prob_alpha

        print("Probability to develop symptoms given infection:\n", prob_alpha)

        return np.array(symptom_cnt), np.array(recov_cnt)

    def compute_exposure_loads(self, beliefs, agg_data=None, day=None):
        eloads = dict()

        if agg_data:
            eloads["pop"] = agg_data["eloads_pop"].ravel()[day]
            eloads["loc-home"] = agg_data["eloads_res"][day]
            eloads["loc-act"] = agg_data["eloads_actloc"][day]
        else:
            eloads["pop"] = beliefs.sum()
            
            val = beliefs[self.le_res_act["pid"]] * self.le_res_act["duration"]/3600
            I = np.zeros(len(self.le_res_act), dtype=int)
            J = self.le_res_act["lid"] - 1000000001
            eloads["loc-home"] = coo_matrix((val, (I, J))).toarray()[0]
            
            val = beliefs[self.le_other_act["pid"]] * self.le_other_act["duration"]/3600
            I = np.zeros(len(self.le_other_act), dtype=int)
            J = self.le_other_act["lid"] - 1
            eloads["loc-act"] = coo_matrix((val, (I, J))).toarray()[0]

        return eloads

    def fit(self, Ytrain, num_days_for_pred=2, impute=True, neg_to_pos_ratio=3, batch_size=1000,
            num_epochs=15, use_adam=True, learn_rate=0.005):

        self.num_days_for_pred = num_days_for_pred
        self.impute = impute

        self.fit_disease_progression(Ytrain)

        self.fit_symptom_development(Ytrain)

        beliefs = (Ytrain == 1).transpose()

        self.train_feat, self.train_label = self.get_train_feat_all_days(beliefs, Ytrain, 
            num_days_for_pred, impute, neg_to_pos_ratio)
        
        self.fit_disease_transmission(self.train_feat, self.train_label, batch_size, num_epochs, 
            use_adam, learn_rate)

    def fed_get_train_feat(self, Ytrain, agg_data, id_map, num_days_for_pred=2, impute=True, neg_to_pos_ratio=3):
        
        self.num_days_for_pred = num_days_for_pred
        self.impute = impute

        beliefs = (Ytrain == 1).transpose()

        self.train_feat, self.train_label = self.get_train_feat_all_days(beliefs, Ytrain, 
            num_days_for_pred, impute, neg_to_pos_ratio, agg_data, id_map)

    def setup_features(self, person, actassign, popnet, id_map=None):
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
        print("Age features:", pe_age_feat.shape)

        pe_sex_feat = np.vstack([person["sex"] == 1, person["sex"] == 2])
        print("Sex features:", pe_sex_feat.shape)

        pids = actassign["pid"]
        if id_map:
            pids = np.array([id_map[v] for v in pids])
        pe_act_feat = coo_matrix((actassign["duration"]/3600, (actassign["activity_type"]-1, pids)), shape=(7, len(person))).toarray()
        print("Activity features:", pe_act_feat.shape)

        pe_base_feat = np.vstack((pe_age_feat, pe_sex_feat, pe_act_feat))
        print("-------------------")
        print("Population exposure:", pe_base_feat.shape)

        self.pe_base_feat = pe_base_feat

        # Ignore contacts shorter than 30 mins
        popnet_30m = popnet.loc[popnet["duration"] > 60*30] 

        # Build contact network adjacency for each type (edges weighted by duration)
        ce_A_list = [None] * 7
        for atype in range(1,8):
            popnet_30m_act = popnet_30m.loc[popnet_30m["activity1"] == atype]
            val = popnet_30m_act["duration"] / 3600 # In units of 1 hour blocks
            I = popnet_30m_act["pid1"]
            J = popnet_30m_act["pid2"]

            if id_map:
                I = np.array([id_map[v] for v in I])
                J = np.array([id_map[v] for v in J])

            A1 = coo_matrix((val.astype(float), (I,J)), shape=(len(person),len(person))) 

            popnet_30m_act = popnet_30m.loc[popnet_30m["activity2"] == atype]
            val = popnet_30m_act["duration"] / 3600 # In units of 1 hour blocks
            I = popnet_30m_act["pid1"]
            J = popnet_30m_act["pid2"]

            if id_map:
                I = np.array([id_map[v] for v in I])
                J = np.array([id_map[v] for v in J])

            A2 = coo_matrix((val.astype(float), (I,J)), shape=(len(person),len(person))) 

            A = (A1 + A2) / 2
            A = (A.transpose() + A) / 2 # Symmetrize
            ce_A_list[atype-1] = A
            print("Type", atype, "processed")

        ce_A_combined = ce_A_list[0]
        for b in ce_A_list[1:]:
            ce_A_combined += b

        self.ce_A_list = ce_A_list
        self.ce_A_combined = ce_A_combined

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

    def compute_contact_feat(self, beliefs, ndays=1):
        out = beliefs.copy()
        for i in range(ndays-1):
            out = self.ce_A_combined @ out
        return np.vstack([self.ce_A_list[atype] @ out for atype in range(len(self.ce_A_list))])

    def beliefs_to_all_features(self, beliefs, agg_data=None, day=None, id_map=None):
        if len(beliefs.shape) == 1:
            beliefs = beliefs[np.newaxis,:]
        ndays = beliefs.shape[0]

        eloads = self.compute_exposure_loads(beliefs[-1], agg_data, day)

        pe_feat = self.eloads_to_pop_feat(eloads)    
        le_feat = self.eloads_to_loc_feat(eloads, id_map)
        ce_feat = np.vstack([self.compute_contact_feat(beliefs[-1-d], d+1) for d in range(ndays)])
        
        return np.vstack((pe_feat,le_feat,ce_feat))

    def predict(self, Ytrain):

        beliefs = (Ytrain == 1).transpose()

        test_feat = self.get_test_feat(beliefs, Ytrain, self.num_days_for_pred, self.impute)

        test_feat = self.scaler.transform(test_feat)

        y_pred = self.model(torch.tensor(test_feat)).detach().numpy()

        return y_pred

    def get_test_feat(self, beliefs, Ytrain, ndays_for_feat=1, impute=False, agg_data=None, id_map=None):

        beliefs_updated = beliefs[-ndays_for_feat:].copy().astype(np.float32)

        if impute and ndays_for_feat > 1:
            for d2 in range(beliefs_updated.shape[0]-1): # except last day
                d_idx = d2 + -ndays_for_feat # original day index
                asym = (Ytrain[:,d_idx] == 0) & (Ytrain[:,d_idx+1] == 2) # recovered on day d_idx+1
                beliefs_updated[:d2+1,asym.ravel()] = self.prob_delta_cumul[-d2-1:][:,np.newaxis]

        test_feat = self.beliefs_to_all_features(beliefs_updated, agg_data, 0, id_map).transpose().astype(np.float32)

        return test_feat

    # Given the beliefs for all training days
    # sample training instances and construct features
    def get_train_feat_all_days(self, beliefs, Ytrain, ndays_for_feat=1, impute=False, neg_to_pos_ratio=3, agg_data=None, id_map=None):
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
            all_feat = self.beliefs_to_all_features(beliefs_updated, agg_data, d-1, id_map)

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
            print("epoch:", epoch)

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
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

                batch += 1
        
        self.model = model
        self.scaler = scaler

    def save(self, model_dir: Path, is_fed=False):
        if is_fed:
            params = {"delta_cumul":self.prob_delta_cumul.tolist(),
                    "alpha":self.prob_alpha,
                    "num_days":self.num_days_for_pred,
                    "impute":self.impute,
                    "weights":self.weights.tolist(),
                    "center":self.center.tolist(),
                    "scale":self.scale.tolist()}
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
        else:
            self.model = torch.load(model_dir / "model.save")
            self.scaler = joblib.load(model_dir / "scaler.save") 
            
        self.prob_delta_cumul = np.array(params["delta_cumul"])
        self.prob_alpha = float(params["alpha"])
        self.num_days_for_pred = params["num_days"]
        self.impute = params["impute"]
    
    def compute_gradient_sum(self, feat, label):
        
        z = (self.weights[0] + feat @ self.weights[1:]).ravel()

        dloss_dz = (1 - 2 / (1 - np.exp(-z)))
        dloss_dz[label == 0] = 1

        feat_with_intercept = np.hstack([np.ones(feat.shape[0])[:,np.newaxis], feat])
        gradient_sum = dloss_dz @ feat_with_intercept

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
    
# Define the model
class PoissonExposureModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PoissonExposureModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)