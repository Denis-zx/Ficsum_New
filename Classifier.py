import numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeClassifierCV
from minirocket import fit, transform
from generate_fingerprint import fingerprint_generator,flaten_fingerprint
from calculate_similarity import get_cosine_distance
from calculate_weight_2 import *
from tsai.basics import *
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *
import torch
from scipy.stats import t,ttest_1samp

# Fingerprint features ['IMF',"mean","stdev","skew",'kurtosis','turning_point_rate','acf','pacf','MI','FI']

class Classifier:
    
    def __init__(self,X_feat,current_datastream,index,GPU_feature = True, max_sim_factor=2,min_sim_factor=2,p_value_st = 0.05):
        #Definition Headers
        self.index = index
        self.classifier = None
        self.cur_fingerprint = None
        self.fingerprint_pool = []
        self.flaten_fingerprint_pool = []
        self.fingerprint_pool_sd = None
        self.fingerprint_vote = []
        self.initial_update = False
        
        self.weight_sd = None
        self.weight_X_sd = None
        self.weight_in_concept = None
        self.weight_between_concept = None

        self.max_similarity_scaling_factor = max_sim_factor
        self.min_similarity_scaling_factor = min_sim_factor
        self.p_value_st = p_value_st
        self.max_similarity = None
        self.allowed_similarity = None
        self.similarity_list = []

        #TODO: Testing need:
        self.accuracy_rate = []

        #Adjust data to minirocket form CPU
        #X_transform,Y_datastream = self.get_adjust_XY(current_datastream=current_datastream)
        #Adjust data to minirocket form GPU
        X_datastream,Y_datastream = self.get_XY(current_datastream=current_datastream)

        #Train classifier CPU
        #self.classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        #self.classifier.fit(X_transform, Y_datastream)
        
        #Train classifier GPU
        self.train(X_feat,Y_datastream)
        
        #Generate First Fingureprint
        #self.predict(current_datastream)
        self.predict_GPU(X_feat,current_datastream)
        self.cur_fingerprint = self.get_fingerprint(current_datastream)
        self.fingerprint_pool.append(self.cur_fingerprint)
        self.flaten_fingerprint_pool.append(flaten_fingerprint(self.cur_fingerprint))

        #Initialized weight
        self.fingerprint_pool_sd = get_fingerpool_sd(self.flaten_fingerprint_pool)
        self.weight_sd = get_weight_sd(self.fingerprint_pool_sd)
        #self.weight_X_sd = get_weight_sd([flat_fingerprint[:-3*10] for flat_fingerprint in self.flaten_fingerprint_pool])

        #Record Accuracy TODO: More implementation needed: include updated accuracy_rate
        current_accuracy = current_datastream["predict_corr"].sum()/current_datastream.shape[0]
        self.accuracy_rate.append(current_accuracy)

    def __str__(self):
        return f"similarity list {self.similarity_list}"
        return f"similarity list {self.similarity_list}\n weight {self.weight_sd}\n fingerprint {self.cur_fingerprint}\n "

    def weighted_sd(self,array,weight):
        average = np.average(array, weights=weight)
        variance = np.average((array-average)**2, weights=weight)
        return math.sqrt(variance)

    def get_XY(self,current_datastream):
        Y_datastream = current_datastream["Y"].to_numpy(dtype = "float32")
        X_datastream = current_datastream.drop("Y", axis=1).to_numpy(dtype = "float32")

        return X_datastream,Y_datastream

    def get_adjust_XY(self,current_datastream):

        Y_datastream = current_datastream["Y"].to_numpy(dtype = "float32")
        X_datastream = current_datastream.drop("Y", axis=1).to_numpy(dtype = "float32")
        parameters = fit(X_datastream)
        X_transform = transform(X_datastream, parameters)

        return X_transform,Y_datastream

    def get_adjust_XY_GPU(self,current_datastream):
        #Historiical version not using now
        Y_datastream = current_datastream["Y"].to_numpy(dtype = "float32")
        X_datastream = current_datastream.drop("Y", axis=1).to_numpy(dtype = "float32")
        reshaped_X = X_datastream.reshape((1, X_datastream.shape[0], X_datastream.shape[1]))
        #Adjust data to minirocket form
        # parameter = Dimension,features
        mrf = MiniRocketFeatures(X_datastream.shape[0], X_datastream.shape[1]).to(default_device())
        PATH = Path(f"./models/MRF_{self.index}.pt")
        mrf.load_state_dict(torch.load(PATH))   
        new_feat = get_minirocket_features(reshaped_X, mrf, chunksize=1024, to_np=True)
        new_feat = np.repeat(new_feat,X_datastream.shape[0],axis=0)
        return new_feat,Y_datastream

    def predict(self,current_datastream):
        X_transform,Y_datastream =self.get_adjust_XY(current_datastream=current_datastream)
        predict_Y = self.classifier.predict(X_transform)
        #generate "Predict_Y" and "Predict_corr(ectness)" Columns
        current_datastream["predict_Y"] = predict_Y
        current_datastream["predict_corr"] = current_datastream.apply(lambda x:1 if x["Y"] == x["predict_Y"] else 0,axis = 1)
        return predict_Y
      
    def predict_GPU (self,X_feat,current_datastream):
        PATH = Path(f'./models/MRL_{self.index}.pkl')
        learn = load_learner(PATH, cpu=False)
        probas, _, predict_Y = learn.get_X_preds(X_feat)
        predict_Y = [float(predict_i) for predict_i in predict_Y]
        current_datastream["predict_Y"] = predict_Y
        current_datastream["predict_corr"] = current_datastream.apply(lambda x:1 if x["Y"] == x["predict_Y"] else 0,axis = 1)
        return predict_Y

    def train(self,X_feat, Y_datastream):
        #Train classifier GPU
        # Using tsai/fastai, create DataLoaders for the features in X_feat.
        tfms = [None, TSClassification()]
        batch_tfms = TSStandardize(by_sample=True)
        dls = get_ts_dls(X_feat, Y_datastream, tfms=tfms, batch_tfms=batch_tfms)
        self.classifier = build_ts_model(MiniRocketHead, dls=dls)
        
        learn = Learner(dls, self.classifier, metrics=accuracy)
        #l_value = learn.lr_find()
        #print(f"learning rate = {l_value}")
        learn.fit_one_cycle(10, 7e-4)
        PATH = Path(f'./models/MRL_{self.index}.pkl')
        PATH.parent.mkdir(parents=True, exist_ok=True)
        learn.export(PATH)

    def get_fingerprint(self,current_datastream):
        return fingerprint_generator(expredicted_datastream=current_datastream)

    def get_similarity(self,new_fingerprint,weight_between_concept,X_only = False):
        self.weight_between_concept = weight_between_concept
        cur_fingerprint_flaten = flaten_fingerprint(new_fingerprint)
        if X_only:
            weight = get_weight(self.weight_X_sd,weight_between_concept)
        else:
            weight = get_weight(self.weight_sd,weight_between_concept)
        cur_similarity = get_cosine_distance(self.flaten_fingerprint_pool[-1],cur_fingerprint_flaten,weight)
        return -cur_similarity

    def similarity_check(self,new_similarity):
        '''
        Return value: 1: Normal case (no further update needed)
        2: New max similarity
        3: No previous similarity (2nd/3rd fingerprint)
        0: Exceed allowed similarity
        '''
        if self.allowed_similarity:
            if new_similarity > self.allowed_similarity:
                if new_similarity > self.max_similarity:
                    return 2
                else:
                    return 1
            else:
                return 0
        else:
            return 3

    def outliner_check(self,new_similarity):
        temp_similarity_list = [x for x in self.similarity_list]
        for idx in range(len(self.fingerprint_vote)):
            temp_similarity_list += [self.similarity_list[idx]] * (self.fingerprint_vote[idx] -1)
        t_critical_value = t.ppf(q=self.p_value_st,df=sum(self.fingerprint_vote))
        max_similarity_allowed = np.max(temp_similarity_list) + self.weighted_sd(self.similarity_list,self.fingerprint_vote)*t_critical_value
        log_info = f"t_critical_value {t_critical_value},max_similarity_allowed {max_similarity_allowed}"
        for idx in range(len(self.similarity_list)-1):
            cur_similarity = self.similarity_list[idx]
            if cur_similarity < max_similarity_allowed:
                log_info += "yes"
                self.similarity_list.pop(idx)
                self.fingerprint_pool.pop(idx)
                self.flaten_fingerprint_pool.pop(idx)
                self.fingerprint_vote.pop(idx)
                #update weight
                self.fingerprint_pool_sd = get_fingerpool_sd(self.flaten_fingerprint_pool)
                self.weight_sd = get_weight_sd(self.fingerprint_pool_sd)
                #Update similarity
                self.max_similarity = np.mean(self.similarity_list) + self.max_similarity_scaling_factor*self.weighted_sd(self.similarity_list,self.fingerprint_vote)
                self.allowed_similarity = np.mean(self.similarity_list) - self.min_similarity_scaling_factor*self.weighted_sd(self.similarity_list,self.fingerprint_vote)
                return (True,log_info)
        return (False,log_info)
        
    def update(self,states,new_similarity,new_fingerprint,X_feat,current_datastream,GPU=False):
        '''
        1: Normal case (no further update needed)
        2: New max similarity
        3: No previous similarity (2nd/3rd fingerprint)
        '''
        log_info = ""
        if states == 1:
            self.fingerprint_vote[-1] += 1
            if not self.initial_update or len(self.similarity_list)>3:
                res,log_info = self.outliner_check(new_similarity)
                if res:
                    self.initial_update = True
        
        elif states == 2:
            #Add fingerprint into the pool
            self.fingerprint_pool.append(new_fingerprint)
            self.flaten_fingerprint_pool.append(flaten_fingerprint(new_fingerprint))
            self.fingerprint_vote.append(1)

            #Update similarity
            self.similarity_list = []
            for fingerprint in self.fingerprint_pool[:-1]:
                self.similarity_list.append(self.get_similarity(fingerprint,self.weight_between_concept))
            #self.similarity_list.append(new_similarity)
            self.max_similarity = np.mean(self.similarity_list) + self.max_similarity_scaling_factor*self.weighted_sd(self.similarity_list,self.fingerprint_vote)
            log_info += f" max_similarity {self.max_similarity}"
            self.allowed_similarity = np.mean(self.similarity_list) - self.min_similarity_scaling_factor*self.weighted_sd(self.similarity_list,self.fingerprint_vote)

            #update weight
            self.fingerprint_pool_sd = get_fingerpool_sd(self.flaten_fingerprint_pool)
            self.weight_sd = get_weight_sd(self.fingerprint_pool_sd)

            '''
            item_to_drop = []
            for idx in range (len(self.similarity_list)):
                if self.similarity_list[idx] < self.allowed_similarity:
                    item_to_drop.append(idx)
            for idx in item_to_drop:
                self.similarity_list.pop(idx)
                self.fingerprint_pool.pop(idx)
            '''

            if GPU:
                current_datastream.drop(["predict_Y","predict_corr"],axis=1,inplace=True)
                Y_datastream = current_datastream["Y"].to_numpy(dtype = "float32")
                self.train(X_feat,Y_datastream)
            else:
                #TODO:Retrain classifier (test for beter result)
                X_transform,Y_datastream =self.get_adjust_XY(current_datastream=current_datastream)
                self.classifier.fit(X_transform, Y_datastream)


        elif states == 3:
            #Add fingerprint into the pool
            self.fingerprint_pool.insert(0,new_fingerprint)
            self.flaten_fingerprint_pool.insert(0,flaten_fingerprint(new_fingerprint))
            self.fingerprint_vote.insert(0,1)

            #update weight
            self.fingerprint_pool_sd = get_fingerpool_sd(self.flaten_fingerprint_pool)
            self.weight_sd = get_weight_sd(self.fingerprint_pool_sd)

            #Update similarity
            self.similarity_list.insert(0,new_similarity)
            self.max_similarity = min(self.similarity_list) + self.max_similarity_scaling_factor*np.std(self.similarity_list)
            log_info += f" max_similarity {self.max_similarity}"
            if len(self.similarity_list) >= 2:
                self.allowed_similarity = np.mean(self.similarity_list) - self.min_similarity_scaling_factor*np.std(self.similarity_list)

        return log_info
