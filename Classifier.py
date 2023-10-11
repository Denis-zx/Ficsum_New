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

# Fingerprint features ['IMF',"mean","stdev","skew",'kurtosis','turning_point_rate','acf','pacf','MI','FI']

class Classifier:
    
    def __init__(self,current_datastream,index,max_sim_factor=2,min_sim_factor=2):
        #Definition Headers
        self.index = index
        self.classifier = None
        self.cur_fingerprint = None
        self.fingerprint_pool = []
        self.flaten_fingerprint_pool = []

        self.fingerprint_pool_sd = None
        self.weight_sd = None

        self.max_similarity_scaling_factor = max_sim_factor
        self.min_similarity_scaling_factor = min_sim_factor
        self.max_similarity = None
        self.allowed_similarity = None
        self.similarity_list = []

        #TODO: Testing need:
        self.accuracy_rate = []

        #Adjust data to minirocket form CPU
        #X_transform,Y_datastream = self.get_adjust_XY(current_datastream=current_datastream)
        #Adjust data to minirocket form GPU
        X_transform,Y_datastream = self.get_adjust_XY_GPU(current_datastream=current_datastream)

        #Train classifier CPU
        #self.classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        #self.classifier.fit(X_transform, Y_datastream)
        
        #Train classifier GPU
        # Using tsai/fastai, create DataLoaders for the features in X_feat.
        tfms = [None, TSClassification()]
        batch_tfms = TSStandardize(by_sample=True)
        dls = get_ts_dls(X_transform, Y_datastream, tfms=tfms, batch_tfms=batch_tfms)
        self.classifier = build_ts_model(MiniRocketHead, dls=dls)
        
        learn = Learner(dls, self.classifier, metrics=accuracy, cbs=ShowGraph())
        learn.lr_find()
        PATH = Path(f'./models/MRL_{self.index}.pkl')
        PATH.parent.mkdir(parents=True, exist_ok=True)
        learn.export(PATH)
        
        #Generate First Fingureprint
        #self.predict(current_datastream)
        self.predict_GPU(current_datastream)
        self.cur_fingerprint = self.get_fingerprint(current_datastream)
        self.fingerprint_pool.append(self.cur_fingerprint)
        self.flaten_fingerprint_pool.append(flaten_fingerprint(self.cur_fingerprint))

        #Initialized weight
        self.fingerprint_pool_sd = get_fingerpool_sd(self.flaten_fingerprint_pool)
        self.weight_sd = get_weight_sd(self.fingerprint_pool_sd)

        #Record Accuracy TODO: More implementation needed: include updated accuracy_rate
        current_accuracy = current_datastream["predict_corr"].sum()/current_datastream.shape[0]
        self.accuracy_rate.append(current_accuracy)

    def __str__(self):
        return f"similarity list {self.similarity_list}"
        return f"similarity list {self.similarity_list}\n weight {self.weight_sd}\n fingerprint {self.cur_fingerprint}\n "

    def get_adjust_XY(self,current_datastream):

        Y_datastream = current_datastream["Y"].to_numpy(dtype = "float32")
        X_datastream = current_datastream.drop("Y", axis=1).to_numpy(dtype = "float32")
        parameters = fit(X_datastream)
        X_transform = transform(X_datastream, parameters)

        return X_transform,Y_datastream

    def get_adjust_XY_GPU(self,current_datastream):
        #Adjust data to minirocket form
        print(current_datastream.shape[0], current_datastream.shape[1]-1)
        mrf = MiniRocketFeatures(current_datastream.shape[0], current_datastream.shape[1]-1).to(default_device())
        PATH = Path("./models/MRF.pt")
        mrf.load_state_dict(torch.load(PATH))   
        
        Y_datastream = current_datastream["Y"].to_numpy(dtype = "float32")
        X_datastream = current_datastream.drop("Y", axis=1).to_numpy(dtype = "float32")

        parameters = mrf.fit(X_datastream)
        X_transform = get_minirocket_features(X_datastream, mrf, chunksize=1024, to_np=True)
        return X_transform,Y_datastream

    def predict(self,current_datastream):
        X_transform,Y_datastream =self.get_adjust_XY(current_datastream=current_datastream)
        predict_Y = self.classifier.predict(X_transform)
        #generate "Predict_Y" and "Predict_corr(ectness)" Columns
        current_datastream["predict_Y"] = predict_Y
        current_datastream["predict_corr"] = current_datastream.apply(lambda x:1 if x["Y"] == x["predict_Y"] else 0,axis = 1)
      
    def predict_GPU (self,current_datastream):
        X_transform,Y_datastream =self.get_adjust_XY_GPU(current_datastream=current_datastream)
        PATH = Path(f'./models/MRL_{self.index}.pkl')
        learn = load_learner(PATH, cpu=False)
        probas, _, predict_Y = learn.get_X_preds(X_transform)
        current_datastream["predict_Y"] = predict_Y
        current_datastream["predict_corr"] = current_datastream.apply(lambda x:1 if x["Y"] == x["predict_Y"] else 0,axis = 1)

    def get_fingerprint(self,current_datastream):
        return fingerprint_generator(expredicted_datastream=current_datastream)

    def get_similarity(self,new_fingerprint):
        cur_fingerprint_flaten = flaten_fingerprint(new_fingerprint)
        cur_similarity = get_cosine_distance(self.flaten_fingerprint_pool[0],cur_fingerprint_flaten,self.weight_sd)
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

    def update(self,states,new_similarity,new_fingerprint,current_datastream):
        '''
        1: Normal case (no further update needed)
        2: New max similarity
        3: No previous similarity (2nd/3rd fingerprint)
        '''
        if states == 2:
            #Add fingerprint into the pool
            self.fingerprint_pool.append(new_fingerprint)
            self.flaten_fingerprint_pool.append(flaten_fingerprint(new_fingerprint))

            #update weight
            self.fingerprint_pool_sd = get_fingerpool_sd(self.flaten_fingerprint_pool)
            self.weight_sd = get_weight_sd(self.fingerprint_pool_sd)

            #Update similarity
            self.similarity_list.append(new_similarity)
            self.max_similarity = new_similarity + 1.5*np.std(self.similarity_list)
            self.allowed_similarity = np.mean(self.similarity_list) - 2*np.std(self.similarity_list)

            #TODO:Retrain classifier (test for beter result)
            X_transform,Y_datastream =self.get_adjust_XY(current_datastream=current_datastream)
            self.classifier.fit(X_transform, Y_datastream)


        elif states == 3:
            #Add fingerprint into the pool
            self.fingerprint_pool.append(new_fingerprint)
            self.flaten_fingerprint_pool.append(flaten_fingerprint(new_fingerprint))

            #update weight
            self.fingerprint_pool_sd = get_fingerpool_sd(self.flaten_fingerprint_pool)
            self.weight_sd = get_weight_sd(self.fingerprint_pool_sd)

            #Update similarity
            self.similarity_list.append(new_similarity)
            self.max_similarity = new_similarity + self.max_similarity_scaling_factor*np.std(self.similarity_list)
            if len(self.similarity_list) >= 2:
                self.allowed_similarity = np.mean(self.similarity_list) - self.min_similarity_scaling_factor*np.std(self.similarity_list)
