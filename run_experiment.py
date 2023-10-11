import pandas as pd
import numpy as np
import warnings
import logging
from pathlib import Path
import torch
from tsai.basics import *
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *

from Classifier import Classifier
from load_datastream import load_data,data_segmentation,load_data_UCR

#Helper Function
def get_error_distance(data_stream):
    prev_error_idx = -1
    error_distances = []
    for i,err in enumerate(data_stream["predict_corr"]):
        if err:
            if prev_error_idx == -1:
                prev_error_idx = i
            else:
                distance = i - prev_error_idx
                error_distances.append(distance)
                prev_error_idx = i
    if len(error_distances) == 0:
        error_distances = [0]
    return error_distances
    
def main():
    logging.basicConfig(filename="newlog.log", level=logging.INFO)
    warnings.filterwarnings("ignore")

    #Parameters
    length = 30
    max_sim_factor = 1.5
    min_sim_factor = 1
    data_path = "./Rawdata/UCR/wafer"
    
    step_counter = 0
    cur_classifier_idx = None
    cur_train_datastream = None
    accuracy_list = []
    classifier_list = [None]

    #TODO:Testing need
    new_classifier_index = []
    
    #Load data 1
    #datastream,type_idx = load_data(data_path)
    #seed = 3

    #Load data 2: UCR-Archive
    file_train,file_test = load_data_UCR(data_path)
    datastream = file_train
    seed = 0
    type_idx = []

    #Data Split (if seed == 0, ignore randomness)
    overlap=0.6
    splited_datastream_list,log_info = data_segmentation(datastream,length,seed,type_idx,overlap=overlap)
    logging.info(f"lenth = {length}, overlap = {overlap}, max_sim_factor = {max_sim_factor}, min_sim_factor = {min_sim_factor}")
    logging.info(log_info)
    logging.info(f"number of run = {len(splited_datastream_list)}")
    
    #NEW
    mrf = MiniRocketFeatures(splited_datastream_list[0].shape[0], splited_datastream_list[0].shape[1]-1).to(default_device())
    PATH = Path("./models/MRF.pt")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mrf.state_dict(), PATH)
    
    cur_idx = 0

    for current_datastream in splited_datastream_list:
        logging.info(f"step_counter {step_counter}, cur_idx = {cur_idx} - {cur_idx+length-1}")
        cur_idx += round(length*(1-overlap))
        step_counter += 1

        if cur_classifier_idx:
            # if classifier exist try current classifier
            cur_classifier = classifier_list[cur_classifier_idx]
            predict_Y = cur_classifier.predict(current_datastream)
                    
            #Get stat for figureprint
            cur_fingerprint = cur_classifier.get_fingerprint(current_datastream)

            #Get similarity
            cur_similarity = cur_classifier.get_similarity(cur_fingerprint)
            state_similarity = cur_classifier.similarity_check(cur_similarity)
            current_accuracy = current_datastream["predict_corr"].sum()/current_datastream.shape[0]
            logging.info(f"Classifier: {cur_classifier_idx}, Accuracy: {current_accuracy}, simiarity = {cur_similarity}, similarity_list = {cur_classifier}")
            
            new_classifier_needed = True
            if state_similarity: 
                # Call current classifier update function when sim check passed
                cur_classifier.update(state_similarity,cur_similarity,cur_fingerprint,current_datastream)
                new_classifier_needed = False
                logging.info(f"Type1 - use current classifier {cur_classifier_idx}, state_similarity: {state_similarity}")
            
            else: 
                # When similarity check not pass, iter through classifier list
                # or init new classifier if no classifier accepted
                for idx in range(1,len(classifier_list)):
                    # TODO: repeated code here, might need restrucure
                    cur_classifier = classifier_list[idx]
                    predict_Y = cur_classifier.predict(current_datastream)

                    #TODO:Shows Accuracy
                    current_accuracy = current_datastream["predict_corr"].sum()/current_datastream.shape[0]
                    
                    #Get stat for figureprint
                    cur_fingerprint = cur_classifier.get_fingerprint(current_datastream)

                    #Get similarity
                    cur_similarity = cur_classifier.get_similarity(cur_fingerprint)
                    state_similarity = cur_classifier.similarity_check(cur_similarity)
                    logging.info(f"Classifier: {idx}, Accuracy: {current_accuracy}, similarity: {cur_similarity}, similarity_list = {cur_classifier}")

                    if state_similarity:
                        cur_classifier.update(state_similarity,cur_similarity,cur_fingerprint,current_datastream)
                        new_classifier_needed = False
                        cur_classifier_idx = idx
                        logging.info((f"Type2 - loop to last classifier{cur_classifier_idx}, similarity: {cur_similarity}, state_similarity: {state_similarity}"))
                        break

            if new_classifier_needed:        
                # New Classifier Needed
                new_classifier_index.append(step_counter-1)
                current_datastream.drop(["predict_Y","predict_corr"],axis=1,inplace=True)
                new_classifier = Classifier(current_datastream,len(classifier_list),max_sim_factor,min_sim_factor)
                classifier_list.append(new_classifier)
                cur_classifier_idx = len(classifier_list)-1
                logging.info("Type3 -new classifier generate")
            
            else:
                #Calculate Accuracy if no new classifier generated
                current_accuracy = current_datastream["predict_corr"].sum()/current_datastream.shape[0]
                accuracy_list.append(current_accuracy)
        
        else:
            #if there is no classifier, train one.
            new_classifier_index.append(step_counter-1)
            new_classifier = Classifier(current_datastream,len(classifier_list),max_sim_factor,min_sim_factor)
            classifier_list.append(new_classifier)
            cur_classifier_idx = 1
    
    logging.info(accuracy_list)
    logging.info(f"Accuracy of n={length}, no shuffle and no update,least accurate = {min(accuracy_list)}")
    logging.info(f"least accuracy found in {accuracy_list.index(min(accuracy_list))}")
    logging.info(f"number of classifier {len(new_classifier_index)}, they are {new_classifier_index}")

    for idx in range(1,len(classifier_list)):
        log_info(f"Classifier: {idx}, similarity_list = {classifier_list[idx]}")
        log_info(f"    Weight = {classifier_list[idx].weight_sd}")
        
if __name__ == "__main__":
    main()     

        