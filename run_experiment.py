import pandas as pd
import numpy as np
import warnings
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tsai.basics import *
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *
from scipy.stats import ttest_1samp
from sklearn.metrics import cohen_kappa_score as kappa

from Classifier import Classifier
from generate_fingerprint import fingerprint_generator
from load_datastream import *
from calculate_weight_2 import *

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

def get_adjust_XY_GPU(current_datastream):
    Y_datastream = current_datastream["Y"].to_numpy(dtype = "float32")
    X_datastream = current_datastream.drop("Y", axis=1).to_numpy(dtype = "float32")
    reshaped_X = X_datastream.reshape((X_datastream.shape[0], 1, X_datastream.shape[1]))
    #Adjust data to minirocket form
    # parameter = Dimension,features
    mrf = MiniRocketFeatures(1, X_datastream.shape[1]).to(default_device())
    PATH = Path(f"./models/MRF_temp.pt")
    mrf.load_state_dict(torch.load(PATH))   
    new_feat = get_minirocket_features(reshaped_X, mrf, chunksize=1024, to_np=True)
    #new_feat = np.repeat(new_feat,X_datastream.shape[0],axis=0)
    return new_feat,Y_datastream

def init_miniRocket(current_datastream):
    Y_datastream = current_datastream["Y"].to_numpy(dtype = "float32")
    X_datastream = current_datastream.drop("Y", axis=1).to_numpy(dtype = "float32")
    mrf_c = MiniRocketFeatures(1,X_datastream.shape[1]).to(default_device())
    reshaped_X = X_datastream.reshape((X_datastream.shape[0],1, X_datastream.shape[1]))
    mrf_c.fit(reshaped_X[0:1])
    X_feat = get_minirocket_features(reshaped_X, mrf_c, chunksize=1024, to_np=True)
    print(X_feat.shape)
    #X_feat = np.repeat(X_feat,X_datastream.shape[0],axis=0)
    PATH = Path(f"./models/MRF_temp.pt")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mrf_c.state_dict(), PATH)

def main():
    logging.basicConfig(filename="log.log", level=logging.INFO)
    warnings.filterwarnings("ignore")

    #Parameters
    length = 50
    max_sim_factor = 1.5
    min_sim_factor = 1.5
    p_value_st= 0.05
    max_num_sim_per_classifier = 4
    data_path = "./Rawdata/Other/AS.csv"
    max_testing_sample = 24000
    
    step_counter = 0
    cur_classifier_idx = None
    cur_train_datastream = None
    accuracy_list = []
    similarity_list =[]
    classifier_selected =[]
    classifier_list = [None]
    #figureprints for all classifier to generate the standard deviation of weight
    flatten_figureprint_repo = []
    fingerrepo_sd = None
    weight_between_concept = None
    weight_between_concept_X = None

    #TODO:Testing need
    new_classifier_index = []
    predict_Y_list= []
    correct_Y_list = []

    #Load data 1
    '''
    datastream,type_idx = load_data(data_path)
    seed = 5
    '''
    
    #Load data 2: UCR-Archive
    '''
    file_train,file_test = load_data_UCR(data_path)
    file_train.append(file_test, ignore_index=True)
    datastream = file_test
    seed = 1
    type_idx = []
    '''

    #Load data 3:
    datastream = load_data_other(data_path)
    '''
    data_length = len(datastream)
    split_length = 1000
    splits = [(x,x+split_length) for x in range(0,data_length,split_length)]
    sample_rate = [0.25]*int(data_length/split_length)
    randomize = True
    train,test =  data_partition(datastream,splits,sample_rate,randomize=randomize)
    datastream = pd.concat([train,test],ignore_index=True)
    '''
    type_idx = []
    seed = 1

    
    #Data Split (if seed == 0, ignore randomness)
    overlap=0.4
    overlap_idx = int(length * overlap)
    splited_datastream_list,log_info,splited_data_stream_indexs = data_segmentation(datastream,length,seed,type_idx,overlap=overlap,max_testing_sample=max_testing_sample)
    logging.info(splited_datastream_list[0])
    #logging.info(splited_datastream_list[1])
    logging.info(f"lenth = {length}, overlap = {overlap}, max_sim_factor = {max_sim_factor}, min_sim_factor = {min_sim_factor}")
    logging.info(log_info)
    logging.info(f"number of run = {len(splited_datastream_list)}")
    
    init_datastream = splited_datastream_list[0].copy()
    init_miniRocket(init_datastream)
    print(init_datastream.shape)


    cur_idx = 0


    for current_datastream in splited_datastream_list:
        logging.info(f"cur_idx = {cur_idx} - {cur_idx+length-1}")
        cur_idx += round(length*(1-overlap))
        step_counter += 1

        cur_feat,cur_Y_datastream = get_adjust_XY_GPU(current_datastream)
        cur_fingerprint = fingerprint_generator(current_datastream)
        

        '''
        #Start with 
        if cur_classifier_idx:
            # if classifier exist try current classifier
            cur_classifier = classifier_list[cur_classifier_idx]
            
            #Get similarity
            cur_X_similarity = cur_classifier.get_similarity(cur_fingerprint,weight_between_concept,X_only=True)
            state_similarity = cur_classifier.similarity_check(cur_X_similarity)

            if state_similarity:
                #if selected current classifier
                predict_Y = cur_classifier.predict_GPU(cur_feat,current_datastream)
            
            else:
                #iterate through classifier and select most appropriate classifier
                for idx in range(1,len(classifier_list)):
                    dd

        else:
            #if there is no classifier, train one.
            new_classifier_index.append(step_counter-1)
            new_classifier = Classifier(cur_feat,current_datastream,len(classifier_list),True,max_sim_factor,min_sim_factor,p_value_st)
            classifier_list.append(new_classifier)
            cur_classifier_idx = 1
            flatten_figureprint_repo.append(new_classifier.flaten_fingerprint_pool[-1])
            fingerrepo_sd = get_fingerpool_sd(flatten_figureprint_repo)
            weight_between_concept = [1] * len(fingerrepo_sd)

            cur_classifier = classifier_list[cur_classifier_idx]
            current_datastream.drop(["predict_Y","predict_corr"],axis=1,inplace=True)
            predict_Y = cur_classifier.predict_GPU(cur_feat,current_datastream)
            
            #Testing Info
            accuracy_list.append(2)
            similarity_list.append(0)
            classifier_selected.append(cur_classifier_idx)
            predict_Y_list+= predict_Y
            correct_Y_list+= list(current_datastream["Y"])

        '''


        if cur_classifier_idx:
            # if classifier exist try current classifier
            cur_classifier = classifier_list[cur_classifier_idx]
            predict_Y = cur_classifier.predict_GPU(cur_feat,current_datastream)
                    
            #Get stat for figureprint

            temp_something = current_datastream.loc[:, ['predict_Y','predict_corr']]
            cur_fingerprint_Y = cur_classifier.get_fingerprint(temp_something)
            cur_fingerprint+=cur_fingerprint_Y

            #Get similarity
            cur_similarity = cur_classifier.get_similarity(cur_fingerprint,weight_between_concept)
            state_similarity = cur_classifier.similarity_check(cur_similarity)
            current_accuracy = current_datastream["predict_corr"].sum()/current_datastream.shape[0]
            logging.info(f"Classifier: {cur_classifier_idx}, Accuracy: {current_accuracy}, simiarity = {cur_similarity}, similarity_list = {cur_classifier}")
            
            new_classifier_needed = True
            if state_similarity: 
                # Similarity check pass
                predict_Y_list+= predict_Y[overlap_idx:]
                correct_Y_list+= list(current_datastream["Y"])[overlap_idx:]
                # Call current classifier update function when sim check passed
                new_classifier_needed = False
                log_info = cur_classifier.update(state_similarity,cur_similarity,cur_fingerprint,cur_feat,current_datastream,GPU=True)
                logging.info(log_info)
                if state_similarity > 1:
                    if state_similarity == 3:
                        flatten_figureprint_repo.append(new_classifier.flaten_fingerprint_pool[0])
                    else:
                        flatten_figureprint_repo.append(new_classifier.flaten_fingerprint_pool[-1])
                    fingerrepo_sd = get_fingerpool_sd(flatten_figureprint_repo)
                    weight_between_concept = get_weight_between_concept(fingerrepo_sd)
                
                #Testing Info
                accuracy_list.append(current_accuracy)
                similarity_list.append(cur_similarity)
                classifier_selected.append(cur_classifier_idx)
                logging.info(f"Type1 - use current classifier {cur_classifier_idx}, state_similarity: {state_similarity}\n")
            
            else: 
                # When similarity check not pass, iter through classifier list
                # or init new classifier if no classifier accepted
                temp_similarity_list = []
                temp_fingerprint_list = []
                accepted_similarity_list = []
                temp_accuracy_list = []
                temp_predict_Y = [0]
                loop_exit = False
                for idx in range(1,len(classifier_list)):
                    # TODO: repeated code here, might need restrucure
                    cur_classifier = classifier_list[idx]
                    current_datastream.drop(["predict_Y","predict_corr"],axis=1,inplace=True)
                    predict_Y = cur_classifier.predict_GPU(cur_feat,current_datastream)
                    temp_predict_Y.append(predict_Y.copy())

                    #TODO:Shows Accuracy
                    current_accuracy = current_datastream["predict_corr"].sum()/current_datastream.shape[0]
                    
                    #Get stat for figureprint
                    cur_fingerprint = cur_classifier.get_fingerprint(current_datastream)
                    temp_fingerprint_list.append(cur_fingerprint)

                    #Get similarity
                    cur_similarity = cur_classifier.get_similarity(cur_fingerprint,weight_between_concept)
                    state_similarity = cur_classifier.similarity_check(cur_similarity)
                    logging.info(f"Classifier: {idx}, Accuracy: {current_accuracy}, state_similarity:{state_similarity}, similarity: {cur_similarity}, similarity_list = {cur_classifier}")

                    temp_similarity_list.append((cur_similarity,idx))
                    temp_accuracy_list.append(current_accuracy)
                    
                    if state_similarity:
                        accepted_similarity_list.append((cur_similarity,idx,state_similarity,cur_fingerprint))
                        loop_exit = True


                if loop_exit:
                    accepted_similarity_list.sort(reverse=True)
                    cur_similarity,idx,state_similarity,cur_fingerprint = accepted_similarity_list[0]
                    cur_classifier = classifier_list[idx]
                    cur_classifier.update(state_similarity,cur_similarity,cur_fingerprint,cur_feat,current_datastream,GPU=True)
                    new_classifier_needed = False
                    cur_classifier_idx = idx
                    accuracy_list.append(temp_accuracy_list[idx-1])
                    similarity_list.append(cur_similarity)
                    logging.info((f"Type2 - loop to last classifier{cur_classifier_idx}, similarity: {cur_similarity}, state_similarity: {state_similarity}\n"))
                    continue
                '''
                temp_similarity_list.sort(reverse=True)
                copy_of_sim = [sim[0] for sim in temp_similarity_list]
                #Method Normal Distributed
                for idx in range(1,len(classifier_list)):
                    temp_idx = temp_similarity_list[idx-1][1]
                    cur_classifier = classifier_list[temp_idx]
                    cur_similarity = temp_similarity_list[idx-1][0]
                    cur_fingerprint = temp_fingerprint_list[temp_idx-1]
                    cur_accuracy = temp_accuracy_list[temp_idx-1]
                    tset, pvalue = ttest_1samp(copy_of_sim, cur_similarity)
                    logging.info(f"Classifier: {temp_idx}, similarity: {cur_similarity}, t-test = {tset}, p-value = {pvalue}")
                    if pvalue < p_value_st and tset < 0:
                        predict_Y_list+= temp_predict_Y[temp_idx][overlap_idx:]
                        correct_Y_list+= list(current_datastream["Y"])[overlap_idx:]
                        new_classifier_needed = False
                        cur_classifier.update(2,cur_similarity,cur_fingerprint,current_datastream,GPU=True)
                        cur_classifier_idx = temp_idx
                        flatten_figureprint_repo.append(cur_classifier.flaten_fingerprint_pool[-1])
                        fingerrepo_sd = get_fingerpool_sd(flatten_figureprint_repo)
                        weight_between_concept = get_weight_between_concept(fingerrepo_sd)
                        
                        #Testing Info
                        accuracy_list.append(cur_accuracy)
                        similarity_list.append(cur_similarity)
                        classifier_selected.append(temp_idx)
                        logging.info((f"Type2 - loop to last classifier{temp_idx}\n"))
                        break
                '''
            if new_classifier_needed:        
                # New Classifier Needed
                new_classifier_index.append(step_counter-1)
                current_datastream.drop(["predict_Y","predict_corr"],axis=1,inplace=True)
                new_classifier = Classifier(cur_feat,current_datastream,len(classifier_list),True,max_sim_factor,min_sim_factor,)
                classifier_list.append(new_classifier)
                cur_classifier_idx = len(classifier_list)-1
                flatten_figureprint_repo.append(new_classifier.flaten_fingerprint_pool[-1])
                fingerrepo_sd = get_fingerpool_sd(flatten_figureprint_repo)
                weight_between_concept = get_weight_between_concept(fingerrepo_sd)

                cur_classifier = classifier_list[cur_classifier_idx]
                current_datastream.drop(["predict_Y","predict_corr"],axis=1,inplace=True)
                predict_Y = cur_classifier.predict_GPU(cur_feat,current_datastream)

                
                #Testing Info
                accuracy_list.append(2)
                similarity_list.append(0)
                classifier_selected.append(cur_classifier_idx)
                predict_Y_list+= predict_Y[overlap_idx:]
                correct_Y_list+= list(current_datastream["Y"])[overlap_idx:]
                logging.info("Type3 -new classifier generate\n")

            '''
            else:
                #Calculate Accuracy if no new classifier generated
                current_accuracy = current_datastream["predict_corr"].sum()/current_datastream.shape[0]
                accuracy_list.append(current_accuracy)
            '''
        
        else:
            #if there is no classifier, train one.
            new_classifier_index.append(step_counter-1)
            new_classifier = Classifier(cur_feat,current_datastream,len(classifier_list),True,max_sim_factor,min_sim_factor,p_value_st)
            classifier_list.append(new_classifier)
            cur_classifier_idx = 1
            flatten_figureprint_repo.append(new_classifier.flaten_fingerprint_pool[-1])
            fingerrepo_sd = get_fingerpool_sd(flatten_figureprint_repo)
            weight_between_concept = [1] * len(fingerrepo_sd)

            cur_classifier = classifier_list[cur_classifier_idx]
            current_datastream.drop(["predict_Y","predict_corr"],axis=1,inplace=True)
            predict_Y = cur_classifier.predict_GPU(cur_feat,current_datastream)
            
            #Testing Info
            accuracy_list.append(2)
            similarity_list.append(0)
            classifier_selected.append(cur_classifier_idx)
            predict_Y_list+= predict_Y
            correct_Y_list+= list(current_datastream["Y"])
    
    logging.info(accuracy_list)
    logging.info(f"Accuracy of n={length}, no shuffle and no update,least accurate = {min(accuracy_list)}")
    logging.info(f"least accuracy found in {accuracy_list.index(min(accuracy_list))}")
    logging.info(f"number of classifier {len(new_classifier_index)}, they are {new_classifier_index}")
    logging.info(f"kappa score = {kappa(predict_Y_list,correct_Y_list)}")

    fig,ax = plt.subplots()
    ax.plot(splited_data_stream_indexs, accuracy_list,color="blue", marker="o")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Accuracy",color="blue")
    ax2=ax.twinx()
    ax2.plot(splited_data_stream_indexs, similarity_list,color="red",marker="o")
    ax2.set_ylabel("Similarity",color="red",fontsize=14)
    plt.show()


    '''
    for idx in range(1,len(classifier_list)):
        log_info(f"Classifier: {idx}, similarity_list = {str(classifier_list[idx])}")
        log_info(f"    Weight = {classifier_list[idx].weight_sd}")
    '''
        
if __name__ == "__main__":
    main()     

        