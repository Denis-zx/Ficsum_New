import numpy as np
import sklearn.metrics
from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
from load_datastream import *
from sklearn.metrics import mean_squared_error
#from softmax import train,predict
import sklearn
import time
import logging

filename= "AS-T"
data_path = "./Rawdata/Other/"+filename+".csv"
max_testing_sample = 24000

#Load data UCR:
#train,exam = load_data_UCR(data_path)

#load data Real:
data = load_data_other(data_path)
split_length = 1000
splits = [(x,x+split_length) for x in range(0,max_testing_sample,split_length)]


rsme_list = []
accuracy_score_list = []
kappa_score_list = []
train_time_list = []
test_time_list = []

logging.basicConfig(filename="mini"+filename+".log", level=logging.INFO)

def reindex_col(df):
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df

def main(cur_sample_rate):
    sample_rate = [cur_sample_rate]*int(max_testing_sample/split_length) 
    df_train,exam = data_partition(data,splits,sample_rate=sample_rate,randomize=False,shuffle=2)
    '''
    y_exam = exam["Y"].to_numpy(dtype = "float32")

    df_train = reindex_col(df_train)
    exam = reindex_col(exam)
    df_train.to_csv("temp_mini_train.csv",index=False,header=False)
    exam.to_csv("temp_mini_exam.csv",index=False,header=False)

    start_train_time = time.perf_counter()
    model_etc = train("temp_mini_train.csv", num_classes = 6, training_size = df_train.shape[0]-2048)
    end_train_time = time.perf_counter()

    start_test_time = time.perf_counter()
    y_pred, accuracy = predict("temp_mini_exam.csv", *model_etc) 
    end_test_time = time.perf_counter()

    #Data Split (if seed == 0, ignore randomness)
    '''
    y_train = df_train["Y"].to_numpy(dtype = "float32")
    x_train = df_train.drop("Y", axis=1).to_numpy(dtype = "float32")
    y_exam = exam["Y"].to_numpy(dtype = "float32")
    x_exam = exam.drop("Y", axis=1).to_numpy(dtype = "float32")

    start_train_time = time.perf_counter()
    parameters = fit(x_train)
    X_training_transform = transform(x_train, parameters)
    classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
    classifier.fit(X_training_transform, y_train)
    end_train_time = time.perf_counter()

    
    start_test_time = time.perf_counter()
    X_test_transform = transform(x_exam, parameters)
    y_pred = classifier.predict(X_test_transform)
    end_test_time = time.perf_counter()

    rmse = mean_squared_error(y_exam, y_pred, squared=False)
    accuracy_score = sklearn.metrics.accuracy_score(y_exam, y_pred)
    kappa_score = sklearn.metrics.cohen_kappa_score(y_exam,y_pred)
    
    rsme_list.append(rmse)
    accuracy_score_list.append(accuracy_score)
    kappa_score_list.append(kappa_score)
    train_time_list.append(end_train_time - start_train_time)
    test_time_list.append(end_test_time - start_test_time)

sample_rate_list = [0.5,0.6,0.7,0.8]
for item in sample_rate_list:
    main(item)
logging.info(f"rsme_list: {rsme_list}")
logging.info(f"accuracy_score_list:{accuracy_score_list}")
logging.info(f"kappa_score_list:{kappa_score_list}")
logging.info(f"train_time_list:{train_time_list}")
logging.info(f"test_time_list:{test_time_list}")