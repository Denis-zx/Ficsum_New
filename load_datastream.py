#Data Function
import pathlib
import pandas as pd
import random
from sklearn.utils import shuffle

def load_data(file_path):
    file_path = pathlib.Path(file_path)
    file_list = list(file_path.glob('*.csv'))
    data_stream = None
    print("number of file read",file_list)
    type_idx = []
    idx_count = 0

    
    #assign types
    for i in range(len(file_list)):
        type_idx.append(idx_count)

        #Append corresponding y to df
        df = pd.read_csv(file_list[i])
        df["Y"] = i

        idx_count += df.shape[0]

        try:
            data_stream = pd.concat([data_stream,df],ignore_index = True)

        except Exception as error:
            print(error)
            data_stream = df
    
    print("numberof1",data_stream["Y"].sum())

    #Return concated datastream and index list where type separated 
    return data_stream,type_idx


def data_segmentation(data_stream,length,seed,type_idx = [],overlap = 0):
    '''
    Return Value: [pd.DF,pd.DF, ... ] 
    Dataframe of length
    '''
    log_info = ""

    if seed == 0:
        unshuffled_data_stream = data_stream.copy(deep=True)
        splited_shuffled_data_stream =[]
        cur_idx = 0
        while cur_idx + length < unshuffled_data_stream.shape[0]:
            splited_shuffled_data_stream.append(unshuffled_data_stream[cur_idx:cur_idx+length])
            cur_idx += round(length*(1-overlap))
        splited_shuffled_data_stream.append(unshuffled_data_stream[cur_idx:])
        log_info += f"number of X_parameter = {unshuffled_data_stream.shape[1]-1}"
    
    elif seed == 1:
        unshuffled_data_stream = data_stream.copy(deep=True)
        splited_data_stream = [unshuffled_data_stream[i:i+length] for i in range(0,unshuffled_data_stream.shape[0],length)]
    
    elif seed == 2:
        # two repeated 
        if len(type_idx) == 0:
            raise ValueError
        unshuffled_data_stream = data_stream.copy(deep=True)
        splited_shuffled_data_stream = pd.DataFrame(columns=unshuffled_data_stream.columns)
        type_idx.sort(reverse = True)
        #get two set of unchange data
        splited_shuffled_data_stream = []
        for idx in type_idx:
            splited_shuffled_data_stream.append(unshuffled_data_stream.iloc[idx:idx+length])
        for idx in type_idx:
            splited_shuffled_data_stream.append(unshuffled_data_stream.iloc[idx+length:idx+length*2])
            unshuffled_data_stream.drop(index = [i for i in range (idx,idx+length*2)],inplace = True)
        shuffled_data_stream = shuffle(unshuffled_data_stream,random_state=seed)
        splited_shuffled_data_stream += [shuffled_data_stream[i:i+length] for i in range(0,shuffled_data_stream.shape[0],length)]
        
    else:
        #divided according to overlap ratio and random breakpoint 
        empty_list = [list() for x in range(len(type_idx))]
        type_list = [x for x in range(len(type_idx))]
        segment_point = dict(zip(type_list,empty_list))
        
        random.seed(seed)
        type_idx.append(data_stream.shape[0]-1)
        #Random generate break point 
        for t in range (len(type_idx)-1):
            for i in range (seed-1):  
                segment_point[t].append(random.randint(type_idx[t],type_idx[t+1]))
            segment_point[t].append(type_idx[t])
            segment_point[t].append(type_idx[t+1])
            segment_point[t].sort()

        log_info = str(segment_point)
        
        #new dataframe 
        shuffled_data_stream = pd.DataFrame(None,columns=data_stream.columns)
        for i in range (seed):
            for t in range (len(type_idx)-1):
                next_data_stream = data_stream.iloc[segment_point[t][i]:segment_point[t][i+1]]
                shuffled_data_stream = pd.concat([shuffled_data_stream, next_data_stream], ignore_index=True)
        shuffled_data_stream = shuffled_data_stream.astype("float")
        
        cur_idx = 0
        splited_shuffled_data_stream =[]
        while cur_idx + length < shuffled_data_stream.shape[0]:
            splited_shuffled_data_stream.append(shuffled_data_stream[cur_idx:cur_idx+length])
            cur_idx += round(length*(1-overlap))
        splited_shuffled_data_stream.append(shuffled_data_stream[cur_idx:])

    return splited_shuffled_data_stream,log_info


def load_data_UCR (file_path):
    file_path = pathlib.Path(file_path)
    file_list = list(file_path.glob('*.csv'))
    print(*file_list)
    file_train = pd.read_csv(file_list[1],header=None)
    file_test = pd.read_csv(file_list[0],header=None)
    header = [f"X_{i}" for i in range(1,file_train.shape[1])]
    header = ["Y"] + header
    file_train.columns = header
    file_test.columns = header
    return (file_train,file_test)