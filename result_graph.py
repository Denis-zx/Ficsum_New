import linecache
import os
import pandas as pd
import matplotlib.pyplot as plt


path = './Result'
dataframe = []
for round in os.listdir(path):
    cur_dict = {}
    for file in os.listdir(path+'/'+round):
        f = open(path+'/'+round+"/"+file,"rb")
        f.seek(0,2)
        f.seek(-10000, 2)
        s = f.readlines()
        s= s[-3:]
        num_classifier = str(s[0]).split(",")[0].split()[-1]
        kappa = float(str(s[1]).split("=")[-1].strip("\\r\\n\'"))
        test_time = float(str(s[2]).split("=")[-1].strip("\\r\\n\'"))
        cur_dict[file] = {"num_classifier":num_classifier,"kappa":kappa,"time":test_time}
        f.close()
    dataframe.append(pd.DataFrame(cur_dict))

result = pd.concat(dataframe)
result.to_csv("result.csv")
'''
result = pd.read_csv("result.csv",index_col=0)

for filename in ["HW"]:

    selected = [x for x in result.columns if filename in x]
    df_selected = result[selected]

    color1 = "red"
    color2 = "blue"
    labels = [0.6,0.7,0.8]
    #labels = [50,75,100]
    fig,ax = plt.subplots()
    ax.boxplot(df_selected[df_selected.index == "num_classifier"],labels=labels, boxprops=dict(color=color2),whiskerprops=dict(color=color2),capprops=dict(color=color2),medianprops=dict(color=color2))
    ax.set_xlabel("Overlap")
    #ax.set_xlabel("Window Size")
    ax.set_ylabel("#Classifier",color = color2)
    ax2=ax.twinx()
    ax2.boxplot(df_selected[df_selected.index == "kappa"],labels=labels, boxprops=dict(color=color1),whiskerprops=dict(color=color1),capprops=dict(color=color1),medianprops=dict(color=color1))
    ax2.set_ylabel("Kappa_Score",color = color1)
    plt.savefig("./Graph_result/"+filename)

 '''