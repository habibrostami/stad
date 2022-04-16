from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
from settings import RESULTS_XCELL
import numpy as np




def add_curve(true_value,predicted_value, headers,patterns, title=''):
    plt.figure()
    plt.subplots(1, figsize=(12, 12))
    plt.title(title, fontsize=18)

    for t, p, h, ptrs in zip(true_value, predicted_value, headers,patterns):
        auc = roc_auc_score(t, p)
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(t,p)
        plt.plot(false_positive_rate1, true_positive_rate1,ls=ptrs, label=h + "-auc="+str(round(auc,2)))


    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.legend(loc="lower right", fontsize=18)
    plt.show()

def extract_title(word):
    i = word.find('-')
    title = word[0:i]
    return title

def draw_roc(file_path):

    df = pd.read_excel(file_path)
    headers = list(df.columns.values)
    #print(df['VGG16_CRC.h5patient_base-predicted-value'])
    count = len(df.index)
    res_dict = {}
    for t in headers:
        res_dict[t] = []
        for item in df[t]:
            if not np.isnan(item):
                res_dict[t].append(item)
    return res_dict

def report_auc(true_value, predicted_value):
    target_names = ['MSIMUT', 'MSS']
    print(confusion_matrix(true_value, np.round(predicted_value)))
    print(classification_report(true_value, np.round(predicted_value), target_names=target_names))
    print('AUC=', roc_auc_score(true_value, predicted_value))

if __name__ == "__main__":

    res = draw_roc(RESULTS_XCELL)
    headers = list(res.keys())
    # 20-22 - 24 - 26 xception
    for ind in [0, 8, 16, 24, 32, 40]:

        indices = [ind , ind+2, ind+4, ind+6]
        patterns = ['solid', '--', '-.', ':']
        true_list = []
        predict_list = []
        headers_list = []
        for i in indices:
            true_list.append(res[headers[i]])
            predict_list.append(res[headers[i+1]])
            headers_list.append(headers[i+1])
        #print(headers[i])
        #print(headers[i+1])
        #add_curve([res[headers[i]]], [res[headers[i+1]]])
        add_curve(true_list,predict_list, headers_list,patterns,extract_title(headers[i+1]))
        #report_auc(res[headers[i]], res[headers[i+1]])
    #print(res)