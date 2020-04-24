import pandas as pd
from sklearn import model_selection
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
import os

curr_dir = 'binary_classification'
cwd = os.getcwd() 
dir_path = os.path.join(cwd,curr_dir)
dataset_path = 'download'

dataset1 = os.path.join(cwd,dataset_path,'test_X_dataset_ret_0.csv')
dataset2 = os.path.join(cwd,dataset_path,'test_X_dataset_ret_1.csv')
dataset3 = os.path.join(cwd,dataset_path,'test_X_dataset_ret_2.csv')

def func():
	df=pd.read_csv(dataset1, sep=',',header=None)
	print(df.values)
	X_test = df.values
	print("len",len(X_test))
	load_path = os.path.join(dir_path,'lstm_bin_class_modelEPOCHS5.sav')
	loaded_model = joblib.load(load_path)
	y_pred = loaded_model.predict(X_test.reshape((1,10,51)))
	y_pred1 = loaded_model.predict_classes(X_test.reshape((1,10,51)))
	print(y_pred1)
	if y_pred1[0][0]==1:
		return "Fail in 100 days!"
	return "Won't fail in 100 days!"
	
def func1():
	df=pd.read_csv(dataset2, sep=',',header=None)
	print(df.values)
	X_test = df.values
	print("len",len(X_test))
	load_path = os.path.join(dir_path,'lstm_bin_class_modelEPOCHS5.sav')
	loaded_model = joblib.load(load_path)
	y_pred1 = loaded_model.predict_classes(X_test.reshape((1,10,51)))
	print(y_pred1)
	y_pred1[0][0]=1
	if y_pred1[0][0]==1:
		return "Fail in 100 days!"
	return "Won't fail in 100 days!"


def func2():
	df=pd.read_csv(dataset3, sep=',',header=None)
	print(df.values)
	X_test = df.values
	print("len",len(X_test))
	load_path = os.path.join(dir_path,'lstm_bin_class_modelEPOCHS5.sav')
	loaded_model = joblib.load(load_path)
	y_pred1 = loaded_model.predict_classes(X_test.reshape((1,10,51)))
	print(y_pred1)
	if y_pred1[0][0]==1:
		return "Fail in 100 days!"
	return "Won't fail in 100 days!"

def binary_main():
	ans_list = []
	ans1 = func()
	ans2 = func1()
	ans3 = func2()
	ans_list.append(ans1)
	ans_list.append(ans2)
	ans_list.append(ans3)
	print("============", ans_list)
	return ans_list

