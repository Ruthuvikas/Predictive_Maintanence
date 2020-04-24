import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
import os

curr_dir = 'cnn'
cwd = os.getcwd() 
dir_path = os.path.join(cwd,curr_dir)
dataset_path = 'download'

dataset1 = os.path.join(cwd,dataset_path,'test_X_dataset_ret_0.csv')
dataset2 = os.path.join(cwd,dataset_path,'test_X_dataset_ret_1.csv')
dataset3 = os.path.join(cwd,dataset_path,'test_X_dataset_ret_2.csv')

def func():
	#read_path = os.path.join(dir_path,'test_X_dataset_ret_0.csv')
	df=pd.read_csv(dataset1, sep=',',header=None)
	print(df.values)
	X_test = df.values
	load_path = os.path.join(dir_path,'lstm_reg3_modelEPOCHS10.sav')
	loaded_model = joblib.load(load_path)
	y_pred = loaded_model.predict(X_test.reshape((1,10,51)))
	y_pred[0][0] = y_pred[0][0] * 15
	days = y_pred[0][0] /(60*24)
	rem  = days - int(days)
	hrs = rem*24
	rem1 = hrs - int(hrs)
	mins = rem1 * 60
	output = str(int(days)) + "	days	" + str(int(hrs)) + "	hours	" + str(int(mins)) + "	minutes"
	return output


def func1():
	#read_path = os.path.join(dir_path,'test_X_dataset_ret_1.csv')
	df=pd.read_csv(dataset2, sep=',',header=None)
	print(df.values)
	X_test = df.values
	load_path = os.path.join(dir_path,'lstm_reg3_modelEPOCHS10.sav')
	loaded_model = joblib.load(load_path)
	y_pred = loaded_model.predict(X_test.reshape((1,10,51)))
	y_pred[0][0] = y_pred[0][0] * 15
	days = y_pred[0][0] /(60*24)
	rem  = days - int(days)
	hrs = rem*24
	rem1 = hrs - int(hrs)
	mins = rem1 * 60
	output = str(int(days)) + "	days	" + str(int(hrs)) + "	hours	" + str(int(mins)) + "	minutes"
	print("==========", output)
	return output

def func2():
	#read_path = os.path.join(dir_path,'test_X_dataset_ret_2.csv')
	df=pd.read_csv(dataset3, sep=',',header=None)
	print(df.values)
	X_test = df.values
	load_path = os.path.join(dir_path,'lstm_reg3_modelEPOCHS10.sav')
	loaded_model = joblib.load(load_path)
	y_pred = loaded_model.predict(X_test.reshape((1,10,51)))
	y_pred[0][0] = y_pred[0][0] * 150
	days = y_pred[0][0] /(60*24)
	rem  = days - int(days)
	hrs = rem*24
	rem1 = hrs - int(hrs)
	mins = rem1 * 60
	output = str(int(days)) + "	days	" + str(int(hrs)) + "	hours	" + str(int(mins)) + "	minutes"
	return output

def cnn_main():
	ans_list = []
	ans1 = func()
	ans2 = func1()
	ans3 = func2()
	ans_list.append(ans1)
	ans_list.append(ans2)
	ans_list.append(ans3)
	print("============", ans_list)
	return ans_list


