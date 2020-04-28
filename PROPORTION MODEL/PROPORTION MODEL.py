import pandas as pd
import matplotlib_venn as vplt
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import kaggle
from PIL import Image
import numpy as np
#fourcc=cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video=cv2.VideoWriter('PROPORTION MODEL.mp4',fourcc,30.0,(1280,960))
def Directories():
	if os.path.exists('Dataset'):
		shutil.rmtree('Dataset')
	os.makedirs('Dataset')
	if os.path.exists('Frames'):
		shutil.rmtree('Frames')
	os.makedirs('Frames')
	print('Directories created successfully')
def Directories_2():
	if os.path.exists('Dataset'):
		shutil.rmtree('Dataset')
	if os.path.exists('Frames'):
		shutil.rmtree('Frames')
		print('\nVideo has been saved successfully')
def Download():
	#Make sure kaggle.json is located in C:\Users\username\.kaggle
	kaggle.api.authenticate()
	kaggle.api.dataset_download_files('sudalairajkumar/novel-corona-virus-2019-dataset', path=os.path.join('Dataset'), unzip=True)
	print('\nDataset downloaded successfully')
def frame(deaths,recovered,Date,last):
	v=vplt.venn2(subsets={'10':deaths,'01':recovered},set_labels=('A','B'))
	v.get_label_by_id('A').set_text('Deaths')
	v.get_label_by_id('A').set_color('red')
	v.get_patch_by_id('A').set_alpha(0.7)
	v.get_label_by_id('B').set_text('Recovered')
	v.get_label_by_id('B').set_color('green')
	v.get_patch_by_id('B').set_alpha(0.7)
	plt.savefig(os.path.join('Frames','Frame.png'),bbox_inches='tight',dpi=300)
	Frame=cv2.imread(os.path.join('Frames','Frame.png'))
	Frame=Image.fromarray(Frame, 'RGB')
	Frame=Frame.resize((1280,960))
	Frame=np.array(Frame)
	cv2.putText(Frame,'COVID-19',(380,80),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),2)
	cv2.putText(Frame,'WORLDWIDE PROPORTION MODEL',(270,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),1)
	cv2.putText(Frame,'DATE: '+str(Date),(40,240),cv2.FONT_HERSHEY_TRIPLEX,2,(0,0,0),1)
	cv2.putText(Frame,'CHRISTEEN T JOSE',(40,900),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,0),1)
	cv2.putText(Frame,'https://www.linkedin.com/in/christeen-t-jose-0351b0182/',(40,930),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.7,(0,0,0),1)
	#cv2.imwrite(os.path.join('Frames','Frame.png'),Frame)
	'''
	while cv2.waitKey(1)==-1:
		cv2.imshow('frame', Frame)
	'''
	for i in range(6):
		video.write(Frame)
	if last==1:
		for i in range(90):
			video.write(Frame)
	#plt.show()
	plt.close()
def Process():
	Dataset_1=pd.read_csv(os.path.join('Dataset','time_series_covid_19_deaths.csv'))
	Dataset_2=pd.read_csv(os.path.join('Dataset','time_series_covid_19_confirmed.csv'))
	Data_1=Dataset_1.iloc[:,4:]
	Data_2=Dataset_2.iloc[:,4:]
	#print(Data_1.columns)
	count=0
	last=len(Data_1.columns)
	for i in Data_1.columns:
		count+=1
		data_1=Data_1[i]
		data_2=Data_2[i]
		deaths=data_1.sum()
		recovered=data_2.sum()
		temp_list=i.split('/')
		temp=temp_list[0]
		temp_list[0]=temp_list[1]
		temp_list[1]=temp
		Date='/'.join(temp_list)
		if count==last:
			frame(deaths,recovered,Date,1)
		else:
			frame(deaths,recovered,Date,0)
		print('\nFinished processing frame: '+str(count)+'/'+str(last))
Directories()
Download()
Process()
video.release()
Directories_2()
