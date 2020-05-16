import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib_venn as vplt
import pandas as pd
import os
import cv2
import numpy as np
from PIL import Image
import shutil
import kaggle
BORDER_CONSTANT=150/255
MARGIN_X=45
MARGIN_Y=76
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video=cv2.VideoWriter('MODEL COVID-19 CHN.mp4',fourcc,30.0,(1920,1080))

def Directories():
	if os.path.exists('Dataset'):
		shutil.rmtree('Dataset')
	os.makedirs('Dataset')
	print('Directories created successfully')
	
def Directories_2():
	if os.path.exists('Dataset'):
		shutil.rmtree('Dataset')
		os.remove('Frame.png')
		os.remove('frame_1.png')
		os.remove('frame_2.png')
		print('\nVideo has been saved successfully')
		
def Download():
	#Make sure kaggle.json is located in C:\Users\username\.kaggle
	kaggle.api.authenticate()
	kaggle.api.dataset_download_files('sudalairajkumar/novel-corona-virus-2019-dataset', path=os.path.join('Dataset'), unzip=True)
	print('\nDataset downloaded successfully')

def frame_1(deaths,recovered):
	v=vplt.venn2(subsets={'10':deaths,'01':recovered},set_labels=('A','B'))
	v.get_label_by_id('A').set_text('Deaths')
	v.get_label_by_id('A').set_color('red')
	v.get_patch_by_id('A').set_alpha(0.7)
	v.get_label_by_id('B').set_text('Recovered')
	v.get_label_by_id('B').set_color('green')
	v.get_patch_by_id('B').set_alpha(0.7)
	plt.savefig('frame_1.png',transparent=True,bbox_inches='tight',dpi=300)
	plt.close()
	
def frame_2(confirmed,meta):
	WorldMap=Basemap(llcrnrlat=17,llcrnrlon=70,urcrnrlat=55,urcrnrlon=138)
	WorldMap.drawcountries(linewidth=0.7,color=(BORDER_CONSTANT,BORDER_CONSTANT,1))
	WorldMap.drawcoastlines(linewidth=0.7,color=(BORDER_CONSTANT,BORDER_CONSTANT,1))
	for i in range(len(meta)):
		x,y=meta.iloc[i,3],meta.iloc[i,2]
		if confirmed.iloc[i] > 0:#confirmed[i] not working
			if meta.iloc[i,0]=='Hubei':
				WorldMap.plot(x,y,color=(0,0,1),marker='o',markersize=6)
				plt.text(x,y+1,str(confirmed.iloc[i]),fontsize=7,color=(0,0,1))
				plt.text(x,y-2,meta.iloc[i,0],fontsize=12,color=(0,0,1))
			else:
				WorldMap.plot(x,y,color=(0,0,1),marker='o',markersize=2,alpha=0.8)
				plt.text(x+1,y,str(confirmed.iloc[i]),fontsize=5,color=(0,0,0))
	plt.savefig('frame_2.png',transparent=True,bbox_inches='tight',facecolor=(0,1,1),dpi=300)
	#plt.show()
	plt.close('all')
	
def frame(Date):
	Frame=np.zeros((1080,1920,3), dtype=np.uint8)
	for i in range(Frame.shape[0]):
		for j in range(Frame.shape[1]):
			Frame[i][j]=(255,255,0)
	Frame_1=cv2.imread('frame_1.png')
	Frame_2=cv2.imread('frame_2.png')
	#Resizing Frame_1
	IMAGE=Image.fromarray(Frame_1, 'RGB')
	Frame_1=IMAGE.resize((994,500))#Width X Height [PIL]
	Frame_1=np.array(Frame_1)
	#Resizing Frame_2
	IMAGE=Image.fromarray(Frame_2, 'RGB')
	Frame_2=IMAGE.resize((1200,696))#Width X Height [PIL]
	Frame_2=np.array(Frame_2)
	#Image Processing
	for i in range(384+MARGIN_X,1080-MARGIN_X):
		for j in range(720+MARGIN_Y,1920-MARGIN_Y):
			Frame[i][j]=Frame_2[i-384][j-720]
	for i in range(18,518):
		for j in range(806,1800):
			count=0
			for k in Frame_1[i-18][j-806]:
				if k == 255:
					count+=1
			if count!=3:
				Frame[i][j]=Frame_1[i-18][j-806]	
	cv2.putText(Frame,'PROPORTION MODEL',(50,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2)
	cv2.putText(Frame,'HUBEI PROVINCE',(50,200),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2)
	cv2.putText(Frame,'MODEL',(50,880),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,0),2)
	cv2.putText(Frame,'COVID-19 CHN',(50,1030),cv2.FONT_HERSHEY_COMPLEX,3.3,(0,0,0),3)
	cv2.putText(Frame,'DATE: ',(40,400),cv2.FONT_HERSHEY_TRIPLEX,3.5,(0,0,0),2)
	cv2.putText(Frame,Date,(40,550),cv2.FONT_HERSHEY_TRIPLEX,3.5,(0,0,0),2)
	cv2.imwrite('Frame.png',Frame)
	for i in range(8):
		video.write(Frame)
		
def final_frame():
	Frame=cv2.imread('Frame.png')
	for i in range(90):
			video.write(Frame)
	
def process():
	Dataset_1=pd.read_csv(os.path.join('Dataset','time_series_covid_19_deaths.csv'))
	Dataset_2=pd.read_csv(os.path.join('Dataset','time_series_covid_19_recovered.csv'))
	Dataset_3=pd.read_csv(os.path.join('Dataset','time_series_covid_19_confirmed.csv'))
	Data_1=Dataset_1[Dataset_1['Province/State']=='Hubei'].iloc[:,4:]
	Data_2=Dataset_2[Dataset_2['Province/State']=='Hubei'].iloc[:,4:]
	Data_3=Dataset_3[Dataset_3['Country/Region']=='China'].iloc[:,4:]
	Data_3_meta=Dataset_3[Dataset_3['Country/Region']=='China'].iloc[:,:4]
	#print(Data_1.head())
	#print(Data_2.head())
	#print(Data_3.head())
	count=0
	last=len(Data_1.columns)
	for i in Data_1.columns:
		count+=1
		deaths=int(Data_1[i])
		recovered=int(Data_2[i])
		confirmed=Data_3[i]
		temp_list=i.split('/')
		temp=temp_list[0]
		temp_list[0]=temp_list[1]
		temp_list[1]=temp
		Date='/'.join(temp_list)
		frame_1(deaths,recovered)
		frame_2(confirmed,Data_3_meta)
		frame(Date)
		print('\nFinished processing frame: '+str(count)+'/'+str(last))
	final_frame()

Directories()
Download()
process()
video.release()
Directories_2()
