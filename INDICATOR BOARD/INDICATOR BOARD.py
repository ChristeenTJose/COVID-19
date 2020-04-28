import cv2
import pandas as pd
import os
import numpy as np
import shutil
import kaggle
SPACING_Y=50
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video=cv2.VideoWriter('INDICATOR BOARD.mp4',fourcc,30.0,(1920,1080))
def Directories():
	if os.path.exists('Dataset'):
		shutil.rmtree('Dataset')
	os.makedirs('Dataset')
	print('Directories created successfully')
def Directories_2():
	if os.path.exists('Dataset'):
		shutil.rmtree('Dataset')
		os.remove('template.png')
		print('\nVideo has been saved successfully')
def Download():
	#Make sure kaggle.json is located in C:\Users\username\.kaggle
	kaggle.api.authenticate()
	kaggle.api.dataset_download_files('sudalairajkumar/novel-corona-virus-2019-dataset', path=os.path.join('Dataset'), unzip=True)
	print('\nDataset downloaded successfully')
def generate_template(Date):
	template=np.zeros((1080,1920,3), dtype=np.uint8)
	#Purple
	for i in range(80):
		for j in range(template.shape[1]):
			template[i][j]=(255,128,213)
	#Orange
	for i in range(80,template.shape[0]):
		for j in range(720):
			template[i][j]=(102,163,255)
	#red
	for i in range(80,template.shape[0]):
		for j in range(720,1120):
			template[i][j]=(77,77,255)
	#blue
	for i in range(80,template.shape[0]):
		for j in range(1120,1520):
			template[i][j]=(255,184,77)
	#green
	for i in range(80,template.shape[0]):
		for j in range(1520,template.shape[1]):
			template[i][j]=(92, 214, 92)
	#Horizontal lines
	cv2.line(template,(0,80),(1920,80),(0,0,0),8)
	for i in range(1,19):
		if i==1:
			cv2.line(template,(0,180),(1920,180),(0,0,0),5)
		else:
			y=180+SPACING_Y*(i-1)
			cv2.line(template,(0,y),(1920,y),(0,0,0),1)
	#Vertical lines
	cv2.line(template,(720,80),(720,1080),(0,0,0),5)
	cv2.line(template,(1120,80),(1120,1080),(0,0,0),5)
	cv2.line(template,(1520,80),(1520,1080),(0,0,0),5)
	cv2.putText(template,'DATE: ',(20,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),3)
	cv2.putText(template,'COVID-19 INDICATOR BOARD',(650,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),3)
	cv2.putText(template,Date,(250,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),3)
	cv2.putText(template,'Country/Region',(20,150),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,0),2)
	cv2.putText(template,'Deaths',(770,150),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,0),2)
	cv2.putText(template,'Confirmed',(1170,150),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,0),2)
	cv2.putText(template,'Recovered',(1570,150),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,0),2)
	cv2.imwrite('template.png',template)

Directories()
Download()

Dataset={}
Data={}
KEYS=('deaths','confirmed','recovered')
for key in KEYS:
	Dataset[key]=pd.read_csv(os.path.join('Dataset','time_series_covid_19_'+key+'.csv'))
	#inplace: It is a boolean which makes the changes in data frame itself if True.
	Dataset[key].iloc[:,0].fillna(value='',inplace=True)
	#print(len(Dataset[key]))
	if key != 'recovered':
		Data[key]=Dataset[key].iloc[:,-1]
	Dataset[key]['Country']=Dataset[key].iloc[:,1]
	for i in range(len(Dataset[key])):
		text=Dataset[key].loc[i,'Country']
		if Dataset[key].iloc[i,0] != '':
			text+= ' ('+Dataset[key].iloc[i,0]+')'
		if len(text)>39:
			text=text[:36]+'...'
		Dataset[key].loc[i,'Country']=text
	#print(Dataset[key].iloc[-20:,-1])
#print(Dataset['confirmed']['Country']==Dataset['deaths']['Country'])
#print(Dataset['confirmed']['Country']==Dataset['recovered']['Country'])	Can only compare identically-labeled Series objects
Date=Dataset['confirmed'].columns[-2]##Country is last column
temp_list=Date.split('/')
temp=temp_list[0]
temp_list[0]=temp_list[1]
temp_list[1]=temp
Date='/'.join(temp_list)

generate_template(Date)

count=0
for i in range(len(Dataset['confirmed'])-17):
	Frame=cv2.imread('template.png')
	for j in range(i,i+19):
		if j==len(Dataset['confirmed']):
			break
		text=Dataset['confirmed'].iloc[j,-1]
		y=220+SPACING_Y*((count%19))
		cv2.putText(Frame,text,(20,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
		count+=1
		for k in range(len(KEYS)):
			if k != 2:
				value=str(Data[KEYS[k]].iloc[j])
				if int(value) < 0:
					value='U/A'
			else:
				Data_recovered=Dataset['recovered']['Country']==text#True or False
				flag=0
				for l in Data_recovered:
					if l==True:
						Data_recovered=Dataset['recovered'][Data_recovered]
						value=str(Data_recovered.iloc[0,-2])
						flag=1
				if flag==0:
					value='U/A'
			cv2.putText(Frame,value,(790+(400*k),y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
	for m in range(70):
		video.write(Frame)
	print('\nFinished Processing: '+str(count)+'/'+str((len(Dataset['confirmed'])-17)*19))
video.release()
Directories_2()


