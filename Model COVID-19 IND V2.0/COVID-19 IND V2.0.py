import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import cv2
import numpy as np
import pandas as pd
import shutil
import kaggle
import sys
import time
from INDIA_lat_long import INDIA_lat_long
FourCC=cv2.VideoWriter_fourcc('m','p','4','v')
video=cv2.VideoWriter('MODEL COVID-19 IND V2.0.mp4',FourCC,30,(1920,1080))
def Directories():
	if os.path.exists('Dataset'):
		shutil.rmtree('Dataset')
	os.makedirs('Dataset')
	if os.path.exists('Frames'):
		shutil.rmtree('Frames')
	os.makedirs('Frames')
	print('All directories have been created')
def Download():
	#Make sure kaggle.json is located in C:\Users\username\.kaggle
	kaggle.api.authenticate()
	kaggle.api.dataset_download_files('sudalairajkumar/covid19-in-india', path=os.path.join('Dataset'), unzip=True)
	print('Download completed')
def Directories_2():
	if os.path.exists('Dataset'):
		shutil.rmtree('Dataset')
	if os.path.exists('Frames'):
		shutil.rmtree('Frames')
	print('\nDirectories deleted successfully !!!')
def Process():
	Dataset=pd.read_csv(os.path.join('Dataset','covid_19_india.csv'))
	#print(Dataset.head(5))
	Places=Dataset.iloc[:,3].unique()
	Dates=Dataset.iloc[:,1].unique()
	#print(Dates)
	for key in ('Confirmed','Cured','Deaths'):
		if key=='Confirmed':
			counter=0
			color_1=(0,0,1)
			color_2=(255,0,0)
		elif key=='Cured':
			color_1=(0,1,0)
			color_2=(0,255,0)
		elif key=='Deaths':
			color_1=(1,0,0)
			color_2=(0,0,255)
		for i in Dates:
			counter+=1
			temp_Dataset=Dataset[Dataset['Date']==i]
			WorldMap=Basemap(llcrnrlat=8,llcrnrlon=65,urcrnrlat=36,urcrnrlon=100)
			WorldMap.drawcountries(linewidth=1)
			WorldMap.bluemarble()
			for j in Places:
				total=0
				if j in temp_Dataset['State/UnionTerritory'].unique():
					row=temp_Dataset[temp_Dataset['State/UnionTerritory']==j]
					total=int(row[key])
				if j !='Unassigned' and j != 'Cases being reassigned to states':
					y,x=INDIA_lat_long[j]
					if total==0:
						total=''
					else:
						total=str(total)
						WorldMap.plot(x,y,color=color_1,marker='^',markersize=6)
						plt.text(x,y,total,fontsize=8,color=(1,1,1))
						plt.text(x,y-1,j,fontsize=6,color=(1,1,1),alpha=0.7)
			plt.savefig(os.path.join('Frames','Frame.png'),transparent=True,bbox_inches='tight',dpi=300,facecolor=(1,1,1))
			#plt.show()
			plt.close('all')
			Image=cv2.imread(os.path.join('Frames','Frame.png'))
			Image=cv2.resize(Image,(1332,1080))
			template=np.full(shape=(1080,1920,3),fill_value=(255,255,255),dtype=np.uint8)
			for k in range(1080):
				for l in range(1332):
					template[k][l+588]=Image[k][l]
			total=temp_Dataset[key].sum()
			cv2.putText(template,'DATE: '+i,(20,90),cv2.FONT_HERSHEY_COMPLEX,2.2,(0,0,0),2)
			cv2.putText(template,'Number of '+key+': ',(20,200),cv2.FONT_HERSHEY_COMPLEX,1.5,color_2,2)
			cv2.putText(template,str(total),(150,300),cv2.FONT_HERSHEY_TRIPLEX,2.3,color_2,2)
			IMAGE2=cv2.imread('Model.png')
			for k in range(400,400+IMAGE2.shape[0]):
				for l in range(IMAGE2.shape[1]):
					template[k][l]=IMAGE2[k-400][l]
			#cv2.imwrite(os.path.join('Frames','Frame.png'),template)
			if total != 0:
				for m in range(10):
					video.write(template)
			print('Finished processing Image: '+str(counter))
		for i in range(75):
			video.write(template)
Directories()
Download()
Process()
video.release()
Directories_2()


