from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow,QMessageBox,QProgressBar
from PyQt5.QtCore import QThread,pyqtSignal
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from mpl_toolkits.basemap import Basemap
import pandas as pd
import shutil
import kaggle
import sys
import time
def Directories():
	if os.path.exists('Dataset'):
		shutil.rmtree('Dataset')
	os.makedirs('Dataset')
	if os.path.exists('Images'):
		shutil.rmtree('Images')
	os.makedirs('Images')
	os.makedirs(os.path.join('Images','confirmed'))
	os.makedirs(os.path.join('Images','deaths'))
	os.makedirs(os.path.join('Images','recovered'))
	#print('All directories have been created')
def Download():
	#Make sure kaggle.json is located in C:\Users\username\.kaggle
	kaggle.api.authenticate()
	kaggle.api.dataset_download_files('sudalairajkumar/novel-corona-virus-2019-dataset', path=os.path.join('Dataset'), unzip=True)
	#print('Download completed')
def GenerateImages(key,counter,Coordinates,data,total):
	if key=='confirmed':
		color=(0,0,1)
	elif key=='recovered':
		color=(0,1,0)
	elif key=='deaths':
		color=(1,0,0)
	WorldMap=Basemap()
	WorldMap.drawcoastlines()
	WorldMap.fillcontinents(color='black')
	WorldMap.drawcountries(color='white')
	for i in range(len(Coordinates)):
		x,y=Coordinates.iloc[i,1],Coordinates.iloc[i,0]
		if data[i] > 0:
			WorldMap.plot(x,y,color=color,marker='o',markersize=1)
	plt.savefig(os.path.join('Images',key,'Image'+str(counter)+'.png'),transparent=True,bbox_inches='tight',dpi=300)
	#plt.show()
	plt.close()
def ImageProcessing(key,counter,Date,total):
	if key=='confirmed':
		color=(255,0,0)
	elif key=='recovered':
		color=(0,255,0)
	elif key=='deaths':
		color=(0,0,255)
	IMAGE=cv2.imread(os.path.join('Images',key,'Image'+str(counter)+'.png'))
	IMAGE=Image.fromarray(IMAGE, 'RGB')
	IMAGE=IMAGE.resize((930,510))#Width X Height [PIL]
	IMAGE=np.array(IMAGE)
	temp=np.zeros((IMAGE.shape[0],IMAGE.shape[1]+350,3),dtype=np.uint8)
	for i in range(IMAGE.shape[0]):
		for j in range(IMAGE.shape[1]):
			temp[i][j+350]=IMAGE[i][j]
	IMAGE=temp
	for i in range(IMAGE.shape[0]):
		for j in range(350):
			IMAGE[i][j]=(255,255,255)
	cv2.putText(IMAGE,'DATE: '+str(Date),(20,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5,(0,0,0),1)
	cv2.putText(IMAGE,'Number of '+key+': ',(20,180),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
	cv2.putText(IMAGE,str(total),(50,250),cv2.FONT_HERSHEY_COMPLEX,1,color,1)
	IMAGE2=cv2.imread('Model.png')
	for i in range(384,384+IMAGE2.shape[0]):
		for j in range(20,20+IMAGE2.shape[1]):
			IMAGE[i][j]=IMAGE2[i-384][j-20]
	cv2.imwrite(os.path.join('Images',key,'Image'+str(counter)+'.png'),IMAGE)
	'''
	while cv2.waitKey(1)==-1:
		cv2.imshow('Model Corona',IMAGE)
	'''
class Thread_Updating(QThread):
	Signal=pyqtSignal(int,int)
	def run(self):
		IMAGE=cv2.imread('template.png')
		cv2.putText(IMAGE,'Updating',(300,200),cv2.FONT_HERSHEY_COMPLEX,4,(0,0,0),3)
		cv2.imwrite(os.path.join('Images','Updating.png'),IMAGE)
		self.Signal.emit(-1,0)
		Directories()
		Download()
		for key in ('confirmed','recovered','deaths'):
			Dataset=pd.read_csv(os.path.join('Dataset','time_series_covid_19_'+key+'.csv'))
			Dataset=Dataset.iloc[:,2:]
			Coordinates=Dataset.iloc[:,:2]
			Data=Dataset.iloc[:,2:]
			#print(Coordinates.head(10))
			#print(Data.head(10))
			#print(Data.columns)
			counter=0
			last=len(Data.columns)
			if key=='confirmed':
				start=0
			elif key=='recovered':
				start=last
			elif key=='deaths':
				start=last*2
			for i in Data.columns:
				counter+=1
				data=Data[i]
				#print(data)
				total=data.sum()
				#print(total)
				temp_list=i.split('/')
				#print(temp_list)
				temp=temp_list[0]
				temp_list[0]=temp_list[1]
				temp_list[1]=temp
				Date='/'.join(temp_list)
				#print(Date)
				GenerateImages(key,counter,Coordinates,data,total)
				ImageProcessing(key,counter,Date,total)
				IMAGE=cv2.imread('template.png')
				cv2.putText(IMAGE,'Updating',(300,200),cv2.FONT_HERSHEY_COMPLEX,4,(0,0,0),3)
				cv2.putText(IMAGE,'Finished processing Image: '+str(counter+start)+'/'+str(last*3),(400,300),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),1)
				cv2.imwrite(os.path.join('Images','Updating.png'),IMAGE)
				'''
					while cv2.waitKey(1)==-1:
					cv2.imshow('Model Corona',IMAGE)
				'''
				self.Signal.emit(counter+start,last*3)
class Thread_Model(QThread):
	Signal=pyqtSignal(str,int,int)
	def __init__(self,key):
		super(Thread_Model,self).__init__()
		self.key=key
	def run(self):
		val2=len(os.listdir(os.path.join('Images',self.key)))
		for i in range(1,val2+1):
			val1=i
			time.sleep(0.3)
			self.Signal.emit(self.key,val1,val2)
class MyWindow(QMainWindow):
	def __init__(self):
		super(MyWindow,self).__init__()
		self.setGeometry(300,100,1402,700)
		self.setWindowTitle("Model Corona")
		self.build()
	def Disable(self):
		self.Confirmed.setEnabled(False)
		self.Deaths.setEnabled(False)
		self.Recovered.setEnabled(False)
		self.Update.setEnabled(False)
	def Enable(self):
		self.Confirmed.setEnabled(True)
		self.Deaths.setEnabled(True)
		self.Recovered.setEnabled(True)
		self.Update.setEnabled(True)
	def startModel(self,key):
		self.Disable()
		self.thread=Thread_Model(key)
		self.thread.Signal.connect(self.setModel)
		self.thread.start()
	def setModel(self,key,val1,val2):
		self.Model(key,val1,val2)
	def Model(self,key,val1,val2):
		self.Image.setPixmap(QtGui.QPixmap(os.path.join('Images',key,'Image'+str(val1)+'.png')))
		if val1==val2:
			self.Enable()
	def startUpdating(self):
		self.Disable()
		self.thread=Thread_Updating()
		self.thread.Signal.connect(self.setUpdating)
		self.thread.start()
	def setUpdating(self, val1,val2):
		self.Updating(val1,val2)	
	def Updating(self,val1,val2):
		self.Image.setPixmap(QtGui.QPixmap(os.path.join('Images','Updating.png')))
		#print('Finished processing Image: '+str(val1)+'/'+str(val2))
		if val1==val2:
			self.Enable()
	def build(self):
		#Fonts
		font1=QtGui.QFont()
		font1.setFamily("Verdana Pro")
		font1.setPointSize(16)
		font2=QtGui.QFont()
		font2.setPointSize(10)
		font2.setFamily("Times New Roman")
		#Text-colors
		brush1=QtGui.QBrush(QtGui.QColor(0, 0, 186))
		brush1.setStyle(QtCore.Qt.SolidPattern)
		brush2=QtGui.QBrush(QtGui.QColor(186, 0, 0))
		brush2.setStyle(QtCore.Qt.SolidPattern)
		brush3=QtGui.QBrush(QtGui.QColor(0, 186, 0))
		brush3.setStyle(QtCore.Qt.SolidPattern)
		#Image
		self.Image=QtWidgets.QLabel(self)
		self.Image.setGeometry(QtCore.QRect(61, 20, 1280, 510))
		self.Image.setText('')
		self.Image.setPixmap(QtGui.QPixmap('template.png'))
		self.Image.setScaledContents(False)
		#Confirmed
		self.Confirmed=QtWidgets.QPushButton(self)
		self.Confirmed.setGeometry(QtCore.QRect(590, 590, 251, 71))
		palette = QtGui.QPalette()
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush1)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush1)
		#palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
		self.Confirmed.setPalette(palette)
		self.Confirmed.setFont(font1)
		self.Confirmed.setText('Confirmed')
		self.Confirmed.clicked.connect(lambda: self.startModel('confirmed'))
		#Deaths
		self.Deaths=QtWidgets.QPushButton(self)
		self.Deaths.setGeometry(QtCore.QRect(240, 590, 251, 71))
		palette = QtGui.QPalette()
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush2)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush2)
		self.Deaths.setPalette(palette)
		self.Deaths.setFont(font1)
		self.Deaths.setText("Deaths")
		self.Deaths.clicked.connect(lambda: self.startModel('deaths'))
        #Recovered
		self.Recovered=QtWidgets.QPushButton(self)
		self.Recovered.setGeometry(QtCore.QRect(940, 590, 251, 71))
		palette = QtGui.QPalette()
		palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush3)
		palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush3)
		self.Recovered.setPalette(palette)
		self.Recovered.setFont(font1)
		self.Recovered.setText("Recovered")
		self.Recovered.clicked.connect(lambda: self.startModel('recovered'))
        #Update
		self.Update=QtWidgets.QPushButton(self)
		self.Update.setGeometry(QtCore.QRect(1280, 633, 93, 28))
		self.Update.setFont(font2)
		self.Update.setText("Update")
		self.Update.clicked.connect(self.startUpdating)
app=QApplication(sys.argv)
Window=MyWindow()
Window.show()
sys.exit(app.exec_())
