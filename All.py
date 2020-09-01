import numpy as np
import cv2
import time
import os
from PIL import Image
import pickle
import pandas as pd
import datetime
#from firebase.firebase import FirebaseAuthentication
#from firebase import firebase
import firebase_admin
from firebase_admin import db
from firebase_admin import credentials


cred = credentials.Certificate('E:/python_save/Opencv/JAV-stars-Face-Recognition-master/firebase-sdk.json')
firebase_admin.initialize_app(cred,{
	'databaseURL' : 'https://testinpython.firebaseio.com/'
	})


def main():
	print('Chon phuong thuc bat dau : ')
	print('1 : Bat dau tu dau')
	print('2 : Face detect')
	a = int(input('Ban chon : '))
	if a == 1:
		makeData()
		facesTrain()
	elif a == 2 :
		print(a)
		Face()

	else :
		print('Input Error')


def makeData():
	global nameInput
	nameInput = (input('Nhap ten : '))
	global idInput
	idInput = input('Nhap ID : ')
	label = nameInput
	cap = cv2.VideoCapture(0)

	# Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
	i=0
	while(True):
		# Capture frame-by-frame
		#
		i+=1
		ret, frame = cap.read()
		if not ret:
			continue
		frame = cv2.resize(frame, dsize=None,fx=0.3,fy=0.3)

		# Hiển thị
		cv2.imshow('frame',frame)

		# Lưu dữ liệu
		if i>=60:
			print("Số ảnh capture = ",i-60)
			# Tạo thư mục nếu chưa có
			if not os.path.exists('images/' + str(label)):
				os.mkdir('images/' + str(label))

			cv2.imwrite('images/' + str(label) + "/" + str(i) + ".png",frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def facesTrain():
	#Define directory
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	image_dir = os.path.join(BASE_DIR, "images")

	#Face cascade
	face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

	#Face recognizer
	recognizer = cv2.face.LBPHFaceRecognizer_create()

	#Define data and label
	y_labels = []
	x_train = []
	current_id = 0
	label_ids = {}


	for root, dirs, files, in os.walk(image_dir):
		for file in files:
			if file.endswith("png") or file.endswith("jpg"):
				path = os.path.join(root, file)
				#label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()
				label = os.path.basename(os.path.dirname(path)).replace(" ", "_") 
				# print("Path: ",path)
				# print("Label: ",label)

				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1
					#label_ids[label] = idInput
				global id_ 
				id_ = label_ids[label]
				#print(label_ids)
				#Add data and label
				#labels.append(label) #some number
				#images.append(path) #verify this image, and turn it into Numpy array, GRAY
				
				pil_image = Image.open(path).convert("L")
				size = (550, 550)
				final_image = pil_image.resize(size, Image.ANTIALIAS)

				image_array = np.array(final_image, "uint8")
				#print(image_array)

				faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

				for (x, y, w, h) in faces:
					roi = image_array[y:y+h, x:x+w]
					x_train.append(roi)
					y_labels.append(id_)

	# print(x_train)
	# print(y_labels)

	with open("labels.pickle", "wb") as f:
		pickle.dump(label_ids, f)


	recognizer.train(x_train, np.array(y_labels))
	recognizer.save("trainer.yml")
	# global QLDL
	# QLDL = {
	# 	nameInput : {
	# 		'Name' : nameInput,
	# 		'ID' : _id,
	# 		'Images' : image_dir
	# 	}
	# }
	Data_ToServer()

def Data_ToServer():
	dataGo={
		'Name' : nameInput,
		'ID'	: idInput,
		'Status' : ''
	}
	ref = db.reference('/Customers/'+nameInput)
	ref.set(dataGo)


def Face():
	#Define Haar cascade
	face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
	eyes_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

	#Face recognizer
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("trainer.yml")

	labels = {"person_name": 1}
	with open("labels.pickle", "rb") as f:
		og_labels = pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}


	# video_src = 'BAR.mp4'
	# cap = cv2.VideoCapture(video_src)

	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	col_names = ['Name','Date','Time']
	attendance = pd.DataFrame(columns = col_names)

	while True:
		ret, frame = cap.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		#Checking attendance
		
		for (x, y, w, h) in faces:
			#print(x,y,w,h)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]

			#Recognizer deep learned model
			id_, conf = recognizer.predict(roi_gray)

			# if conf >= 0.9: #and conf <= 85:
			# 	print(id_)
			# 	print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			#cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
			cv2.putText(frame, str(id_)+": "+name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

			#img_item = 'my-image.png'

			#cv2.imwrite(img_item, roi_gray)

			color = (255,0,0)
			stroke = 2
			end_cord_x = x + w
			end_cord_y = y + h
			cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

			#Detect eyes
			eyes = eyes_cascade.detectMultiScale(roi_gray)
			for (ex, ey, ew, eh) in eyes:
				cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

			

			#Time
			time_s = time.time()
			date = str(datetime.datetime.fromtimestamp(time_s).strftime('%Y-%m-%d'))
			timeStamp = datetime.datetime.fromtimestamp(time_s).strftime('%H:%M:%S')

			#Write csv file
			attendance.loc[len(attendance)] = [name,date,timeStamp]
			#data = pd.DataFrame({"Name": name, "Date": date, "Time": timeStamp}, index=[0])


			#attendance = attendance.append(data, ignore_index=True, sort=False)
			
		
		print(attendance.head())

		cv2.imshow('video', frame)

		if cv2.waitKey(20) & 0xFF == ord("q"):
			break
	attendance=attendance.drop_duplicates(keep='first',subset=['Name'])
	attendance.to_csv("Checking_attendance.csv")

	#ref2 = db.reference('/')
	#ref2.child('Customers').child(name).child('Status').update(attendance)
	ref2 = db.reference('/Customers/'+name+'/'+'Status')
	print(attendance.loc[attendance['Name'] == name, 'Date'].iloc[0]	)
			
	ref2.update(dict(attendance[attendance["Name"]==name]["Time"]))
	

	cv2.destroyAllWindows()
main()