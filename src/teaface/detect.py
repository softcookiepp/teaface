import cv2
import sys
import numpy as np

from sklearn.cluster import DBSCAN

import os

#path_to_image = sys.argv[1]

from deepface import DeepFace

def get_face_embeddings(imgs):
	file_refs = []
	out_list = []
	for img in imgs:
		for f in DeepFace.represent(img_path = img, enforce_detection = False, model_name = "ArcFace"):
			# account for multiple faces in single image
			file_refs.append(img_path)
			out_list.append(f)
	file_refs = np.array(file_refs)
	arr_list = []
	for out in out_list:
		face_confidence = out["face_confidence"]
		embedding = np.array(out["embedding"]).reshape(1, -1)
		arr_list.append(embedding)
	out = np.concatenate(arr_list, axis = 0)
	input(out.shape)
	return file_refs, out

def extract_faces(path_to_image: str):
	# Load the Haar Cascade model for face detection
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	
	# Read your image
	img = cv2.imread(path_to_image)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	
	# Draw rectangles around faces
	for (x, y, w, h) in faces:
		face_img = img[y:y + h, x:x + w]
		#print(i)
		
		yield face_img
		#i += 1
		#cv2.imshow("croopted face", face_img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		#out_path = "/dev/shm/tmp.jpg"
		#cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

	#cv2.imshow('Detected Faces', img)
	#cv2.waitKey(0)

def write_image(face_img):
	cv2.imwrite(f'face_{x}_{y}.jpg', face_img)

def _test_face_embedding():
	img = cv2.imread(sys.argv[1])
	out = get_face_embedding(img)
	

def batch_extract_embeddings(img_paths: list, batch_size = 64, model = "ArcFace"):
	embedding_batches = []
	file_ref_batches = []
	for i in range(len(img_paths) // batch_size + 1):
		end = (i+1)*batch_size
		if end > len(img_paths):
			end = len(img_paths)
		img_paths_batched = img_paths[i*batch_size:end]
		
		file_refs, embeddings = get_face_embeddings(img_paths_batched)
		embedding_batches.append(embeddings)
		file_ref_batches.append(file_refs)
	file_refs = np.concatenate(file_ref_batches, axis = 0)
	embeddings = np.concatenate(embedding_batches, axis = 0)
	return file_refs, embeddings

folder = sys.argv[1]
files = os.listdir(folder)
os.chdir(folder)
batch_extract_embeddings(files)
