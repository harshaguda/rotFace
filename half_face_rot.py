"""
This code is for cropping of half face from a rotated face.
"""
import cv2
import face_recognition
import math
import os

def slope(box):
	print (box)
	if (box[1][0]-box[0][0]) == 0:
		f_slope = 999		
	else:
		f_slope = (-box[1][1]+box[0][1])/(-box[1][0]+box[0][0])
	#print(math.atan(f_slope))
	return math.atan(f_slope)
def rotate(img, angle, cor):
	angle = angle*57.2958
	rows,cols,r = img.shape
	if angle < 0:
		r_angle = 90 + angle
	else:
		r_angle = 90 - angle 
		r_angle = -r_angle
	M = cv2.getRotationMatrix2D(cor,int(r_angle),1)
	dst = cv2.warpAffine(img,M,(cols,rows))
	return dst
	#cv2.imshow("rot", dst)




def face_crop(image):
	face_locations = face_recognition.face_locations(image)
	for face_location in face_locations:
		print(face_location)
    # Print the location of each face in this image
		top, right, bottom, left = face_location
		face_image = image[top:bottom, left:right]
		return face_image, bottom

def nose(face_image):		
	face_landmarks_list = face_recognition.face_landmarks(face_image)
	angle = 90/57.2958
	#face_landmarks[facial_feature] = []
	print(face_landmarks_list)
	for face_landmarks in face_landmarks_list:
		sum = 0
	# Print the location of each facial feature in this image
		for facial_feature in face_landmarks.keys():
			if facial_feature == "nose_bridge":
				print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
				for i in range(4):
					sum = sum + face_landmarks[facial_feature][i][0]
				#print(face_landmarks[facial_feature][i][0])
				print(sum/4)
				sum = int(sum/4)
				(top,bottom,color) = face_image.shape
				print(bottom,sum)
				cropped_l = face_image[0:bottom, 0:sum]
				cropped_r = face_image[0:bottom,sum:top]
				angle = slope(face_landmarks[facial_feature])
				print("Land marks:",face_landmarks[facial_feature])
		print(face_landmarks)
	if len(face_landmarks_list) == 0:
		flag = False  
		return flag, 90, 0,0,0
	else:
		flag = True
		return flag, angle, face_landmarks[facial_feature], cropped_l,cropped_r

	

if __name__ == '__main__':
	angle = 90
	k = 0
	#image = cv2.imread("26.jpg")
	imgList = os.listdir("/home/harsha/rotFace/croppedDataset/vara/")
	#image = face_recognition.load_image_file("/home/harsha/rotFace/croppedDataset/amoolya/28.jpg")
	for imageN in imgList:
		k = k+2 
		image = cv2.imread("/home/harsha/rotFace/croppedDataset/vara/"+imageN)
		print(imageN)
		cv2.imshow("image", image)
		#cv2.waitKey(0)
		#print(image)
		if image is None:
			continue
		#face_image, bottom = face_crop(image)
		#cv2.imshow("face_image1",face_image)
		#cv2.waitKey(0)
		flag, angle, nose_cor, cropped_l,cropped_r = nose(image)
		print("angle:",angle*57.2958)
		if flag:
			dst = rotate(image,angle,nose_cor[2])
		#r_face_image, bottom = face_crop(dst)
			#cv2.imshow("face_image2",dst)
			flag, angle, nose_cor,cropped_l,cropped_r = nose(dst)
			cv2.imwrite(str(k)+".jpg",cropped_l)
			cv2.imwrite(str(k+1)+".jpg",cropped_r)
			#cv2.imshow("cropped_l", cropped_l)
			#cv2.imshow("cropped_r",cropped_r)
		#angle, cropped = nose(r_face_image,bottom)
		#print("angle:",angle*57.2958)
		cv2.waitKey(1)
		#cv2.imshow("cropped", cropped)
	#new_img = rotate(image,angle)
	cv2.waitKey(1)
	
