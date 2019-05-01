"""
This code is for cropping of half face from a rotated face.
"""
import cv2
import face_recognition
import math

def slope(box):
	print (box)
	if (box[1][0]-box[0][0]) == 0:
		f_slope = 999		
	else:
		f_slope = (-box[1][1]+box[0][1])/(-box[1][0]+box[0][0])
	#print(math.atan(f_slope))
	return math.atan(f_slope)
def rotate(img, angle):
	angle = angle*57.2958
	rows,cols,r = img.shape
	if angle < 0:
		r_angle = 90 + angle
	else:
		r_angle = 90 - angle 
	M = cv2.getRotationMatrix2D((cols/2,rows/2),int(r_angle),1)
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

def nose(face_image, bottom):		
	face_landmarks_list = face_recognition.face_landmarks(face_image)

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
				#print(bottom,sum)
				cropped = face_image[0:bottom, 0:sum]
				angle = slope(face_landmarks[facial_feature])
	return angle, cropped
if __name__ == '__main__':
	
	#image = cv2.imread("26.jpg")
	image = face_recognition.load_image_file("27.jpg")
	face_image, bottom = face_crop(image)
	cv2.imshow("face_image",face_image)
	cv2.waitKey(0)
	angle, cropped = nose(image,bottom)
	dst = rotate(image,angle)
	face_image = face_crop(image)
	angle, cropped = nose(face_image,bottom)
	print("angle:",angle)

	cv2.imshow("cropped", cropped)
	#new_img = rotate(image,angle)
	cv2.waitKey(0)
	
