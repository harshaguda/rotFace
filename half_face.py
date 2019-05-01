"""
This code is for cropping of half face from a perfectly aligned face.
"""
import cv2
import face_recognition
image = face_recognition.load_image_file("25.jpg")

face_locations = face_recognition.face_locations(image)

for face_location in face_locations:
	print(face_location)
    # Print the location of each face in this image
	top, right, bottom, left = face_location
	face_image = image[top:bottom, left:right]

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
			print(bottom,sum)
			cropped = face_image[0:bottom, 0:sum]
cv2.imshow("cropped", cropped)
cv2.waitKey(00)
	
