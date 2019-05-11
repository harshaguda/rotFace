import cv2
import face_recognition
import os
c = 0
def face_crop(image,c):
	face_locations = face_recognition.face_locations(image)
	face_image = []
	for face_location in face_locations:
		print(face_location)
    	# Print the location of each face in this image
		top, right, bottom, left = face_location
		face_image = image[top:bottom, left:right]
		c = c+1
		img_name = 'arunima'+'/'+str(c)+'.jpg'
		cv2.imwrite(img_name,face_image)
	if face_image is not None: 
		return face_image,c

if __name__ == "__main__":
	imgList = []
	c = 700
	k = 0 
	path_to_data = '/home/harsha/rotFace/new'
	imgList = os.listdir(path_to_data)
	length = len(imgList)
	print(imgList)
	print(length)
	if imgList is not []:
		for image in imgList:
			k = k+ 1
			percentage = k/length 
			image = path_to_data + '/' + image
			img = cv2.imread(image)
			out_img,c = face_crop(img,c)
			#print(k)
			print("[INFO]Percentage of completion:",int(percentage*100),"%")
		#c=+1

	else:
		print("[error]There are no images in the path specified (or) check the path again")
