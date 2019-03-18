# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png

# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2,sys,re

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []
le=[]
ri=[]
to=[]
bo=[]
un_count=0

# loop over the facial embeddings
for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
	matches = face_recognition.compare_faces(data["encodings"],
		encoding)
	name = "Unknown"

	# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)


	if name == 'Unknown':
		un_count+=1

	names.append(name)


mosaic_rate = 30

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	if name == 'Unknown':
		le.append(left)
		ri.append(right)
		to.append(top)
		bo.append(bottom)

	# 	face_img = image[top:bottom, left:right]
	# 	cv2.imshow("im", face_img)

	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)



	# if name == 'chang' :
	# 	print('chang')
	# 	face_img = image[top:bottom, left:right]
	# 	cv2.imshow("im", face_img)
	#
	#
	# 	face_img = cv2.resize(face_img, ((bottom-top)//mosaic_rate, (right-left)//mosaic_rate))
	#
	# 	face_img = cv2.resize(face_img, (bottom-top, right-left), interpolation=cv2.INTER_AREA)
	# 	image[top:bottom, left:right] = face_img



# print(le)
# print(ri)
# print(to)
# print(bo)

print(un_count)
i=0
while i < un_count:
	print(i)
	face_img = image[to[i]:bo[i], le[i]:ri[i]]
	cv2.imshow("im", face_img)
	face_img = cv2.resize(face_img, ((bo[i]-to[i])//mosaic_rate, (ri[i]-le[i])//mosaic_rate))
	face_img = cv2.resize(face_img, ((ri[i]-le[i]),bo[i]-to[i]), interpolation=cv2.INTER_AREA)
	image[to[i]:bo[i], le[i]:ri[i]] = face_img
	i+=1



cv2.imshow("Image", image)
cv2.waitKey(0)
