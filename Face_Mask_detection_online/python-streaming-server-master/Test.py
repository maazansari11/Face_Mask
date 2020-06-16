import cv2
import base64


img = cv2.imread("images\\test.jpg")
#base64_encoded_data = base64.b64encode()
#print(base64_encoded_data)

img_str = cv2.imencode('.jpg', img)[1].tostring()
base64_encoded_data = base64.b64encode(img_str)

print(base64_encoded_data)