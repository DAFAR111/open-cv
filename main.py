import cv2

capture = cv2.VideoCapture(0)
face_haar_cascade = cv2.CascadeClassifier('enter/haarcascade_frontalface_default.xml')

while True:
    _, image = capture.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



