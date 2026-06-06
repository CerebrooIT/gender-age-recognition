import cv2 #OpenCv library

#Face detector (OpenCV)
path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(path)

# models for gender and age
#readNet(knowledge, recept): first .caffemodel then .prototxt

gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")

#Average values for network
AVERAGE = (78.4263377603, 87.7689143744, 114.895847746)

#List of possible answers - network return number (index), we convert in text
GENDERS = ["Male", "Female"] # 0=male, 1=female
AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-26)","(27-32)", "(38-43)", "(48-53)", "(60-100)"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_detector.detectMultiScale(gray, 1.1, 5,minSize=(60,60))

    for (x,y,w,h) in face: #for every face detected
        face = frame[y:y+h, x:x+w] #crop face

        #prepare face for network: scale on 227x277 and balance colors
        blob = cv2.dnn.blobFromImage(face,1.0,(227,227),AVERAGE, swapRB=False)

        gender_net.setInput(blob) #Enter face-to-gender network
        res_gender = gender_net.forward() #Get answer from network
        gender = GENDERS[res_gender[0].argmax()]

        age_net.setInput(blob)
        res_age = age_net.forward()
        age = AGE_LIST[res_age[0].argmax()]

        text = gender + " " + age  #Example: "Male (25-32)"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) #Green rectangle

        # Text above rectangle
        cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Face detection - press Q for exit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
