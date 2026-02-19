import cv2
import mediapipe as mp

mpose = mp.solutions.pose
pose = mpose.Pose()
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hasil = pose.process(imgRGB)
    if hasil.pose_landmarks:
        print ("terdeteksi")
    else:
        print("Tidak Terdeteksi")

    cv2.imshow("webcam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()