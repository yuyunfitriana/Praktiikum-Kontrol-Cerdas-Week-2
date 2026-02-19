import cv2
import mediapipe as mp

mpose = mp.solutions.pose
mdraw = mp.solutions.drawing_utils

pose = mpose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()

    if not success:
        print("Kamera tidak terbaca")
        break

    img = cv2.flip(img, 1)

    img.flags.writeable = False
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hasil = pose.process(imgRGB)

    img.flags.writeable = True

    if hasil.pose_landmarks:
        mdraw.draw_landmarks(img, hasil.pose_landmarks, mpose.POSE_CONNECTIONS)

        lm = hasil.pose_landmarks.landmark
        h, w, c = img.shape

        ls = (int(lm[11].x * w), int(lm[11].y * h))  # left shoulder
        rs = (int(lm[12].x * w), int(lm[12].y * h))  # right shoulder
        lw = (int(lm[15].x * w), int(lm[15].y * h))  # left wrist
        rw = (int(lm[16].x * w), int(lm[16].y * h))  # right wrist

        cv2.circle(img, ls, 15, (128, 128, 128), -1)
        cv2.circle(img, rs, 15, (128, 128, 128), -1)
        cv2.circle(img, lw, 15, (128, 128, 128), -1)
        cv2.circle(img, rw, 15, (128, 128, 128), -1)

        if lw[1] < ls[1]:
            cv2.putText(img, "Tangan Kiri Naik",
                        (30, 60),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        1.2, (0, 255, 0), 2)

        if rw[1] < rs[1]:
            cv2.putText(img, "Tangan Kanan Naik",
                        (30, 110),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        1.2, (0, 255, 255), 2)

    cv2.imshow("Pose Detection", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()
