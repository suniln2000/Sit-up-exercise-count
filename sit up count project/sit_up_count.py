import cv2
import mediapipe as md

md_drawing = md.solutions.drawing_utils
md_drawing_styles = md.solutions.drawing_styles
md_pose = md.solutions.pose

count = 0
position = None

cap = cv2.VideoCapture('/home/a/Desktop/computer vision/project/body track project/YouCut_20230215_150350300.mp4')

with md_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Unable to read video file")
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image)

        imlist = []

        if result.pose_landmarks:
            md_drawing.draw_landmarks(
                image, result.pose_landmarks, md_pose.POSE_CONNECTIONS)
            for id, im in enumerate(result.pose_landmarks.landmark):
                h, w_ = image.shape[:2]
                X, Y = int(im.x * w_), int(im.y * h)
                imlist.append([id, X, Y])

        if len(imlist) != 0:
            if (imlist[12][2] and imlist[11][2] >= imlist[26][2] and imlist[25][2]):
                position = "down"
            if (imlist[12][2] and imlist[11][2] <= imlist[26][2] and imlist[25][2]) and position == "down":
                position = "up"
                count += 1
                print(count)

        # Draw count on the frame
        cv2.putText(image, f"Count: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("sit up counter", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
