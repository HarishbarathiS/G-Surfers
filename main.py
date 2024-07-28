import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Get the screen width and height
screen_width, screen_height = pyautogui.size()

# Calculate half the screen width
half_screen_width = int(screen_width / 2)

# Initialize webcam
cap = cv2.VideoCapture(0)


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=10):
    # Calculate the number of segments
    dist = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
    num_segments = int(dist / gap)
    
    # Draw the line segments
    for i in range(num_segments):
        start = (int(pt1[0] + (pt2[0] - pt1[0]) * (i / num_segments)), int(pt1[1] + (pt2[1] - pt1[1]) * (i / num_segments)))
        end = (int(pt1[0] + (pt2[0] - pt1[0]) * ((i + 0.5) / num_segments)), int(pt1[1] + (pt2[1] - pt1[1]) * ((i + 0.5) / num_segments)))
        cv2.line(img, start, end, color, thickness)

right = 0
left = 0
up = 0
down = 0

prev_right = 0
prev_left = 0
prev_up = 0
prev_down = 0

flag_horizontal = 0
flag_vertical = 0

right_line = 0
left_line = 0
top_line = 0
bottom_line = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect pose
    results = pose.process(rgb_frame)

    # Draw landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the landmarks for the left and right shoulders
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        
        # get height and width of the frame
        h, w, _ = frame.shape

        half_frame_width = int(w / 2)
        half_frame_height = int(h / 2)
        quarter_frame_width = int(half_frame_width/2)

        if left_shoulder and right_shoulder:
            l_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            r_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            # Draw circles at the shoulder landmarks
            cv2.circle(frame, l_shoulder_coords, 10, (0, 255, 0), -1)
            cv2.circle(frame, r_shoulder_coords, 10, (0, 0, 255), -1)
        if left_shoulder and right_shoulder :
            if left_line and r_shoulder_coords[0] < left_line:
                right = 1
            if right_line and l_shoulder_coords[0] > right_line:
                left = 1

            if bottom_line and  r_shoulder_coords[1] > bottom_line and r_shoulder_coords[1] > bottom_line:
                #cv2.putText(frame,'Down',(10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                #pyautogui.press('down')
                down = 1

            if top_line and l_shoulder_coords[1] < top_line and r_shoulder_coords[1] < top_line:
                #cv2.putText(frame,'Up',(w-50,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                #pyautogui.press('up')
                up = 1

        if prev_left == 0 and left == 1:
            pyautogui.press("left")
            print("left")
        if prev_right == 0 and right == 1:
            print("right")
            pyautogui.press("right")
        if prev_up == 0 and up == 1:
            print("up")
            pyautogui.press("up")
        if prev_down == 0 and down == 1:
            print("down")
            pyautogui.press("down")

        prev_up = up
        prev_down = down
        prev_right = right
        prev_left = left

        up = 0
        down = 0
        right = 0
        left = 0


        #Draw a vertical line at half the screen width
        cv2.line(frame, (r_shoulder_coords[0] + int((l_shoulder_coords[0] - r_shoulder_coords[0])/2), 0), (r_shoulder_coords[0] + int((l_shoulder_coords[0] - r_shoulder_coords[0])/2), h), (255, 0, 0), 2)
        
        if flag_horizontal == 0:
            right_line = int(left_shoulder.x * w) + 70
            left_line = int(right_shoulder.x * w) - 70

        flag_horizontal = 1

        if right_line and left_line:
            cv2.putText(frame,'LEFT',(right_line, bottom_line + int((top_line-bottom_line)/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame,'RIGHT',(left_line, bottom_line + int((top_line-bottom_line)/2)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (right_line, 0), (right_line, h), (255, 0, 0), 2)
            cv2.line(frame, (left_line, 0), (left_line, h), (255, 0, 0), 2)

        if flag_vertical == 0:
            top_line = int(left_shoulder.y * h) - 70
            bottom_line = int(left_shoulder.y * h) + 70

        flag_vertical = 1

        if top_line and bottom_line:
            cv2.putText(frame,'UP',(half_frame_width,top_line-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame,'DOWN',(half_frame_width,bottom_line-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (0,top_line), (w,top_line), (255, 0, 0), 2)
            cv2.line(frame, (0,bottom_line), (w,bottom_line), (255, 0, 0), 2)

        #draw_dotted_line(frame, (0,half_frame_height-120), (w,half_frame_height-120), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
