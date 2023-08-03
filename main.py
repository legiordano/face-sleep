import cv2
import dlib
import face_recognition

def load_face_landmark_predictor():
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return predictor

def detect_fatigue(facial_landmarks):
    left_eye_closed = facial_landmarks[43][1] > facial_landmarks[44][1] and facial_landmarks[42][1] > facial_landmarks[47][1]
    right_eye_closed = facial_landmarks[38][1] > facial_landmarks[41][1] and facial_landmarks[37][1] > facial_landmarks[46][1]
    return left_eye_closed or right_eye_closed

def process_camera():
    video_capture = cv2.VideoCapture(0)

    face_landmark_predictor = load_face_landmark_predictor()

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture frame.")
            break

        face_locations = face_recognition.face_locations(frame)
        
        if not face_locations:
            continue

        facial_landmarks = face_landmark_predictor(frame, face_locations[0])
        facial_landmarks = [(landmark.x, landmark.y) for landmark in facial_landmarks.parts()]

        is_fatigued = detect_fatigue(facial_landmarks)

        if is_fatigued:
            cv2.putText(frame, "¡Estás quedándote dormido!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow("Detección de Cansancio Facial", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    process_camera()

if __name__ == "__main__":
    main()
