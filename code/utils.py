import os
import cv2
from dotenv import load_dotenv

def rename_video(input_video, env):
    os.rename(input_video, env)
    return env

def load_model(proto_path, model_path):
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)


if __name__ == "__main__" :
    # load_dotenv()

    # MODEL_PATH = os.environ.get('MODEL_PATH')
    # PROTOTXT_PATH = os.environ.get('PROTOTXT_PATH')
    # INPUT_FILE_NAME = os.environ.get('INPUT_FILE_NAME')

    # name = rename_video(INPUT_FILE_NAME, INPUT_FILE_NAME)

    # name = rename_video("video/input_1.mp4")
    # net = load_model(PROTOTXT_PATH,MODEL_PATH)
    # print(name)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        cv2.imshow('Movenet Ligthning', frame)

        while cv2.waitKey(10) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()