import time
import cv2
import numpy as np
import onnxruntime as ort
from imutils.video import VideoStream

def main():
    # constants
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    mean = 0.485 * 255.
    std = 0.229 * 255.

    # create runnable session with exported model
    ort_session = ort.InferenceSession("signlanguage.onnx")
    vc = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        # Capture frame-by-frame
        frame = vc.read()

        width = 700
        height = 480
        frame = cv2.resize(frame, (width,height))

        img = frame[20:250, 20:250]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(img, (28, 28))
        x = (x - mean) / std

        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = ort_session.run(None, {'input': x})[0]

        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]
        window_name = "Sign Language Translator"

        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow(window_name, frame)

        image = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)
        cv2.imshow(window_name, image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()