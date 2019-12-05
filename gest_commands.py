import tensorflow as tf
import cv2
import multiprocessing as _mp
from src.utils import load_graph,  detect_hands, predict, is_in_square, mimic
from src.config import RED, GREEN, YELLOW, CYAN, BLUE
import numpy as np
from threading import Timer

tf.flags.DEFINE_integer("width", 1280, "Screen width")
tf.flags.DEFINE_integer("height", 720, "Screen height")
tf.flags.DEFINE_float("threshold", 0.7, "Threshold for score")
tf.flags.DEFINE_float("alpha", 0.3, "Transparent level")
tf.flags.DEFINE_string("pre_trained_model_path", "src/pretrained_model.pb", "Path to pre-trained model")

FLAGS = tf.flags.FLAGS


def main():
    graph, sess = load_graph(FLAGS.pre_trained_model_path)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FLAGS.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FLAGS.height)

    mp = _mp.get_context("spawn")
    v = mp.Value('i', 0)
    lock = mp.Lock()
    process = mp.Process(target=mimic, args=(v, lock))
    process.start()

    x_center = int(FLAGS.width / 2)
    y_center = int(FLAGS.height / 2)
    radius = int(min(FLAGS.width, FLAGS.height) / 4)


    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        results = predict(boxes, scores, classes, FLAGS.threshold, FLAGS.width, FLAGS.height)
        text="Oof"


        top_left_square_corr = np.array([(0, 0), (FLAGS.width//3, 0),
                                                 (FLAGS.width//3, FLAGS.height//2), (0, FLAGS.height//2)])
        bottom_left_square_corr = np.array([(0,FLAGS.height),(0, FLAGS.height//2),
                                        (FLAGS.width//3, FLAGS.height//2), (FLAGS.width//3, FLAGS.height)])
        bottom_right_square_corr = np.array([(FLAGS.width,FLAGS.height), (FLAGS.width - FLAGS.width//3, FLAGS     .height),
                                (FLAGS.width-FLAGS.width//4, FLAGS.height-FLAGS.height//3), (FLAGS.width, FLAGS.height-FLAGS.height//3)])
        top_right_square_corr = np.array([(FLAGS.width, 0), (FLAGS.width-FLAGS.width//4, 0),
                                        (FLAGS.width-FLAGS.width//4, FLAGS.height//3), (FLAGS.width, FLAGS.height//3)])


        if len(results) == 1:
            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 10, RED, -1)

            if category == "Open" and np.linalg.norm((x - x_center, y - y_center)) <= radius:
                action=1
                text=action
            elif category == "Open" and is_in_square((x,y), top_left_square_corr):
                action = 3
                text = action
            elif category == "Open" and is_in_square((x,y), top_right_square_corr):
                action = 2
                text = action
            elif category == "Closed" and is_in_square((x,y), bottom_right_square_corr):
                action = 4
                text = action
            else:
                action=0
            with lock:
                v.value = action
            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        overlay = frame.copy()
        height = FLAGS.height//3
        width = FLAGS.width//3


        cv2.drawContours(overlay, [top_left_square_corr], 0,
                         CYAN, -1)
        cv2.drawContours(overlay, [bottom_right_square_corr], 0,
                         RED, -1)
        cv2.drawContours(overlay, [bottom_left_square_corr], 0,
                         GREEN, -1)
        cv2.drawContours(overlay, [top_right_square_corr], 0,
                         YELLOW, -1)
        cv2.circle(overlay, (x_center, y_center), radius, BLUE, -1)
        cv2.addWeighted(overlay, FLAGS.alpha, frame, 1 - FLAGS.alpha, 0, frame)
        cv2.imshow('Detection', frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
