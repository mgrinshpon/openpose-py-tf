import cv2
import numpy as np
from openpose_py_tf import Estimator


if __name__ == '__main__':
    # video_path = "tests/resources/this_is_america.mp4"
    video_path = "tests/resources/mattias_kog.avi"
    cap = cv2.VideoCapture(video_path)

    assert cap.isOpened()

    num_frames_to_skip_per_analysis = 2
    original_fps = 30
    output_fps = int(round(original_fps / (1 + num_frames_to_skip_per_analysis)))

    ret, frame = cap.read()
    output_size = np.array(frame).shape
    # print(output_size)
    # outcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter((video_path + "_output.avi"), outcc, output_fps, (output_size[0], output_size[1]))

    graph_path = "models/graph/cmu/graph_opt.pb"
    target_size = (1312, 736)  # Default largest size in width and height

    estimator = Estimator(graph_path, target_size=target_size)

    frame_number = -1

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame', frame)

            if frame_number % (num_frames_to_skip_per_analysis + 1) == 0:
                humans = estimator.inference(np.array(frame), resize_to_default=True, upsample_size=4.0)
                retval = Estimator.draw_humans(np.array(frame), humans)
                # cv2.imshow("retval", retval)
                # out.write(retval)
                img_path = "tests/resources/kog_pics/demo_{}.png".format(frame_number)
                print("Writing image {} to path {}".format(frame_number, img_path))
                cv2.imwrite(img_path, retval)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_number = frame_number + 1
        except:
            print("Deep, dark failures have occured on frame number {}. Forgive me.".format(frame_number))

    cap.release()
    # out.release()
    cv2.destroyAllWindows()
