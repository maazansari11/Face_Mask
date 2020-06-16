import os
import cv2
from datetime import datetime

num = 0
img_array = []
size = []


def database_entry():
    pass


def create_video(video_name, video_violation_list, fps):
    print("[INFO] creating video")
    no_of_videos = 0

    for v in video_violation_list:
        s = (0, 0)
        start_frame, end_frame = v[0], v[1]
        violation_no = "violation_" + str(no_of_videos)

        for fn in range(start_frame, end_frame):
            if start_frame <= end_frame:
                file_name = "no_photo_zone_output" + os.sep + str(fn) + ".jpg"
                img = cv2.imread(file_name)
                try:
                    h, w, l = img.shape
                    s = (w, h)
                    img_array.append(img)
                except AttributeError:
                    break
            else:
                break

        if not os.path.exists(
                "processed_video" + os.sep + video_name.split(".")[-2].split("\\")[
                    -1] + os.sep + datetime.now().strftime(
                    "%Y") + os.sep + datetime.now().strftime("%m") + os.sep + datetime.now().strftime(
                    "%d") + os.sep + datetime.now().strftime("%H") + os.sep + violation_no):
            os.makedirs(
                "processed_video" + os.sep + video_name.split(".")[-2].split("\\")[
                    -1] + os.sep + datetime.now().strftime(
                    "%Y") + os.sep + datetime.now().strftime("%m") + os.sep + datetime.now().strftime(
                    "%d") + os.sep + datetime.now().strftime("%H") + os.sep + violation_no)

        violation_video = video_name.split(".")[-2].split("\\")[-1] + "_violation" + str(no_of_videos) + '.mp4'
        vv_path = "processed_video" + os.sep + video_name.split(".")[-2].split("\\")[
            -1] + os.sep + datetime.now().strftime(
            "%Y") + os.sep + datetime.now().strftime("%m") + os.sep + datetime.now().strftime(
            "%d") + os.sep + datetime.now().strftime("%H") + os.sep + violation_no + os.sep + violation_video
        print("Video Violtion path", vv_path)
        out_file = cv2.VideoWriter(vv_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, s)

        for j in range(len(img_array)):
            out_file.write(img_array[j])

        # TODO: Make entry in mysql datbase
        # database_entry()

        out_file.release()
        img_array.clear()
        no_of_videos += 1


create_video("processed_video\\Video2_20sec.mp4", [[173, 473], [948, 1248]], 60)
