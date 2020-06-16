import os
import glob
import schedule
import time
import shutil
from video_processing import main


def preprocess():
    # print("[INFO] Checking directories...")
    if not os.path.exists('processing_video'):
        os.makedirs('processing_video')
    if not os.path.exists("processed_video"):
        os.makedirs('processed_video')
    if not os.path.exists("video_for_process"):
        os.makedirs("video_for_process")

    # if not glob.glob(os.path.join(os.getcwd(), "video_for_process\\*.mp4")):
    #     print("[INFO] Paste a video in the 'video_for_process' Directory OR press 'CTRL + C' to exit from "
    #           "the program")

    if glob.glob(os.path.join(os.getcwd(), "video_for_process\\*.mp4")):
        for filepath in glob.glob(os.path.join(os.getcwd(), "video_for_process\\*.mp4")):
            # print(colored("[INFO] Found video.", 'blue'))
            print("[INFO] Found video.")
            print("[INFO] Copying video to the processing folder...")
            shutil.move(filepath, os.getcwd() + os.sep + "processing_video" + os.sep + filepath.split("\\")[-1])
            break

        for videopath in glob.glob(os.path.join(os.getcwd(), "processing_video\\*.mp4")):
            main(videopath)
            shutil.move(videopath, os.getcwd() + os.sep + "processed_video" + os.sep + videopath.split("\\")[-1])
            print("[INFO] Video Processed.")
    else:
        pass


try:
    schedule.every(5).seconds.do(preprocess)
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    print("[INFO] Exiting from the program.")
    time.sleep(1)
