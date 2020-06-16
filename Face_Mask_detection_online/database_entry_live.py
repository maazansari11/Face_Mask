from mysql.connector import connect
import time
import os


def get_max(table_name):
    dashboard_db = connect(user='root', passwd="admin", db='dashboard')
    cur = dashboard_db.cursor()
    cur.execute("SELECT MAX(ID) FROM " + table_name)
    temp = cur.fetchall()[0][0] + 1
    cur.close()
    return str(temp)


def entry_to_db_live(video_name, video_path, violation_video_path, violation_frame_path, camera_ip, camera_user, camera_port, camera_password):
    # print("[INFO] Making entries.")
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("Video Name: \t\t\t\t", video_name)
    # print("Video Path: \t\t\t\t", video_path)
    # print("Violation Video path: \t\t", violation_video_path)
    # print("Violation frame path:\t\t", violation_frame_path)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    dashboard_db = connect(user='root', passwd="admin", db='dashboard')

    videos_ins = get_max("videos")
    violation_ins = get_max("violation_tbl")
    print("video ins", videos_ins, violation_ins)

    cur12 = dashboard_db.cursor()
    get_config_desc_id = "SELECT ID from dashboard.camera_configuration_tbl where camera_ip_fid = '" + str(camera_ip) + "' and camera_user_id = '"+str(camera_user)+"' and camera_port_no_fid = '"+str(camera_port)+"' and camera_password_fid = '"+str(camera_password)+"';"
    cur12.execute(get_config_desc_id)
    camera_config_pk = cur12.fetchall()[0][0]

    accessTimesinceEpoc = os.path.getatime(video_path)
    video_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(accessTimesinceEpoc))

    # print("INSERT INTO videos VALUES('"+videos_ins+"', '"+video_name+"', '"+video_path+"', '"+video_datetime+"', '"+str(camera_config_pk)+"');")
    insert_video = "INSERT INTO videos VALUES('"+videos_ins+"', '"+video_name+"', '"+video_path+"', '"+video_datetime+"', '"+str(camera_config_pk)+"');"
    # print(insert_video)
    cur13 = dashboard_db.cursor()
    cur13.execute(insert_video)
    cur13.close()

    accessTimesinceEpoc = os.path.getatime(violation_video_path)
    accessTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(accessTimesinceEpoc))

    cur14 = dashboard_db.cursor()
    insert_violation = "INSERT INTO violation_tbl VALUES('"+violation_ins+"', '"+violation_video_path.replace("\\", "\\\\")+"', '"+violation_frame_path.replace("\\", "\\\\")+"', '"+accessTime+"', '"+videos_ins+"')"
    # print(insert_violation)
    cur14.execute(insert_violation)
    cur14.close()
    dashboard_db.commit()   # TODO: uncomment this line for actual process
    dashboard_db.close()
    print("[INFO] Database updated.")

# v_video_path = os.getcwd() + os.sep + "video_violations\\YEAR 2020\\MONTH 05\\DATE 20\\HOUR 20\\violation0.mp4"
# video_path = os.getcwd() + os.sep + "live_camera_recordings\\2020\\05\\21\\18\\Video_18_02.mp4"
# a_video_path = video_path.replace("\\", "\\\\")
# # print(a_video_path)
# entry_to_db_live("demo_15_30", a_video_path, v_video_path.replace("\\", "\\\\"), "c:\\\\violation\\\\image\\\\path.mp4", "195.229.90.110", "admin", "4444", "India12345")
# cwd = os.getcwd() + os.sep
# print(cwd.replace("\\", "\\\\"))
