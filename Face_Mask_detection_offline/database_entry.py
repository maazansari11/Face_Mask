from mysql.connector import connect
import time


def get_max(table_name):
    dashboard_db = connect(user='root', passwd="admin", db='dashboard')
    cur = dashboard_db.cursor()
    cur.execute("SELECT MAX(ID) FROM " + table_name)
    temp = cur.fetchall()[0][0] + 1
    cur.close()
    return str(temp)


def entry_to_db(folder_path, video_name, video_path, violation_video_path, violation_frame_path):
    dashboard_db = connect(user='root', passwd="admin", db='dashboard')
    cur2 = dashboard_db.cursor()
    query = "SELECT ID FROM dashboard.configuration_type_tbl where configuration_description_fld = 'Face Mask';"
    cur2.execute(query)
    config_type_id = cur2.fetchall()[0][0]
    cur2.close()

    site_ins = get_max("configuration_tbl")
    camera_ins = get_max("camera_configuration_tbl")
    videos_ins = get_max("videos")
    violation_ins = get_max("violation_tbl")

    cur3 = dashboard_db.cursor()
    cur3.execute(
        "SELECT EXISTS (SELECT configuaration_description_fld FROM configuration_tbl WHERE "
        "configuaration_description_fld = 'Bur Dubai' AND config_type_id = 12);")
    if cur3.fetchall()[0][0] == 0:
        insert_query_ct = "INSERT INTO configuration_tbl VALUES(" + site_ins + ", " + str(
            config_type_id) + ", 'Bur Dubai', 'Mask Configuration', '', '', '', '', '')"
        print(insert_query_ct)
        cur6 = dashboard_db.cursor()
        cur6.execute(insert_query_ct)
        cur6.close()
        cur3.close()
    else:
        cur7 = dashboard_db.cursor()
        cur7.execute(
            "SELECT ID FROM configuration_tbl WHERE configuaration_description_fld = 'Bur Dubai' AND config_type_id = "
            "12;")
        site_ins = str(cur7.fetchall()[0][0])
        cur3.close()
        cur7.close()

    cur1 = dashboard_db.cursor()
    cur1.execute(
        "SELECT EXISTS(SELECT camera_ip_fid FROM camera_configuration_tbl WHERE camera_ip_fid = 'ip' AND "
        "config_id_fld = " + site_ins + ");")
    if cur1.fetchall()[0][0] == 0:
        insert_query_cct = 'INSERT INTO camera_configuration_tbl VALUES(' + camera_ins + ', "' + folder_path + '", "ip", "", "", 0, 1, ' + str(
            site_ins) + ')'
        print(insert_query_cct)
        cur8 = dashboard_db.cursor()
        cur8.execute(insert_query_cct)
        cur8.close()
        cur1.close()
    else:
        cur9 = dashboard_db.cursor()
        cur9.execute(
            "SELECT ID FROM camera_configuration_tbl WHERE camera_ip_fid = 'ip' AND config_id_fld = " + site_ins + ";")
        camera_ins = str(cur9.fetchall()[0][0])
        cur9.close()
        cur1.close()

    cur4 = dashboard_db.cursor()
    cur4.execute("SELECT EXISTS(SELECT video_name_fld FROM videos WHERE video_name_fld = '" +
                 video_name.split("\\")[-1].split(".")[-2] + "' AND camera_config_id = '" + camera_ins + "');")
    if cur4.fetchall()[0][0] == 0:
        insert_query_v = "INSERT INTO videos VALUES('" + videos_ins + "', '" + video_name.split("\\")[-1].split(".")[
            -2] + "', '" + video_path + "', '" + str(time.strftime('%Y-%m-%d %H:%M:%S')) + "', " + str(camera_ins) + ")"
        print(insert_query_v)
        cur10 = dashboard_db.cursor()
        cur10.execute(insert_query_v)
        cur10.close()
        cur4.close()
    else:
        cur11 = dashboard_db.cursor()
        cur11.execute("SELECT ID FROM videos WHERE video_name_fld = '" + video_name.split("\\")[-1].split(".")[
            -2] + "' AND camera_config_id = " + camera_ins + ";")
        videos_ins = str(cur11.fetchall()[0][0])
        cur11.close()
        cur4.close()

    cur5 = dashboard_db.cursor()
    insert_query_vt = "INSERT INTO violation_tbl VALUES(" + violation_ins + ", '" + violation_video_path + "', '" + violation_frame_path + "', '" + str(
        time.strftime('%Y-%m-%d %H:%M:%S')) + "', " + str(videos_ins) + ")"
    print(insert_query_vt)
    cur5.execute(insert_query_vt)
    cur5.close()
    # print(insert_query_ct)

    dashboard_db.commit()  # TODO: uncomment this line for actual process
    dashboard_db.close()

# entry_to_db("c:\\\\", "demo.mp4", "c:\\\\video\\\\path.mp4", "c:\\\\violation\\\\video\\\\path.mp4", "c:\\\\violation\\\\image\\\\path.mp4")
