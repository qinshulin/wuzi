# -*- coding: utf-8 -*-
import cv2
import numpy as np
from yolo.detect     import Yolo_detect_wuzi
from yolo.utils.plots import plot_one_box
import time
import os
import cv2
import io
import requests
from flask import request
from log.MyTextLog import serverLogger
import json
from flask import Flask
import time
import base64
import argparse
import shutil

# 配置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#创建文件路径
basedir = os.path.abspath(os.path.dirname(__file__))
#print(basedir)
json_path=os.path.join(basedir,"result_json")
wuzi_path=os.path.join(basedir,"result_wuzi")
no_wuzi_path=os.path.join(basedir,"result_no_wuzi")
for path in [json_path,wuzi_path,no_wuzi_path]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


#def parse_opt():
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default=r'yolo/weight/last.pt',help='model path(s)')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=640, help='inference size h,w')
parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--max', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--augment', default=True, action='store_true', help='augmented inference')
opt = parser.parse_args()

#yolo初始化
detect =Yolo_detect_wuzi(opt.weights,imgsz=opt.imgsz)
log = serverLogger
app = Flask(__name__)

def cv_imread(image_path):
    cv_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    return cv_img

@app.route('/Detect_wuzi', methods=['POST'])
def Detect_wuzi():
    timestr = time.strftime('%F-%T', time.localtime()).replace(':', '-')
    try:
        img_str = request.form['wuzi']
        if img_str is None  :
            log.logger.error(">>>>>>>>>>>>>>>>>>img base64为空")
            return json.dumps({'status': "003", "message": "base64为空", "data":None})

        img_byte = base64.b64decode(img_str)
        image = np.frombuffer(img_byte, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # conf_thres = opt.conf,  # confidence threshold
        # iou_thres = opt.iou,  # NMS IOU threshold
        # max_det = opt.max,  # maximum detections per image
        # augment = opt.augment
        result=detect.run(image,conf_thres=opt.conf,iou_thres=opt.iou,max_det=opt.max,augment=opt.augment)
        wuzi_cout=len(result[0])

        ################ 没有检测到污渍
        if wuzi_cout==0:
            tmp_path=os.path.join(json_path,timestr+".json")
            return_dict={'status': "002", "message": "no wuzi",  "data":None}
            with open(tmp_path, "w") as fp:
                fp.write(str(return_dict))
            fp.close()
            cv2.imwrite(os.path.join(no_wuzi_path,timestr+".png"),image)
            return json.dumps(return_dict)

        ################检测到污渍
        else:

            #show_re=False
            #if show_re:
            return_coor=[]
            for idx, res in enumerate(result[0]):
                plot_one_box(res[:4], image, label="wuzi", color=(0, 0, 255), line_thickness=3)
                return_coor.append({'position':res[:4],"similarity":res[5] })
                # cv2.namedWindow('image',0)
                # cv2.imshow('image', image)
                # cv2.waitKey(11111)
                # cv2.destroyAllWindows()

            tmp_path = os.path.join(json_path, timestr + ".json")
            data={}
            ##########################   暂时为空
            data["imagePath"]=''
            one_wuzi_save_path=os.path.join(wuzi_path, timestr + ".png")
            data["markImagePath"]=one_wuzi_save_path
            data["wuzi"]=return_coor
            return_dict ={'status': "001", "message":"成功", "data":data}
            with open(tmp_path, "w") as fp:
                fp.write(str(return_dict))
            fp.close()
            cv2.imwrite(one_wuzi_save_path, image)
            return json.dumps(return_dict)

    except Exception as e:
        log.logger.error(">>>>>>>>>>>>>>>>>>系统出错 {}  ".format(str(e)))
        return json.dumps({'status': "004", "message": "系统出错", "data":None})



if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8888)


