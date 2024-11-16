import json
import os

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils.utils import cvtColor, preprocess_input, resize_image
from yolo import YOLO

#---------------------------------------------------------------------------#
#   map_mode specifies the content to compute when running this file.
#   map_mode = 0 represents the entire mAP calculation process, including obtaining prediction results and calculating mAP.
#   map_mode = 1 represents obtaining prediction results only.
#   map_mode = 2 represents calculating mAP only.
#---------------------------------------------------------------------------#
map_mode            = 0
#-------------------------------------------------------#
#   Paths to the validation dataset annotations and images
#-------------------------------------------------------#
cocoGt_path         = 'coco_dataset/annotations/instances_val2017.json'
dataset_img_path    = 'coco_dataset/val2017'
#-------------------------------------------------------#
#   Folder for output results, default is map_out
#-------------------------------------------------------#
temp_save_path      = 'map_out/coco_eval'

class mAP_YOLO(YOLO):
    #---------------------------------------------------#
    #   Detect image
    #---------------------------------------------------#
    def detect_image(self, image_id, image, results, clsid2catid):
        #---------------------------------------------------#
        #   Calculate the height and width of the input image
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Convert the image to RGB to prevent errors during prediction with grayscale images.
        #   This code only supports prediction for RGB images; all other image types will be converted to RGB.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image for undistorted resizing.
        #   Alternatively, resize directly for recognition.
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Feed the image into the network for prediction!
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   Stack the prediction boxes, then perform non-maximum suppression
            #---------------------------------------------------------#
            outputs = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if outputs[0] is None: 
                return outputs

            top_label   = np.array(outputs[0][:, 5], dtype = 'int32')
            top_conf    = outputs[0][:, 4]
            top_boxes   = outputs[0][:, :4]

        for i, c in enumerate(top_label):
            result                      = {}
            top, left, bottom, right    = top_boxes[i]

            result["image_id"]      = int(image_id)
            result["category_id"]   = clsid2catid[c]
            result["bbox"]          = [float(left),float(top),float(right-left),float(bottom-top)]
            result["score"]         = float(top_conf[i])
            results.append(result)
        return results

if __name__ == "__main__":
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    cocoGt      = COCO(cocoGt_path)
    ids         = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()

    if map_mode == 0 or map_mode == 1:
        yolo = mAP_YOLO(confidence = 0.001, nms_iou = 0.65)

        with open(os.path.join(temp_save_path, 'eval_results.json'),"w") as f:
            results = []
            for image_id in tqdm(ids):
                image_path  = os.path.join(dataset_img_path, cocoGt.loadImgs(image_id)[0]['file_name'])
                image       = Image.open(image_path)
                results     = yolo.detect_image(image_id, image, results, clsid2catid)
            json.dump(results, f)

    if map_mode == 0 or map_mode == 2:
        cocoDt      = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))
        cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox') 
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print("Get mAP done.")
