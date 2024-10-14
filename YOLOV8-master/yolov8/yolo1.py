import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config, array_to_image)
from utils.utils_bbox import DecodeBox

from arcface.nets.arcface import Arcface as arcface
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt


from arcface.arcface import Arcface
'''
训练自己的数据集必看注释！
'''


# 面部对齐模块
import cv2, dlib, argparse
import numpy as np
from face_alignment.utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image1
predictor = dlib.shape_predictor(r"E:\graduation project\yoloV5-arcface_forlearn-master\yoloV5_face\shape_predictor_68_face_landmarks.dat.tar")
detector = dlib.get_frontal_face_detector()


class YOLO(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path_yolo": r'E:\graduation project1\YOLOV8-master\yolov8\logs\best_epoch_weights.pth',
        "classes_path": 'target.txt',
        "model_path_arcface":r"E:\Arcface\model\arcface_iresnet50.pth",
        "backbone_arcface": "iresnet50",
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape_yolo": [1440, 1440],
        "input_shape_arcface": [640, 640, 3],
        # ------------------------------------------------------#
        #   所使用到的yolov8的版本：
        #   n : 对应yolov8_n
        #   s : 对应yolov8_s
        #   m : 对应yolov8_m
        #   l : 对应yolov8_l
        #   x : 对应yolov8_x
        # ------------------------------------------------------#
        "phi": 'n',
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.4,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image_yolo": False,
        "letterbox_image_arcface": False,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)

        self.bbox_util = DecodeBox(self.num_classes, (self.input_shape_yolo[0], self.input_shape_yolo[1]))

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate_yolo()
        self.generate_arcface()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate_yolo(self, onnx=False):
        # ---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        # ---------------------------------------------------#
        self.net_yolo = YoloBody(self.input_shape_yolo, self.num_classes, self.phi, True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net_yolo.load_state_dict(torch.load(self.model_path_yolo, map_location=device))
        self.net_yolo = self.net_yolo.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path_yolo))
        if not onnx:
            if self.cuda:
                self.net_yolo = nn.DataParallel(self.net_yolo)
                self.net_yolo = self.net_yolo.cuda()

    def generate_arcface(self):
        # ---------------------------------------------------#
        #   建立arcface模型，载入arcface模型的权重
        # ---------------------------------------------------#
        print('Arcface: Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net_arcface = arcface(backbone=self.backbone_arcface, mode="predict").eval()
        self.net_arcface.load_state_dict(torch.load(self.model_path_arcface, map_location=device), strict=False)
        print('{} model loaded.'.format(self.model_path_arcface))

        if self.cuda:
            self.net_arcface = torch.nn.DataParallel(self.net_arcface)
            cudnn.benchmark = True
            self.net_arcface = self.net_arcface.cuda()


    # ---------------------------------------------------#
    #   识别图片
    # ---------------------------------------------------#
    def detect_image_arcface(self, image_1, image_2):
        # ---------------------------------------------------#
        #   图片预处理，归一化
        # ---------------------------------------------------#
        with torch.no_grad():
            image_1 = array_to_image(image_1, [self.input_shape_arcface[1], self.input_shape_arcface[0]],
                                   letterbox_image=self.letterbox_image_arcface)
            image_2 = array_to_image(image_2, [self.input_shape_arcface[1], self.input_shape_arcface[0]],
                                   letterbox_image=self.letterbox_image_arcface)

            photo_1 = torch.from_numpy(
                np.expand_dims(np.transpose(preprocess_input(np.array(image_1, np.float32)), (2, 0, 1)), 0))
            photo_2 = torch.from_numpy(
                np.expand_dims(np.transpose(preprocess_input(np.array(image_2, np.float32)), (2, 0, 1)), 0))

            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            output1 = self.net_arcface(photo_1).cpu().numpy()
            output2 = self.net_arcface(photo_2).cpu().numpy()

            # ---------------------------------------------------#
            #   计算二者之间的距离
            # ---------------------------------------------------#
            l1 = np.linalg.norm(output1 - output2, axis=1)

        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(image_1))
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(image_2))
        # plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va='bottom', fontsize=11)
        # plt.show()
        return l1

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image_yolo(self, image, crop=True, count=False):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#

        image = cvtColor(image)

        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape_yolo[1], self.input_shape_yolo[0]), self.letterbox_image_yolo)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net_yolo(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape_yolo,
                                                         image_shape, self.letterbox_image_yolo, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image


            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(1e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape_yolo), 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])

                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # ---------------------------------------------------------#
        #   人脸识别 + 人脸对齐 + 图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            # 人脸裁剪加人脸对齐 PIL.Image

            crop_image = image.crop([left, top, right, bottom])

            # 转化为np.array
            crop_image = np.array(crop_image)

            # left, top, right, bottom = rect_to_tuple(det)

            dets = detector(crop_image, 1)

            for i, det in enumerate(dets):

                det = dlib.rectangle(max(det.left(), 0),
                                     max(det.top(), 0),
                                     max(det.right(), 0),
                                     max(det.bottom(), 0))

                shape = predictor(crop_image, det)
                left_eye = extract_left_eye_center(shape)
                right_eye = extract_right_eye_center(shape)
                M = get_rotation_matrix(left_eye, right_eye)
                rotated = cv2.warpAffine(crop_image, M, (crop_image.shape[1], crop_image.shape[0]),
                                         flags=cv2.INTER_CUBIC)
                image_1 = crop_image1(rotated, det)


            # 指定数据库路径
            data_base = r'E:\graduation project1\YOLOV8-master\yolov8\database'
            # 读取数据库中文件
            data_base_files = os.listdir(data_base)
            # 初始化距离和身份
            pro_min = 2
            name = "None"
            # 遍历文件列表
            for file in data_base_files:
                # 拼接文件的完整路径
                file_path = os.path.join(data_base, file)
                # 检查文件是否为图片文件
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    # 打开数据库图片并且face_alignment
                    image_2 = Image.open(file_path)
                    image_2 = np.array(image_2)

                    # dets = detector(image_2, 1)
                    #
                    # for i, det in enumerate(dets):
                    #
                    #     det = dlib.rectangle(max(det.left(), 0),
                    #                          max(det.top(), 0),
                    #                          max(det.right(), 0),
                    #                          max(det.bottom(), 0))
                    #
                    #     shape = predictor(image_2, det)
                    #     left_eye = extract_left_eye_center(shape)
                    #     right_eye = extract_right_eye_center(shape)
                    #     M = get_rotation_matrix(left_eye, right_eye)
                    #     rotated = cv2.warpAffine(image_2, M, (image_2.shape[1], image_2.shape[0]),
                    #                              flags=cv2.INTER_CUBIC)
                    #     image_2 = crop_image1(rotated, det)


                    probability = self.detect_image_arcface(image_1, image_2)


                    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    # # 显示第一张图像
                    # ax1.imshow(image_1)
                    # ax1.axis('off')  # 关闭坐标轴
                    # # 显示第二张图像
                    # ax2.imshow(image_2)
                    # ax2.axis('off')  # 关闭坐标轴
                    # fig.text(0.5, 0.5, f'probability: {probability}', color='black', fontsize=14, ha='center',
                    #          va='center')
                    # plt.show()

                    if float(probability) < 0.5 and probability < pro_min:
                        name = file.split('.')[0]
                        pro_min = min(pro_min, probability)


            label = '{}_{}_{:.2f}_{:.2f}'.format(predicted_class, name, score, float(pro_min))
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape_yolo[1], self.input_shape_yolo[0]), self.letterbox_image_yolo)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net_yolo(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape_yolo,
                                                         image_shape, self.letterbox_image_yolo, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                # ---------------------------------------------------------#
                outputs = self.net_yolo(images)
                outputs = self.bbox_util.decode_box(outputs)
                # ---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                # ---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape_yolo,
                                                             image_shape, self.letterbox_image_yolo,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape_yolo[1], self.input_shape_yolo[0]), self.letterbox_image_yolo)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            dbox, cls, x, anchors, strides = self.net_yolo(images)
            outputs = [xi.split((xi.size()[1] - self.num_classes, self.num_classes), 1)[1] for xi in x]

        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, -1, h, w]), [0, 2, 3, 1])[0]
            score = np.max(sigmoid(sub_output[..., :]), -1)
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches=-0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path_yolo):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape_yolo).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net_yolo,
                          im,
                          f=model_path_yolo,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path_yolo)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path_yolo)

        print('Onnx model save as {}'.format(model_path_yolo))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape_yolo[1], self.input_shape_yolo[0]), self.letterbox_image_yolo)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net_yolo(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape_yolo,
                                                         image_shape, self.letterbox_image_yolo, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return
