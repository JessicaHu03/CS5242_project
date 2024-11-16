# -----------------------------------------------------------------------#
#   `predict.py` integrates functionalities such as single-image prediction, 
#   video detection, FPS testing, and directory traversal detection into one Python file.
#   The mode can be changed by modifying the `mode` variable.
# -----------------------------------------------------------------------#

import time
import cv2
import numpy as np
from PIL import Image
from yolo1 import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    # ----------------------------------------------------------------------------------------------------------#
    #   `mode` specifies the testing mode:
    #   'predict'           Single-image prediction. For modifications like saving images or cropping objects, 
    #                       refer to the detailed comments below.
    #   'video'             Video detection. Use a camera or a video file for detection. See comments below for details.
    #   'fps'               FPS testing using the `street.jpg` image in the `img` folder. See comments below for details.
    #   'dir_predict'       Traverse a folder for detection and save results. By default, it processes the `img` folder 
    #                       and saves results to the `img_out` folder. See comments below for details.
    #   'heatmap'           Heatmap visualization of prediction results. See comments below for details.
    #   'export_onnx'       Export the model to ONNX format. Requires PyTorch 1.7.1 or higher.
    # ----------------------------------------------------------------------------------------------------------#
    mode = "heatmap"

    # -------------------------------------------------------------------------#
    #   `crop`              Indicates whether to crop targets after single-image prediction.
    #   `count`             Indicates whether to count detected objects.
    #   These options are effective only when `mode='predict'`.
    # -------------------------------------------------------------------------#
    crop = False
    count = False
    # ----------------------------------------------------------------------------------------------------------#
    #   `video_path`        Specifies the video path. Set `video_path=0` to use the camera.
    #                       To detect from a video file, set `video_path = "xxx.mp4"` to read `xxx.mp4` in the root directory.
    #   `video_save_path`   Specifies the save path for the output video. If empty, the video is not saved.
    #                       To save the video, set `video_save_path = "yyy.mp4"` to save as `yyy.mp4` in the root directory.
    #   `video_fps`         Specifies the FPS for the saved video.
    #
    #   These options are effective only when `mode='video'`.
    #   When saving videos, press Ctrl+C to exit or run to the last frame to complete the saving process.
    # ----------------------------------------------------------------------------------------------------------#
    video_path = r"YOLOV8-master/yolov8/img/video_crop.mp4"
    video_save_path = r"YOLOV8-master/yolov8/img/Fro_30_Unfro_30.mp4"
    video_fps = 30.0
    # ----------------------------------------------------------------------------------------------------------#
    #   `test_interval`     Specifies the number of image detections for FPS measurement. A larger value leads to more accurate FPS.
    #   `fps_image_path`    Specifies the image path for FPS testing.
    #
    #   These options are effective only when `mode='fps'`.
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   `dir_origin_path`   Specifies the folder path containing images for detection.
    #   `dir_save_path`     Specifies the folder path to save detected images.
    #
    #   These options are effective only when `mode='dir_predict'`.
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    # -------------------------------------------------------------------------#
    #   `heatmap_save_path` Specifies the save path for heatmap visualization. Defaults to the `model_data` folder.
    #
    #   This option is effective only when `mode='heatmap'`.
    # -------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    # -------------------------------------------------------------------------#
    #   `simplify`          Indicates whether to simplify the ONNX model.
    #   `onnx_save_path`    Specifies the save path for the ONNX model.
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        '''
        1. To save the detected image, use `r_image.save("img.jpg")`. Modify this directly in `predict.py`.
        2. To obtain the coordinates of the bounding boxes, access the `yolo.detect_image` function 
           and read the `top`, `left`, `bottom`, and `right` values during the drawing process.
        3. To crop targets based on the bounding boxes, access the `yolo.detect_image` function, 
           and use the `top`, `left`, `bottom`, and `right` values to crop from the original image using matrix operations.
        4. To add additional text (e.g., the number of detected objects) to the prediction image, 
           access the `yolo.detect_image` function, check the `predicted_class` (e.g., `if predicted_class == 'car':`), 
           and record the count. Use `draw.text` to add text to the image.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image_yolo(image, crop=crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read the camera (or video). Ensure the camera is correctly installed or the video path is valid.")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Saved processed video to the path: " + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
