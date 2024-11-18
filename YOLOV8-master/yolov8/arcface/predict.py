from PIL import Image

from arcface import Arcface

if __name__ == "__main__":
    model = Arcface()

    # ----------------------------------------------------------------------------------------------------------#
    #   The `mode` is used to specify the testing mode:
    #   'predict' indicates single image prediction. If you want to modify the prediction process,
    #   such as saving images or cropping objects, refer to the detailed comments below.
    #   'fps' is used for testing FPS. The test image used is street.jpg in the `img` folder. 
    #   Refer to the comments below for more details.
    # ----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    # -------------------------------------------------------------------------#
    #   `test_interval` specifies the number of times the image is tested for FPS measurement.
    #   In theory, the larger the `test_interval`, the more accurate the FPS.
    #   `fps_test_image` is the image used for FPS testing.
    # -------------------------------------------------------------------------#
    test_interval = 100
    fps_test_image = 'img/somebody.jpg'

    if mode == "predict":
        while True:
            image_1 = input('Input image_1 filename:')
            try:
                image_1 = Image.open(image_1)
            except:
                print('Image_1 Open Error! Try again!')
                continue

            image_2 = input('Input image_2 filename:')
            try:
                image_2 = Image.open(image_2)
            except:
                print('Image_2 Open Error! Try again!')
                continue

            probability = model.detect_image(image_1, image_2)
            print(probability)

    elif mode == "fps":
        img = Image.open(fps_test_image)
        tact_time = model.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
