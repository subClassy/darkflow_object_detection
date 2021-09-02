from darkflow.net.build import TFNet
import cv2
import tensorflow as tf

config = tf.compat.v1.ConfigProto(log_device_placement = False)
config.gpu_options.allow_growth = False 

with tf.compat.v1.Session(config=config) as sess:
    options = {
                'model': './cfg/yolo.cfg',
                'load': './data/yolov2.weights',
                'threshold': 0.6
               }
    tfnet = TFNet(options)    

img = cv2.imread('./test/sample_scream.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = tfnet.return_predict(img)

for (i, result) in enumerate(results):
    x = result['topleft']['x']
    w = result['bottomright']['x']-result['topleft']['x']
    y = result['topleft']['y']
    h = result['bottomright']['y']-result['topleft']['y']
    cv2.rectangle(img, (x, y),(x+w, y+h),(0, 255, 0), 2)
    label_position = (x + int(w/2)), abs(y - 10)
    cv2.putText(img, result['label'], label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

cv2.imshow("Objet Detection YOLO", img)
cv2.waitKey(0)
cv2.destroyAllWindows()