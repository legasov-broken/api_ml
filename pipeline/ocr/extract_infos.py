from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image


class ocr:
    def __init__(self, url_model):
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['weights'] = url_model
        config['cnn']['pretrained'] = False
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        self.predictor = Predictor(config)

    def OCR(self, img, data):
        boxes = list(data.values())
        labels = list(data.keys())

        dict = {'errorCode': 0, 'data': {
            'data': {}, 'confidence': {}}, 'errorMsg': 'success'}
        data_label = dict['data']
        for i in range(len(boxes)):
            boxes[i] = sorted(boxes[i], key=lambda k: [k[0], k[1]])
            box = boxes[i]
            for j in range(len(box)):
                text_img = img[int(box[j][1]):int(box[j][3]),
                               int(box[j][0]):int(box[j][2])]
                # cv2.imwrite('text_img' + str(i) + '.png', text_img)
                
                if labels[i] in data_label['data']:
                    text, conf = self.predictor.predict(
                        Image.fromarray(text_img), True)
                    data_label['data'][labels[i]] = text + \
                        ', ' + data_label['data'][labels[i]]
                    data_label['confidence'][labels[i]] = (
                        conf + data_label['confidence'][labels[i]])/2
                else:
                    text, conf = self.predictor.predict(
                        Image.fromarray(text_img), True)
                    data_label['data'][labels[i]] = text
                    data_label['confidence'][labels[i]] = conf
        dict['data'] = data_label
        return dict
