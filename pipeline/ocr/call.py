import json
from flask import jsonify, request
import numpy as np

from pipeline import app
from pipeline.ocr import corner, extract_infos, text
from processing_tools import url_to_image
# from pipeline.ocr.functions import url_to_image

THRESHOLD = 0.3
imageSize = 512
# corner_url = 'http://localhost:8501/v1/models/corner:predict'
# text_url = 'http://localhost:8501/v1/models/text:predict'
seq2seq_url = 'pipeline/ocr/vietocr/seq2seqocr.pth'
targetSize = { 'w': imageSize, 'h': imageSize }
   
@app.route('/api/v1/ocr', methods=['POST'])
def api_ocr():
    data = json.loads(request.data)
    img_url = data['img_url']
    corner_url = data['corner_url']
    text_url = data['text_url']
    image = url_to_image(img_url)
    image = np.array(image)

    targetSize['h'] = image.shape[0]
    targetSize['w'] = image.shape[1]

    corner_img = corner.corner(image, corner_url, THRESHOLD, targetSize)

    targetSize['h'] = corner_img.shape[0]
    targetSize['w'] = corner_img.shape[1]

    textt = text.text(corner_img, text_url, THRESHOLD, targetSize)

    test = extract_infos.ocr(seq2seq_url)

    data = test.OCR(corner_img, textt)
    
    return jsonify(data)