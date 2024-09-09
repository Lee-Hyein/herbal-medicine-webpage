from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
from rembg import remove
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# 모델 로드
bp_dense_model = load_model('sv_BP_CAM_DenseNet_AUG_100_0422.h5', compile=False)
sj_loaded_model = load_model('sv_SJ_CAM_ResNet_AUG_95__0411.h5', compile=False)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    print("분석 요청 받음")  # 로깅 추가
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다'}), 400

    file = request.files['image']
    herb_type = request.form.get('herbType')
    
    # 이미지 전처리
    img = image_prepro(file)
    
    # 모델 예측
    if herb_type == 'sanjo':
        predictions, predicted_class = predict_sanjo(img)
    else:
        predictions, predicted_class = predict_bangpung(img)
    
    # Grad-CAM 분석
    heatmap = make_gradcam_heatmap(img, bp_dense_model if herb_type == 'bangpung' else sj_loaded_model, 'conv2d_5' if herb_type == 'bangpung' else 'conv2d_15', predicted_class)
    
    # LIME 분석
    lime_exp = explain_with_lime(img, bp_dense_model if herb_type == 'bangpung' else sj_loaded_model)
    
    # 결과 이미지 생성
    result_images = generate_result_images(img, heatmap, lime_exp)
    
    # 텍스트 결과 생성
    text_result = generate_text_result(predictions, predicted_class, herb_type)

    results = {
        'prediction': text_result,
        'images': result_images
    }

    return jsonify(results)

@app.route('/')
def home():
    return "한약재 분석 API 서버가 실행 중입니다."

def image_prepro(file):
    image = Image.open(file)
    image = remove(image)
    width, height = image.size
    square_size = max(width, height)
    background = Image.new('RGB', (square_size, square_size), (0, 0, 0))
    offset = ((square_size - width) // 2, (square_size - height) // 2)
    background.paste(image, offset)
    image = background.resize((299, 299))
    image = image.convert('RGB')
    img_array = keras_image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

def predict_sanjo(img):
    predictions = sj_loaded_model.predict(img)
    predicted_class = np.argmax(predictions[0])
    return predictions[0], predicted_class

def predict_bangpung(img):
    predictions = bp_dense_model.predict(img)
    predicted_class = np.argmax(predictions[0])
    return predictions[0], predicted_class

def make_gradcam_heatmap(img, model, layer_name, pred_index):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def explain_with_lime(img, model):
    def segment_image_with_slic(image, n_segments=150, compactness=10, sigma=1):
        segments_slic = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma)
        return segments_slic
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img[0].astype('double'), 
                                             lambda x: model.predict(x), 
                                             top_labels=5, hide_color=0, num_samples=250,
                                             segmentation_fn=lambda x: segment_image_with_slic(x, n_segments=150, compactness=10, sigma=1))
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    return mark_boundaries(temp / 2 + 0.5, mask)

def generate_result_images(original_img, heatmap, lime_exp):
    # 원본 이미지
    original_img = (original_img[0] * 255).astype(np.uint8)
    
    # Grad-CAM 히트맵
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras_image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = keras_image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.4 + original_img
    superimposed_img = keras_image.array_to_img(superimposed_img)
    
    # 결과 이미지를 base64로 인코딩
    buffered = io.BytesIO()
    superimposed_img.save(buffered, format="PNG")
    gradcam_image = base64.b64encode(buffered.getvalue()).decode()
    
    # LIME 결과 이미지
    lime_image = keras_image.array_to_img(lime_exp)
    buffered = io.BytesIO()
    lime_image.save(buffered, format="PNG")
    lime_image = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        'original': base64.b64encode(cv2.imencode('.png', original_img)[1]).decode(),
        'gradcam': gradcam_image,
        'lime': lime_image
    }

def generate_text_result(predictions, predicted_class, herb_type):
    if herb_type == 'sanjo':
        classes = ['면조인', '산조인']
        group = '산조인류'
    else:
        classes = ['해방풍', '식방풍', '방풍']
        group = '방풍류'
    
    result = f"이 이미지가 '{classes[predicted_class]}'일 확률은 {predictions[predicted_class]*100:.2f}% 입니다.\n"
    result += f"{classes[predicted_class]}({get_scientific_name(classes[predicted_class])})\n\n"
    result += f"그룹: {group}\n\n"
    
    for i, class_name in enumerate(classes):
        result += f"{class_name} 판정 확률: {predictions[i]*100:.2f}%\n"
    
    result += f"\n활성화 부위: {get_activation_area(predictions, herb_type)}\n"
    result += get_characteristics(classes[predicted_class], get_activation_area(predictions, herb_type))
    
    return result

def get_scientific_name(herb_name):
    scientific_names = {
        '산조인': '산조 Ziziphus jujuba var. spinosa',
        '면조인': '전자조 Ziziphus mauritiana',
        '해방풍': '갯방풍 Glehnia littoralis',
        '식방풍': '갯기름나물 Peucedanum japonicum',
        '방풍': 'Saposhnikovia divaricata'
    }
    return scientific_names.get(herb_name, '')

def get_activation_area(predictions, herb_type):
    if herb_type == 'sanjo':
        center_mean_ratio = predictions[0]  # 이 부분은 실제 center_mean_ratio 계산 로직으로 대체해야 합니다
        if center_mean_ratio > 0.6:
            return 'center + line' if predictions[1] >= 0.6 else 'center'
        else:
            return 'edge'
    else:
        center_mean_ratio = predictions[0]  # 이 부분은 실제 center_mean_ratio 계산 로직으로 대체해야 합니다
        edges_above_mean = sum(p > 0.5 for p in predictions[1:])  # 이 부분도 실제 로직으로 대체해야 합니다
        if center_mean_ratio > 0.6:
            return 'center + edge' if edges_above_mean >= 2 else 'center'
        else:
            return 'edge'

def get_characteristics(herb_name, activation_area):
    characteristics = {
        '면조인': {
            'center + line': '1. 황갈색의 반점이 산재해있고 중간에 세로 주름이 없음\n2. 회황색을 띰\n3. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n4. 산조인에 비해 두께가 얇고 납작함\n5. 흔히 비늘 모양 무늬가 있음\n',
            'center': '1. 회황색을 띰\n2. 황갈색의 반점이 산재해있고 중간에 세로 주름이 없음\n3. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n4. 산조인에 비해 두께가 얇고 납작함\n5. 흔히 비늘 모양 무늬가 있음\n',
            'edge': '1. 산조인에 비해 두께가 얇고 납작함\n2. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n3. 황갈색의 반점이 산재해있고 중간에 세로 주름이 없음\n4. 회황색을 띰\n5. 흔히 비늘 모양 무늬가 있음\n'
        },
        '산조인': {
            'center + line': '1. 자색 또는 자갈색을 띰\n2. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n3. 융기되지 않은 평탄한 면에 능선(세로 줄)이 있음\n4. 매끄럽고 광택이 있으며 어떤 것은 벌어진 무늬가 있음\n',
            'center': '1. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n2. 매끄럽고 광택이 있으며 어떤 것은 벌어진 무늬가 있음\n3. 자색 또는 자갈색을 띰\n4. 융기되지 않은 평탄한 면에 능선(세로 줄)이 있음\n',
            'edge': '1. 융기되지 않은 평탄한 면에 능선(세로 줄)이 있음\n2. 한쪽 면은 조금 융기되어 있고 한쪽 면은 비교적 평탄함\n3. 자색 또는 자갈색을 띰\n4. 매끄럽고 광택이 있으며 어떤 것은 벌어진 무늬가 있음\n'
        },
        '해방풍': {
            'center + edge': '1. 가늘고 긴 원기둥 모양\n2. 목부는 연한 노란색이며 조직이 치밀함\n3. 피부는 가루성이며, 갈색의 분비도가 작은 점으로 흩어져 있음\n4. 여러 개의 벌어진 틈이 있음\n',
            'center': '1. 목부는 연한 노란색이며 조직이 치밀함\n2. 여러 개의 벌어진 틈이 있음\n3. 가늘고 긴 원기둥 모양\n4. 피부는 가루성이며, 갈색의 분비도가 작은 점으로 흩어져 있음\n',
            'edge': '1. 피부는 가루성이며, 갈색의 분비도가 작은 점으로 흩어져 있음\n2. 가늘고 긴 원기둥 모양\n3. 여러 개의 벌어진 틈이 있음\n4. 목부는 연한 노란색이며 조직이 치밀함\n'
        },
        '식방풍': {
            'center + edge': '1. 식방풍은 피부와 목부의 색깔이 비슷하다\n2. 목부는 황갈색이며 치밀함\n3. 피부는 회갈색,연한 갈색이며 빈틈이 보이고 갈색의 형성층이 뚜렷함\n',
            'center': '1. 목부는 황갈색이며 치밀함\n2. 피부는 회갈색,연한 갈색이며 빈틈이 보이고 갈색의 형성층이 뚜렷함\n3. 식방풍은 피부와 목부의 색깔이 비슷함\n',
            'edge': '1. 피부는 회갈색,연한 갈색이며 빈틈이 보이고 갈색의 형성층이 뚜렷함\n2. 목부는 황갈색이며 치밀함\n3. 식방풍은 피부와 목부의 색깔이 비슷함\n'
        },
        '방풍': {
            'center + edge': '1. 목부는 연한 노란색임\n2. 피부는 연한 갈색이고 회갈색의 벌어진 틈이 여러개 있음\n3. 목부에 비해 피부의 색깔이 뚜렷하게 진함\n',
            'center': '1. 목부는 연한 노란색임\n2. 피부에 비해 목부의 색깔이 뚜렷하게 연함\n3. 피부는 연한 갈색이고 회갈색의 벌어진 틈이 여러개 있음\n',
            'edge': '1. 피부는 연한 갈색이고 회갈색의 벌어진 틈이 여러개 있음\n2. 목부에 비해 피부의 색깔이 뚜렷하게 진함\n3. 목부는 연한 노란색임\n'
        }
    }
    return characteristics.get(herb_name, {}).get(activation_area, '특성 정보가 없습니다.')

if __name__ == '__main__':
    app.run(debug=True, port=5000)