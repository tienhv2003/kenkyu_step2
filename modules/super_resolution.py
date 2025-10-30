import cv2
import cv2.dnn_superres
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "super_resolution", "FSRCNN_x4.pb")

#AIモデルの読み込み
model_FSRCNN = cv2.dnn_superres.DnnSuperResImpl_create()

#モデルの設定
model_FSRCNN.readModel(MODEL_PATH)
model_FSRCNN.setModel("fsrcnn", 4)

# 画像フォルダ内のすべての画像に対して「超解像＋コントラスト強調」を行う関数
def super_resolution_and_contrast_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        img = cv2.imread(input_path)
        if img is None:
            print(f"画像を読み込めませんでした: {input_path}")
            continue
        
        upscaled_img = model_FSRCNN.upsample(img)

        # Convert to grayscale and enhance contrast using CLAHE
        gray_img = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(gray_img)

        # Add white border to stabilize downstream feature extraction
        padded = cv2.copyMakeBorder(enhanced_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)

        cv2.imwrite(output_path, padded)
        print(f"超解像＋前処理（グレースケール・コントラスト強調＋余白追加）画像を保存しました: {output_path}")




