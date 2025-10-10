import cv2
import cv2.dnn_superres
import numpy as np
import os

#AIモデルの読み込み
model_FSRCNN = cv2.dnn_superres.DnnSuperResImpl_create()

#モデルの設定
model_FSRCNN.readModel("models/super_resolution/FSRCNN_x4.pb")
model_FSRCNN.setModel("fsrcnn", 4)

#画像フォルダ内のすべての画像を処理する関数
def superres_images_in_folder(input_folder, output_folder):
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
        
        cv2.imwrite(output_path, upscaled_img)
        print(f"超解像画像を保存しました: {output_path}")




