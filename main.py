import os
import cv2
from modules.auto_cut_image import process_all_images  # 自動切り出し
from modules.super_resolution import superres_images_in_folder  # 超解像処理

output_folder = "73"  # または任意の値

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(BASE_DIR, f"data/number/{output_folder}")
mid_folder = os.path.join(BASE_DIR, f"data/plates_after_cut/{output_folder}")
output_folder_superres = os.path.join(BASE_DIR, f"data/plates_after_cut_super_resolution/{output_folder}")

# ステップ1: Cannyによる自動切り出し（good1/...に画像がなければ実行）
if not os.path.exists(mid_folder) or len(os.listdir(mid_folder)) == 0:
    print("--- Cannyによる自動切り出しを実行中 ---")
    process_all_images(output_folder)
else:
    print("--- 既に自動切り出し済みの画像が存在するため、このステップをスキップ ---")

# ステップ2: 超解像処理
if not os.path.exists(output_folder_superres):
    os.makedirs(output_folder_superres)

superres_images_in_folder(mid_folder, output_folder_superres)
