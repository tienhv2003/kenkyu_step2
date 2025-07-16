import os
import cv2
from superres import superres_images_in_folder
from cut_image_test import process_all_images

output_folder = "112"  # hoặc giá trị bạn muốn

input_folder = f"number/{output_folder}"
mid_folder = f"good1/{output_folder}"
output_folder_superres = f"good_superres/{output_folder}"

# Bước 1: Cắt tự động bằng Canny
if not os.path.exists(mid_folder) or len(os.listdir(mid_folder)) == 0:
    print("--- Đang thực hiện cắt tự động bằng Canny ---")
    process_all_images(output_folder)
else:
    print("--- Đã có ảnh cắt tự động, bỏ qua bước này ---")

# Bước 2: Siêu phân giải
if not os.path.exists(output_folder_superres):
    os.makedirs(output_folder_superres)

superres_images_in_folder(mid_folder, output_folder_superres)
