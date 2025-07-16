import os
import cv2
from daikeihosei import process_images_in_folder, output_folder
from superres import superres_images_in_folder

# Đường dẫn thư mục input và output trung gian
input_folder = f"number/{output_folder}"  # Ảnh gốc
mid_folder = f"good/{output_folder}"      # Ảnh đã chỉnh góc
output_folder_superres = f"good_superres/{output_folder}"  # Ảnh sau siêu phân giải

# Bước 1: Chỉnh góc biển số (nếu chưa có ảnh trong good/...)
if not os.path.exists(mid_folder) or len(os.listdir(mid_folder)) == 0:
    print("--- Đang thực hiện chỉnh góc biển số ---")
    process_images_in_folder(input_folder)
else:
    print("--- Đã có ảnh chỉnh góc, bỏ qua bước này ---")

# Bước 2: Siêu phân giải từng ảnh trong good/...
if not os.path.exists(output_folder_superres):
    os.makedirs(output_folder_superres)



superres_images_in_folder(mid_folder, output_folder_superres)
