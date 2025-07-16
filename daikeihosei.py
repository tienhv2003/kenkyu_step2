import cv2
import numpy as np
import math
import os

# クリックした点を保存するリスト
clicked_points = []
img = None  # グローバル変数として宣言
img_copy = None
output_folder = "127"  # 出力フォルダ
folder = "a"


# マウスイベントを処理するコールバック関数
def click_event(event, x, y, flags, param):
    global clicked_points, img
    # 左クリックが検出されたら座標をリストに追加
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        # 選択された点を表示する
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # type: ignore
        cv2.imshow("Image", img)  # type: ignore

        # 4点がクリックされたら処理を開始
        if len(clicked_points) == 4:
            cv2.destroyAllWindows()
            perform_trapezoid_correction(clicked_points)

# 台形補正を実行する関数
def perform_trapezoid_correction(points):
    global img_copy, output_file_path, img_num
    # 選択した4点を順番に整理（左上、右上、左下、右下）
    sorted_points = sort_vertices(np.array(points, dtype=np.float32))

    # クリックした4点
    p1, p2, p3, p4 = sorted_points

    # 比率調整（ここでは比率を変えない）
    w_ratio = 1.1

    # 幅取得
    o_width = np.linalg.norm(p2 - p1)
    o_width = math.floor(o_width * w_ratio)

    # 高さ取得
    o_height = np.linalg.norm(p3 - p1)
    o_height = math.floor(o_height)

    # 変換前の4点
    src = np.float32([p1, p2, p3, p4])

    # 変換後の4点
    dst = np.float32([[0, 0], [o_width, 0], [0, o_height], [o_width, o_height]])

    # 変換行列を取得
    M = cv2.getPerspectiveTransform(src, dst)

    # 射影変換を行う
    corrected_img = cv2.warpPerspective(img_copy, M, (o_width, o_height))

    # 画像の保存
    output_file_path = os.path.join(f"good/{output_folder}", f"{output_folder}_{folder}_{img_num}.jpg")
    cv2.imwrite(output_file_path, corrected_img)
    print(f"補正した画像を保存しました: {output_file_path}")

# 頂点を左上、右上、左下、右下の順にソートする関数
def sort_vertices(vertices):
    # x座標とy座標の和が最も小さい点が左上、最も大きい点が右下
    sum_coords = vertices.sum(axis=1)
    diff_coords = np.diff(vertices, axis=1)

    top_left = vertices[np.argmin(sum_coords)]
    bottom_right = vertices[np.argmax(sum_coords)]
    top_right = vertices[np.argmin(diff_coords)]
    bottom_left = vertices[np.argmax(diff_coords)]

    return np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

# 画像フォルダ内の画像を処理する関数
def process_images_in_folder(input_folder):
    global img, img_copy, clicked_points, img_num
    # フォルダが存在しなければ作成
    if not os.path.exists(f"good/{output_folder}"):
        os.makedirs(f"good/{output_folder}")

    # フォルダ内の画像ファイルを取得
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 画像を一枚ずつ処理
    img_num = 1
    for image_file in image_files:
        input_file_path = os.path.join(input_folder, image_file)
        print(f"処理中の画像: {input_file_path}")
        
        img = cv2.imread(input_file_path)
        if img is None:
            print(f"画像が見つかりません: {input_file_path}")
            continue

        img_copy = img.copy()
        clicked_points = []  # クリックした点をリセット

        # 画像を表示してクリックイベントを登録
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", click_event)

        # ユーザーがクリックして4点選択するまで待機
        cv2.waitKey(0)

        img_num += 1

    cv2.destroyAllWindows()

# メイン関数
if __name__ == "__main__":
    input_folder = f"number/{output_folder}"  # 入力フォルダ

    process_images_in_folder(input_folder)

    
    
