# image_preprocess.py
import os
import numpy as np
from PIL import Image
from skimage.feature import hog


# trích xuất HOG features
def extract_hog(img, img_size=(64, 64)):
    """
    Trích xuất đặc trưng HOG từ ảnh
    """
    img = img.resize(img_size).convert("L")  # grayscale
    img_np = np.array(img)
    features = hog(
        img_np,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features, img_np   # trả thêm ảnh gốc (grayscale)

# Load ảnh trong 1 folder

def load_images_from_folder(folder, label):
    data, labels, raw_imgs = [], [], []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            img = Image.open(filepath)
            features, img_np = extract_hog(img)
            data.append(features)
            labels.append(label)
            raw_imgs.append(img_np)
        except Exception as e:
            print(f"Không đọc được ảnh {filepath}: {e}")
    return data, labels, raw_imgs


if __name__ == "__main__":
    folder_xe_to = r"D:\STUDY\HK1-25-26\thi-giac-may-tinh\crawlKNN\car\xe_to"
    folder_xe_nho = r"D:\STUDY\HK1-25-26\thi-giac-may-tinh\crawlKNN\car\xe_nho"

    data, labels, raw_imgs = [], [], []

    d1, l1, imgs1 = load_images_from_folder(folder_xe_to, 0)
    d2, l2, imgs2 = load_images_from_folder(folder_xe_nho, 1)

    data.extend(d1); labels.extend(l1); raw_imgs.extend(imgs1)
    data.extend(d2); labels.extend(l2); raw_imgs.extend(imgs2)

    data = np.array(data)
    labels = np.array(labels)
    raw_imgs = np.array(raw_imgs, dtype=object)

    print("Tổng số ảnh:", len(labels))
    print("Số chiều đặc trưng HOG:", data.shape[1])

    # Lưu ra file .npz
    np.savez("car_dataset.npz", data=data, labels=labels, raw_imgs=raw_imgs)
    print(" Đã lưu dữ liệu xử lý vào car_dataset.npz")
