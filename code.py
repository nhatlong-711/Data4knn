# from icrawler.builtin import GoogleImageCrawler


# # elon_crawler = GoogleImageCrawler(storage={'root_dir': 'car/xemay'})
# # elon_crawler.crawl(keyword='xe máy', max_num=120) 


# bill_crawler = GoogleImageCrawler(storage={'root_dir': 'car/sedan'})
# bill_crawler.crawl(keyword='hình ảnh taxi Việt Nam', max_num=100)

# train_knn.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV



# Load dữ liệu đã xử lý
dataset = np.load("car_dataset.npz", allow_pickle=True)
data = dataset["data"]
labels = dataset["labels"]
raw_imgs = dataset["raw_imgs"]

print("Dataset shape:", data.shape)
print("Số lượng ảnh:", len(labels))

# Train/test

X_train, X_test, y_train, y_test, imgs_train, imgs_test = train_test_split(
    data, labels, raw_imgs, test_size=0.2, random_state=42, stratify=labels
)


# 3. Dùng GridSearchCV để tìm k tối ưu
param_grid = {"n_neighbors": list(range(1, 21))}  # thử k từ 1 → 20
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

print("K tốt nhất:", grid.best_params_["n_neighbors"])
print("Độ chính xác cross-validation:", grid.best_score_)

# Huấn luyện lại với k tối ưu
knn = grid.best_estimator_
knn.fit(X_train, y_train)

# knn = KNeighborsClassifier(n_neighbors=3) 
# knn.fit(X_train, y_train)

# Đánh giá

y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Xe To", "Xe Nho"]))


# # Trực quan hóa confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Xe To", "Xe Nho"], yticklabels=["Xe To", "Xe Nho"])
# plt.xlabel("Dự đoán")
# plt.ylabel("Thực tế")
# plt.title("Confusion Matrix - KNN + HOG")
# plt.show()

# 6. Hiển thị một vài ảnh test
plt.figure(figsize=(10, 6))
for i in range(6):
    idx = np.random.randint(0, len(X_test))
    img = imgs_test[idx]

    plt.subplot(2, 3, i+1)
    plt.imshow(np.array(img, dtype=np.uint8), cmap="gray")  # 🔧 fix lỗi dtype object
    plt.title(f"Thực tế: {'Xe To' if y_test[idx]==0 else 'Xe Nho'}\n"
              f"Dự đoán: {'Xe To' if y_pred[idx]==0 else 'Xe Nho'}")
    plt.axis("off")
plt.tight_layout()
plt.show()
