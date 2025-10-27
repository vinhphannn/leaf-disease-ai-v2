# Giới thiệu bộ dữ liệu

## **Tên bộ dữ liệu:**

**PlantCustom – New Combined Plant Disease Dataset**

Bộ dữ liệu này được nhóm tổng hợp và làm sạch nhằm phục vụ cho đề tài **nhận diện bệnh lá cây bằng học sâu**.
Nguồn dữ liệu được kết hợp từ nhiều bộ công khai trên Kaggle cùng với các ảnh tự thu thập ngoài thực tế, giúp tăng độ đa dạng và tính khái quát của mô hình.

---

## **Nguồn dữ liệu và cấu trúc**

Bộ dữ liệu được chia thành 2 phần chính:

```
data/
│
├── train/
│   ├── Apple___healthy/
│   ├── Apple___Apple_scab/
│   ├── ...
└── val/
    ├── Apple___healthy/
    ├── Apple___Apple_scab/
    ├── ...
```

- **Tổng số ảnh:** ~90.000 ảnh
- **Số lớp:** 38 lớp (Healthy + các bệnh khác nhau của Apple, Tomato, Corn, Grape, Potato, v.v.)
- **Tỷ lệ chia:** 80% train – 20% val
- **Định dạng ảnh:** RGB, chất lượng cao, kích thước đa dạng

---

## **Nguồn tổng hợp gồm:**

1. **New Plant Diseases Dataset** (nguồn chính)→ Lấy 1.000 ảnh train và 200 ảnh val cho mỗi lớp.
2. **PlantDoc Dataset**→ Ảnh ngoài tự nhiên, tăng cường dữ liệu ×4, lấy 200 train + 40 val mỗi lớp.
3. **PlantVillage Dataset**→ Ảnh lá cây đã tách nền, thêm 100 ảnh train mỗi lớp.
4. **Ảnh tự thu thập ngoài thực tế**
   → Chụp thủ công để bổ sung tính đa dạng môi trường.

---

## **Cách tải dữ liệu**

Bạn có thể tải trực tiếp bộ dữ liệu tại Kaggle:
 [https://www.kaggle.com/datasets/phanvnvinhs2564cntt/plantcustom/data](https://www.kaggle.com/datasets/phanvnvinhs2564cntt/plantcustom/data)

### **Cách 1: Tải trực tiếp trên trình duyệt**

1. Truy cập link trên.
2. Đăng nhập tài khoản Kaggle.
3. Nhấn **“Download”** để tải toàn bộ dữ liệu `.zip`.

### **Cách 2: Tải bằng lệnh Kaggle CLI**

Nếu bạn đã cài Kaggle API:

```bash
kaggle datasets download -d phanvnvinhs2564cntt/plantcustom -p ./data
unzip data/plantcustom.zip -d ./data
```

## **Sử dụng trong dự án**

Sau khi tải về, giải nén thư mục `data/` vào gốc dự án để mã nguồn có thể tự động đọc đúng đường dẫn dữ liệu:

```
project_root/
│
├── data/
│   ├── train/
│   └── val/
├── models/
├── notebooks/
└── main.py
```
