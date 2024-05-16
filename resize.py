from PIL import Image
import os

# Đường dẫn tới thư mục chứa ảnh
input_folder = './TRAIN/HoangPhat'
output_folder = './TRAIN/HoangPhat_resized'

# Kích thước mới
new_size = (64, 64)  # Ví dụ: (800, 600)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpg'):  
        img = Image.open(os.path.join(input_folder, filename))
        img_resized = img.resize(new_size)
        img_resized.save(os.path.join(output_folder, filename))

print('Resize completed.')
