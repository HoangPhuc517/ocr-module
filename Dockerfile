# Sử dụng base image Python chính thức
FROM python:3.11-slim

# Thiết lập biến môi trường để Python không cache output và không dùng .pyc
ENV PYTHONUNBUFFERED 1

# Tạo và chuyển đến thư mục làm việc trong container
WORKDIR /app

# Copy file requirements.txt và cài đặt các dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy tất cả mã nguồn còn lại vào container
COPY . .

# Khai báo cổng mà ứng dụng sẽ lắng nghe (Fly.io sẽ ánh xạ)
ENV PORT 8080

# Healthcheck: Kiểm tra cổng 8080 cứ mỗi 5 giây
# HEALTHCHECK --interval=5s --timeout=3s \
#   CMD curl --fail http://localhost:8080/ || exit 1

# Chạy ứng dụng bằng Gunicorn
# Thay 'ocr:app' bằng 'tên_file_python:tên_biến_flask_app'
CMD exec gunicorn --bind :$PORT --workers 1 ocr:app