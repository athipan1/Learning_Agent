# ใช้ base image ที่มี Python
FROM python:3.12-slim

# ตั้ง working directory
WORKDIR /app

# Copy requirements ก่อน (เพื่อ cache layer)
COPY requirements.txt .

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code ทั้งหมดเข้าไป
COPY . .

# กำหนดคำสั่งเริ่มต้น
CMD ["python", "main.py"]
