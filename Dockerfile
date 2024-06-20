# Gunakan image Python 3.11.5 sebagai base image
FROM python:3.11.5

# Install dependensi sistem yang diperlukan
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Set environment variable untuk menghindari masalah interaksi pengguna
ENV DEBIAN_FRONTEND=noninteractive

# Buat direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt ke direktori kerja
COPY requirements.txt .

# Install dependensi Python dari requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Salin seluruh kode proyek ke direktori kerja
COPY . .

# Expose port yang digunakan oleh Streamlit
EXPOSE 8501

# Perintah untuk menjalankan aplikasi Streamlit
CMD ["streamlit", "run", "inference_classifier.py"]
