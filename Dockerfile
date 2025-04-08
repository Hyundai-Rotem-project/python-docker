# 1. 베이스 이미지
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# 2. 필수 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    git \
    libgl1-mesa-glx \             
    libglib2.0-0 \                
    libsm6 \                     
    libxrender1 \
    libxext6 \
    && rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. 파이썬 패키지 설치
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install ultralytics

# 4. 작업 디렉토리 설정
WORKDIR /app

# 5. 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. 코드 복사
COPY . .

# 7. 실행 명령 (실행 스크립트는 나중에 제거 가능)
CMD ["tail", "-f", "/dev/null"]
