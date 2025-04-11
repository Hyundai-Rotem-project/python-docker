# 1. 베이스 이미지
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 2. 시스템 패키지 설치
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
    && ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. pip 업그레이드
RUN pip3 install --upgrade pip

# 4. 작업 디렉토리 설정
WORKDIR /app

# 5. 의존성 복사 및 설치
COPY requirements.txt .

# torch/torchvision/torchaudio는 미리 설치 (CUDA 11.8과 호환)
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 6. 코드 복사
COPY . .

# 7. 실행 명령 (테스트용으로 tail 사용, 추후 수정 가능)
CMD ["tail", "-f", "/dev/null"]
