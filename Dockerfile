# 1. 파이썬 이미지 기반으로 시작
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# 2. 패키지 목록을 업데이트하고 Git을 설치
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install ultralytics

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. requirements.txt 파일을 컨테이너로 복사
COPY requirements.txt .

# 5. 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 6. 애플리케이션 코드를 컨테이너로 복사
COPY . .

# 7. 기본 실행 명령 설정 (여기서는 파이썬 애플리케이션 실행)
CMD ["python", "app.py"]
