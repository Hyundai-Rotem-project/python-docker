# 1. 파이썬 이미지 기반으로 시작
FROM python:3.12.7-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 파일을 컨테이너로 복사
COPY requirements.txt .

# 4. 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 코드를 컨테이너로 복사
COPY . .

# 6. 기본 실행 명령 설정 (여기서는 파이썬 애플리케이션 실행)
CMD ["python", "app.py"]
