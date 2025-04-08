# 🐳 python-docker: 원격 GPU 도커 환경 공유 프로젝트

> 여러 명이 함께 사용하는 B 서버의 GPU를 원격에서 활용하기 위한 도커 환경 구성

---

## 📌 개요

이 프로젝트는 B 서버에 구축된 GPU 도커 환경에 A 사용자들이 원격으로 접속하여 **자신만의 컨테이너를 생성하고**, **코드 테스트 및 모델 학습**을 수행할 수 있게 구성됩니다.  
모든 코드는 Git으로 관리되며, 학습 데이터는 Syncthing으로 실시간(또는 수시) 동기화합니다.

---

## 🛠️ 사전 준비 (A 사용자의 입장에서)

### 1. Zerotier로 B와 네트워크 연결

1. [https://www.zerotier.com](https://www.zerotier.com) 접속 → 회원가입 후 네트워크 생성
2. A, B 양쪽 모두 Zerotier 설치 후 같은 네트워크에 가입
3. Zerotier Central에서 각 장비 승인 (Allow)
4. B의 Zerotier IP 확인 → 이후 SSH 접속 및 Syncthing 공유에 사용

---

### 2. SSH 연결 설정 (VSCode용)

1. B에서 **SSH 서버 실행** (WSL2 기준):
   ```bash
   sudo apt update && sudo apt install openssh-server
   sudo service ssh start
   ```
2. A에서 VSCode 확장 프로그램 `Remote - SSH` 설치
3. `~/.ssh/config`에 B의 Zerotier IP로 연결 설정
   ```
   Host bserver
       HostName <B의 Zerotier IP>
       User <WSL 유저명>
   ```
4. VSCode → Remote Explorer → bserver 접속

---

### 3. Git 클론 (A에서 작업 시작)

```bash
git clone https://github.com/Hyundai-Rotem-project/python-docker.git
cd python-docker
```

---

## 🚀 도커 컨테이너 실행 (원격 B 서버)

### A 사용자가 직접 실행

각 사용자는 `docker-compose.yml`의 컨테이너 이름과 포트를 사용자 기준으로 수정 후 아래 명령어로 실행:

```bash
docker-compose -f docker-compose.a.yaml up -d --build
```

예시 (`docker-compose.a.yaml`):

```yaml
version: "3"
services:
  python-env-a:
    build: .
    container_name: python-env-a
    ports:
      - "5010:5000"  # A 사용자는 http://localhost:5010 접근
    volumes:
      - ./shared_data:/app/shared_data
    working_dir: /app
    command: tail -f /dev/null
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

---

## 🔁 Syncthing을 이용한 학습 데이터 공유

1. A, B 모두 Syncthing 설치:
   - [https://syncthing.net](https://syncthing.net) 에서 다운로드
2. A에서 **학습 데이터 폴더 공유 설정**
3. B에서 공유 승인 → B 내의 `shared_data` 디렉토리로 설정
4. 도커 `volumes`에서 `/app/shared_data`로 마운트하면 학습 데이터 자동 반영

---

## 📂 디렉토리 구성 예시

```
python-docker/
├── app.py                # A에서 실행 (클라이언트)
├── docker-compose.a.yaml # 사용자 A 전용 컨테이너 실행 설정
├── Dockerfile
├── requirements.txt
├── shared_data/          # Syncthing으로 동기화되는 폴더 (학습 데이터)
```

---

## 🎯 워크플로우 예시

1. A 사용자가 VSCode Remote-SSH로 B에 접속
2. `docker-compose`로 본인 컨테이너 실행
3. B의 GPU가 필요한 코드(ex. 학습 코드)는 도커에서 실행
4. 시뮬레이터는 A에서 실행 (localhost 고정된 클라이언트 대응)

---

## 📎 기타 명령어

```bash
# 컨테이너 확인
docker ps

# 컨테이너 접속
docker exec -it python-env-a bash

# 컨테이너 중지
docker-compose -f docker-compose.a.yaml down
```
