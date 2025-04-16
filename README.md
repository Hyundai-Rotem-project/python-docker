# 🐳 python-docker: 원격 GPU 도커 환경 공유 프로젝트

> 여러 명이 함께 사용하는 B 서버의 GPU를 원격에서 활용하기 위한 도커 환경 구성

---

## 📌 개요

이 프로젝트는 B 서버에 구축된 GPU 도커 환경에 A 사용자들이 원격으로 접속하여 **자신만의 컨테이너를 생성하고**, **코드 테스트 및 모델 학습**을 수행할 수 있게 구성됩니다.  
모든 코드는 Git으로 관리되며, 학습 데이터는 NAS 공유 폴더를 통해 직접 접근 가능합니다.

---

## 🛠️ 사전 준비 (A 사용자의 입장에서)

### 1. Zerotier로 B와 네트워크 연결

1. [https://www.zerotier.com](https://www.zerotier.com) 접속 → 회원가입 후 네트워크 생성
2. A, B 양쪽 모두 Zerotier 설치 후 같은 네트워크에 가입
3. Zerotier Central에서 각 장비 승인 (Allow)
4. B의 Zerotier IP 확인 → 이후 SSH 접속 등에 사용

---

### 2. SSH 연결 설정 (VSCode용)

1. B에서 **SSH 서버 실행** (WSL2 기준):
   ```bash
   sudo apt update && sudo apt install openssh-server
   sudo service ssh start
   ```
2. A에서 VSCode 확장 프로그램 `Remote - SSH` 설치

3. VSCode → Crtl+shift+P → Remote-ssh: Connect to host...

---

### 3. Git 클론 (A에서 작업 시작)

```bash
git clone https://github.com/Hyundai-Rotem-project/python-docker.git
cd python-docker
```

---

## 📂 디렉토리 구성 예시

```
python-docker/
├── client/                        # A에서 실행 (클라이언트)
│   └── app.py                    # 시뮬레이터와 통신하는 Flask API
│
├── server/                        # B에서 실행 (서버 측 코드)
│   ├── train/                    # 학습용 코드 및 스크립트
│   │   ├── train.py
│   │   └── augment.py           # 데이터 증강 코드
│   ├── inference/               # 추론용 코드
│   │   └── inference.py
│   └── docker-compose.yml       # B에서 사용하는 docker-compose 설정
│
├── Dockerfile                    # 공통 도커 환경 (server용)
├── requirements.txt
├── shared_data/                  # NAS에 연결된 학습 데이터 폴더
└── README.md
```

---

## 🎯 워크플로우 예시 (Branch 전략 포함)

1. **A 사용자가 VSCode Remote-SSH로 B 서버에 접속**
   - Zerotier 네트워크를 통해 SSH 연결
   - `/mnt/...`에 마운트된 NAS 공유 폴더 접근 가능

2. **각자 브랜치를 생성하여 독립 작업**
   - 본인의 이름 혹은 작업 목적에 따라 브랜치 생성  
     예: `feature/user-a-inference`, `feature/user-b-augmentation`
   ```bash
   git checkout -b feature/<이름-작업내용>
   ```

3. **도커 컨테이너 실행 및 개발**
    - 컨테이너는 개인별로 분리 (`docker-compose.yml` 파일을 수정해 생성)
    예시 (`docker-compose.yml`):
      ```yaml
      version: "3"
      services:
        python-env-test:
          build: .
          container_name: python-env-a # 컨테이너 이름 설정
          ports:
            - "5010:5000"  # A 사용자는 http://localhost:5010 접근
          volumes:
            - /mnt/c/Users/user/shared_data:/app/shared_data # NAS mount
            - /home/<사용자명>/workspace/python-docker:/app # 개인 작업 디렉토리 설정
          working_dir: /app
          command: tail -f /dev/null
          runtime: nvidia
          environment:
            - NVIDIA_VISIBLE_DEVICES=all
      ```
    - 학습 또는 추론 코드는 `/server/` 경로 아래에 위치
    ```bash
    docker compose up -d --build 
    ```

4. **시뮬레이터는 A에서 로컬로 실행**
    - `client/app.py`를 실행하여 시뮬레이터와 통신

5. **작업 완료 후 PR(Pull Request) 생성**
    - 본인 브랜치를 `main` 브랜치로 PR 요청
    - 리뷰 승인 후 `main`에 병합
---

## 📎 도커 명령어

```bash
# 컨테이너 확인
docker ps

# 컨테이너 접속
docker exec -it python-env-a bash

# 컨테이너 중지
docker-compose -f docker-compose.yml down
```