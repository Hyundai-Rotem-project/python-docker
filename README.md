# python-docker
docker for CI/CD

# 프로젝트 환경 설정

이 프로젝트는 도커를 사용하여 동일한 개발 환경을 설정합니다. 아래의 지침에 따라 도커 환경을 설정하고 애플리케이션을 실행할 수 있습니다.

## 1. 프로젝트 클론
먼저, 이 프로젝트를 클론하세요:
```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```
## 2. 도커 설치
* 윈도우 : Docker Desktop 다운로드 및 설치.
* 리눅스 : 아래 명령어 입력
```bash
sudo apt update
sudo apt install docker.io docker-compose
```
* 도커 버전 확인 : 28.0.1
```bash
docker --version
```
## 3. 도커 환경 실행
1. 도커 파일이 있는 디렉토리로 이동
```bash
cd python-docker
```
2. 도커 이미지 빌드
```bash
docker-compose build
```
3. 도커 컨테이너 실행
```bash
docker-compose up
```
## 4. 애플리케이션 확인
애플리케이션은 http://localhost:5000에서 실행됩니다.

## 5. 도커환경 종료
```bash
docker-compose down
```
