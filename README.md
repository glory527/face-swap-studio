본 프로젝트는 오픈소스 관례 및 GitHub 용량 제한(파일당 100MB)을 준수하기 위해 대용량 모델 바이너리 파일을 저장소에 포함하지 않습니다.

애플리케이션 구동을 위해 아래 모델 파일들을 직접 다운로드하여 models/ 폴더 내에 배치해 주세요.

👨Face Swap Model
파일명: inswapper_128.onnx (InsightFace 공식 제공 모델)

🪜Upscale Models (Real-ESRGAN)
파일명: realesrgan_x4plus.pth
파일명: realesrgan_x4plus_anime_6B.pth

(참고: 외부 런타임 구동을 위해 필요한 가중치 파일로, 공식 저장소 릴리즈 페이지 등에서 다운로드 가능합니다.)

---

🚀 시작하기 (How to Run)
1) 환경 구축 (Conda 환경 이용 시)
Bash
conda env create -f environment.yml
conda activate faceswap
2) 환경 구축 (Pip 이용 시)
Bash
pip install -r requirements.txt
3) 애플리케이션 실행
Bash
python app.py

---

# 🎭 FaceSwap Studio Pro (VRAM Stabilization Version)

InsightFace 모델을 기반으로 한 고성능 이미지 및 비디오/GIF 페이스 스왑(Face Swap) 단일 실행 애플리케이션입니다.  
배포 편의성을 위해 **Gradio 기반의 단일 파일 구조(`app.py`)**를 채택하면서도, 내부적으로는 백엔드 아키텍처 원칙을 준수하여 계층 분리 및 자원 관리를 최적화했습니다.

---

## ✨ 핵심 기능 및 기술적 차별점 (Backend Engineering)

1. **관심사 분리 및 계층화 구조 (Layered Architecture)**
   * 프로토타입의 스파게티 코드를 지양하고, 단일 파일 내부를 환경 설정(`AppConfig`), AI 추론 엔진(`FaceSwapCore`), 미디어 가공 레이어(`MediaProcessor`), 이벤트 기반 GUI(`Gradio UI`)로 계층화하여 유지보수성을 극대화했습니다.

2. **LAB 색 공간 기반 Auto-Color Matching**
   * 단순히 얼굴 영역을 이질적으로 이어 붙이는 방식에서 벗어나, 타겟 이미지의 LAB 색 공간 내부 채도 및 명도 평균/표준편차를 분석하여 스왑된 얼굴의 톤을 목 전반까지 자연스럽게 매칭하는 `ColorAdjuster` 알고리즘을 구현했습니다.

3. **VRAM 누수 최적화 및 가비지 컬렉션 제어**
   * 대용량 비디오 및 GIF 프레임 추론 시 발생하는 ONNX Runtime과 인코딩 버퍼의 메모리 축적 문제를 해결하기 위해, 15프레임 주기마다 `torch.cuda.empty_cache()` 및 `gc.collect()`를 강제 수행하여 VRAM 데드락을 방지했습니다.

4. **프레임 간 떨림 방지 (BBox Smoothing)**
   * 비디오 프레임 단위로 얼굴을 재탐지할 때 생기는 바운딩 박스의 경계면 흔들림 현상을 해결하고자, 이전 프레임 좌표와의 선형 보간 알고리즘(`alpha=0.55`)을 적용하여 안정적인 영상 결과물을 도출합니다.

5. **의존성 경량화 (Dependency Refactoring)**
   * 초기 가상환경 세팅(`environment.yml`)에 포함되어 있던 무겁고 불안정한 외부 AI 업스케일러 패키지(`basicsr`, `facexlib`)를 걷어내고, OpenCV 큐빅 보간법(`INTER_CUBIC`) 기반의 최적화 파이프라인으로 내부 로직을 커스텀 빌드하여 런타임 안정성을 확보하고 불필요한 의존성을 제거했습니다.


---

## ⚠️ 면책 조항 (Disclaimer)
* 본 프로젝트는 학습 및 개인 포트폴리오 목적으로 제작된 오픈소스 소프트웨어입니다.
* 본 프로그램을 악의적인 목적(AI 생성물 사기, 타인의 명예훼손, 범죄 이용 등)으로 사용함으로써 발생하는 모든 법적 책임은 사용자 본인에게 있습니다.