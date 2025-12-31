# 🏗️ AI Mass Designer: 텍스트 기반 건축 매스 생성기

**AI Mass Designer**는 LLM(GPT-4o)과 Graph-GAN(Generative Adversarial Network)을 결합하여, 사용자의 자연어 묘사를 3차원 건축 매스로 자동 생성하는 AI 솔루션입니다.  
"7층 규모의 오피스를 설계해줘, 1층은 필로티 구조야"와 같은 텍스트 입력을 해석하여, 건축적 제약조건(FAR, 프로그램 비율)을 만족하는 복셀 단위의 3D 모델을 제안합니다.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red?logo=pytorch)
![OpenAI](https://img.shields.io/badge/AI-GPT--4o-green?logo=openai)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit)

---

## 주요 기능 (Key Features)

### 1. 자연어 기반 설계 조건 생성 (Text-to-Graph)
* **LLM 에이전트**: `make_local.py`의 OpenAI API 연동을 통해 사용자의 건축 요구사항을 해석합니다.
* **자동 라벨링 및 구조화**: 층수, 코어(계단/엘리베이터) 위치, 오피스 배치, 돌출(테라스) 등의 텍스트를 파싱하여 `Local Graph`(노드 및 연결 관계) JSON 데이터로 변환합니다.

### 2. Graph-GAN 기반 매스 생성 (Generative Core)
* **복셀 생성 모델**: 학습된 Generator(`Model/models.py`)가 입력된 그래프 제약 조건을 바탕으로 3D 공간상의 복셀 배치를 생성합니다.
* **Gumbel Softmax**: `util.py`에 구현된 미분 가능한 샘플링을 통해 이산적인(Discrete) 복셀 타입을 결정합니다.
* **제약 조건 최적화**: `util_eval.py`를 통해 연결성(Connectivity)과 용적률(FAR)을 평가하고, 가장 적합한 결과물을 선별(`evaluate_best_of_n`)합니다.

### 3. 건축 프로그램 로직 및 후처리 (Rule-based Logic)
* **프로그램 자동 할당**: `Lobby`, `Office`, `core` 3가지 건축 프로그램을 기능적 위계에 맞춰 배치합니다.
* **코어 통합 알고리즘**: 생성된 결과물에서 흩어진 코어(계단실 등)를 수직으로 정렬하거나 불필요한 노이즈를 제거하는 후처리 로직(`postprocess_core_labels`)이 적용됩니다.

### 4. 3D 시각화 및 대시보드 (Visualization)
* **3D 복셀 렌더링**: `test.py`를 통해 생성된 복셀 데이터를 Matplotlib으로 3차원 시각화합니다.
* **Streamlit 앱**: `app.py`를 통해 웹 인터페이스에서 로고와 함께 시스템을 구동할 수 있는 기초 환경을 제공합니다.

---


<img width="1291" height="386" alt="image" src="https://github.com/user-attachments/assets/2e8750c0-dc00-4bb2-8cf1-c633e77ce143" />
<img width="1256" height="606" alt="화면 캡처 2025-08-10 205546" src="https://github.com/user-attachments/assets/5fcec2ba-d164-4bfd-88e8-4e268ddf5209" />


1. '.env' 파일을 생성하고 'OPENAI_API_KEY' 변수에 자신의 GPT API키를 입력합니다.
2.  run_massDesigner 파일을 클릭해서 MassDesigner 웹 프로그램을 실행합니다
