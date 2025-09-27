# WebVTT 자막 생성 파이프라인

Gemini AI와 wav2vec2를 사용하여 오디오 파일과 텍스트 대본으로부터 고품질 WebVTT 자막 파일을 생성하는 도구입니다.

## 📋 개요

이 파이프라인은 논리적인 순서로 다음 과정을 거칩니다:

1. **스텝 1**: Gemini AI를 사용한 지능적 텍스트 분할 (의미 단위, 2줄 자막)
2. **스텝 2**: wav2vec2를 사용한 시간 정렬
3. **스텝 3**: WebVTT 형식으로 변환

각 스텝은 독립적으로 실행 가능하며, 읽기 편한 고품질 자막을 생성합니다.

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
# 가상환경 생성 (이미 있다면 생략)
python3 -m venv venv
source venv/bin/activate

# 필요한 패키지 설치
pip install torch torchaudio transformers librosa soundfile
pip install google-generativeai python-dotenv
```

### 2. 환경변수 설정

`.env` 파일을 생성하고 Gemini API 키를 설정하세요:

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 3. 의존성 확인

```bash
python pipeline.py check
```

## 📁 파일 구조

```
podcast/
├── README.md                    # 이 파일
├── .env                        # API 키 (생성 필요)
├── podcast1.mp3                # 오디오 파일
├── script_clean.txt            # 정제된 대본 텍스트
├── pipeline.py                 # 메인 파이프라인 관리자
├── step1_gemini_text_split.py  # 1단계: Gemini 텍스트 분할
├── step2_wav2vec2_timing.py    # 2단계: wav2vec2 시간 정렬
├── step3_generate_vtt.py       # 3단계: WebVTT 생성
├── out/                        # 결과물 디렉토리
│   ├── step1_subtitles.json    # 1단계 결과 (분할된 자막)
│   ├── step2_timed_subtitles.json # 2단계 결과 (시간 정보 포함)
│   └── podcast.vtt             # 최종 WebVTT 파일
├── archive/                    # 기존 시스템 결과물
└── old_system/                 # 기존 시스템 스크립트들
```

## 🚀 사용법

### 전체 파이프라인 실행

```bash
# 모든 스텝을 순차적으로 실행
python pipeline.py all
```

### 개별 스텝 실행

```bash
# 스텝 1: 텍스트 분할 (Gemini AI 사용)
python pipeline.py step --step-num 1

# 스텝 2: 시간 정렬 (wav2vec2 사용)
python pipeline.py step --step-num 2

# 스텝 3: WebVTT 생성
python pipeline.py step --step-num 3
```

### 특정 스텝부터 실행

```bash
# 스텝 2부터 끝까지 실행
python pipeline.py from --from-step 2

# 스텝 1부터 끝까지 실행 (전체 파이프라인과 동일)
python pipeline.py from --from-step 1
```

### 현재 상태 확인

```bash
# 각 스텝의 완료 상태와 진행률 확인
python pipeline.py status
```

### 실패한 항목 재시도

```bash
# 스텝 1에서 실패한 항목들만 재처리
python pipeline.py retry1
```

## 📊 스텝별 상세 설명

### 스텝 0: 오디오-텍스트 정렬

**입력**:
- `podcast1.mp3` (오디오 파일)
- `script_clean.txt` (대본 텍스트)

**출력**:
- `alignment_result_fixed.json` (정렬 결과)

**설명**: wav2vec2 모델을 사용하여 오디오와 텍스트를 시간적으로 정렬합니다.

```bash
# 직접 실행 (고급 사용자용)
python wav2vec2_alignment.py --audio podcast1.mp3 --text script_clean.txt --output alignment_result_fixed.json
```

### 스텝 1: 자막 분할

**입력**:
- `alignment_result_fixed.json`

**출력**:
- `step1_split_result.json` (분할된 자막)
- `step1_progress.json` (진행상황)

**설명**: Gemini AI를 사용하여 긴 텍스트를 자막에 적합한 길이로 지능적으로 분할합니다.

**특징**:
- 의미 단위로 분할
- 최대 2줄, 각 줄 최대 25글자
- AI 결과 검증 및 fallback 처리
- 진행상황 저장으로 중단 시 이어서 처리 가능

```bash
# 직접 실행 (고급 사용자용)
python step1_split_subtitles.py --input alignment_result_fixed.json --output step1_split_result.json --progress step1_progress.json

# 실패한 항목만 재처리
python step1_retry_failed.py --progress step1_progress.json --original alignment_result_fixed.json --output step1_split_result.json
```

### 스텝 2: WebVTT 생성

**입력**:
- `step1_split_result.json`

**출력**:
- `podcast1.vtt` (최종 WebVTT 파일)

**설명**: 분할된 자막을 표준 WebVTT 형식으로 변환합니다.

```bash
# 직접 실행 (고급 사용자용)
python step2_generate_webvtt.py --input step1_split_result.json --output podcast1.vtt --validate
```

## ⚙️ 고급 옵션

### 자막 분할 옵션 조정

```bash
# 한 줄 최대 20글자, 최대 3줄로 설정
python pipeline.py step --step-num 1 --extra-args --max-chars 20 --max-lines 3
```

### 진행상황 파일에서 WebVTT 생성

스텝 1이 완료되지 않았더라도 중간 결과로 WebVTT를 생성할 수 있습니다:

```bash
python step2_generate_webvtt.py --input step1_progress.json --output temp_podcast.vtt
```

## 🐛 문제 해결

### 1. API 할당량 초과

```
Error: 429 You exceeded your current quota
```

**해결법**:
- 잠시 기다린 후 재시도
- `python pipeline.py retry1` 명령으로 실패한 부분만 재처리

### 2. 의존성 오류

```
ModuleNotFoundError: No module named 'torch'
```

**해결법**:
```bash
python pipeline.py check  # 의존성 확인
pip install torch torchaudio transformers librosa soundfile google-generativeai python-dotenv
```

### 3. 가상환경 문제

파이프라인이 자동으로 `venv/bin/python`을 찾아 사용합니다. 가상환경이 활성화되지 않았어도 작동합니다.

### 4. 진행상황 복구

스텝 1 실행 중 중단된 경우:

```bash
# 현재 상태 확인
python pipeline.py status

# 실패한 부분만 재시도
python pipeline.py retry1

# 또는 처음부터 다시 시작
python pipeline.py step --step-num 1
```

## 📋 출력 파일 설명

- **`alignment_result_fixed.json`**: 원본 오디오-텍스트 정렬 결과
- **`step1_progress.json`**: 스텝 1 진행상황 (중간 결과 포함)
- **`step1_split_result.json`**: 최종 분할된 자막 데이터
- **`podcast1.vtt`**: 완성된 WebVTT 자막 파일

## 💡 팁

1. **큰 파일 처리**: 긴 오디오의 경우 스텝 1에서 시간이 오래 걸릴 수 있습니다. `Ctrl+C`로 중단 후 `retry1`로 이어서 처리하세요.

2. **품질 확인**: 스텝 2에서 `--validate` 옵션을 사용하여 생성된 WebVTT의 유효성을 검사하세요.

3. **API 효율성**: Gemini API는 분당 요청 제한이 있으므로, 대량 처리 시 재시도 기능을 활용하세요.

4. **커스터마이징**: 각 스크립트는 독립적으로 실행 가능하므로, 필요에 따라 매개변수를 조정하여 사용하세요.

## 🔍 모니터링

실행 중 진행상황을 모니터링하려면:

```bash
# 다른 터미널에서
python pipeline.py status

# 또는 진행상황 파일을 직접 확인
cat step1_progress.json | jq .
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:

1. `python pipeline.py check` - 의존성 확인
2. `python pipeline.py status` - 현재 상태 확인
3. 로그 메시지 확인
4. `.env` 파일의 API 키 확인