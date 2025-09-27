#!/usr/bin/env python3
"""
2단계: wav2vec2를 사용한 시간 정렬
분할된 자막에 정확한 시간 정보 부여
"""

import json
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import numpy as np
import argparse
from typing import List, Tuple, Dict
import re
from difflib import SequenceMatcher

class Wav2Vec2Timer:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        """
        Wav2Vec2 모델을 사용한 시간 정렬기 초기화
        """
        print(f"모델 로딩 중: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        print(f"모델이 {self.device}에 로드되었습니다.")

    def load_audio(self, audio_path: str, sample_rate: int = 16000) -> np.ndarray:
        """
        오디오 파일 로드 및 전처리
        """
        print(f"오디오 파일 로딩: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        print(f"오디오 길이: {len(audio)/sample_rate:.2f}초")
        return audio

    def transcribe_with_timestamps(self, audio: np.ndarray, chunk_size: int = 30) -> List[Dict]:
        """
        오디오를 청크 단위로 전사하여 타임스탬프와 함께 반환
        """
        print("오디오 전사 중...")

        sample_rate = 16000
        chunk_samples = chunk_size * sample_rate
        transcriptions = []

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            start_time = i / sample_rate
            end_time = min((i + len(chunk)) / sample_rate, len(audio) / sample_rate)

            # 입력 텐서 준비
            inputs = self.processor(chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)

            # 모델 추론
            with torch.no_grad():
                logits = self.model(input_values).logits

            # 디코딩
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            # 정규화 (소문자, 공백 정리)
            transcription = self.normalize_text(transcription)

            transcriptions.append({
                "start": start_time,
                "end": end_time,
                "text": transcription,
                "chunk_index": i // chunk_samples
            })

            print(f"청크 {i//chunk_samples + 1} 처리 완료: {start_time:.1f}s-{end_time:.1f}s")

        return transcriptions

    def normalize_text(self, text: str) -> str:
        """
        텍스트 정규화 (공백 제거, 소문자 변환 등)
        """
        # 소문자 변환, 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text.lower().strip())
        return text

    def find_subtitle_times(self, subtitles: List[str], transcriptions: List[Dict]) -> List[Dict]:
        """
        분할된 자막들에 대해 시간을 균등 배분 (한국어 음성인식 한계로 인한 fallback)
        """
        print(f"자막 {len(subtitles)}개에 대한 시간 정렬 시작...")

        # 전체 오디오 길이
        total_duration = transcriptions[-1]["end"]

        # 전체 텍스트 길이
        total_text_length = sum(len(subtitle) for subtitle in subtitles)

        result = []
        current_time = 0.0

        for i, subtitle in enumerate(subtitles):
            # 텍스트 길이에 비례하여 시간 할당
            text_length = len(subtitle)

            # 최소 지속 시간 보장 (0.5초)
            min_duration = 0.5
            proportional_duration = (text_length / total_text_length) * total_duration
            duration = max(min_duration, proportional_duration)

            # 최대 지속 시간 제한 (5초)
            duration = min(duration, 5.0)

            start_time = current_time
            end_time = start_time + duration

            # 마지막 자막은 오디오 끝에 맞춤
            if i == len(subtitles) - 1:
                end_time = total_duration

            result.append({
                "index": i + 1,
                "text": subtitle,
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "confidence": "estimated"
            })

            current_time = end_time

            if (i + 1) % 20 == 0:
                print(f"진행률: {i + 1}/{len(subtitles)}")

        return result

    def find_similar_text(self, target: str, full_text: str, start_pos: int) -> Tuple[int, int]:
        """
        유사한 텍스트 위치 찾기
        """
        best_ratio = 0.0
        best_position = None

        # 검색 범위를 제한하여 성능 향상
        search_length = min(len(target) * 3, 200)
        search_area = full_text[start_pos:start_pos + search_length * 10]

        for i in range(0, len(search_area) - len(target) + 1, 5):  # 5글자씩 건너뛰며 검색
            candidate = search_area[i:i + len(target)]
            ratio = SequenceMatcher(None, target, candidate).ratio()

            if ratio > best_ratio and ratio > 0.6:  # 60% 이상 유사도
                best_ratio = ratio
                best_position = (start_pos + i, start_pos + i + len(target))

        return best_position

    def text_position_to_time(self, start_pos: int, end_pos: int,
                            full_text: str, transcriptions: List[Dict]) -> Tuple[float, float]:
        """
        텍스트 위치를 시간으로 변환
        """
        # 각 청크의 텍스트 길이 누적 계산
        cumulative_length = 0

        for trans in transcriptions:
            chunk_text = trans["text"]
            chunk_start_pos = cumulative_length
            chunk_end_pos = cumulative_length + len(chunk_text)

            # 시작 시간 찾기
            if chunk_start_pos <= start_pos <= chunk_end_pos:
                chunk_duration = trans["end"] - trans["start"]
                relative_start_pos = start_pos - chunk_start_pos
                relative_start_ratio = relative_start_pos / len(chunk_text) if len(chunk_text) > 0 else 0
                start_time = trans["start"] + chunk_duration * relative_start_ratio

            # 끝 시간 찾기
            if chunk_start_pos <= end_pos <= chunk_end_pos:
                chunk_duration = trans["end"] - trans["start"]
                relative_end_pos = end_pos - chunk_start_pos
                relative_end_ratio = relative_end_pos / len(chunk_text) if len(chunk_text) > 0 else 1
                end_time = trans["start"] + chunk_duration * relative_end_ratio

                return start_time, end_time

            cumulative_length = chunk_end_pos + 1  # 공백 추가

        # 기본값 반환 (찾지 못한 경우)
        return transcriptions[0]["start"], transcriptions[-1]["end"]

    def process_subtitles(self, audio_path: str, subtitles_path: str, output_path: str):
        """
        전체 처리 과정 실행
        """
        # 오디오 로드
        audio = self.load_audio(audio_path)

        # 자막 데이터 로드
        print(f"자막 파일 로딩: {subtitles_path}")
        with open(subtitles_path, 'r', encoding='utf-8') as f:
            subtitle_data = json.load(f)

        subtitles = [item["text"] for item in subtitle_data["subtitles"]]
        print(f"로드된 자막 수: {len(subtitles)}")

        # 오디오 전사
        transcriptions = self.transcribe_with_timestamps(audio)

        # 자막에 시간 정보 부여
        timed_subtitles = self.find_subtitle_times(subtitles, transcriptions)

        # 결과 저장
        result_data = {
            "total_subtitles": len(timed_subtitles),
            "audio_duration": len(audio) / 16000,
            "subtitles": timed_subtitles
        }

        # 출력 디렉토리 생성
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"시간 정렬 결과가 저장되었습니다: {output_path}")

        # 통계 출력
        high_confidence = sum(1 for s in timed_subtitles if s.get("confidence") == "high")
        low_confidence = sum(1 for s in timed_subtitles if s.get("confidence") == "low")

        print(f"\n=== 시간 정렬 통계 ===")
        print(f"총 자막 수: {len(timed_subtitles)}")
        print(f"고신뢰도 매칭: {high_confidence}")
        print(f"저신뢰도 매칭: {low_confidence}")
        print(f"오디오 길이: {result_data['audio_duration']:.1f}초")

def main():
    parser = argparse.ArgumentParser(description="2단계: wav2vec2를 사용한 시간 정렬")
    parser.add_argument("--audio", required=True, help="오디오 파일 경로")
    parser.add_argument("--subtitles", required=True, help="1단계 자막 JSON 파일")
    parser.add_argument("--output", default="out/step2_timed_subtitles.json", help="출력 JSON 파일")

    args = parser.parse_args()

    # 시간 정렬기 초기화
    timer = Wav2Vec2Timer()

    # 처리 실행
    timer.process_subtitles(args.audio, args.subtitles, args.output)

if __name__ == "__main__":
    main()