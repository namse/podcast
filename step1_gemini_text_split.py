#!/usr/bin/env python3
"""
1단계: Gemini를 사용한 텍스트 분할
전체 텍스트를 자막에 적합한 의미 단위로 분할
"""

import json
import os
import argparse
import sys
import re
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

class GeminiTextSplitter:
    def __init__(self):
        """
        Gemini API를 사용한 텍스트 분할기 초기화
        """
        # .env 파일에서 환경변수 로드
        load_dotenv()

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

        # Gemini API 설정
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-flash-latest')

    def create_splitting_prompt(self, text: str) -> str:
        """
        Gemini에게 보낼 텍스트 분할 프롬프트 생성
        """
        prompt = f"""
다음 한국어 팟캐스트 대본을 자막에 적합하게 분할해주세요.

## 분할 규칙:
1. 한 자막은 최대 3줄까지 가능
2. 각 줄은 whitespace 제외한 한글만 보았을 때 15-20글자가 적절하며, 25글자를 절대 넘지 않도록 함

## 줄바꿈 규칙:
- **1번 줄바꿈(\\n)**: 같은 자막 내에서 다음 줄로 (최대 3줄 자막 만들기)
- **2번 줄바꿈(\\n\\n)**: 완전히 다른 자막으로 분리

## 출력 형식:
예시:
```
안녕하세요! 남세의 한국어 팟캐스트는
한국어를 공부하시는 분들을 대상으로,

한국어 듣기 공부를 좀 더 재미있게 하실 수 있도록
다양한 정보를 담아 전달해 드리는 방송입니다.
```

## 주의사항:
- 설명은 하지 말고 분할된 텍스트만 출력하세요
- 절대로 줄바꿈 추가 외에는 텍스트 내용은 수정하지 말고 그대로 출력하세요.
- 괄호 같은 특수문자도 그대로 출력합니다.

입력 텍스트:
"{text}"

분할 결과:
"""
        return prompt

    def create_refine_prompt(self) -> str:
        """
        검토 및 개선을 위한 간단한 후속 요청
        """
        return "적절했는지 검토하고 refine 해주세요. refine 설명 없이 분할된 텍스트만 출력하세요."

    def normalize_text(self, text: str) -> str:
        """
        텍스트 정규화: 공백, 줄바꿈 제거하여 순수 내용만 비교
        """
        # 모든 공백문자와 줄바꿈 제거
        normalized = re.sub(r'\s+', '', text)
        return normalized.strip()

    def validate_content_integrity(self, original_text: str, split_texts: List[str]) -> bool:
        """
        원본 텍스트와 분할 결과의 내용 일치성 검증
        """
        # 원본 텍스트 정규화
        original_normalized = self.normalize_text(original_text)

        # 분할된 텍스트들을 연결하여 정규화
        combined_split = ''.join(split_texts)
        split_normalized = self.normalize_text(combined_split)

        # 내용 일치성 검사
        is_match = original_normalized == split_normalized

        if not is_match:
            print(f"❌ 내용 불일치 - 원본: {len(original_normalized)}자, 분할: {len(split_normalized)}자")

            # 디버깅을 위한 상세 출력
            debug_file = "debug_content_mismatch.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== 원본 텍스트 ===\n")
                f.write(original_text + "\n\n")
                f.write("=== 분할 결과 (연결) ===\n")
                f.write('\n'.join(split_texts) + "\n\n")
                f.write("=== 정규화된 원본 ===\n")
                f.write(original_normalized + "\n\n")
                f.write("=== 정규화된 분할 ===\n")
                f.write(split_normalized + "\n\n")

                # 차이점 찾기
                min_len = min(len(original_normalized), len(split_normalized))
                for i in range(min_len):
                    if original_normalized[i] != split_normalized[i]:
                        f.write(f"=== 첫 차이점 위치: {i} ===\n")
                        start = max(0, i - 50)
                        end = min(len(original_normalized), i + 50)
                        f.write(f"원본 주변: {original_normalized[start:end]}\n")
                        end_split = min(len(split_normalized), i + 50)
                        f.write(f"분할 주변: {split_normalized[start:end_split]}\n")
                        break

            print(f"디버그 정보 저장: {debug_file}")

        return is_match

    def split_and_refine_text(self, text: str) -> List[str]:
        """
        Gemini를 사용하여 텍스트를 분할하고 검토/개선하는 2단계 처리 (대화 형식)
        """
        try:
            # 1단계: 초기 분할
            prompt = self.create_splitting_prompt(text)
            chat = self.model.start_chat()
            response = chat.send_message(prompt)
            initial_result = response.text.strip()

            # 2단계: 검토 및 개선
            refine_prompt = self.create_refine_prompt()
            refine_response = chat.send_message(refine_prompt)
            refined_result = refine_response.text.strip()

            # 2번 줄바꿈(\n\n)으로 자막 분할
            split_texts = [part.strip() for part in refined_result.split('\n\n') if part.strip()]

            # 분할 결과가 없으면 에러 처리
            if not split_texts:
                print(f"❌ Gemini 분할 실패")
                sys.exit(1)

            # 내용 일치성 검증
            if not self.validate_content_integrity(text, split_texts):
                print(f"❌ 내용 불일치")
                sys.exit(1)

            # 글자 수 검증
            for i, text_part in enumerate(split_texts):
                lines = text_part.split('\n')
                if len(lines) > 3:
                    print(f"❌ 3줄 초과: {len(lines)}줄")
                    print(f"문제 자막 #{i+1}: {text_part}")
                    sys.exit(1)

                for j, line in enumerate(lines):
                    if len(line) > 25:
                        print(f"❌ 25글자 초과: {len(line)}글자")
                        print(f"문제 자막 #{i+1}, 줄 #{j+1}: '{line}'")
                        sys.exit(1)

            return split_texts

        except Exception as e:
            print(f"❌ API 오류: {e}")
            sys.exit(1)


    def process_text_file(self, input_path: str, output_path: str):
        """
        텍스트 파일을 줄별로 처리하여 분할된 자막 생성
        """
        # 출력 디렉토리 생성
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        print(f"처리: {len(lines)}줄 (4줄씩 배치)")

        # 4줄씩 묶어서 처리
        all_subtitles = []
        batch_size = 4

        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(lines) + batch_size - 1) // batch_size

            print(f"배치 {batch_num}/{total_batches}")

            # 4줄을 하나의 텍스트로 결합
            batch_text = '\n'.join(batch_lines)

            # 배치를 2단계 처리 (분할 → 검토/개선)
            batch_subtitles = self.split_and_refine_text(batch_text)
            all_subtitles.extend(batch_subtitles)

        print(f"완료: {len(all_subtitles)}개 자막")

        # 결과를 JSON으로 저장
        result_data = {
            "total_subtitles": len(all_subtitles),
            "subtitles": []
        }

        for i, subtitle_text in enumerate(all_subtitles):
            result_data["subtitles"].append({
                "index": i + 1,
                "text": subtitle_text,
                "char_count": len(subtitle_text),
                "line_count": len(subtitle_text.split('\n'))
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        # 전체 내용 검증
        original_all = '\n'.join(lines)
        if not self.validate_content_integrity(original_all, all_subtitles):
            print("❌ 전체 내용 불일치")
            sys.exit(1)

        # 통계
        char_counts = [len(sub) for sub in all_subtitles]
        print(f"저장: {output_path}")
        print(f"통계: 평균 {sum(char_counts) / len(char_counts):.1f}글자, 최대 {max(char_counts)}글자")

def main():
    parser = argparse.ArgumentParser(description="1단계: Gemini를 사용한 텍스트 분할")
    parser.add_argument("--input", required=True, help="입력 텍스트 파일")
    parser.add_argument("--output", default="out/step1_subtitles.json", help="출력 JSON 파일")

    args = parser.parse_args()

    # 분할기 초기화
    splitter = GeminiTextSplitter()

    # 처리 실행
    splitter.process_text_file(args.input, args.output)

if __name__ == "__main__":
    main()