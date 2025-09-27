#!/usr/bin/env python3
"""
1단계: Gemini를 사용한 텍스트 분할
전체 텍스트를 자막에 적합한 의미 단위로 분할
"""

import json
import os
import argparse
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

        print("Gemini Flash Latest 모델이 로드되었습니다.")

    def create_splitting_prompt(self, text: str) -> str:
        """
        Gemini에게 보낼 텍스트 분할 프롬프트 생성
        """
        prompt = f"""
다음 한국어 팟캐스트 대본을 WebVTT 자막에 적합하게 분할해주세요.

## 분할 규칙:
1. 한 자막은 최대 2줄까지 가능
2. 각 줄은 15-20글자가 적절하며, 25글자를 절대 넘지 않도록 함
3. 의미가 완결되는 어구나 절을 기준으로 분할
4. 문장의 자연스러운 호흡과 쉼표, 조사, 접속어 등을 고려
5. 긴 텍스트는 2줄로 나누어 더 읽기 좋게 만들어주세요

## 줄바꿈 규칙:
- **1번 줄바꿈(\\n)**: 같은 자막 내에서 다음 줄로 (2줄 자막 만들기)
- **2번 줄바꿈(\\n\\n)**: 완전히 다른 자막으로 분리

## 출력 형식:
예시:
```
안녕하세요.

남세의 한국어 팟캐스트는
한국어를 공부하시는 분들을 대상으로,

한국어 듣기 공부를 좀 더
재미있게 하실 수 있도록

다양한 정보를 담아
전달해 드리는 방송입니다.
```

## 주의사항:
- 설명은 하지 말고 분할된 텍스트만 출력하세요
- 각 자막이 독립적으로 이해 가능하도록 분할하세요
- 긴 내용은 적극적으로 2줄로 분할하여 읽기 편하게 만드세요

입력 텍스트:
"{text}"

분할 결과:
"""
        return prompt

    def split_text_with_gemini(self, text: str) -> List[str]:
        """
        Gemini를 사용하여 텍스트를 지능적으로 분할
        """
        try:
            prompt = self.create_splitting_prompt(text)
            response = self.model.generate_content(prompt)

            # 응답에서 분할된 텍스트 추출
            result = response.text.strip()
            print(f"Gemini 응답: {result[:100]}...")

            # 2번 줄바꿈(\n\n)으로 자막 분할
            split_texts = [part.strip() for part in result.split('\n\n') if part.strip()]

            # 분할 결과가 없으면 원본을 적절히 분할
            if not split_texts:
                print("Gemini 분할 실패, 기본 분할 적용")
                return self.fallback_split(text)

            # 글자 수 검증
            validated_texts = []
            for text_part in split_texts:
                lines = text_part.split('\n')
                if len(lines) <= 2 and all(len(line) <= 25 for line in lines):
                    validated_texts.append(text_part)
                else:
                    print(f"글자 수 초과 텍스트 재분할: {text_part[:30]}...")
                    validated_texts.extend(self.fallback_split(text_part))

            return validated_texts

        except Exception as e:
            print(f"Gemini API 오류: {e}")
            print("기본 분할 방식 사용")
            return self.fallback_split(text)

    def fallback_split(self, text: str) -> List[str]:
        """
        Gemini 실패 시 기본 분할 방식
        """
        # 문장 부호로 분할
        import re
        sentences = re.split(r'([.!?。？！])', text)

        result = []
        current = ""

        for part in sentences:
            part = part.strip()
            if not part:
                continue

            if part in '.!?。？！':
                current += part
                if len(current) <= 25:
                    result.append(current)
                    current = ""
                else:
                    # 너무 길면 더 세분화
                    words = current[:-1].split()  # 문장부호 제외
                    temp = ""
                    for word in words:
                        if len(temp + word) <= 20:
                            temp += word + " "
                        else:
                            if temp:
                                result.append(temp.strip() + part)
                            temp = word + " "
                    if temp:
                        result.append(temp.strip() + part)
                    current = ""
            else:
                if len(current + part) <= 20:
                    current += part
                else:
                    if current:
                        result.append(current)
                    current = part

        if current:
            result.append(current)

        return result

    def process_text_file(self, input_path: str, output_path: str):
        """
        텍스트 파일을 처리하여 분할된 자막 생성
        """
        print(f"입력 파일 로딩: {input_path}")

        # 출력 디렉토리 생성
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as f:
            full_text = f.read().strip()

        print(f"전체 텍스트 길이: {len(full_text)}글자")

        # 전체 텍스트를 Gemini로 분할
        print("Gemini로 텍스트 분할 시작...")
        split_subtitles = self.split_text_with_gemini(full_text)

        print(f"분할 완료: {len(split_subtitles)}개 자막 생성")

        # 결과를 JSON으로 저장
        result_data = {
            "total_subtitles": len(split_subtitles),
            "subtitles": []
        }

        for i, subtitle_text in enumerate(split_subtitles):
            result_data["subtitles"].append({
                "index": i + 1,
                "text": subtitle_text,
                "char_count": len(subtitle_text),
                "line_count": len(subtitle_text.split('\n'))
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"분할 결과가 저장되었습니다: {output_path}")

        # 통계 출력
        char_counts = [len(sub) for sub in split_subtitles]
        line_counts = [len(sub.split('\n')) for sub in split_subtitles]

        print(f"\n=== 분할 통계 ===")
        print(f"총 자막 수: {len(split_subtitles)}")
        print(f"평균 글자 수: {sum(char_counts) / len(char_counts):.1f}")
        print(f"최대 글자 수: {max(char_counts)}")
        print(f"최대 줄 수: {max(line_counts)}")
        print(f"25글자 초과 자막: {sum(1 for c in char_counts if c > 25)}")

        # 샘플 출력
        print(f"\n=== 첫 5개 자막 샘플 ===")
        for i, subtitle in enumerate(split_subtitles[:5]):
            print(f"{i+1:2d}. [{len(subtitle):2d}글자] {subtitle}")

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