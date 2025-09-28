#!/usr/bin/env python3
"""
5단계: Gemini 2.5 Pro를 활용한 이미지 개념 분석
전체 자막 맥락을 고려한 통합적 이미지 개념 제안
"""

import json
import os
import argparse
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

class ImageConceptAnalyzer:
    def __init__(self):
        """
        Gemini 2.5 Pro 이미지 개념 분석기 초기화
        """
        # .env 파일에서 환경변수 로드
        load_dotenv()

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

        # Gemini API 설정
        genai.configure(api_key=api_key)

        # Gemini 2.5 Pro 모델 설정
        self.model = genai.GenerativeModel('models/gemini-2.5-pro')

        print("Gemini 2.5 Pro 모델이 로드되었습니다.")

    def analyze_all_groups_for_image_concepts(self, groups: List[Dict]) -> List[str]:
        """
        전체 자막 그룹을 한번에 분석하여 맥락을 고려한 이미지 개념 제안
        """
        # 전체 자막 정보 구성
        subtitle_info = []
        for group in groups:
            start_time = group['start_time']
            text = group['text']
            subtitle_info.append(f"[{start_time:.1f}s] {text}")

        full_subtitles = "\n".join(subtitle_info)

        analysis_prompt = f"""
각 대사 그룹마다 어떤 이미지를 화면에 보여주면 좋을지 제안해줘.
단, 이미지에는 텍스트를 직접적으로 보여주지 않았으면 좋겠어.
시청자에게 이미지로 모든 내용을 자세하게 설명하려고 하지 않았으면 좋겠어.
그렇다고 너무 은유적으로 표현하는데에 집중하진 않았으면 좋겠어.
또한 캐릭터를 등장시키고, 특히 주인공 캐릭터는 청년 한국 남자캐릭터이며 영상 전반을 그 캐릭터가 이어가도록 해줬으면 좋겠어.

답변은 한국어로 해줘.

전체 자막:
{full_subtitles}

출력 형식:
반드시 다음과 같은 형식으로만 답변해줘. 다른 설명이나 헤더는 포함하지 말고, 정확히 이 형식만 사용해:

GROUP1: [이미지 설명]
GROUP2: [이미지 설명]
...

각 이미지 설명은 '\n' 없이 작성해줘. 절대로 다른 형식이나 추가 설명을 포함하지 마.
"""

        try:
            print("전체 자막 통합 분석 중...")
            response = self.model.generate_content(analysis_prompt)
            analysis_result = response.text.strip()

            print("=== AI 분석 결과 ===")
            print(analysis_result)

            # 결과 파싱
            image_concepts = self.parse_analysis_result(analysis_result, len(groups))

            return image_concepts

        except Exception as e:
            print(f"이미지 개념 분석 실패: {e}")
            raise e

    def parse_analysis_result(self, analysis_result: str, expected_count: int) -> List[str]:
        """
        AI 분석 결과를 파싱하여 개별 이미지 개념 추출
        """
        concepts = []
        lines = analysis_result.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('GROUP') and ':' in line:
                # GROUP1: [이미지 설명] 형식에서 설명 부분 추출
                concept = line.split(':', 1)[1].strip()
                concepts.append(concept)

        # 파싱 결과 검증
        if len(concepts) != expected_count:
            print(f"디버그: 파싱된 개념들:")
            for i, concept in enumerate(concepts):
                print(f"  {i+1}: {concept}")
            raise ValueError(f"파싱된 이미지 개념 수({len(concepts)})가 예상 그룹 수({expected_count})와 일치하지 않습니다.")

        return concepts

    def process_subtitle_groups(self, groups_file_path: str):
        """
        자막 그룹 파일을 처리하여 통합적 이미지 개념 분석
        """
        print(f"자막 그룹 파일 로딩: {groups_file_path}")

        # step4 결과 파일 읽기
        with open(groups_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        groups = []
        for line in lines:
            line = line.strip()
            if line and '\t' in line:
                start_ms, text = line.split('\t', 1)
                groups.append({
                    'start_ms': int(start_ms),
                    'start_time': int(start_ms) / 1000.0,
                    'text': text
                })

        print(f"로드된 그룹 수: {len(groups)}")

        # 전체 그룹 통합 분석
        image_concepts = self.analyze_all_groups_for_image_concepts(groups)

        # 출력 라인 생성
        output_lines = []
        for i, group in enumerate(groups):
            if i < len(image_concepts):
                concept = image_concepts[i]
                # 불필요한 줄바꿈 제거
                concept = " ".join(concept.split())
                output_line = f"{group['start_ms']}\t{concept}"
                output_lines.append(output_line)

        # 결과를 파일로 저장
        output_file = "out/step5_image_concepts.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

        print(f"\n=== 이미지 개념 분석 완료 ===")
        print(f"총 분석된 그룹 수: {len(groups)}")
        print(f"결과 저장 경로: {output_file}")

        # 출력 결과 표시
        print(f"\n=== 출력 결과 ===")
        for line in output_lines:
            print(line)

        return output_lines

def main():
    parser = argparse.ArgumentParser(description="5단계: Gemini 2.5 Pro를 활용한 이미지 개념 분석")
    parser.add_argument("--input", default="out/step4_subtitle_groups.txt", help="4단계 자막 그룹 파일")

    args = parser.parse_args()

    # 이미지 개념 분석기 초기화
    analyzer = ImageConceptAnalyzer()

    # 처리 실행
    analyzer.process_subtitle_groups(args.input)

if __name__ == "__main__":
    main()