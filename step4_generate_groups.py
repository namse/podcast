#!/usr/bin/env python3
"""
4단계: Gemini Flash Latest를 활용한 자막 그룹핑
AI 기반 의미론적 그룹핑으로 정확한 자막 구간 분할
"""

import json
import os
import argparse
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

class SubtitleGrouper:
    def __init__(self):
        """
        Gemini 자막 그룹핑 초기화
        """
        # .env 파일에서 환경변수 로드
        load_dotenv()

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")

        # Gemini API 설정
        genai.configure(api_key=api_key)

        # 텍스트 분석용 모델
        self.model = genai.GenerativeModel('models/gemini-flash-latest')

        print("Gemini Flash Latest 모델이 로드되었습니다.")

    def analyze_and_group_subtitles(self, subtitles: List[Dict]) -> List[Dict]:
        """
        전체 자막을 분석하여 의미론적 그룹으로 분할
        """
        # 시간 정보와 함께 자막 리스트 생성
        subtitle_with_time = []
        for i, sub in enumerate(subtitles):
            subtitle_with_time.append(f"{i+1}. [{sub['start']:.1f}s-{sub['end']:.1f}s] {sub['text'].replace(chr(10), ' ')}")

        # 그룹핑 분석 프롬프트
        analysis_prompt = f"""
다음은 한국어 팟캐스트 자막입니다. 이미지 생성을 위한 최적의 그룹핑을 제안해주세요.

전체 자막 ({len(subtitles)}개):
{chr(10).join(subtitle_with_time)}

## 그룹핑 요청:
- 의미적으로 완결되는 주제별로 자막들을 그룹화
- 12-18개 그룹으로 분할 (너무 많지도 적지도 않게)
- 각 그룹은 15-45초 범위로 유지
- 문장 중간에서 끊지 않음

## 출력 형식 (반드시 이 형식을 준수하세요):
그룹1: 1-6 | 팟캐스트 소개 및 주제 발표
그룹2: 7-10 | 주요 개념의 정의와 설명
그룹3: 11-15 | 구체적인 사례와 예시
...

- 반드시 "그룹N: 시작번호-끝번호 | 주제설명" 형식으로만 출력
- 표나 기타 형식 사용 금지
- 각 그룹을 한 줄씩 작성
"""

        try:
            print("자막 그룹핑 분석 중...")
            response = self.model.generate_content(analysis_prompt)
            analysis_result = response.text.strip()

            print("=== AI 그룹핑 결과 ===")
            print(analysis_result)

            # 분석 결과를 파싱하여 그룹 생성
            groups = self.parse_grouping_result(analysis_result, subtitles)

            return groups

        except Exception as e:
            print(f"그룹핑 분석 실패: {e}")
            return []

    def parse_grouping_result(self, analysis_result: str, subtitles: List[Dict]) -> List[Dict]:
        """
        AI 분석 결과를 파싱하여 실제 그룹 데이터로 변환
        """
        groups = []
        lines = analysis_result.split('\n')

        for line in lines:
            line = line.strip()
            if '그룹' in line and ':' in line and '|' in line:
                try:
                    # 파싱: 그룹1: 1-6 | 주제 설명
                    parts = line.split('|')
                    if len(parts) >= 2:
                        # 자막 번호 부분 추출
                        left_part = parts[0].strip()  # "그룹1: 1-6"
                        subtitle_range = left_part.split(':')[1].strip()  # "1-6"

                        # 자막 번호 파싱
                        if '-' in subtitle_range:
                            start_idx, end_idx = map(int, subtitle_range.split('-'))
                            subtitle_indices = list(range(start_idx - 1, end_idx))  # 0-based index
                        else:
                            subtitle_indices = [int(subtitle_range) - 1]

                        # 해당 자막들 추출
                        group_subtitles = []
                        for idx in subtitle_indices:
                            if 0 <= idx < len(subtitles):
                                group_subtitles.append(subtitles[idx])

                        if group_subtitles:
                            # 그룹 데이터 생성
                            start_time = group_subtitles[0]["start"]
                            end_time = group_subtitles[-1]["end"]
                            combined_text = " ".join([s["text"].replace('\n', ' ') for s in group_subtitles])

                            group_data = {
                                "group_id": len(groups) + 1,
                                "start_time": start_time,
                                "end_time": end_time,
                                "duration": end_time - start_time,
                                "subtitle_count": len(group_subtitles),
                                "subtitles": group_subtitles,
                                "combined_text": combined_text,
                                "topic": parts[1].strip(),
                                "subtitle_indices": subtitle_indices
                            }

                            groups.append(group_data)
                            print(f"그룹 {len(groups)}: {start_time:.1f}s-{end_time:.1f}s ({len(group_subtitles)}개) - {parts[1].strip()}")

                except Exception as e:
                    print(f"그룹 파싱 오류: {line} - {e}")
                    continue

        return groups

    def validate_groups(self, groups: List[Dict], subtitles: List[Dict]) -> List[Dict]:
        """
        생성된 그룹들의 유효성 검사 및 조정
        """
        if not groups:
            return []

        validated_groups = []
        covered_indices = set()

        for group in groups:
            # 중복 체크
            group_indices = set(group["subtitle_indices"])
            if group_indices & covered_indices:
                print(f"중복된 자막 인덱스 발견, 건너뜀: {group_indices & covered_indices}")
                continue

            # 너무 짧은 그룹 제외
            if group["duration"] < 3.0:
                print(f"그룹이 너무 짧음, 건너뜀: {group['duration']:.1f}초")
                continue

            covered_indices.update(group_indices)
            validated_groups.append(group)

        # 빠진 자막들이 있으면 추가 그룹 생성
        all_indices = set(range(len(subtitles)))
        missing_indices = all_indices - covered_indices

        if missing_indices:
            print(f"빠진 자막들을 추가 그룹으로 생성: {sorted(missing_indices)}")
            # 연속된 구간들로 분할
            missing_sorted = sorted(missing_indices)
            current_group_indices = []

            for i, idx in enumerate(missing_sorted):
                current_group_indices.append(idx)

                # 다음 인덱스가 연속되지 않거나 마지막 인덱스인 경우 그룹 생성
                if (i == len(missing_sorted) - 1 or
                    missing_sorted[i + 1] != idx + 1):

                    if current_group_indices:
                        missing_subtitles = [subtitles[i] for i in current_group_indices]

                        extra_group = {
                            "group_id": len(validated_groups) + 1,
                            "start_time": missing_subtitles[0]["start"],
                            "end_time": missing_subtitles[-1]["end"],
                            "duration": missing_subtitles[-1]["end"] - missing_subtitles[0]["start"],
                            "subtitle_count": len(missing_subtitles),
                            "subtitles": missing_subtitles,
                            "combined_text": " ".join([s["text"].replace('\n', ' ') for s in missing_subtitles]),
                            "topic": "추가 내용",
                            "subtitle_indices": current_group_indices
                        }
                        validated_groups.append(extra_group)

                    current_group_indices = []

        # 시간 순으로 정렬
        validated_groups.sort(key=lambda x: x["start_time"])

        # 그룹 ID 재할당
        for i, group in enumerate(validated_groups):
            group["group_id"] = i + 1

        return validated_groups

    def process_subtitles(self, subtitles_path: str):
        """
        전체 처리 과정 실행
        """
        # 자막 데이터 로드
        print(f"자막 파일 로딩: {subtitles_path}")
        with open(subtitles_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        subtitles = data["subtitles"]
        print(f"로드된 자막 수: {len(subtitles)}")

        # AI 기반 자막 그룹화
        groups = self.analyze_and_group_subtitles(subtitles)

        if not groups:
            print("그룹핑 실패")
            return

        # 그룹 검증 및 조정
        validated_groups = self.validate_groups(groups, subtitles)

        print(f"\n=== 최종 그룹핑 결과 ===")
        print(f"총 그룹 수: {len(validated_groups)}")

        # 요청된 포맷으로 출력 및 파일 저장: {시작ms}\t{텍스트}\n
        output_lines = []
        print(f"\n=== 출력 결과 ===")
        for group in validated_groups:
            start_ms = int(group["start_time"] * 1000)
            text_without_linebreak = group["combined_text"]
            output_line = f"{start_ms}\t{text_without_linebreak}"
            print(output_line)
            output_lines.append(output_line)

        # 파일로 저장
        output_file = "out/step4_subtitle_groups.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

        print(f"\n결과가 저장되었습니다: {output_file}")

        return validated_groups

def main():
    parser = argparse.ArgumentParser(description="4단계: Gemini를 활용한 자막 그룹핑")
    parser.add_argument("--input", default="out/step2_timed_subtitles.json", help="시간 정렬된 자막 JSON 파일")

    args = parser.parse_args()

    # 자막 그룹핑 초기화
    grouper = SubtitleGrouper()

    # 처리 실행
    grouper.process_subtitles(args.input)

if __name__ == "__main__":
    main()