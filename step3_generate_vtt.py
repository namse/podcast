#!/usr/bin/env python3
"""
3단계: WebVTT 생성
시간 정보가 포함된 자막을 WebVTT 형식으로 변환
"""

import json
import argparse
from typing import List, Dict

class WebVTTGenerator:
    def __init__(self):
        """
        WebVTT 생성기 초기화
        """
        print("WebVTT 생성기가 초기화되었습니다.")

    def format_time(self, seconds: float) -> str:
        """
        초를 WebVTT 시간 형식(HH:MM:SS.mmm)으로 변환
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = seconds % 60

        return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"

    def validate_subtitle_timing(self, subtitles: List[Dict]) -> List[Dict]:
        """
        자막 타이밍 검증 및 수정
        """
        validated = []

        for i, subtitle in enumerate(subtitles):
            start = subtitle["start"]
            end = subtitle["end"]

            # 시작 시간이 끝 시간보다 늦으면 수정
            if start >= end:
                end = start + 0.5  # 최소 0.5초 지속

            # 이전 자막과 겹치지 않도록 조정
            if i > 0 and start < validated[-1]["end"]:
                start = validated[-1]["end"]
                if start >= end:
                    end = start + 0.5

            validated_subtitle = subtitle.copy()
            validated_subtitle["start"] = round(start, 3)
            validated_subtitle["end"] = round(end, 3)

            validated.append(validated_subtitle)

        return validated

    def generate_webvtt(self, subtitles: List[Dict]) -> str:
        """
        자막 리스트를 WebVTT 형식으로 변환
        """
        webvtt_content = ["WEBVTT", ""]

        for subtitle in subtitles:
            # 자막 번호
            webvtt_content.append(str(subtitle["index"]))

            # 시간 정보
            start_time = self.format_time(subtitle["start"])
            end_time = self.format_time(subtitle["end"])
            webvtt_content.append(f"{start_time} --> {end_time}")

            # 자막 텍스트
            webvtt_content.append(subtitle["text"])

            # 빈 줄로 구분
            webvtt_content.append("")

        return "\n".join(webvtt_content)

    def process_timed_subtitles(self, input_path: str, output_path: str, validate: bool = True):
        """
        시간 정보가 포함된 자막 파일을 WebVTT로 변환
        """
        print(f"시간 정렬된 자막 파일 로딩: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        subtitles = data["subtitles"]
        print(f"로드된 자막 수: {len(subtitles)}")

        if validate:
            print("자막 타이밍 검증 중...")
            subtitles = self.validate_subtitle_timing(subtitles)

        # WebVTT 생성
        print("WebVTT 생성 중...")
        webvtt_content = self.generate_webvtt(subtitles)

        # 출력 디렉토리 생성
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(webvtt_content)

        print(f"WebVTT 파일이 저장되었습니다: {output_path}")

        # 통계 출력
        total_duration = subtitles[-1]["end"] if subtitles else 0
        avg_duration = sum(sub["end"] - sub["start"] for sub in subtitles) / len(subtitles) if subtitles else 0

        print(f"\n=== WebVTT 생성 통계 ===")
        print(f"총 자막 수: {len(subtitles)}")
        print(f"전체 길이: {total_duration:.1f}초")
        print(f"평균 자막 길이: {avg_duration:.1f}초")

        # 샘플 출력
        print(f"\n=== 첫 5개 자막 샘플 ===")
        for i, subtitle in enumerate(subtitles[:5]):
            start_time = self.format_time(subtitle["start"])
            end_time = self.format_time(subtitle["end"])
            print(f"{subtitle['index']:2d}. {start_time} --> {end_time}")
            print(f"    {subtitle['text']}")

    def validate_webvtt_file(self, vtt_path: str) -> Dict:
        """
        생성된 WebVTT 파일의 유효성 검사
        """
        print(f"WebVTT 파일 유효성 검사: {vtt_path}")

        with open(vtt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.strip().split('\n')
        issues = []
        stats = {
            "total_subtitles": 0,
            "issues": [],
            "is_valid": True
        }

        # 첫 줄이 WEBVTT인지 확인
        if not lines[0].strip() == "WEBVTT":
            issues.append("파일이 'WEBVTT'로 시작하지 않습니다")
            stats["is_valid"] = False

        # 자막 블록 검사
        i = 2  # WEBVTT 다음 빈 줄 이후부터
        subtitle_count = 0

        while i < len(lines):
            if lines[i].strip().isdigit():  # 자막 번호
                subtitle_count += 1

                # 시간 라인 검사
                if i + 1 < len(lines):
                    time_line = lines[i + 1]
                    if " --> " not in time_line:
                        issues.append(f"자막 {subtitle_count}: 시간 형식 오류")
                        stats["is_valid"] = False

                # 텍스트 라인 검사
                if i + 2 < len(lines):
                    text_line = lines[i + 2]
                    if not text_line.strip():
                        issues.append(f"자막 {subtitle_count}: 텍스트가 비어있음")
                        stats["is_valid"] = False

                i += 4  # 번호, 시간, 텍스트, 빈줄
            else:
                i += 1

        stats["total_subtitles"] = subtitle_count
        stats["issues"] = issues

        if stats["is_valid"]:
            print("✓ WebVTT 파일이 유효합니다")
        else:
            print("✗ WebVTT 파일에 문제가 있습니다:")
            for issue in issues:
                print(f"  - {issue}")

        print(f"총 자막 수: {subtitle_count}")

        return stats

def main():
    parser = argparse.ArgumentParser(description="3단계: WebVTT 생성")
    parser.add_argument("--input", required=True, help="시간 정렬된 자막 JSON 파일")
    parser.add_argument("--output", default="out/podcast.vtt", help="출력 WebVTT 파일")
    parser.add_argument("--validate", action="store_true", help="타이밍 검증 수행")
    parser.add_argument("--check-vtt", action="store_true", help="생성된 VTT 파일 유효성 검사")

    args = parser.parse_args()

    # WebVTT 생성기 초기화
    generator = WebVTTGenerator()

    # WebVTT 생성
    generator.process_timed_subtitles(args.input, args.output, args.validate)

    # VTT 파일 유효성 검사
    if args.check_vtt:
        generator.validate_webvtt_file(args.output)

if __name__ == "__main__":
    main()