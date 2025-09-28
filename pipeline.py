#!/usr/bin/env python3
"""
새로운 WebVTT 자막 생성 파이프라인
올바른 순서: Gemini 분할 → wav2vec2 시간 정렬 → WebVTT 생성
"""

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path

class NewPipelineManager:
    """새로운 파이프라인 관리 클래스"""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.steps = {
            1: {
                "name": "텍스트 분할 (Gemini)",
                "script": "step1_gemini_text_split.py",
                "input": "script_clean.txt",
                "output": "out/step1_subtitles.json",
                "description": "Gemini AI를 사용하여 텍스트를 자막에 적합한 단위로 분할"
            },
            2: {
                "name": "시간 정렬 (wav2vec2)",
                "script": "step2_wav2vec2_timing.py",
                "input": ["podcast1.mp3", "out/step1_subtitles.json"],
                "output": "out/step2_timed_subtitles.json",
                "description": "wav2vec2를 사용하여 분할된 자막에 시간 정보 부여"
            },
            3: {
                "name": "WebVTT 생성",
                "script": "step3_generate_vtt.py",
                "input": "out/step2_timed_subtitles.json",
                "output": "out/podcast.vtt",
                "description": "시간 정보가 포함된 자막을 WebVTT 형식으로 변환"
            },
            4: {
                "name": "자막 그룹핑",
                "script": "step4_generate_groups.py",
                "input": "out/step2_timed_subtitles.json",
                "output": "out/step4_subtitle_groups.txt",
                "description": "Gemini Flash Latest를 사용한 의미론적 자막 그룹핑"
            },
            5: {
                "name": "이미지 개념 분석",
                "script": "step5_generate_image_concepts.py",
                "input": "out/step4_subtitle_groups.txt",
                "output": "out/step5_image_concepts.txt",
                "description": "Gemini 2.5 Pro를 사용한 시청자 흥미도 기반 이미지 개념 분석"
            }
        }

    def check_dependencies(self):
        """필요한 파일과 패키지들이 존재하는지 확인"""
        print("=== 의존성 확인 ===")

        # Python 패키지 확인
        required_packages = [
            ("google-generativeai", "google.generativeai"),
            ("python-dotenv", "dotenv"),
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("librosa", "librosa")
        ]

        missing_packages = []
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
                print(f"✓ {package_name}")
            except ImportError:
                print(f"✗ {package_name} (누락)")
                missing_packages.append(package_name)

        if missing_packages:
            print(f"\n누락된 패키지를 설치하세요:")
            print(f"pip install {' '.join(missing_packages)}")
            return False

        # .env 파일 확인
        env_file = self.base_dir / ".env"
        if env_file.exists():
            print("✓ .env 파일 존재")
        else:
            print("✗ .env 파일 누락")
            print("GEMINI_API_KEY를 .env 파일에 설정하세요")
            return False

        # 필수 입력 파일 확인
        required_files = ["script_clean.txt", "podcast1.mp3"]
        for file_name in required_files:
            file_path = self.base_dir / file_name
            if not file_path.exists():
                print(f"✗ 필요한 파일이 없습니다: {file_name}")
                return False
            print(f"✓ {file_name}")

        return True

    def check_step_requirements(self, step_num: int):
        """특정 스텝의 요구사항 확인"""
        step = self.steps[step_num]

        if step_num == 1:
            # 스텝 1: 원본 텍스트 파일 확인
            input_file = step["input"]
            input_path = self.base_dir / input_file
            if not input_path.exists():
                print(f"✗ 입력 파일이 없습니다: {input_file}")
                return False
            print(f"✓ {input_file}")
        elif step_num == 2:
            # 스텝 2: 오디오 파일과 1단계 결과 확인
            for input_file in step["input"]:
                input_path = self.base_dir / input_file
                if not input_path.exists():
                    print(f"✗ 입력 파일이 없습니다: {input_file}")
                    return False
                print(f"✓ {input_file}")
        else:
            # 스텝 3: 이전 스텝의 출력 확인
            input_file = step["input"]
            input_path = self.base_dir / input_file
            if not input_path.exists():
                print(f"✗ 입력 파일이 없습니다: {input_file}")
                print(f"이전 스텝을 먼저 실행하세요")
                return False
            print(f"✓ {input_file}")

        return True

    def run_step(self, step_num: int, extra_args: list = None):
        """특정 스텝 실행"""
        if step_num not in self.steps:
            print(f"잘못된 스텝 번호: {step_num}")
            return False

        step = self.steps[step_num]
        print(f"\n=== 스텝 {step_num}: {step['name']} ===")
        print(f"설명: {step['description']}")

        # 요구사항 확인
        if not self.check_step_requirements(step_num):
            return False

        # 명령어 구성
        script_path = self.base_dir / step["script"]
        if not script_path.exists():
            print(f"✗ 스크립트 파일이 없습니다: {step['script']}")
            return False

        # 가상환경 Python 경로 찾기
        venv_python = self.base_dir / "venv" / "bin" / "python"
        if venv_python.exists():
            python_cmd = str(venv_python)
        else:
            python_cmd = "python"

        cmd = [python_cmd, str(script_path)]

        # 각 스텝별 매개변수 설정
        if step_num == 1:
            cmd.extend([
                "--input", step["input"],
                "--output", step["output"]
            ])
        elif step_num == 2:
            cmd.extend([
                "--audio", step["input"][0],
                "--subtitles", step["input"][1],
                "--output", step["output"]
            ])
        elif step_num == 3:
            cmd.extend([
                "--input", step["input"],
                "--output", step["output"],
                "--validate",
                "--check-vtt"
            ])
        elif step_num == 4:
            cmd.extend([
                "--input", step["input"]
            ])
        elif step_num == 5:
            cmd.extend([
                "--input", step["input"]
            ])

        # 추가 인자가 있으면 추가
        if extra_args:
            cmd.extend(extra_args)

        print(f"실행 명령어: {' '.join(cmd)}")
        print("실행 중...")

        try:
            env = os.environ.copy()
            result = subprocess.run(cmd, cwd=self.base_dir, check=True,
                                  capture_output=False, text=True, env=env)
            print(f"✓ 스텝 {step_num} 완료")
            return True

        except subprocess.CalledProcessError as e:
            print(f"✗ 스텝 {step_num} 실패 (종료 코드: {e.returncode})")
            return False
        except Exception as e:
            print(f"✗ 스텝 {step_num} 실행 중 오류: {e}")
            return False

    def show_status(self):
        """현재 파이프라인 상태 표시"""
        print("\n=== 새로운 파이프라인 상태 ===")

        for step_num, step in self.steps.items():
            output_file = self.base_dir / step["output"]
            status = "완료" if output_file.exists() else "미완료"

            print(f"스텝 {step_num}: {step['name']} - {status}")

            if output_file.exists():
                # 파일 크기와 수정 시간 표시
                size = output_file.stat().st_size
                mtime = output_file.stat().st_mtime
                import datetime
                mtime_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  파일: {step['output']} ({size} bytes, {mtime_str})")

        # 최종 결과 확인
        final_vtt = self.base_dir / "out" / "podcast.vtt"

        print(f"\n=== 최종 결과 ===")
        if final_vtt.exists():
            print(f"생성된 VTT: {final_vtt.name} ({final_vtt.stat().st_size} bytes)")
        else:
            print("VTT 파일: 아직 생성되지 않음")

    def compare_vtt_files(self):
        """기존 VTT와 새로운 VTT 파일 비교"""
        old_vtt = self.base_dir / "archive" / "podcast1.vtt"
        new_vtt = self.base_dir / "out" / "podcast.vtt"

        if not old_vtt.exists():
            print("기존 VTT 파일이 없습니다")
            return

        if not new_vtt.exists():
            print("새로운 VTT 파일이 없습니다")
            return

        print("\n=== VTT 파일 비교 ===")

        # 파일 크기 비교
        old_size = old_vtt.stat().st_size
        new_size = new_vtt.stat().st_size
        print(f"파일 크기: 기존 {old_size} bytes → 새로운 {new_size} bytes")

        # 자막 수 비교 (간단한 카운트)
        with open(old_vtt, 'r', encoding='utf-8') as f:
            old_content = f.read()
        with open(new_vtt, 'r', encoding='utf-8') as f:
            new_content = f.read()

        old_subtitle_count = old_content.count(' --> ')
        new_subtitle_count = new_content.count(' --> ')
        print(f"자막 수: 기존 {old_subtitle_count}개 → 새로운 {new_subtitle_count}개")

        # 첫 몇 줄 비교
        print(f"\n첫 10줄 비교:")
        old_lines = old_content.split('\n')[:10]
        new_lines = new_content.split('\n')[:10]

        print("기존:")
        for i, line in enumerate(old_lines, 1):
            print(f"  {i:2d}: {line}")

        print("\n새로운:")
        for i, line in enumerate(new_lines, 1):
            print(f"  {i:2d}: {line}")

def main():
    parser = argparse.ArgumentParser(description="새로운 WebVTT 생성 파이프라인 관리")
    parser.add_argument("command", choices=[
        "check", "status", "step", "from", "all", "compare"
    ], help="실행할 명령")
    parser.add_argument("--step-num", type=int, help="실행할 스텝 번호")
    parser.add_argument("--from-step", type=int, help="시작할 스텝 번호")
    parser.add_argument("--extra-args", nargs="*", help="스크립트에 전달할 추가 인자")

    args = parser.parse_args()

    pipeline = NewPipelineManager()

    if args.command == "check":
        # 의존성 확인
        success = pipeline.check_dependencies()
        sys.exit(0 if success else 1)

    elif args.command == "status":
        # 현재 상태 표시
        pipeline.show_status()

    elif args.command == "step":
        # 특정 스텝 실행
        if args.step_num is None:
            print("--step-num을 지정해주세요")
            sys.exit(1)
        success = pipeline.run_step(args.step_num, args.extra_args)
        sys.exit(0 if success else 1)

    elif args.command == "from":
        # 특정 스텝부터 실행
        if args.from_step is None:
            print("--from-step을 지정해주세요")
            sys.exit(1)

        for step_num in range(args.from_step, max(pipeline.steps.keys()) + 1):
            success = pipeline.run_step(step_num, args.extra_args)
            if not success:
                print(f"스텝 {step_num}에서 실패했습니다.")
                sys.exit(1)
        print("지정된 스텝들이 완료되었습니다.")

    elif args.command == "all":
        # 전체 파이프라인 실행
        for step_num in sorted(pipeline.steps.keys()):
            success = pipeline.run_step(step_num, args.extra_args)
            if not success:
                print(f"스텝 {step_num}에서 실패했습니다.")
                sys.exit(1)
        print("전체 파이프라인이 완료되었습니다.")

    elif args.command == "compare":
        # VTT 파일 비교
        pipeline.compare_vtt_files()

if __name__ == "__main__":
    main()