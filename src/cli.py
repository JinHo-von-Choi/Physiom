"""
<summary>명령행 인터페이스 엔트리 포인트</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>카메라 또는 디렉토리 입력을 통해 토큰을 생성한다.</remarks>
"""

import argparse
import sys

from src import generate_token_from_camera, generate_token_from_directory
from src.config import default_config
from src.api_http import run_server


def build_parser() -> argparse.ArgumentParser:
    """
    <summary>argparse 파서 구성</summary>
    <returns>구성된 ArgumentParser (서브커맨드 포함)</returns>
    <remarks>
    - 서브커맨드 camera/directory/server를 제공한다.
    - 필수/선택 인자를 명시적으로 정의하여 CLI 사용성을 높인다.
    
    서브커맨드:
        camera: 카메라로부터 임베딩 생성
            --max-frames: 캡처할 최대 프레임 수 (기본: 5)
        
        directory: 디렉토리 이미지로부터 임베딩 생성
            --path: 이미지 디렉토리 경로 (필수)
            --max-images: 로드할 최대 이미지 수 (기본: 5)
        
        server: FastAPI HTTP 서버 실행
            --host: 리스닝 호스트 (기본: 127.0.0.1)
            --port: 리스닝 포트 (기본: 23535)
    
    사용 예시:
        python main.py camera --max-frames 3
        python main.py directory --path "C:/faces" --max-images 10
        python main.py server --host 0.0.0.0 --port 8080
        python main.py --help
    </remarks>
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Face token generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    camera_parser = subparsers.add_parser("camera", help="카메라에서 얼굴 캡처 후 토큰 생성")
    camera_parser.add_argument("--max-frames", type=int, default=default_config.max_frames, dest="max_frames")

    directory_parser = subparsers.add_parser("directory", help="디렉토리 이미지로 토큰 생성")
    directory_parser.add_argument("--path", type=str, required=True, dest="path")
    directory_parser.add_argument("--max-images", type=int, default=default_config.max_images, dest="max_images")

    server_parser = subparsers.add_parser("server", help="FastAPI 서버 실행")
    server_parser.add_argument("--host", type=str, default=default_config.server_host)
    server_parser.add_argument("--port", type=int, default=default_config.server_port)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    <summary>CLI 진입점</summary>
    <param name="argv">명령행 인자 리스트 (None이면 sys.argv 사용)</param>
    <returns>프로세스 종료 코드 (0=성공, 1=실패)</returns>
    <remarks>
    - 예외는 포착하여 stderr에 출력 후 비정상 종료 코드(1)를 반환한다.
    - server 모드는 uvicorn을 직접 실행하며, Ctrl+C로 종료 가능.
    
    종료 코드:
        0: 정상 종료
        1: 오류 발생 (예외 메시지 stderr 출력)
    
    출력:
        - camera/directory 모드: 임베딩 리스트를 stdout에 출력
        - server 모드: uvicorn 로그 출력
    
    예외 처리:
        - 모든 예외를 포착하여 오류 메시지만 stderr에 출력
        - 스택 트레이스는 출력하지 않음 (사용자 친화적)
        - 디버깅 필요 시 --verbose 플래그 추가 권장 (향후 구현)
    
    사용 예시:
        # Python 스크립트에서 호출
        from src.cli import main
        exit_code = main(["camera", "--max-frames", "3"])
        
        # 명령행에서 직접 호출
        python main.py camera --max-frames 5
        
        # Unity Process 실행
        Process process = new Process();
        process.StartInfo.FileName = "python";
        process.StartInfo.Arguments = "main.py camera --max-frames 5";
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.RedirectStandardOutput = true;
        process.Start();
        string output = process.StandardOutput.ReadToEnd();
    
    성능:
        - camera: ~2초 (모델 로딩) + 캡처 시간
        - directory: ~2초 (모델 로딩) + 로딩/추론 시간
        - server: ~2초 (모델 로딩) 후 대기 상태
    
    제한사항:
        - 단일 프로세스에서 서브커맨드 1회만 실행 가능
        - 환경변수 설정 미지원 (config.py 직접 수정 필요)
    </remarks>
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "camera":
            embedding = generate_token_from_camera(max_frames=args.max_frames, config=default_config)
            print(embedding)
        elif args.command == "directory":
            embedding = generate_token_from_directory(dir_path=args.path, max_images=args.max_images, config=default_config)
            print(embedding)
        elif args.command == "server":
            run_server(host=args.host, port=args.port, config=default_config)
        else:
            parser.print_help()
            return 1
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

