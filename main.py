"""
<summary>프로젝트 CLI 실행 엔트리</summary>
<author>최진호</author>
<date>2025-12-16</date>
<version>1.0.0</version>
<remarks>python main.py ... 형태로 CLI를 제공한다.</remarks>
"""

import sys

from src.cli import main


if __name__ == "__main__":
    sys.exit(main())

