"""
检查Python文件语法正确性
"""

import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """检查单个文件的语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        ast.parse(source, filename=str(file_path))
        return True, None
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def check_all_files():
    """检查所有Python文件"""
    python_files = []

    # 收集所有Python文件
    for pattern in ["*.py", "**/*.py"]:
        python_files.extend(Path(".").glob(pattern))

    # 排除一些文件
    exclude_patterns = ["__pycache__", ".git", "venv", "env"]
    python_files = [f for f in python_files
                   if not any(pattern in str(f) for pattern in exclude_patterns)]

    print(f"检查 {len(python_files)} 个Python文件...")

    errors = []
    success_count = 0

    for file_path in python_files:
        success, error = check_syntax(file_path)
        if success:
            success_count += 1
            print(f"✅ {file_path}")
        else:
            errors.append((file_path, error))
            print(f"❌ {file_path}: {error}")

    print(f"\n📊 检查结果:")
    print(f"✅ 成功: {success_count} 个文件")
    print(f"❌ 失败: {len(errors)} 个文件")

    if errors:
        print(f"\n❌ 语法错误文件:")
        for file_path, error in errors:
            print(f"  {file_path}: {error}")
        return False
    else:
        print(f"\n🎉 所有文件语法检查通过！")
        return True

if __name__ == "__main__":
    success = check_all_files()
    sys.exit(0 if success else 1)