import sys
import re
from collections import defaultdict

def parse_line(line):
    """解析一行日志，返回 (x, y, core, message) 或 None"""
    pattern = r'^\d+:\(x=(\d+),y=(\d+)\):([^:]+):(.*)$'
    match = re.match(pattern, line.strip())
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        core = match.group(3).strip()
        msg = match.group(4).strip()
        return x, y, core, msg
    return None

def main():
    # 获取输出文件路径（如果提供了）
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    fout = open(out_path, 'w', encoding='utf-8') if out_path else sys.stdout

    data = defaultdict(list)

    for line in sys.stdin:
        parsed = parse_line(line)
        if parsed:
            x, y, core, msg = parsed
            data[(x, y, core)].append(msg)

    # 按坐标排序输出
    for (x, y, core) in sorted(data.keys()):
        msgs = data[(x, y, core)][-5:]  # 取最后5条
        fout.write(f"({x},{y}) {core}:\n")
        for m in msgs:
            fout.write(f"  {m}\n")
        fout.write("\n")  # 坐标之间空一行

    if out_path:
        fout.close()

if __name__ == "__main__":
    main()
