#!/usr/bin/env python
import json
import os
import sys
import argparse

def validate_and_fix_jsonl(input_file, output_file=None, verbose=True):
    """
    验证JSONL文件并尝试修复常见问题
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出修复后文件路径，如果为None则使用原文件名加后缀_fixed
        verbose: 是否显示详细信息
    
    Returns:
        修复结果，包含：
        - 总行数
        - 有效行数
        - 修复行数
        - 无法修复行数
    """
    # 检查是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return {"status": "error", "message": f"找不到文件 {input_file}"}
    
    # 设置输出文件名
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_fixed{ext}"
    
    # 统计信息
    stats = {
        "total_lines": 0,
        "valid_lines": 0, 
        "fixed_lines": 0,
        "unfixable_lines": 0
    }
    
    # 有效的JSON行
    valid_json_lines = []
    
    # 读取文件并验证每一行
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            stats["total_lines"] += 1
            line = line.strip()
            
            # 跳过空行
            if not line:
                if verbose:
                    print(f"警告：第{i}行为空行，已跳过")
                continue
            
            # 尝试解析JSON
            try:
                json_obj = json.loads(line)
                valid_json_lines.append(line)
                stats["valid_lines"] += 1
                
            except json.JSONDecodeError as e:
                # 尝试修复常见问题
                fixed = False
                
                # 问题1：可能缺少引号
                try:
                    # 替换单引号为双引号
                    fixed_line = line.replace("'", "\"")
                    json_obj = json.loads(fixed_line)
                    valid_json_lines.append(fixed_line)
                    stats["fixed_lines"] += 1
                    fixed = True
                    if verbose:
                        print(f"修复：第{i}行 - 单引号替换为双引号")
                except:
                    pass
                
                # 问题2：尾部可能有多余逗号
                if not fixed:
                    try:
                        # 检查和去除结尾的逗号
                        if line.rstrip().endswith(","):
                            fixed_line = line.rstrip().rstrip(",")
                            json_obj = json.loads(fixed_line)
                            valid_json_lines.append(fixed_line)
                            stats["fixed_lines"] += 1
                            fixed = True
                            if verbose:
                                print(f"修复：第{i}行 - 去除尾部逗号")
                    except:
                        pass
                
                # 其他问题无法修复
                if not fixed:
                    stats["unfixable_lines"] += 1
                    if verbose:
                        print(f"错误：第{i}行JSON解析失败 - {e}")
                        print(f"  行内容: {line[:100]}..." if len(line) > 100 else f"  行内容: {line}")
    
    # 写入修复后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in valid_json_lines:
            f.write(line + '\n')
    
    # 打印摘要
    if verbose:
        print("\n=== 验证与修复摘要 ===")
        print(f"总行数: {stats['total_lines']}")
        print(f"有效行数: {stats['valid_lines']}")
        print(f"修复行数: {stats['fixed_lines']}")
        print(f"无法修复行数: {stats['unfixable_lines']}")
        print(f"修复后文件已保存至: {output_file}")
    
    return {
        "status": "success",
        "stats": stats,
        "output_file": output_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证和修复JSONL文件")
    parser.add_argument("input_file", help="输入JSONL文件")
    parser.add_argument("-o", "--output", help="输出修复后的文件路径")
    parser.add_argument("-q", "--quiet", action="store_true", help="关闭详细输出")
    
    args = parser.parse_args()
    
    result = validate_and_fix_jsonl(
        args.input_file, 
        args.output,
        not args.quiet
    )
    
    if result["status"] == "error":
        sys.exit(1)
    else:
        sys.exit(0) 