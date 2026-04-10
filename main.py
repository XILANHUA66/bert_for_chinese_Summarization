"""
bart-chinese-summarization 本地推理脚本（CPU 优化版）
用法：
    python summarize.py                          # 交互模式（循环输入）
    python summarize.py --text "你的文章内容..."  # 单次输入模式
    python summarize.py --fast                   # 快速模式（beam=2，速度约快 1 倍，质量略降）
    python summarize.py --model /自定义/模型路径  # 指定模型目录
"""

import sys
import time
import argparse
import torch
from transformers import BertTokenizer, BartForConditionalGeneration

# ── 默认配置（与训练时保持一致）────────────────────────────
DEFAULT_MODEL_DIR = "./bart-chinese-summarization"
MAX_INPUT_LENGTH  = 512
MAX_TARGET_LENGTH = 64
NO_REPEAT_NGRAM   = 2

# ── ANSI 颜色（终端美化）────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def load_model(model_dir: str, num_beams: int):
    """加载 Tokenizer 和微调后的模型，并做 CPU 专项优化"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        n_threads = torch.get_num_threads()
        torch.set_num_threads(n_threads)
        device_label = f"CPU（{n_threads} 线程）"
    else:
        device_label = f"GPU（{torch.cuda.get_device_name(0)}）"

    print(f"{YELLOW}📦 加载模型：{model_dir}{RESET}")
    print(f"   运行设备：{device_label}")
    print(f"   Beam 数量：{num_beams}{'  ⚡ 快速模式' if num_beams < 4 else ''}")

    try:
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model     = BartForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,  # CPU 只支持 float32
        )
    except Exception as e:
        print(f"\n❌ 模型加载失败：{e}")
        print("   请确认 --model 路径下包含 config.json / pytorch_model.bin 等权重文件。")
        sys.exit(1)

    model.to(device)
    model.eval()

    # torch.compile 加速（PyTorch >= 2.0，首次调用会额外编译约 30 秒）
    if device == "cpu" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, backend="inductor", mode="reduce-overhead")
            print(f"   torch.compile：已启用（首次推理会多花 ~30 秒编译）")
        except Exception:
            pass  # 低版本 PyTorch 不支持，静默跳过

    print(f"{GREEN}✅ 模型加载完成{RESET}\n")
    return tokenizer, model, device


def generate_summary(text: str, tokenizer, model, device: str, num_beams: int) -> str:
    """对输入文本生成摘要"""
    inputs = tokenizer(
        text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            num_beams=num_beams,
            max_length=MAX_TARGET_LENGTH,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
            early_stopping=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # BertTokenizer 解码后字符间有空格，去掉
    return summary.replace(" ", "")


def interactive_mode(tokenizer, model, device: str, num_beams: int):
    """交互循环模式"""
    print(f"{BOLD}{CYAN}═══════════════════════════════════════════{RESET}")
    print(f"{BOLD}{CYAN}   🇨🇳  中文自动摘要 · 交互模式{RESET}")
    print(f"{BOLD}{CYAN}   粘贴文章后输入空行回车 → 生成摘要{RESET}")
    print(f"{BOLD}{CYAN}   输入 q 或 exit 退出{RESET}")
    print(f"{BOLD}{CYAN}═══════════════════════════════════════════{RESET}\n")

    while True:
        try:
            first_line = input(f"{YELLOW}📝 请输入文章内容：\n{RESET}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GREEN}👋 已退出。{RESET}")
            break

        # 支持多行粘贴，空行结束
        lines = [first_line]
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip() == "":
                break
            lines.append(line)

        text = "\n".join(lines).strip()

        if text.lower() in ("q", "exit", "quit", "退出"):
            print(f"{GREEN}👋 已退出。{RESET}")
            break

        if not text:
            print(f"{YELLOW}⚠️  输入为空，请重新输入。{RESET}\n")
            continue

        char_count = len(text)
        print(f"\n{CYAN}⏳ 生成摘要中（输入 {char_count} 字，CPU 需要数秒请稍候）...{RESET}")

        t0 = time.time()
        summary = generate_summary(text, tokenizer, model, device, num_beams)
        elapsed = time.time() - t0

        print(f"\n{BOLD}{'─' * 48}{RESET}")
        print(f"{BOLD}📄 原文（前 80 字）：{RESET}{text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"{BOLD}{GREEN}✨ 摘要：{RESET}{GREEN}{summary}{RESET}")
        print(f"   耗时：{elapsed:.1f} 秒")
        print(f"{BOLD}{'─' * 48}{RESET}\n")


def single_mode(text: str, tokenizer, model, device: str, num_beams: int):
    """单次推理模式"""
    print(f"{CYAN}⏳ 生成摘要中...{RESET}")
    t0 = time.time()
    summary = generate_summary(text, tokenizer, model, device, num_beams)
    elapsed = time.time() - t0

    print(f"\n{BOLD}{'─' * 48}{RESET}")
    print(f"{BOLD}📄 原文（前 80 字）：{RESET}{text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"{BOLD}{GREEN}✨ 摘要：{RESET}{GREEN}{summary}{RESET}")
    print(f"   耗时：{elapsed:.1f} 秒")
    print(f"{BOLD}{'─' * 48}{RESET}")


def main():
    parser = argparse.ArgumentParser(description="bart-chinese-summarization 本地推理脚本")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_DIR,
                        help=f"微调权重目录（默认：{DEFAULT_MODEL_DIR}）")
    parser.add_argument("--text", "-t", default=None,
                        help="要摘要的文本；不传则进入交互循环模式")
    parser.add_argument("--fast", action="store_true",
                        help="快速模式：num_beams 降为 2，速度约快 1 倍，摘要质量略降")
    args = parser.parse_args()

    num_beams = 2 if args.fast else 4
    tokenizer, model, device = load_model(args.model, num_beams)

    if args.text:
        single_mode(args.text, tokenizer, model, device, num_beams)
    else:
        interactive_mode(tokenizer, model, device, num_beams)


if __name__ == "__main__":
    main()