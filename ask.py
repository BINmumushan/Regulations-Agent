"""知识库问答入口：python ask.py "问题"，或直接运行进入交互模式。"""

from __future__ import annotations

import argparse

from kb_agent.qa import AnswerResult, answer_question


def print_result(result: AnswerResult) -> None:
    print(f"回答：{result.answer}")
    if result.sources:
        print("来源：")
        for source in result.sources:
            print(f"  - {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库问答 Agent")
    parser.add_argument("question", nargs="?", help="要问的问题；省略则进入交互模式")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="检索返回的片段数")
    parser.add_argument(
        "-t", "--score-threshold", type=float, default=None, help="相似度阈值，低于该值视为不知道"
    )
    parser.add_argument(
        "-i", "--index-dir", default=None, help="向量库目录（默认 data/vectorstore）"
    )
    args = parser.parse_args()

    if args.question:
        result = answer_question(
            args.question,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            index_directory=args.index_dir,
        )
        print_result(result)
        return

    print("进入交互问答模式，输入 exit 或 Ctrl+C 退出。")
    history: list[dict[str, object]] = []
    while True:
        try:
            question = input("问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        try:
            result = answer_question(
                question,
                top_k=args.top_k,
                score_threshold=args.score_threshold,
                index_directory=args.index_dir,
                history=history,
            )
            print_result(result)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result.answer})
            if result.sources:
                history[-1]["sources"] = result.sources
            history = history[-16:]
        except Exception as exc:
            print(f"出错了：{exc}")


if __name__ == "__main__":
    main()
