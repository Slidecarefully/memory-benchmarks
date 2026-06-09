"""
LongMemEval Benchmark Runner
=============================

Combined ingest + search + answer + judge pipeline for the LongMemEval
benchmark (ICLR 2025).

Each of the 500 questions has its own conversation context (haystack_sessions)
that must be ingested into Mem0 before the question can be answered.

Supports two evaluation modes:
    - answerer (default): Generate answer from memories, then judge correctness
    - retrieval: Judge whether retrieved memories alone are sufficient

Flow:
    1. Download dataset (auto-download from HuggingFace if missing)
    2. Sample / filter questions by type
    3. For each question:
        a. Ingest haystack sessions into Mem0 (pair-level checkpoint)
        b. Search Mem0 -> retrieved memories
        c. (answerer) Generate answer, then judge correctness
           (retrieval) Judge if memories suffice
        d. Save per-question checkpoint
    4. Compute metrics (by question type, by cutoff)
    5. Write unified result JSON

Usage:
    python -m benchmarks.longmemeval.run --project-name test
    python -m benchmarks.longmemeval.run --project-name full --all-questions
    python -m benchmarks.longmemeval.run --project-name full --mode retrieval
    python -m benchmarks.longmemeval.run --project-name test --per-type 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from benchmarks.common.llm_client import LLMClient
from benchmarks.common.mem0_client import Mem0Client, format_search_results
from benchmarks.common.metrics import compute_overall_metrics
from benchmarks.common.schema import (
    CutoffResult,
    EvalItem,
    GenerationData,
    JudgmentData,
    Metadata,
    Metrics,
    RetrievalData,
    UnifiedResult,
)
from benchmarks.common.utils import (
    Checkpoint,
    GracefulShutdown,
    IngestionCheckpoint,
    cutoff_label,
    download_file,
    parse_cutoffs,
    save_result_json,
    setup_logging,
)

from .prompts import (
    QUESTION_TYPES,
    get_answer_generation_prompt,
    get_judge_prompt,
)

load_dotenv(override=True)

# ===============================================================================
# CONSTANTS
# ===============================================================================

DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)
DEFAULT_DATASET_DIR = "datasets/longmemeval"
DEFAULT_DATASET_FILE = "longmemeval_s_cleaned.json"
CHUNK_SIZE = 2  # messages per ingestion chunk (user+assistant pair)


# ===============================================================================
# RETRIEVAL JUDGE PROMPT
# ===============================================================================

# Step 1: 定义 retrieval 模式下 judge LLM 使用的 system prompt。
# 这个 system prompt 用来约束 judge 的角色：
# 它不是回答问题的模型，而是“严格但公平”的检索结果评估器；
# 同时要求输出必须是 JSON，方便后续代码结构化解析。
RETRIEVAL_JUDGE_SYSTEM = (
    "You are a strict but fair evaluator for a memory retrieval system. "
    "Return JSON only with the format requested."
)

# Step 2: 定义 retrieval judge 的用户提示词模板。
# 这个 prompt 的作用是：让 judge LLM 判断“检索到的 memories 是否足够支持正确回答问题”。
# 它不要求 judge 生成最终答案，而是要求 judge 判断检索结果是否覆盖了 ground truth answer 所需的核心信息。
RETRIEVAL_JUDGE_PROMPT = """Determine whether the retrieved memories contain enough information to correctly answer the question.

## Evaluation Steps

1. **Core Intent**: What fact, preference, or piece of information must the memories provide?

2. **Abstention Case**: If the ground truth says the question is unanswerable ("I don't know", "not mentioned"):
   - PASS if memories are empty or would not lead to a confidently wrong answer.
   - FAIL only if memories would cause a hallucinated/incorrect answer.

3. **Evidence Check**: Do retrieved memories contain the key facts needed?
   - Action completion: "received X", "switched to X" implies current state
   - Semantic equivalence counts
   - Most recent memory is authoritative for conflicts
   - Reasonable inference counts
   - Date format variations and off-by-one on day counts are acceptable
   - Extra context beyond what is needed does not cause FAIL

4. **Double-Check**: Re-read memories. Does quoted text actually appear? Does it actually support the core intent?

5. **Verdict**: PASS if core intent is supported; FAIL if not.

## Input

Question: {question}
Question Date: {question_date}
Expected Answer: {answer}
{profile_section}
Retrieved Memories ({num_memories} total):
{memories_text}

## Output

Return exactly this JSON:
{{
    "core_intent": "<What the user needs to know>",
    "core_intent_supported": true/false,
    "supporting_evidence": "<Quote exact text from memories or explain what is missing>",
    "judgment": "PASS or FAIL",
    "reason": "<One sentence explanation>"
}}"""


def _format_retrieval_memories(
    search_results: list[dict],
    question_date: str = "",
) -> str:
    """Format search results into a numbered list for the retrieval judge."""

    # Step 3: 将 Mem0 search 返回的 memories 格式化成 judge prompt 里可读的文本。
    # 输入 search_results 是一个 memory 结果列表；
    # 输出是一个编号列表字符串，例如：
    # 1. xxx (score=0.8231) [created: 2023-05-01T...]
    #
    # question_date 参数当前没有被实际使用，但保留在函数签名中，
    # 方便后续如果需要按问题日期做格式化或过滤，可以直接扩展。

    # Step 3.1: 如果没有检索结果，则返回 "(None)"。
    # 这样 judge LLM 能明确知道当前没有任何 retrieved memories。
    if not search_results:
        return "(None)"

    # Step 3.2: 初始化 lines，用于存放每条 memory 的格式化文本。
    lines = []

    # Step 3.3: 遍历 search_results，并从 1 开始编号。
    for i, r in enumerate(search_results, 1):
        # Step 3.4: 取出 memory 文本。
        # 如果没有 memory 字段，则使用空字符串兜底。
        mem = r.get("memory", "")

        # Step 3.5: 取出检索分数。
        # 如果没有 score 字段，则默认为 0。
        score = r.get("score", 0)

        # Step 3.6: 取出 memory 创建时间。
        # created_at 后续会帮助 judge 判断冲突信息中的时间先后关系。
        created = r.get("created_at", "")

        # Step 3.7: 先构造当前 memory 的主体部分：
        # "{编号}. {memory内容}"。
        parts = [f"{i}. {mem}"]

        # Step 3.8: 如果存在 score，则追加分数。
        # 这里保留 4 位小数，避免 prompt 太长，同时给 judge 一个相关性参考。
        if score:
            parts.append(f"(score={score:.4f})")

        # Step 3.9: 如果存在 created_at，则追加创建时间。
        # 这对于 LongMemEval 中有时间顺序的问题尤其重要。
        if created:
            parts.append(f"[created: {created}]")

        # Step 3.10: 将当前 memory 的各部分拼成一行，并加入 lines。
        lines.append(" ".join(parts))

    # Step 3.11: 用换行符拼接所有 memory 行，作为 prompt 中的 Retrieved Memories 部分。
    return "\n".join(lines)


def get_retrieval_judge_prompt(
    question: str,
    answer: str,
    search_results: list[dict],
    question_date: str = "",
    user_profile: dict | None = None,
) -> str:
    """Build the retrieval mode judge prompt."""

    # Step 4: 构造 retrieval judge 的完整 prompt。
    # 这个函数把 question、ground truth answer、search results、question date、
    # 以及可选 user_profile 填入 RETRIEVAL_JUDGE_PROMPT 模板。

    # Step 4.1: 先把 search_results 格式化成 judge 可读的 memories 文本。
    memories_text = _format_retrieval_memories(search_results, question_date)

    # Step 4.2: 初始化 user profile 文本区域。
    # 默认没有 user profile。
    profile_section = ""

    # Step 4.3: 如果提供了 user_profile，则将其格式化为多行文本。
    # 这些 profile 信息会进入 judge prompt，帮助 judge 判断检索结果是否足够。
    if user_profile:
        # Step 4.3.1: 写入 User Profile 标题。
        profile_lines = ["User Profile:"]

        # Step 4.3.2: 遍历 user_profile 的键值对。
        for k, v in user_profile.items():
            # Step 4.3.3: 只写入非 None 的 profile 字段。
            if v is not None:
                profile_lines.append(f"  {k}: {v}")

        # Step 4.3.4: 将 profile 行拼接成字符串。
        profile_section = "\n".join(profile_lines)

    # Step 4.4: 将所有变量填入 RETRIEVAL_JUDGE_PROMPT 模板。
    # 注意：
    # - question_date 为空时填 "(not specified)"；
    # - answer 会转成字符串；
    # - num_memories 使用 search_results 的长度；
    # - memories_text 是格式化后的 memory 列表。
    return RETRIEVAL_JUDGE_PROMPT.format(
        question=question,
        question_date=question_date or "(not specified)",
        answer=str(answer),
        profile_section=profile_section,
        num_memories=len(search_results),
        memories_text=memories_text,
    )


# ===============================================================================
# DATASET
# ===============================================================================


def download_dataset(dataset_dir: str, logger: Any) -> str:
    """Download LongMemEval dataset from HuggingFace if not present."""

    # Step 5: 确定 LongMemEval 数据集本地保存路径。
    # DEFAULT_DATASET_FILE 是默认数据集文件名。
    path = os.path.join(dataset_dir, DEFAULT_DATASET_FILE)

    # Step 5.1: 如果本地已经存在数据集文件，则直接返回路径，不重复下载。
    if os.path.exists(path):
        logger.info("Dataset already exists: %s", path)
        return path

    # Step 5.2: 如果数据集不存在，则创建数据集目录。
    os.makedirs(dataset_dir, exist_ok=True)

    # Step 5.3: 记录开始下载日志。
    logger.info("Downloading LongMemEval dataset...")

    # Step 5.4: 从 DATASET_URL 下载数据集到本地 path。
    # download_file 通常会处理进度条、HTTP 下载和文件写入。
    download_file(DATASET_URL, path, description="Downloading LongMemEval")

    # Step 5.5: 下载完成后，读取 JSON 文件做基本合法性校验。
    with open(path) as f:
        data = json.load(f)

    # Step 5.6: 校验数据集格式。
    # 这里期望数据是 list，并且至少有 500 个问题。
    # 如果校验失败，说明下载内容不对或文件损坏。
    if not isinstance(data, list) or len(data) < 500:
        # Step 5.6.1: 删除无效文件，避免后续误用。
        os.remove(path)

        # Step 5.6.2: 抛出异常，提示数据集不合法。
        raise RuntimeError(
            f"Invalid dataset: expected 500 questions, got {len(data)}"
        )

    # Step 5.7: 记录下载成功日志。
    logger.info("Downloaded: %s (%d questions)", path, len(data))

    # Step 5.8: 返回数据集路径。
    return path


def load_dataset(path: str) -> list[dict]:
    # Step 6: 从指定路径加载 LongMemEval 数据集。
    # 数据集文件是 JSON 格式，返回值是 list[dict]。
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===============================================================================
# QUESTION SAMPLING
# ===============================================================================


def sample_questions_stratified(
    questions: list[dict],
    per_type: int = 5,
    seed: int = 42,
    selected_types: list[str] | None = None,
) -> list[dict]:
    """Sample questions stratified by question_type."""

    # Step 7: 按 question_type 分层抽样。
    # 目标是从每一种问题类型中抽取 per_type 个问题，
    # 避免只抽到某一种类型，导致小样本实验不均衡。

    # Step 7.1: 确定要参与抽样的问题类型集合。
    # 如果用户指定 selected_types，就只保留这些类型；
    # 否则使用全局 QUESTION_TYPES。
    type_filter = set(selected_types) if selected_types else set(QUESTION_TYPES)

    # Step 7.2: 创建 groups。
    # key 是 question_type，value 是该类型下的问题列表。
    groups: dict[str, list[dict]] = defaultdict(list)

    # Step 7.3: 遍历所有问题，把符合类型要求的问题放入对应分组。
    for q in questions:
        if q["question_type"] in type_filter:
            groups[q["question_type"]].append(q)

    # Step 7.4: 对每个类型内部的问题按 question_id 排序。
    # 这样可以保证在相同 seed 下抽样更加稳定、可复现。
    for qtype in groups:
        groups[qtype].sort(key=lambda q: q["question_id"])

    # Step 7.5: 创建固定随机种子的随机数生成器。
    # 使用 random.Random(seed) 而不是全局 random，
    # 可以避免影响程序其他地方的随机性。
    rng = random.Random(seed)

    # Step 7.6: 初始化 sampled，用于保存最终抽样结果。
    sampled = []

    # Step 7.7: 按 question_type 的字典序遍历分组。
    # 这样不同运行之间类型处理顺序稳定。
    for qtype in sorted(groups.keys()):
        # Step 7.7.1: 取出当前类型的问题列表。
        group = groups[qtype]

        # Step 7.7.2: 当前类型最多抽 per_type 个；
        # 如果该类型问题数量不足 per_type，则全部抽取。
        n = min(per_type, len(group))

        # Step 7.7.3: 从当前类型中随机抽取 n 个问题。
        selected = rng.sample(group, n)

        # Step 7.7.4: 加入最终 sampled 列表。
        sampled.extend(selected)

    # Step 7.8: 对所有抽样结果按 question_id 排序。
    # 这样最终处理顺序稳定，便于复现实验和对比输出。
    sampled.sort(key=lambda q: q["question_id"])

    # Step 7.9: 返回分层抽样后的问题列表。
    return sampled


# ===============================================================================
# SESSION AND TURN PROCESSING
# ===============================================================================


def parse_longmemeval_date(date_str: str) -> int | None:
    """Parse '2023/05/01 (Mon) 21:05' -> Unix epoch int (treated as UTC)."""

    # Step 8: 将 LongMemEval 的日期字符串解析成 Unix epoch 秒。
    # 原始格式类似：
    # "2023/05/01 (Mon) 21:05"
    # 输出是 int timestamp，并且按 UTC 处理。
    try:
        # Step 8.1: 删除日期字符串中的星期部分，例如 "(Mon)"。
        # 删除后得到类似 "2023/05/01 21:05"。
        cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()

        # Step 8.2: 按 "%Y/%m/%d %H:%M" 格式解析时间。
        # replace(tzinfo=timezone.utc) 表示将其视为 UTC 时间。
        dt = datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)

        # Step 8.3: 转成 Unix epoch 秒并返回。
        return int(dt.timestamp())

    except (ValueError, TypeError):
        # Step 8.4: 如果日期格式错误，或 date_str 类型不对，则返回 None。
        return None


def parse_longmemeval_date_human(date_str: str) -> str:
    """Parse '2023/05/01 (Mon) 21:05' -> 'Monday, May 01, 2023'."""

    # Step 9: 将 LongMemEval 的日期字符串转成更适合 prompt 阅读的日期格式。
    # 原始格式：
    # "2023/05/01 (Mon) 21:05"
    # 输出格式：
    # "Monday, May 01, 2023"
    try:
        # Step 9.1: 删除星期缩写部分，例如 "(Mon)"。
        cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()

        # Step 9.2: 按数据集原始时间格式解析。
        # 注意这里没有设置 timezone，因为这里只是为了生成可读字符串。
        dt = datetime.strptime(cleaned, "%Y/%m/%d %H:%M")

        # Step 9.3: 格式化成人类可读日期。
        return dt.strftime("%A, %B %d, %Y")

    except (ValueError, TypeError):
        # Step 9.4: 如果解析失败，则返回原始字符串。
        # 这样至少不会丢失原始 question_date 信息。
        return date_str


def sort_sessions_chronologically(
    question: dict,
) -> list[tuple[str, str, list[dict]]]:
    """Sort haystack_sessions by their corresponding haystack_dates.

    Returns list of (session_id, date_str, session) tuples sorted by date.
    """

    # Step 10: 将一个问题中的 haystack sessions 按时间排序。
    # LongMemEval 中一个问题会包含多个 haystack session；
    # 为了按真实时间顺序 ingest，需要根据 haystack_dates 对 sessions 排序。

    # Step 10.1: 取出所有 haystack_sessions。
    # 每个 session 是一段对话 turn 列表。
    sessions = question["haystack_sessions"]

    # Step 10.2: 取出每个 session 对应的日期字符串。
    dates = question["haystack_dates"]

    # Step 10.3: 取出每个 session 的 id。
    session_ids = question["haystack_session_ids"]

    # Step 10.4: 将 session_id、date、session 三者配对。
    # paired 中每个元素是：
    # (session_id, date_str, session)
    paired = list(zip(session_ids, dates, sessions))

    # Step 10.5: 定义排序 key。
    # 能成功解析日期的 session 排在前面，并按 timestamp 升序；
    # 不能解析日期的 session 排在后面，并按原始 date_str 做稳定排序。
    def sort_key(item: tuple) -> tuple:
        # Step 10.5.1: 尝试解析当前 session 的日期。
        parsed = parse_longmemeval_date(item[1])

        # Step 10.5.2: 如果日期解析成功，返回 (0, timestamp, 原始日期字符串)。
        # 第一个元素 0 表示“可解析日期”，会排在不可解析日期之前。
        if parsed is not None:
            return (0, parsed, item[1])

        # Step 10.5.3: 如果日期解析失败，返回 (1, 0, 原始日期字符串)。
        # 第一个元素 1 表示“不可解析日期”，排在后面。
        return (1, 0, item[1])

    # Step 10.6: 原地按 sort_key 排序。
    paired.sort(key=sort_key)

    # Step 10.7: 返回排序后的 session 列表。
    return paired


def pair_turns(session: list[dict]) -> list[list[dict]]:
    """Pair consecutive user/assistant turns, stripping 'has_answer' field.

    Returns list of message pairs for Mem0 add() calls.
    """

    # Step 11: 将一个 session 中的连续 turn 切分成 pair。
    # LongMemEval 的 session 是一段多轮对话；
    # Mem0 ingestion 时不是整个 session 一次 add，
    # 而是每两个 turn 组成一个 pair 后调用一次 mem0.add。

    # Step 11.1: 清理每个 turn，只保留 role 和 content。
    # 这里会丢弃 has_answer 等数据集标注字段，
    # 因为 mem0.add 只需要标准 message 格式。
    cleaned = [{"role": t["role"], "content": t["content"]} for t in session]

    # Step 11.2: 初始化 pairs。
    pairs = []

    # Step 11.3: 每隔两个 turn 取一组。
    # 通常就是：
    # [
    #   {"role": "user", "content": "..."},
    #   {"role": "assistant", "content": "..."}
    # ]
    for i in range(0, len(cleaned), 2):
        # Step 11.3.1: 取当前位置开始的两个 turn。
        # 如果 session 长度是奇数，最后一个 pair 可能只有 1 条 message。
        pair = cleaned[i : i + 2]

        # Step 11.3.2: 加入 pairs。
        pairs.append(pair)

    # Step 11.4: 返回 pair 列表，供 ingest_question() 逐个调用 mem0.add。
    return pairs

# ===============================================================================
# INGESTION
# ===============================================================================


async def ingest_question(
    question: dict,
    mem0: Mem0Client,
    logger: Any,
    run_id: str,
    output_dir: str,
    shutdown: GracefulShutdown,
    debug: bool = True,
) -> tuple[bool, str, int]:
    """Ingest all haystack sessions of a LongMemEval question into Mem0.

    Each question gets its own user_id so memories don't leak between questions.

    Returns: (success, user_id, total_pairs_processed)
    """

    # Step 1: 取出当前问题的 question_id。
    question_id = question["question_id"]

    # Step 2: 为当前问题构造独立的 user_id。
    # 这样每个 LongMemEval question 都有自己的 Mem0 用户作用域，
    # 避免不同问题之间的 memories 互相污染。
    user_id = f"longmemeval_{question_id}_{run_id}"

    # Step 3: 初始化 ingestion checkpoint。
    # checkpoint 用来支持断点续跑，避免重复 ingest 已完成的 pair。
    checkpoint = IngestionCheckpoint(output_dir)
    key = question_id

    # Step 4: 检查该问题是否已经完整 ingest 过。
    # Check if already complete
    is_done, cp_data = checkpoint.is_complete(key, CHUNK_SIZE)

    # Step 5: 如果 checkpoint 显示已完成，则直接返回成功。
    if is_done and cp_data:
        pairs_done = cp_data.get("total_pairs_processed", 0)
        user_id = cp_data.get("user_id", user_id)

        logger.info(
            "Question %s already ingested (user_id=%s, %d pairs)",
            question_id, user_id, pairs_done,
        )

        return True, user_id, pairs_done

    # Step 6: 检查是否存在部分完成的进度。
    # Check for partial progress
    chunks_already_done, resumed_uid = checkpoint.load_progress(key, CHUNK_SIZE)

    # Step 7: 如果有部分进度，并且 checkpoint 里保存了 user_id，
    # 则继续使用之前的 user_id，保证续跑时写入同一个 Mem0 用户作用域。
    if resumed_uid and chunks_already_done:
        user_id = resumed_uid

        logger.info(
            "Resuming question %s from %d completed pairs",
            question_id, len(chunks_already_done),
        )

    # Step 8: 将当前问题里的 sessions 按时间顺序排序。
    sorted_sessions = sort_sessions_chronologically(question)

    # Step 9: 统计所有 session 中一共有多少个 user/assistant pair。
    # Count total pairs for progress
    total_pairs = sum(len(pair_turns(s)) for _, _, s in sorted_sessions)

    # Step 10: 打印 ingestion 开始日志。
    logger.info(
        "Ingesting question %s: %d sessions, %d pairs",
        question_id, len(sorted_sessions), total_pairs,
    )

    # Step 11: 如果开启 debug，则准备 debug 日志文件。
    # Debug log
    debug_file = None
    if debug:
        # Step 11.1: 创建 debug 输出目录。
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Step 11.2: 构造当前 question 的 debug 文件路径。
        debug_path = os.path.join(debug_dir, f"{question_id}_ingestion.txt")

        # Step 11.3: 如果是续跑，则追加写入；否则重新写入。
        debug_mode = "a" if chunks_already_done else "w"
        debug_file = open(debug_path, debug_mode, encoding="utf-8")

        # Step 11.4: 如果不是续跑，则写入 debug 文件头部信息。
        if not chunks_already_done:
            debug_file.write(f"{'=' * 80}\n")
            debug_file.write(f"QUESTION {question_id} (type={question['question_type']})\n")
            debug_file.write(f"Sessions: {len(sorted_sessions)}, Pairs: {total_pairs}\n")
            debug_file.write(f"User ID: {user_id}\n")
            debug_file.write(f"{'=' * 80}\n\n")

    # Step 12: 初始化 ingestion 进度条。
    # initial 设置为已经完成的 chunk 数量，用于断点续跑。
    pbar = tqdm(
        total=total_pairs,
        desc=f"Ingest {question_id}",
        initial=len(chunks_already_done),
        leave=True,
    )

    # Step 13: 初始化处理成功和失败计数。
    total_processed = len(chunks_already_done)
    total_failed = 0

    # Step 14: 遍历按时间排序后的所有 session。
    for session_idx, (session_id, date_str, session) in enumerate(sorted_sessions):

        # Step 14.1: 空 session 直接跳过。
        if not session:
            continue

        # Step 14.2: 解析 session 的时间戳。
        # 这个 timestamp 会在 mem0.add 时传入，用于 memory 的时间信息。
        session_timestamp = parse_longmemeval_date(date_str) if date_str else None

        # Step 14.3: 将 session 拆成多个对话 pair。
        pairs = pair_turns(session)

        # Step 14.4: 如果开启 debug，且该 session header 没写过，则写入 session 头部。
        if debug_file and f"s{session_idx}_header" not in chunks_already_done:
            debug_file.write(f"\n{'---' * 27}\n")
            debug_file.write(
                f"SESSION {session_idx} ({session_id})  |  "
                f"Date: {date_str}  |  Pairs: {len(pairs)}\n"
            )
            debug_file.write(f"{'---' * 27}\n\n")

        # Step 15: 遍历当前 session 中的每个 pair。
        for pair_idx, messages in enumerate(pairs):
            # Step 15.1: 构造当前 pair 的 chunk key。
            # 例如 s0_p3 表示第 0 个 session 的第 3 个 pair。
            chunk_key = f"s{session_idx}_p{pair_idx}"

            # Step 15.2: 如果该 chunk 已经处理过，则跳过。
            if chunk_key in chunks_already_done:
                continue

            # Step 15.3: 如果收到优雅退出信号，则保存当前进度并返回。
            if shutdown.requested:
                logger.info(
                    "Shutdown requested at question %s, chunk %s",
                    question_id, chunk_key,
                )

                pbar.close()

                if debug_file:
                    debug_file.close()

                return True, user_id, total_processed

            # Step 15.4: 如果 pair 里有空 content，则跳过该 pair。
            # Skip pairs with empty content
            if any(not msg.get("content", "").strip() for msg in messages):
                chunks_already_done.add(chunk_key)
                total_processed += 1
                pbar.update(1)
                continue

            # Step 15.5: 如果开启 debug，则把当前 pair 的原始消息写入 debug 文件。
            if debug_file:
                debug_file.write(f"--- Pair {pair_idx} ({len(messages)} messages) ---\n")

                for msg in messages:
                    debug_file.write(f"  {msg['role']}: {msg['content'][:200]}\n")

                debug_file.write("\n")

            # Step 15.6: 调用 mem0.add，将当前 pair 写入 Mem0。
            # 这里会触发 Mem0 的 memory extraction / add pipeline。
            response = await mem0.add(messages, user_id, timestamp=session_timestamp)

            # Step 15.7: 如果 mem0.add 有返回结果，则认为当前 pair 处理成功。
            if response is not None:
                total_processed += 1

                # Step 15.8: 如果开启 debug，则把本次抽取出的 memories 写入 debug 文件。
                if debug_file:
                    results = response.get("results", [])

                    if results:
                        debug_file.write(f"--- Pair {pair_idx} (extracted) ---\n")

                        for mem_item in results:
                            mem_text = mem_item.get("memory", "")
                            event_type = mem_item.get("event", "")
                            debug_file.write(f"  [{event_type}] {mem_text}\n")

                        debug_file.write("\n")
            else:
                # Step 15.9: 如果 mem0.add 返回 None，则认为 ingestion 失败。
                total_failed += 1

                logger.warning(
                    "Ingestion failed: %s session %d pair %d",
                    question_id, session_idx, pair_idx,
                )

            # Step 15.10: 无论成功或失败，都把当前 chunk 标记为已完成。
            chunks_already_done.add(chunk_key)

            # Step 15.11: 保存当前 question 的 ingestion 进度。
            checkpoint.save_progress(key, {
                "question_id": question_id,
                "user_id": user_id,
                "run_id": run_id,
                "chunk_size": CHUNK_SIZE,
                "completed_chunks": list(chunks_already_done),
            })

            # Step 15.12: 更新进度条描述，若有失败则显示失败数量。
            pbar.set_description(
                f"Ingest {question_id}"
                + (f" [!fail={total_failed}]" if total_failed else "")
            )

            # Step 15.13: 更新进度条。
            pbar.update(1)

    # Step 16: 所有 session / pair 处理完成，关闭进度条。
    pbar.close()

    # Step 17: 如果开启 debug，则写入 summary 并关闭文件。
    if debug_file:
        debug_file.write(
            f"\nSUMMARY: {total_processed}/{total_pairs} OK, {total_failed} failed\n"
        )
        debug_file.close()

    # Step 18: 保存完整完成的 checkpoint。
    checkpoint.save_complete(key, {
        "question_id": question_id,
        "user_id": user_id,
        "run_id": run_id,
        "chunk_size": CHUNK_SIZE,
        "total_pairs_processed": total_processed,
        "total_pairs_failed": total_failed,
    })

    # Step 19: 返回 ingestion 结果。
    # total_failed == 0 表示整体成功。
    return total_failed == 0, user_id, total_processed


# ===============================================================================
# SEARCH + ANSWER + JUDGE
# ===============================================================================


async def process_question_answerer(
    question: dict,
    user_id: str,
    mem0: Mem0Client,
    answerer: LLMClient,
    judge_llm: LLMClient,
    cutoffs: list[int],
    top_k: int,
    user_profile: dict | None,
    predict_only: bool,
    logger: Any,
    score_debug: bool = False,
    existing_search_results: list | None = None,
) -> dict[str, Any]:
    """Process a question in answerer mode: search + generate answer + judge.

    Returns a result dict suitable for serialization.
    """

    # Step 1: 从 question 中取出基础字段。
    question_id = question["question_id"]
    question_text = question["question"]
    question_type = question["question_type"]
    answer = str(question["answer"])
    question_date = question.get("question_date", "")

    # Step 2: 将问题日期转换成人类可读格式，供 answerer prompt 使用。
    # Human-readable question date for the answerer prompt
    question_date_human = (
        parse_longmemeval_date_human(question_date) if question_date else ""
    )

    # Step 3: 执行搜索阶段。
    # --- Search ---
    if existing_search_results is not None:
        # Step 3.1: 如果已有 search 结果，则直接复用，不再调用 Mem0 search。
        formatted = existing_search_results
        query_debug = None
        search_latency = 0.0
    else:
        # Step 3.2: 否则调用 mem0.search 搜索相关 memories。
        start = time.monotonic()

        search_results = await mem0.search(
            question_text, user_id, top_k=top_k, score_debug=score_debug,
        )

        # Step 3.3: 计算搜索耗时，单位毫秒。
        search_latency = (time.monotonic() - start) * 1000

        # Step 3.4: 格式化 search 结果，并提取 debug 信息。
        formatted, query_debug = format_search_results(search_results)

    # Step 4: 构造当前问题的基础 result。
    result: dict[str, Any] = {
        "question_id": question_id,
        "question_type": question_type,
        "question": question_text,
        "ground_truth_answer": answer,
        "question_date": question_date,
        "is_abstention": question_id.endswith("_abs"),
        "user_id": user_id,
        "answer_session_ids": question.get("answer_session_ids", []),
        "retrieval": {
            "search_query": question_text,
            "search_results": formatted,
            "search_latency_ms": round(search_latency, 1),
            "total_results": len(formatted),
        },
    }

    # Step 5: 如果存在 query_debug，则写入 result。
    if query_debug:
        result["retrieval"]["query_debug"] = query_debug

    # Step 6: 如果存在 user_profile，则写入 result。
    if user_profile:
        result["user_profile"] = user_profile

    # Step 7: 如果是 predict_only 模式，只返回搜索结果，不生成答案、不 judge。
    if predict_only:
        return result

    # Step 8: 初始化不同 cutoff 下的评测结果。
    # --- Answer + Judge at each cutoff ---
    cutoff_results: dict[str, dict] = {}

    # Step 9: 遍历每个 cutoff，例如 top_5 / top_10 / top_20。
    for c in cutoffs:
        # Step 9.1: 截取前 c 条搜索结果。
        sliced = formatted[:c]

        # Step 9.2: 按 created_at 排序，让 answerer 看到自然时间线。
        # Sort chronologically for the answerer (natural timeline)
        sliced_chrono = sorted(sliced, key=lambda x: x.get("created_at") or "")

        # Step 9.3: 构造 cutoff label。
        label = cutoff_label(c)

        # Step 9.4: 构造 answer generation prompt。
        # Generate answer
        gen_prompt = get_answer_generation_prompt(
            question=question_text,
            search_results=sliced_chrono,
            question_date=question_date_human,
            user_profile=user_profile,
        )

        # Step 9.5: 调用 answerer LLM 生成答案。
        generated_answer = await answerer.generate(system="", user=gen_prompt)

        # Step 9.6: 删除可能存在的 chain-of-thought / mem_thinking 标签内容。
        # Strip chain-of-thought tags
        generated_answer = re.sub(
            r"[<\[]mem_thinking[>\]].*?[<\[]/mem_thinking[>\]]",
            "",
            generated_answer,
            flags=re.DOTALL,
        ).strip()

        # Step 9.7: 如果答案中包含 ANSWER:，只保留 ANSWER: 后面的内容。
        if "ANSWER:" in generated_answer:
            generated_answer = generated_answer.rsplit("ANSWER:", 1)[-1].strip()

        # Step 9.8: 构造 judge prompt。
        # Judge: yes/no correctness
        judge_prompt = get_judge_prompt(
            question_type=question_type,
            question_id=question_id,
            question=question_text,
            answer=answer,
            response=generated_answer,
            question_date=question_date_human,
        )

        # Step 9.9: 调用 judge LLM 判断答案是否正确。
        correct, judge_raw = await judge_llm.judge_yes_no(judge_prompt)

        # Step 9.10: 将 judge 结果转换为 score 和 judgment。
        score = 1.0 if correct else 0.0
        judgment = "PASS" if correct else "FAIL"

        # Step 9.11: 保存当前 cutoff 下的评测结果。
        cutoff_results[label] = {
            "judgment": judgment,
            "score": score,
            "generated_answer": generated_answer,
            "judge_raw": judge_raw,
            "memories_evaluated": len(sliced),
            "reason": f"Generated answer: {generated_answer[:500]}",
        }

    # Step 10: 将所有 cutoff 结果写入 result。
    result["cutoff_results"] = cutoff_results

    # Step 11: 返回完整结果。
    return result


async def process_question_retrieval(
    question: dict,
    user_id: str,
    mem0: Mem0Client,
    judge_llm: LLMClient,
    cutoffs: list[int],
    top_k: int,
    user_profile: dict | None,
    predict_only: bool,
    logger: Any,
    score_debug: bool = False,
) -> dict[str, Any]:
    """Process a question in retrieval mode: search + judge memories directly.

    Returns a result dict suitable for serialization.
    """

    # Step 1: 从 question 中取出基础字段。
    question_id = question["question_id"]
    question_text = question["question"]
    question_type = question["question_type"]
    answer = str(question["answer"])
    question_date = question.get("question_date", "")

    # Step 2: 调用 Mem0 search 检索相关 memories。
    # --- Search ---
    start = time.monotonic()

    search_results = await mem0.search(
        question_text, user_id, top_k=top_k, score_debug=score_debug,
    )

    # Step 3: 计算 search 耗时，单位毫秒。
    search_latency = (time.monotonic() - start) * 1000

    # Step 4: 格式化搜索结果和 query debug 信息。
    formatted, query_debug = format_search_results(search_results)

    # Step 5: 构造基础 result。
    result: dict[str, Any] = {
        "question_id": question_id,
        "question_type": question_type,
        "question": question_text,
        "ground_truth_answer": answer,
        "question_date": question_date,
        "is_abstention": question_id.endswith("_abs"),
        "user_id": user_id,
        "answer_session_ids": question.get("answer_session_ids", []),
        "retrieval": {
            "search_query": question_text,
            "search_results": formatted,
            "search_latency_ms": round(search_latency, 1),
            "total_results": len(formatted),
        },
    }

    # Step 6: 如果有 query_debug，则写入 retrieval。
    if query_debug:
        result["retrieval"]["query_debug"] = query_debug

    # Step 7: 如果有 user_profile，则写入 result。
    if user_profile:
        result["user_profile"] = user_profile

    # Step 8: predict_only 模式下，只返回检索结果，不做 judge。
    if predict_only:
        return result

    # Step 9: 初始化各个 cutoff 下的 judge 结果。
    # --- Judge at each cutoff ---
    cutoff_results: dict[str, dict] = {}

    # Step 10: 遍历每个 cutoff。
    for c in cutoffs:
        # Step 10.1: 取前 c 条检索结果。
        sliced = formatted[:c]

        # Step 10.2: 构造 cutoff label。
        label = cutoff_label(c)

        # Step 10.3: 构造 retrieval judge prompt。
        prompt = get_retrieval_judge_prompt(
            question=question_text,
            answer=answer,
            search_results=sliced,
            question_date=question_date,
            user_profile=user_profile,
        )

        # Step 10.4: 调用 judge LLM，要求输出结构化结果。
        raw = await judge_llm.generate_structured(
            system=RETRIEVAL_JUDGE_SYSTEM,
            user=prompt,
        )

        # Step 10.5: 如果 raw 是 dict，则读取 judgment 字段。
        if isinstance(raw, dict):
            judgment_str = raw.get("judgment", "").upper()
            passed = judgment_str == "PASS"
        else:
            # Step 10.6: 如果返回格式异常，则认为失败。
            passed = False

        # Step 10.7: 转换为 score 和 judgment。
        score = 1.0 if passed else 0.0
        judgment = "PASS" if passed else "FAIL"

        # Step 10.8: 保存当前 cutoff 的评测结果。
        cutoff_results[label] = {
            "judgment": judgment,
            "score": score,
            "generated_answer": raw.get("supporting_evidence", "") if isinstance(raw, dict) else "",
            "memories_evaluated": len(sliced),
            "reason": raw.get("reason", "") if isinstance(raw, dict) else "",
            "core_intent": raw.get("core_intent", "") if isinstance(raw, dict) else "",
            "core_intent_supported": raw.get("core_intent_supported", False) if isinstance(raw, dict) else False,
        }

    # Step 11: 写入 cutoff_results。
    result["cutoff_results"] = cutoff_results

    # Step 12: 返回完整结果。
    return result


async def apply_longmemeval_answerer_judge_to_saved_result(
    result: dict,
    answerer: LLMClient,
    judge_llm: LLMClient,
    cutoffs: list[int],
) -> None:
    """Fill ``cutoff_results`` from ``retrieval.search_results`` (no Mem0)."""

    # Step 1: 从已有 result 中读取 search_results。
    # 该函数不再调用 Mem0，只基于已保存的 retrieval 结果重新生成 cutoff_results。
    formatted = list(result["retrieval"]["search_results"])

    # Step 2: 读取问题基础字段。
    question_text = result["question"]
    question_id = result["question_id"]
    question_type = result["question_type"]
    answer = str(result["ground_truth_answer"])
    question_date = result.get("question_date", "")
    user_profile = result.get("user_profile")

    # Step 3: 将 question_date 转成人类可读形式。
    question_date_human = (
        parse_longmemeval_date_human(question_date) if question_date else ""
    )

    # Step 4: 初始化 cutoff_results。
    cutoff_results: dict[str, dict] = {}

    # Step 5: 遍历每个 cutoff。
    for c in cutoffs:
        # Step 5.1: 取前 c 条 search results。
        sliced = formatted[:c]

        # Step 5.2: 按时间顺序排序，供 answerer 生成答案。
        sliced_chrono = sorted(sliced, key=lambda x: x.get("created_at") or "")

        # Step 5.3: 构造 cutoff label。
        label = cutoff_label(c)

        # Step 5.4: 构造 answer generation prompt。
        gen_prompt = get_answer_generation_prompt(
            question=question_text,
            search_results=sliced_chrono,
            question_date=question_date_human,
            user_profile=user_profile,
        )

        # Step 5.5: 调用 answerer 生成答案。
        generated_answer = await answerer.generate(system="", user=gen_prompt)

        # Step 5.6: 清理 mem_thinking 标签内容。
        generated_answer = re.sub(
            r"[<\[]mem_thinking[>\]].*?[<\[]/mem_thinking[>\]]",
            "",
            generated_answer,
            flags=re.DOTALL,
        ).strip()

        # Step 5.7: 如果包含 ANSWER:，只保留最终答案部分。
        if "ANSWER:" in generated_answer:
            generated_answer = generated_answer.rsplit("ANSWER:", 1)[-1].strip()

        # Step 5.8: 构造 judge prompt。
        judge_prompt = get_judge_prompt(
            question_type=question_type,
            question_id=question_id,
            question=question_text,
            answer=answer,
            response=generated_answer,
            question_date=question_date_human,
        )

        # Step 5.9: 调用 judge LLM 判断答案是否正确。
        correct, judge_raw = await judge_llm.judge_yes_no(judge_prompt)

        # Step 5.10: 转换 judge 结果。
        score = 1.0 if correct else 0.0
        judgment = "PASS" if correct else "FAIL"

        # Step 5.11: 保存当前 cutoff 结果。
        cutoff_results[label] = {
            "judgment": judgment,
            "score": score,
            "generated_answer": generated_answer,
            "judge_raw": judge_raw,
            "memories_evaluated": len(sliced),
            "reason": f"Generated answer: {generated_answer[:500]}",
        }

    # Step 6: 将生成的 cutoff_results 写回 result。
    result["cutoff_results"] = cutoff_results


async def apply_longmemeval_retrieval_judge_to_saved_result(
    result: dict,
    judge_llm: LLMClient,
    cutoffs: list[int],
) -> None:
    """Fill ``cutoff_results`` using retrieval-judge prompts (no Mem0)."""

    # Step 1: 从已有 result 中读取 search_results。
    # 这个函数只重新 judge，不再调用 Mem0 search。
    formatted = list(result["retrieval"]["search_results"])

    # Step 2: 读取问题基础字段。
    question_text = result["question"]
    answer = str(result["ground_truth_answer"])
    question_date = result.get("question_date", "")
    user_profile = result.get("user_profile")

    # Step 3: 初始化 cutoff_results。
    cutoff_results: dict[str, dict] = {}

    # Step 4: 遍历每个 cutoff。
    for c in cutoffs:
        # Step 4.1: 取前 c 条检索结果。
        sliced = formatted[:c]

        # Step 4.2: 构造 cutoff label。
        label = cutoff_label(c)

        # Step 4.3: 构造 retrieval judge prompt。
        prompt = get_retrieval_judge_prompt(
            question=question_text,
            answer=answer,
            search_results=sliced,
            question_date=question_date,
            user_profile=user_profile,
        )

        # Step 4.4: 调用 judge LLM 生成结构化判断结果。
        raw = await judge_llm.generate_structured(
            system=RETRIEVAL_JUDGE_SYSTEM,
            user=prompt,
        )

        # Step 4.5: 解析 judgment 字段。
        if isinstance(raw, dict):
            judgment_str = raw.get("judgment", "").upper()
            passed = judgment_str == "PASS"
        else:
            # Step 4.6: 返回结构异常时，默认判失败。
            passed = False

        # Step 4.7: 转换为 score 和 judgment。
        score = 1.0 if passed else 0.0
        judgment = "PASS" if passed else "FAIL"

        # Step 4.8: 保存当前 cutoff 的 retrieval judge 结果。
        cutoff_results[label] = {
            "judgment": judgment,
            "score": score,
            "generated_answer": raw.get("supporting_evidence", "") if isinstance(raw, dict) else "",
            "memories_evaluated": len(sliced),
            "reason": raw.get("reason", "") if isinstance(raw, dict) else "",
            "core_intent": raw.get("core_intent", "") if isinstance(raw, dict) else "",
            "core_intent_supported": raw.get("core_intent_supported", False) if isinstance(raw, dict) else False,
        }

    # Step 5: 将 cutoff_results 写回 result。
    result["cutoff_results"] = cutoff_results

def longmemeval_predict_outputs_complete(
    output_dir: str,
    question_ids: list[str],
) -> tuple[bool, list[str]]:
    # Step 1: 初始化 missing 列表。
    # 用于记录缺失、不可读、或没有 search_results 的 question_id。
    missing: list[str] = []

    # Step 2: 遍历预期应该存在预测结果的所有 question_id。
    for qid in question_ids:
        # Step 2.1: 构造该 question 对应的结果文件路径。
        path = os.path.join(output_dir, f"{qid}.json")

        # Step 2.2: 如果文件不存在，则记录为 missing。
        if not os.path.isfile(path):
            missing.append(qid)
            continue

        try:
            # Step 2.3: 尝试读取并解析 JSON 文件。
            data = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            # Step 2.4: 如果 JSON 不可读或文件读取失败，也记录为 missing。
            missing.append(f"{qid} (unreadable)")
            continue

        # Step 2.5: 检查 retrieval 字段。
        retr = data.get("retrieval") or {}

        # Step 2.6: 如果没有 search_results，说明 predict 输出不完整。
        if "search_results" not in retr:
            missing.append(f"{qid} (no search_results)")

    # Step 3: 如果 missing 为空，说明所有输出完整。
    # 返回：
    # - complete: bool
    # - missing: list[str]
    return len(missing) == 0, missing


# ===============================================================================
# METRICS + DISPLAY
# ===============================================================================


def compute_longmemeval_metrics(
    evaluations: list[dict],
    cutoffs: list[int],
) -> dict:
    """Compute per-question-type and overall metrics at each cutoff."""

    # Step 1: 初始化总的 metrics 容器。
    # 最终结构大致是：
    # {
    #   "top_5": {
    #       "overall": {...},
    #       "by_question_type": {...}
    #   },
    #   "top_10": {...}
    # }
    metrics_by_cutoff = {}

    # Step 2: 遍历每一个 cutoff。
    # cutoff 表示只看前 c 条检索结果时的评测效果。
    for c in cutoffs:
        # Step 2.1: 将 cutoff 数字转换成 label。
        # 例如 c=5 可能变成 "top_5"。
        label = cutoff_label(c)

        # Step 2.2: 统计当前参与评测的问题总数。
        total = len(evaluations)

        # Step 2.3: 提取当前 cutoff 下每个 evaluation 的 score。
        # 如果某条 evaluation 没有 cutoff_results / label / score，
        # 则默认按 0.0 处理。
        scores = [
            e.get("cutoff_results", {}).get(label, {}).get("score", 0.0)
            for e in evaluations
        ]

        # Step 2.4: 统计正确数量。
        # 这里约定 score >= 0.5 就算正确 / PASS。
        correct = sum(1 for s in scores if s >= 0.5)

        # Step 3: 按 question_type 分组收集 score。
        # key 是 question_type，value 是该类型下所有问题的 score 列表。
        by_type: dict[str, list] = defaultdict(list)

        # Step 3.1: 遍历所有 evaluation。
        for e in evaluations:
            # Step 3.2: 获取问题类型。
            # 如果没有 question_type，则归为 "unknown"。
            qtype = e.get("question_type", "unknown")

            # Step 3.3: 将当前 cutoff 下的 score 加入对应 question_type 分组。
            by_type[qtype].append(
                e.get("cutoff_results", {}).get(label, {}).get("score", 0.0)
            )

        # Step 4: 初始化每种 question_type 的指标容器。
        type_metrics = {}

        # Step 4.1: 按 question_type 字母序遍历，保证输出顺序稳定。
        for qtype in sorted(by_type):
            # Step 4.2: 取出当前类型下所有分数。
            type_scores = by_type[qtype]

            # Step 4.3: 统计当前类型下正确数量。
            # 同样使用 score >= 0.5 作为正确标准。
            type_correct = sum(1 for s in type_scores if s >= 0.5)

            # Step 4.4: 计算当前 question_type 的指标。
            # total: 当前类型问题总数
            # correct: 当前类型正确数
            # accuracy: 正确率，百分比形式
            # avg_score: 平均分，百分比形式
            type_metrics[qtype] = {
                "total": len(type_scores),
                "correct": type_correct,
                "accuracy": type_correct / len(type_scores) * 100 if type_scores else 0.0,
                "avg_score": statistics.mean(type_scores) * 100 if type_scores else 0.0,
            }

        # Step 5: 汇总当前 cutoff 下的 overall 指标和按类型指标。
        metrics_by_cutoff[label] = {
            "overall": {
                # Step 5.1: 当前 cutoff 下参与评测的问题总数。
                "total": total,

                # Step 5.2: 当前 cutoff 下整体正确数量。
                "correct": correct,

                # Step 5.3: 当前 cutoff 下整体 accuracy。
                # 如果 total=0，则返回 0.0，避免除零。
                "accuracy": correct / total * 100 if total else 0.0,

                # Step 5.4: 当前 cutoff 下整体平均 score。
                # scores 为空时返回 0.0。
                "avg_score": statistics.mean(scores) * 100 if scores else 0.0,
            },

            # Step 5.5: 当前 cutoff 下，按 question_type 细分的指标。
            "by_question_type": type_metrics,
        }

    # Step 6: 返回所有 cutoff 的指标结果。
    return metrics_by_cutoff


def display_results(metrics_by_cutoff: dict, cutoffs: list[int]) -> None:
    """Print metrics to console."""

    # Step 7: 将 compute_longmemeval_metrics() 计算好的指标打印到控制台。
    # 该函数不做计算，只负责展示。

    # Step 7.1: 按传入的 cutoffs 顺序逐个展示。
    for c in cutoffs:
        # Step 7.2: 将 cutoff 转换成 label，例如 "top_5"。
        label = cutoff_label(c)

        # Step 7.3: 取出当前 cutoff 对应的 metrics。
        # 如果不存在，则使用空 dict 兜底。
        m = metrics_by_cutoff.get(label, {})

        # Step 7.4: 取出 overall 指标。
        overall = m.get("overall", {})

        # Step 7.5: 打印当前 cutoff 标题。
        print(f"\n--- {label} ---")

        # Step 7.6: 打印整体结果。
        # 格式类似：
        # Overall: 42/50 (84.0%) avg=84.0%
        print(
            f"  Overall: {overall.get('correct', 0)}/{overall.get('total', 0)} "
            f"({overall.get('accuracy', 0):.1f}%) "
            f"avg={overall.get('avg_score', 0):.1f}%"
        )

        # Step 7.7: 遍历并打印每种 question_type 的结果。
        # sorted(...) 保证展示顺序稳定。
        for qtype, tm in sorted(m.get("by_question_type", {}).items()):
            # Step 7.8: 打印当前 question_type 的正确数、总数和 accuracy。
            print(
                f"  {qtype}: {tm['correct']}/{tm['total']} ({tm['accuracy']:.1f}%)"
            )

# ===============================================================================
# CLI
# ===============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark: ingest + search + answer + judge",
    )
    parser.add_argument(
        "--project-name", required=True,
        help="Name for this eval run",
    )
    parser.add_argument(
        "--answerer-model", default="gpt-5",
        help="Model for answer generation",
    )
    parser.add_argument(
        "--judge-model", default="gpt-5",
        help="Model for judging",
    )
    parser.add_argument(
        "--provider", default="openai",
        help="LLM provider (openai, anthropic, azure)",
    )
    parser.add_argument(
        "--judge-provider", default=None,
        help="Judge provider (defaults to --provider)",
    )
    parser.add_argument(
        "--mode", default="answerer", choices=["retrieval", "answerer"],
        help="Evaluation mode: retrieval (judge memories) or answerer (generate+judge)",
    )
    parser.add_argument(
        "--top-k", type=int, default=200,
        help="Number of search results to retrieve",
    )
    parser.add_argument(
        "--top-k-cutoffs", default="10,20,50,200",
        help="Comma-separated cutoffs for evaluation",
    )
    parser.add_argument(
        "--max-workers", type=int, default=10,
        help="Max parallel workers",
    )
    parser.add_argument(
        "--output-dir", default="results/longmemeval",
        help="Output directory",
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Skip answer+judge, only ingest+search",
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Judge only: requires all predict JSONs on disk. No Mem0.",
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="With --evaluate-only: re-run judge even if cutoff_results exist",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from checkpoint (default: True)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--score-debug", action="store_true",
        help="Include score breakdowns in output",
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help="Path to local longmemeval dataset JSON",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Reuse a specific run_id for resume",
    )
    parser.add_argument(
        "--all-questions", action="store_true",
        help="Process all 500 questions (ignores --per-type)",
    )
    parser.add_argument(
        "--per-type", type=int, default=5,
        help="Questions to sample per question_type (default: 5, yielding 30)",
    )
    parser.add_argument(
        "--question-types", default=None,
        help="Comma-separated question types to include (default: all 6)",
    )
    parser.add_argument(
        "--user-profile", action="store_true",
        help="Fetch user profiles for use in prompts",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified sampling",
    )
    parser.add_argument(
        "--rpm", type=int, default=200,
        help="Requests per minute for LLM",
    )
    parser.add_argument(
        "--backend", default="oss", choices=["oss", "cloud"],
        help="Mem0 backend: 'oss' for self-hosted server (default), 'cloud' for api.mem0.ai",
    )
    parser.add_argument(
        "--mem0-host", default=None,
        help="Mem0 server URL",
    )
    parser.add_argument(
        "--mem0-api-key", default=None,
        help="Mem0 API key (cloud mode only)",
    )
    return parser.parse_args()


# ===============================================================================
# MAIN
# ===============================================================================


async def async_main() -> None:
    # Step 1: 解析命令行参数。
    # args 中包含 benchmark 模式、模型配置、数据集路径、top_k、并发数、输出目录等配置。
    args = parse_args()

    # Step 2: 初始化日志系统。
    # 如果 args.debug=True，则开启更详细的 debug 日志。
    logger = setup_logging("longmemeval", debug=args.debug)

    # Step 3: 解析 top_k_cutoffs。
    # 例如用户传入 "5,10,20"，这里会转成可用于评测的 cutoff 列表。
    cutoffs = parse_cutoffs(args.top_k_cutoffs)

    # Step 4: 解析要处理的问题类型。
    # 如果 args.question_types 非空，就按逗号拆分成列表；
    # 如果为空，则 selected_types=None，表示不限制问题类型。
    selected_types = (
        [t.strip() for t in args.question_types.split(",") if t.strip()]
        if args.question_types
        else None
    )

    # Step 5: 确定本次运行的 run_id。
    # 如果用户传了 args.run_id，就用用户指定的；
    # 否则随机生成一个 8 位短 uuid。
    run_id = args.run_id or uuid.uuid4().hex[:8]

    # Step 6: 构造本次预测结果输出目录。
    # 目录名格式为 predicted_{project_name}。
    output_dir = os.path.join(args.output_dir, f"predicted_{args.project_name}")

    # Step 7: 确保输出目录存在。
    os.makedirs(output_dir, exist_ok=True)

    # Step 8: 打印本次 benchmark 的基本运行信息。
    print(f"LongMemEval Benchmark | project={args.project_name} run_id={run_id}")
    print(f"  Mode: {args.mode}")
    print(f"  Answerer: {args.answerer_model} ({args.provider})")
    print(f"  Judge: {args.judge_model} ({args.judge_provider or args.provider})")
    print(f"  Cutoffs: {args.top_k_cutoffs}")
    print(f"  Top-K: {args.top_k}")

    # Step 9: 加载数据集。
    # Load dataset
    if args.dataset_path:
        # Step 9.1: 如果用户指定了数据集路径，则直接使用该路径。
        dataset_path = args.dataset_path
    else:
        # Step 9.2: 否则下载默认数据集。
        dataset_path = download_dataset(DEFAULT_DATASET_DIR, logger)

    # Step 9.3: 从数据集路径中加载所有问题。
    all_questions = load_dataset(dataset_path)

    # Step 9.4: 打印加载到的问题总数。
    print(f"  Dataset: {len(all_questions)} questions loaded")

    # Step 10: 根据参数决定处理全部问题，还是抽样处理部分问题。
    # Sample / filter questions
    if args.all_questions:
        # Step 10.1: 如果指定处理全部问题，并且指定了问题类型，则按类型过滤。
        if selected_types:
            questions_to_process = [
                q for q in all_questions
                if q["question_type"] in set(selected_types)
            ]
        else:
            # Step 10.2: 如果没有指定类型，则处理全部问题。
            questions_to_process = all_questions

        # Step 10.3: 打印将要处理的问题数量。
        print(f"  Processing all {len(questions_to_process)} questions")
    else:
        # Step 10.4: 如果不处理全部问题，则按问题类型分层抽样。
        questions_to_process = sample_questions_stratified(
            all_questions,
            per_type=args.per_type,
            seed=args.seed,
            selected_types=selected_types,
        )

        # Step 10.5: 打印抽样后的问题数量。
        print(
            f"  Sampled {len(questions_to_process)} questions "
            f"({args.per_type} per type)"
        )

    # Step 11: 统计并打印本次处理的问题类型分布。
    # Print type distribution
    type_counts: dict[str, int] = defaultdict(int)
    for q in questions_to_process:
        type_counts[q["question_type"]] += 1

    for qtype in sorted(type_counts.keys()):
        print(f"    {qtype}: {type_counts[qtype]}")

    # Step 12: 初始化 answerer LLM。
    # answerer 负责根据检索到的 memory 生成答案。
    answerer = LLMClient(
        model=args.answerer_model, provider=args.provider, rpm=args.rpm,
    )

    # Step 13: 初始化 judge LLM。
    # 如果用户指定了 judge_provider，就用 judge_provider；
    # 否则默认和 answerer 使用同一个 provider。
    judge_provider = args.judge_provider or args.provider
    judge_llm = LLMClient(
        model=args.judge_model, provider=judge_provider, rpm=args.rpm,
    )

    # Step 14: 如果开启 evaluate_only，则只做评测，不再重新执行 Mem0 ingest/search。
    if args.evaluate_only:
        # Step 14.1: 如果没有要处理的问题，直接结束。
        if not questions_to_process:
            print("No questions in scope.")
            return

        # Step 14.2: 提取本次应该评测的问题 id。
        expected_ids = [q["question_id"] for q in questions_to_process]

        # Step 14.3: 检查预测结果文件是否已经全部存在。
        complete, missing = longmemeval_predict_outputs_complete(output_dir, expected_ids)

        # Step 14.4: 如果预测结果不完整，则不能 evaluate-only，直接退出。
        if not complete:
            print(
                "Evaluate-only aborted: not all predict outputs are on disk. "
                "Finish ingest+search for every in-scope question first."
            )
            print(f"  Missing or invalid: {len(missing)} (showing up to 25): {missing[:25]}")
            return

        # Step 14.5: 预测结果完整，开始只跑 judge 阶段。
        print(f"  Predict complete ({len(expected_ids)} questions). Running judge phase (no Mem0)...")

        # Step 14.6: 创建并发控制 semaphore，限制 judge 并发数量。
        sem = asyncio.Semaphore(args.max_workers)

        # Step 14.7: 初始化进度统计。
        progress = {"done": 0, "total": len(questions_to_process)}

        # Step 14.8: 初始化 live_scores，用于实时展示不同 cutoff 下的通过率。
        live_scores = {
            cutoff_label(c): {"seen": 0, "passed": 0}
            for c in cutoffs
        }

        # Step 14.9: 创建异步锁，保护进度更新。
        progress_lock = asyncio.Lock()

        # Step 14.10: 初始化进度条。
        pbar = tqdm(total=progress["total"], desc="Rejudge", leave=True)

        # Step 14.11: 定义内部函数，用于更新进度条右侧的实时分数摘要。
        def update_progress_postfix(data: dict) -> None:
            # Step 14.11.1: 取出当前问题不同 cutoff 下的 judge 结果。
            cutoff_results = data.get("cutoff_results", {})

            # Step 14.11.2: 遍历每个 cutoff，累计 seen / passed。
            for label in live_scores:
                result = cutoff_results.get(label)
                if not result:
                    continue

                live_scores[label]["seen"] += 1

                # Step 14.11.3: score >= 0.5 视为通过。
                if result.get("score", 0.0) >= 0.5:
                    live_scores[label]["passed"] += 1

            # Step 14.11.4: 构造进度条 postfix 中展示的通过率摘要。
            summary = {}
            for label, stats in live_scores.items():
                seen = stats["seen"]
                if not seen:
                    continue
                summary[label.replace("top_", "t")] = f"{(stats['passed'] / seen) * 100:.1f}%"

            # Step 14.11.5: 如果有摘要，则更新进度条展示。
            if summary:
                pbar.set_postfix(summary)

        # Step 14.12: 定义内部异步函数，负责重新 judge 单个问题。
        async def judge_one(question: dict) -> None:
            # Step 14.12.1: 获取问题 id，并定位该问题已有的预测结果 JSON 文件。
            qid = question["question_id"]
            path = os.path.join(output_dir, f"{qid}.json")

            # Step 14.12.2: 读取已有预测结果。
            data = json.loads(Path(path).read_text())

            # Step 14.12.3: 如果该结果已经有 cutoff_results，并且没有指定 rejudge，则跳过重评。
            if data.get("cutoff_results") and not args.rejudge:
                async with progress_lock:
                    update_progress_postfix(data)
                    progress["done"] += 1
                    pbar.update(1)
                return

            # Step 14.12.4: 使用 semaphore 控制 judge 并发。
            async with sem:
                # Step 14.12.5: 如果是 retrieval 模式，使用 retrieval judge。
                if args.mode == "retrieval":
                    await apply_longmemeval_retrieval_judge_to_saved_result(
                        data, judge_llm, cutoffs,
                    )
                else:
                    # Step 14.12.6: 否则使用 answerer judge。
                    await apply_longmemeval_answerer_judge_to_saved_result(
                        data, answerer, judge_llm, cutoffs,
                    )

                # Step 14.12.7: 保存更新后的 judge 结果。
                save_result_json(path, data)

            # Step 14.12.8: 更新全局进度条和 live score。
            async with progress_lock:
                update_progress_postfix(data)
                progress["done"] += 1
                pbar.update(1)

        # Step 14.13: 并发执行所有问题的 judge_one。
        await asyncio.gather(*[judge_one(q) for q in questions_to_process])

        # Step 14.14: 关闭进度条。
        pbar.close()

        # Step 14.15: 重新读取所有评测结果。
        all_evaluations = [
            json.loads(Path(os.path.join(output_dir, f"{qid}.json")).read_text())
            for qid in expected_ids
        ]

        # Step 14.16: 计算 LongMemEval 指标。
        metrics = compute_longmemeval_metrics(all_evaluations, cutoffs)

        # Step 14.17: 在终端展示指标结果。
        display_results(metrics, cutoffs)

        # Step 14.18: 记录 metadata 中使用的 run_id。
        run_id_meta = args.run_id or run_id

        # Step 14.19: 生成统一结果文件名。
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unified_path = os.path.join(
            args.output_dir, f"longmemeval_results_{timestamp}.json",
        )

        # Step 14.20: 保存统一汇总结果 JSON。
        save_result_json(unified_path, {
            "metadata": {
                "benchmark": "longmemeval",
                "project_name": args.project_name",
                "run_id": run_id_meta,
                "timestamp": timestamp,
                "mode": args.mode,
                "answerer_model": args.answerer_model,
                "judge_model": args.judge_model,
                "provider": args.provider,
                "top_k": args.top_k,
                "top_k_cutoffs": [cutoff_label(c) for c in cutoffs],
                "total_questions": len(all_evaluations),
                "question_types": sorted(type_counts.keys()),
                "all_questions": args.all_questions,
                "per_type": args.per_type,
                "seed": args.seed,
                "evaluate_only": True,
            },
            "metrics_by_cutoff": metrics,
            "evaluations": all_evaluations,
        })

        # Step 14.21: 打印结果路径和评测数量。
        print(f"\nResults saved to: {unified_path}")
        print(f"\nTotal questions evaluated: {len(all_evaluations)}")

        # Step 14.22: evaluate_only 模式到这里结束。
        return

    # Step 15: 非 evaluate_only 模式下，初始化 Mem0 后端。
    # 优先读取环境变量 MEM0_BACKEND，否则使用 args.backend。
    backend = os.getenv("MEM0_BACKEND", args.backend)

    # Step 16: 创建 Mem0Client。
    # 如果 backend == "cloud"，则传入 api_key；
    # 否则本地模式不使用 api_key。
    mem0 = Mem0Client(
        mode=backend,
        host=args.mem0_host,
        api_key=args.mem0_api_key if backend == "cloud" else None,
        rpm=args.rpm,
    )

    # Step 17: 初始化优雅退出控制器。
    # 用于处理中断信号，避免任务中途硬退出。
    shutdown = GracefulShutdown()

    # Step 18: 初始化 checkpoint 管理器。
    # 注意：这段代码里 checkpoint 被创建，但在当前片段中没有继续使用。
    checkpoint = Checkpoint(output_dir)

    # Step 19: 初始化评测结果列表。
    all_evaluations: list[dict] = []

    # Step 20: 如果开启 resume，则从输出目录中加载已有结果。
    if args.resume:
        # Step 20.1: 遍历输出目录下所有 json 文件。
        for p in sorted(Path(output_dir).glob("*.json")):
            # Step 20.2: 跳过以下划线开头的内部文件。
            if p.name.startswith("_"):
                continue

            try:
                # Step 20.3: 读取已有结果文件。
                data = json.loads(p.read_text())

                # Step 20.4: 如果文件里有 question_type，则认为是有效问题结果。
                if data.get("question_type"):
                    all_evaluations.append(data)
            except (json.JSONDecodeError, KeyError):
                # Step 20.5: 如果 JSON 损坏或字段异常，跳过该文件。
                continue

        # Step 20.6: 打印恢复出的已有结果数量。
        print(f"  Loaded {len(all_evaluations)} existing results")

    # Step 21: 构造已有 question_id 集合，用于避免重复处理。
    existing_ids = {e["question_id"] for e in all_evaluations}

    # Step 22: 进入 Mem0Client 异步上下文。
    # 这里负责打开/关闭 Mem0 相关资源。
    async with mem0:
        # Step 23: 进入优雅退出上下文。
        with shutdown:
            # Step 24: 创建锁，用于保护 all_evaluations 和 existing_ids 的并发写入。
            results_lock = asyncio.Lock()

            # Step 25: 创建问题级 semaphore，限制同时处理的问题数量。
            question_semaphore = asyncio.Semaphore(args.max_workers)

            # Step 26: 初始化处理进度。
            progress = {"done": 0, "total": len(questions_to_process)}

            # Step 27: 创建问题处理进度条。
            pbar = tqdm(total=progress["total"], desc="Questions", leave=True)

            # Step 28: 从已有结果中找出 predict-only 结果。
            # 这些结果已经有 retrieval/search 数据，但还没有 cutoff_results。
            # Build lookup of predict-only results (have search data but no cutoff_results)
            predict_only_results = {
                e["question_id"]: e for e in all_evaluations
                if "retrieval" in e and "cutoff_results" not in e
            }

            # Step 29: 定义单个问题的完整处理流程。
            async def process_single_question(question: dict):
                # Step 29.1: 用 semaphore 限制并发问题数量。
                async with question_semaphore:
                    # Step 29.2: 如果收到退出信号，直接返回。
                    if shutdown.requested:
                        return

                    # Step 29.3: 获取当前问题 id。
                    question_id = question["question_id"]

                    # Step 29.4: 加锁检查当前问题是否已经处理过。
                    async with results_lock:
                        if question_id in existing_ids:
                            progress["done"] += 1
                            pbar.update(1)
                            return

                    # Step 29.5: 检查当前问题是否已有 predict-only 结果。
                    # Check if we have predict-only results (search data already exists)
                    existing_predict = predict_only_results.get(question_id)

                    if existing_predict and existing_predict.get("retrieval"):
                        # Step 29.6: 如果已有 search 结果，则跳过 ingest+search，
                        # 后面直接复用已有 retrieval 结果。
                        # Skip ingest+search, use existing search results
                        user_id = existing_predict.get("user_id", f"longmemeval_{question_id}_{run_id}")
                        user_profile = None
                    else:
                        # Step 29.7: 如果没有可复用结果，则先执行 Mem0 ingestion。
                        # --- Ingest ---
                        success, user_id, pairs = await ingest_question(
                            question=question,
                            mem0=mem0,
                            logger=logger,
                            run_id=run_id,
                            output_dir=output_dir,
                            shutdown=shutdown,
                            debug=args.debug,
                        )

                        # Step 29.8: 如果 ingestion 失败，记录错误。
                        if not success:
                            logger.error(
                                "Ingestion failed for question %s", question_id,
                            )

                        # Step 29.9: ingestion 后再次检查是否收到退出信号。
                        if shutdown.requested:
                            return

                        # Step 29.10: 标记没有已有 predict 结果，后续需要 fresh search。
                        existing_predict = None  # will search fresh below

                    # Step 29.11: 如果用户要求 user_profile，则从 Mem0 读取 user profile。
                    # Fetch user profile if requested
                    user_profile = None
                    if args.user_profile:
                        user_profile = await mem0.get_user_profile(user_id)

                    # Step 29.12: 根据 mode 执行 retrieval-only 或 answerer 模式。
                    # --- Search + Answer/Judge ---
                    if args.mode == "retrieval":
                        # Step 29.13: retrieval 模式：
                        # 只处理检索结果，并用 judge 评估检索是否命中答案。
                        result = await process_question_retrieval(
                            question=question,
                            user_id=user_id,
                            mem0=mem0,
                            judge_llm=judge_llm,
                            cutoffs=cutoffs,
                            top_k=args.top_k,
                            user_profile=user_profile,
                            predict_only=args.predict_only,
                            logger=logger,
                            score_debug=args.score_debug,
                        )
                    else:
                        # Step 29.14: answerer 模式：
                        # 使用检索结果作为上下文，让 answerer 生成答案，再用 judge 评估答案。
                        # Use existing search results from predict-only run if available
                        existing_search = None

                        # Step 29.15: 如果有已有 predict-only retrieval 结果，则复用其中的 search_results。
                        if existing_predict and existing_predict.get("retrieval"):
                            existing_search = existing_predict["retrieval"].get("search_results", [])

                        # Step 29.16: 执行 answerer 流程。
                        result = await process_question_answerer(
                            question=question,
                            user_id=user_id,
                            mem0=mem0,
                            answerer=answerer,
                            judge_llm=judge_llm,
                            cutoffs=cutoffs,
                            top_k=args.top_k,
                            user_profile=user_profile,
                            predict_only=args.predict_only,
                            logger=logger,
                            score_debug=args.score_debug,
                            existing_search_results=existing_search,
                        )

                    # Step 29.17: 保存当前问题的结果到单独 JSON 文件。
                    # Save per-question result
                    result_path = os.path.join(output_dir, f"{question_id}.json")
                    save_result_json(result_path, result)

                    # Step 29.18: 加锁更新全局结果列表和已处理 id 集合。
                    async with results_lock:
                        all_evaluations.append(result)
                        existing_ids.add(question_id)

                    # Step 29.19: 更新进度条。
                    pbar.update(1)

            # Step 30: 为所有问题创建异步任务。
            tasks = [process_single_question(q) for q in questions_to_process]

            # Step 31: 并发执行所有问题处理任务。
            await asyncio.gather(*tasks)

            # Step 32: 所有任务结束后关闭进度条。
            pbar.close()

    # Step 33: 如果不是 predict_only，并且已有评测结果，则计算最终指标。
    # --- Metrics ---
    if not args.predict_only and all_evaluations:
        # Step 33.1: 按 question_id 去重。
        # 如果同一个 question_id 出现多次，保留最后一次结果。
        # Deduplicate by question_id, keeping the latest (last) entry
        seen = {}
        for e in all_evaluations:
            seen[e.get("question_id")] = e

        # Step 33.2: 得到去重后的结果列表。
        deduped = list(seen.values())

        # Step 33.3: 检查是否有 cutoff_results。
        # 只有已经 judge 过的结果才会包含 cutoff_results。
        has_cutoffs = any("cutoff_results" in e for e in deduped)

        if has_cutoffs:
            # Step 33.4: 计算 LongMemEval 指标。
            metrics = compute_longmemeval_metrics(deduped, cutoffs)

            # Step 33.5: 打印指标。
            display_results(metrics, cutoffs)

            # Step 33.6: 保存统一汇总结果。
            # Save unified result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unified_path = os.path.join(
                args.output_dir, f"longmemeval_results_{timestamp}.json",
            )

            # Step 33.7: 写入统一结果 JSON 文件。
            save_result_json(unified_path, {
                "metadata": {
                    "benchmark": "longmemeval",
                    "project_name": args.project_name,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "mode": args.mode,
                    "answerer_model": args.answerer_model,
                    "judge_model": args.judge_model,
                    "provider": args.provider,
                    "top_k": args.top_k,
                    "top_k_cutoffs": [cutoff_label(c) for c in cutoffs],
                    "total_questions": len(all_evaluations),
                    "question_types": sorted(type_counts.keys()),
                    "all_questions": args.all_questions,
                    "per_type": args.per_type,
                    "seed": args.seed,
                },
                "metrics_by_cutoff": metrics,
                "evaluations": all_evaluations,
            })

            # Step 33.8: 打印统一结果保存路径。
            print(f"\nResults saved to: {unified_path}")

    # Step 34: 打印本次最终处理的问题数量。
    print(f"\nTotal questions processed: {len(all_evaluations)}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
