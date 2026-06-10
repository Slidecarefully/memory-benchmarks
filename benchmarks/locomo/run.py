# Source: :contentReference[oaicite:0]{index=0}

"""
LOCOMO Benchmark Runner
=======================

Combined ingest + search + answer + judge pipeline for the LOCOMO-10
benchmark (Snap Research, ACL 2024).

Flow:
    1. Download dataset (auto-download from GitHub if missing)
    2. For each conversation:
        a. Parse into sessions, ingest via Mem0
        b. For each question:
            - Search Mem0 -> retrieved memories
            - Generate answer (answerer model)
            - Judge answer vs ground truth (judge model)
            - Save checkpoint
    3. Compute metrics (by category, by cutoff)
    4. Write unified result JSON

Usage:
    python -m benchmarks.locomo.run --project-name test
    python -m benchmarks.locomo.run --project-name full --answerer-model gpt-4o --judge-model gpt-4o
    python -m benchmarks.locomo.run --project-name full --predict-only
"""

# Step 1: 启用 postponed evaluation of annotations。
# 作用是让类型注解延迟求值，减少运行时循环引用或前向引用问题。
from __future__ import annotations

# Step 2: 导入标准库。
# argparse 用于解析命令行参数；
# asyncio 用于异步并发执行 ingest/search/judge；
# json/os/re/statistics/sys/time/uuid 用于文件、正则、统计、时间、ID 等基础能力。
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import uuid

# Step 3: 导入 defaultdict，用于按类别聚合 metrics。
from collections import defaultdict

# Step 4: 导入 datetime/timezone，用于解析 LOCOMO session 时间并转成 epoch。
from datetime import datetime, timezone

# Step 5: 导入 Path，用于更方便地读取/遍历 JSON 文件。
from pathlib import Path

# Step 6: 导入 Any，用于部分参数的宽泛类型标注，例如 logger。
from typing import Any

# Step 7: 加载 .env 环境变量。
# 例如 OpenAI key、Mem0 backend 配置等可能来自环境变量。
from dotenv import load_dotenv

# Step 8: 导入 tqdm，用于显示 ingest/search/judge 进度条。
from tqdm import tqdm

# Step 9: 导入 benchmark 公共 LLM client。
# answerer 和 judge 都通过这个封装调用不同 provider 的模型。
from benchmarks.common.llm_client import LLMClient

# Step 10: 导入 Mem0 client 以及搜索结果格式化函数。
# Mem0Client 负责调用 Mem0 add/search/profile；
# format_search_results 负责把 Mem0 返回结果整理成统一 JSON 结构。
from benchmarks.common.mem0_client import Mem0Client, format_search_results

# Step 11: 导入通用 metrics 计算函数。
# 注意：当前代码片段中 compute_overall_metrics 被导入，但后面实际没有使用。
from benchmarks.common.metrics import compute_overall_metrics

# Step 12: 导入统一结果 schema。
# 这些 schema 当前片段里主要是为了类型/统一输出结构准备；
# 实际后文更多直接使用 dict 结构。
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

# Step 13: 导入 benchmark 通用工具函数。
# 包括 checkpoint、优雅退出、cutoff label、下载文件、解析 cutoff、保存 JSON、日志初始化等。
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

# Step 14: 导入 LOCOMO 专用 prompt 和常量。
# 这里包含：
# - category 名称映射
# - judge system prompt
# - answer generation prompt
# - judge prompt
# - 带 evidence 的 judge prompt
# - answer 预处理函数
from .prompts import (
    CATEGORIES_TO_EVALUATE,
    CATEGORY_NAMES,
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt,
    get_judge_prompt_with_evidence,
    preprocess_answer,
)

# Step 15: 加载 .env 文件，并允许覆盖已有环境变量。
load_dotenv(override=True)

# ===============================================================================
# CONSTANTS
# ===============================================================================

# Step 16: LOCOMO-10 数据集下载地址。
DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

# Step 17: 默认数据集目录。
DEFAULT_DATASET_DIR = "datasets/locomo"

# Step 18: 默认数据集文件名。
DEFAULT_DATASET_FILE = "locomo10.json"

# Step 19: 每个 ingestion chunk 包含多少 turn。
# 这里是 1，表示每条 message 单独作为一次 mem0.add 的 chunk。
CHUNK_SIZE = 1  # turns per ingestion chunk


# ===============================================================================
# DATASET
# ===============================================================================


def download_dataset(dataset_dir: str, logger: Any) -> str:
    """Download locomo10.json from GitHub if not present."""

    # Step 20: 计算数据集本地路径。
    path = os.path.join(dataset_dir, DEFAULT_DATASET_FILE)

    # Step 21: 如果本地文件已存在，直接复用，不重复下载。
    if os.path.exists(path):
        logger.info("Dataset already exists: %s", path)
        return path

    # Step 22: 如果目录不存在，则先创建目录。
    os.makedirs(dataset_dir, exist_ok=True)

    # Step 23: 打印下载日志。
    logger.info("Downloading LOCOMO-10 dataset...")

    # Step 24: 从 GitHub 下载 LOCOMO-10 数据集到本地 path。
    download_file(DATASET_URL, path, description="Downloading LOCOMO-10")

    # Step 25: 下载后读取 JSON，进行格式校验。
    with open(path) as f:
        data = json.load(f)

    # Step 26: 校验 LOCOMO-10 应该是 list，并且恰好包含 10 个 conversations。
    # 如果不符合，说明下载文件损坏或 URL 内容不对。
    if not isinstance(data, list) or len(data) != 10:
        # Step 26.1: 删除错误文件，避免下次误用。
        os.remove(path)

        # Step 26.2: 抛出明确异常。
        raise RuntimeError(f"Invalid dataset: expected 10 conversations, got {len(data)}")

    # Step 27: 记录下载成功日志。
    logger.info("Downloaded: %s (%d conversations)", path, len(data))

    # Step 28: 返回本地数据集路径。
    return path


def load_dataset(path: str) -> list[dict]:
    # Step 29: 从指定路径读取 LOCOMO 数据集 JSON。
    # 返回值是 list[dict]，每个 dict 对应一个 conversation entry。
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===============================================================================
# CONVERSATION PARSING
# ===============================================================================


def parse_locomo_date(date_str: str) -> datetime | None:
    """Parse LOCOMO date: '1:56 pm on 8 May, 2023'."""

    # Step 30: LOCOMO 日期可能使用完整月份名，也可能使用月份缩写。
    # 所以这里尝试两个格式：
    # - "%I:%M %p on %d %B, %Y"  例如 1:56 PM on 8 May, 2023
    # - "%I:%M %p on %d %b, %Y"  例如 1:56 PM on 8 May, 2023 / abbreviated month
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            # Step 30.1: 尝试按当前格式解析日期字符串。
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            # Step 30.2: 当前格式失败则尝试下一个格式。
            continue

    # Step 30.3: 所有格式都失败则返回 None。
    return None


def locomo_date_to_epoch(date_str: str) -> int | None:
    # Step 31: 将 LOCOMO 日期字符串转成 Unix epoch 秒。
    parsed = parse_locomo_date(date_str)

    # Step 31.1: 如果日期解析成功，则将其视为 UTC 时间并转成 timestamp。
    if parsed:
        return int(parsed.replace(tzinfo=timezone.utc).timestamp())

    # Step 31.2: 解析失败则返回 None。
    return None


def get_sorted_sessions(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    """Extract and sort sessions chronologically."""

    # Step 32: 找出 conversation 中所有 session_x 形式的 key。
    # 例如 session_1、session_2，但不包含 session_1_date_time。
    session_keys = [k for k in conversation if re.match(r"^session_\d+$", k)]

    # Step 33: 初始化配对列表，每项是：
    # (session_key, date_str, turns)
    paired = []

    # Step 34: 遍历每个 session key，找出对应日期和 turns。
    for key in session_keys:
        # Step 34.1: LOCOMO 中 session 的日期字段通常是 session_x_date_time。
        date_key = f"{key}_date_time"

        # Step 34.2: 读取日期字符串。
        date_str = conversation.get(date_key, "")

        # Step 34.3: 读取该 session 的 turn 列表。
        turns = conversation[key]

        # Step 34.4: 加入 paired。
        paired.append((key, date_str, turns))

    # Step 35: 定义 session 排序 key。
    def sort_key(item: tuple) -> tuple:
        # Step 35.1: 尝试解析 session 的日期字符串。
        parsed = parse_locomo_date(item[1])

        # Step 35.2: 如果日期可解析，则优先按真实时间排序。
        if parsed:
            return (0, parsed)

        # Step 35.3: 如果日期不可解析，则退化为按 session 编号排序。
        num = int(re.search(r"\d+", item[0]).group())
        return (1, datetime(2000, 1, num))

    # Step 36: 原地排序，使 sessions 按时间顺序进入 ingestion。
    paired.sort(key=sort_key)

    # Step 37: 返回排序后的 session 列表。
    return paired


def session_to_chunks(turns: list[dict], speaker_a: str, speaker_b: str) -> list[list[dict]]:
    """Convert turns to message chunks for ingestion."""

    # Step 38: 将 LOCOMO 原始 turns 转成 Mem0 add() 需要的 messages。
    messages = []

    # Step 39: 遍历 session 中每个 turn。
    for turn in turns:
        # Step 39.1: 取出说话人、文本、图片描述、图片 query。
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        blip = turn.get("blip_caption", "")
        query = turn.get("query", "")

        # Step 39.2: 如果既有图片 query 又有图片描述，则构造完整图片标签。
        if query and blip:
            photo_tag = f"[Sharing image - query: {query}. The image shows: {blip}]"

        # Step 39.3: 如果只有 query，则说明用户分享了和 query 相关的图片。
        elif query:
            photo_tag = f"[Sharing image - query for: {query}]"

        # Step 39.4: 如果只有 blip_caption，则说明图片内容由 caption 描述。
        elif blip:
            photo_tag = f"[Sharing image that shows: {blip}]"

        # Step 39.5: 没有图片信息则不添加 photo_tag。
        else:
            photo_tag = ""

        # Step 39.6: 如果有图片标签，把它拼接到文本后面；
        # 如果原始文本为空，则直接使用 photo_tag。
        if photo_tag:
            text = f"{text} {photo_tag}" if text else photo_tag

        # Step 39.7: 如果最终 text 仍为空，则跳过这个 turn。
        if not text:
            continue

        # Step 39.8: 将 speaker_a 映射为 user，其余 speaker 映射为 assistant。
        role = "user" if speaker == speaker_a else "assistant"

        # Step 39.9: 保存为 Mem0 标准 message 格式。
        # content 中保留 speaker 名称，避免丢失说话人身份。
        messages.append({"role": role, "content": f"{speaker}: {text}"})

    # Step 40: 按 CHUNK_SIZE 将 messages 切成 chunks。
    chunks = []

    # Step 41: 每 CHUNK_SIZE 条 message 组成一个 chunk。
    for i in range(0, len(messages), CHUNK_SIZE):
        chunk = messages[i : i + CHUNK_SIZE]

        # Step 41.1: 非空 chunk 才加入。
        if chunk:
            chunks.append(chunk)

    # Step 42: 返回 chunk 列表，后续每个 chunk 会调用一次 mem0.add。
    return chunks


# ===============================================================================
# EVIDENCE HELPERS
# ===============================================================================


def load_evidence_lookup(dataset_path: str) -> dict[tuple, str]:
    """Build lookup: (conv_idx, dia_id) -> formatted turn text."""

    # Step 43: 读取 LOCOMO 数据集，用于构建 evidence lookup。
    # evidence lookup 的作用是把 qa 中的 dia_id 引用映射回原始对话文本。
    with open(dataset_path) as f:
        data = json.load(f)

    # Step 44: 初始化 lookup。
    # key: (conv_idx, dia_id)
    # value: 格式化后的证据文本。
    lookup = {}

    # Step 45: 遍历每个 conversation。
    for conv_idx, conv in enumerate(data):
        # Step 45.1: 取出 conversation 内容。
        conversation = conv["conversation"]

        # Step 45.2: 先收集 session_num -> session_date 的映射。
        session_dates = {}

        # Step 45.3: 遍历 conversation 中所有 key，找 session_x_date_time。
        for key in conversation:
            if key.endswith("_date_time") and key.startswith("session_"):
                # Step 45.4: 从 key 中解析 session 编号。
                session_num = key.replace("session_", "").replace("_date_time", "")

                # Step 45.5: 保存该 session 的日期。
                session_dates[session_num] = conversation[key]

        # Step 46: 再遍历所有 session，构造 dia_id -> 原始 turn 文本。
        for key in conversation:
            # Step 46.1: 只处理 session_x，不处理 session_x_date_time。
            if key.startswith("session_") and not key.endswith("date_time"):
                # Step 46.2: 如果当前字段不是 turn list，则跳过。
                if not isinstance(conversation[key], list):
                    continue

                # Step 46.3: 遍历 session 中每个 turn。
                for turn in conversation[key]:
                    # Step 46.4: 取出 dia_id。
                    dia_id = turn.get("dia_id", "")

                    # Step 46.5: 如果没有 dia_id，就无法作为 evidence 引用，跳过。
                    if dia_id:
                        # Step 46.6: 取出 speaker 和 text。
                        speaker = turn.get("speaker", "")
                        text = turn.get("text", "")

                        # Step 46.7: 从 dia_id 中解析 D 后面的数字，用于定位 session date。
                        dia_match = re.match(r"D(\d+):", dia_id)

                        # Step 46.8: 初始化日期后缀。
                        date_suffix = ""

                        # Step 46.9: 如果 dia_id 中有 session 编号，则查对应 session 日期。
                        if dia_match:
                            snum = dia_match.group(1)
                            sdate = session_dates.get(snum, "")

                            # Step 46.10: 如果有 session 日期，则加入 evidence 文本中。
                            if sdate:
                                date_suffix = f", said on {sdate}"

                        # Step 46.11: 构造 evidence lookup 文本。
                        lookup[(conv_idx, dia_id)] = f'[{dia_id}{date_suffix}] {speaker}: "{text}"'

    # Step 47: 返回完整 evidence lookup。
    return lookup


# ===============================================================================
# INGESTION
# ===============================================================================


async def ingest_conversation(
    conv_idx: int,
    entry: dict,
    mem0: Mem0Client,
    logger: Any,
    run_id: str,
    project_name: str,
    output_dir: str,
    shutdown: GracefulShutdown,
    debug: bool = True,
) -> tuple[bool, str, int]:
    """Ingest all sessions of a LOCOMO conversation into Mem0.

    Returns: (success, user_id, total_chunks_processed)
    """

    # Step 48: 取出 conversation 数据和两位说话人名称。
    conversation = entry["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    # Step 49: 为每个 conversation 构造独立 user_id。
    # 这样不同 conversation 的 memories 不会互相污染。
    user_id = f"locomo_{conv_idx}_{run_id}"

    # Step 50: 初始化 ingestion checkpoint。
    # 用于断点续跑，避免重复 ingest 已完成 chunks。
    checkpoint = IngestionCheckpoint(output_dir)
    key = str(conv_idx)

    # Step 51: 检查当前 conversation 是否已经完整 ingest。
    # Check if already complete
    is_done, cp_data = checkpoint.is_complete(key, CHUNK_SIZE)

    # Step 52: 如果已完成，直接返回 checkpoint 中记录的结果。
    if is_done and cp_data:
        chunks_done = cp_data.get("total_chunks_processed", 0)
        user_id = cp_data.get("user_id", user_id)
        logger.info("Conversation %d already ingested (user_id=%s, %d chunks)", conv_idx, user_id, chunks_done)
        return True, user_id, chunks_done

    # Step 53: 如果没完整完成，则检查是否有部分进度。
    # Check for partial progress
    chunks_already_done, resumed_uid = checkpoint.load_progress(key, CHUNK_SIZE)

    # Step 54: 如果存在部分进度，并且 checkpoint 里保存了 user_id，则继续使用旧 user_id。
    if resumed_uid and chunks_already_done:
        user_id = resumed_uid
        logger.info("Resuming conversation %d from %d completed chunks", conv_idx, len(chunks_already_done))

    # Step 55: 将 conversation 的 sessions 按时间顺序排序。
    sorted_sessions = get_sorted_sessions(conversation)

    # Step 56: 计算总 chunk 数，用于进度条。
    total_chunks = sum(len(session_to_chunks(s, speaker_a, speaker_b)) for _, _, s in sorted_sessions)

    # Step 57: 打印 ingestion 开始日志。
    logger.info(
        "Ingesting conversation %d: %s & %s, %d sessions, %d chunks",
        conv_idx, speaker_a, speaker_b, len(sorted_sessions), total_chunks,
    )

    # Step 58: 如果开启 debug，则创建 debug ingestion 日志文件。
    # Debug log
    debug_file = None
    if debug:
        # Step 58.1: 创建 debug 目录。
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Step 58.2: 当前 conversation 的 debug 文件路径。
        debug_path = os.path.join(debug_dir, f"conv_{conv_idx}_ingestion.txt")

        # Step 58.3: 如果是续跑则 append，否则重新写入。
        debug_mode = "a" if chunks_already_done else "w"
        debug_file = open(debug_path, debug_mode, encoding="utf-8")

        # Step 58.4: 非续跑时写入 debug 文件头部。
        if not chunks_already_done:
            debug_file.write(f"{'=' * 80}\n")
            debug_file.write(f"CONVERSATION {conv_idx}: {speaker_a} & {speaker_b}\n")
            debug_file.write(f"Sessions: {len(sorted_sessions)}, Chunks: {total_chunks}\n")
            debug_file.write(f"User ID: {user_id}\n")
            debug_file.write(f"{'=' * 80}\n\n")

    # Step 59: 创建 ingestion 进度条。
    # initial=len(chunks_already_done) 可以让断点续跑时进度条从已完成数量开始。
    pbar = tqdm(total=total_chunks, desc=f"Ingest conv {conv_idx}", initial=len(chunks_already_done), leave=True)

    # Step 60: 初始化处理计数。
    total_processed = len(chunks_already_done)
    total_failed = 0

    # Step 61: 遍历排序后的每个 session。
    for session_key, date_str, turns in sorted_sessions:
        # Step 61.1: 将当前 session 转成 Mem0 ingestion chunks。
        chunks = session_to_chunks(turns, speaker_a, speaker_b)

        # Step 61.2: 如果当前 session 没有有效 chunk，跳过。
        if not chunks:
            continue

        # Step 61.3: 将 session 日期转为 epoch timestamp。
        # 后续传给 mem0.add，作为 memory 时间信息。
        session_epoch = locomo_date_to_epoch(date_str)

        # Step 61.4: 如果开启 debug，并且当前 session header 没写过，则写入 session header。
        if debug_file and f"{session_key}_header" not in chunks_already_done:
            debug_file.write(f"\n{'---' * 27}\n")
            debug_file.write(f"SESSION: {session_key}  |  Date: {date_str}  |  Chunks: {len(chunks)}\n")
            debug_file.write(f"{'---' * 27}\n\n")

        # Step 62: 遍历当前 session 的每个 chunk。
        for chunk_idx, messages in enumerate(chunks):
            # Step 62.1: 构造当前 chunk 的唯一 key。
            chunk_key = f"{session_key}_c{chunk_idx}"

            # Step 62.2: 如果该 chunk 已经完成，则跳过。
            if chunk_key in chunks_already_done:
                continue

            # Step 62.3: 如果收到优雅退出信号，则关闭资源并返回当前进度。
            if shutdown.requested:
                logger.info("Shutdown requested at conv %d, chunk %s", conv_idx, chunk_key)
                pbar.close()
                if debug_file:
                    debug_file.close()
                return True, user_id, total_processed

            # Step 62.4: 如果 chunk 里有空 content，则跳过并标记已处理。
            # Skip empty messages
            if any(not msg.get("content", "").strip() for msg in messages):
                chunks_already_done.add(chunk_key)
                total_processed += 1
                pbar.update(1)
                continue

            # Step 62.5: 如果开启 debug，则写入当前 chunk 的原始 messages。
            if debug_file:
                debug_file.write(f"--- Chunk {chunk_idx} ({len(messages)} messages) ---\n")
                for msg in messages:
                    debug_file.write(f"  {msg['role']}: {msg['content']}\n")
                debug_file.write("\n")

            # Step 62.6: 调用 mem0.add 写入当前 chunk。
            # 这里会触发 Mem0 的 memory extraction + vector store insert 流程。
            response = await mem0.add(messages, user_id, timestamp=session_epoch)

            # Step 62.7: 如果 response 不为 None，认为当前 chunk 处理成功。
            if response is not None:
                total_processed += 1

                # Step 62.8: 如果开启 debug，则把 Mem0 抽取出的 memories 写入 debug 文件。
                if debug_file:
                    results = response.get("results", [])
                    if results:
                        debug_file.write(f"--- Chunk {chunk_idx} (extracted) ---\n")
                        for mem_item in results:
                            mem_text = mem_item.get("memory", "")
                            event_type = mem_item.get("event", "")
                            debug_file.write(f"  [{event_type}] {mem_text}\n")
                        debug_file.write("\n")
            else:
                # Step 62.9: 如果 mem0.add 返回 None，记录失败。
                total_failed += 1
                logger.warning("Ingestion failed: conv %d %s chunk %d", conv_idx, session_key, chunk_idx)

            # Step 62.10: 无论成功或失败，都把当前 chunk 标记为已完成。
            chunks_already_done.add(chunk_key)

            # Step 62.11: 保存当前 conversation 的部分进度。
            checkpoint.save_progress(key, {
                "conversation_idx": conv_idx,
                "user_id": user_id,
                "run_id": run_id,
                "chunk_size": CHUNK_SIZE,
                "completed_chunks": list(chunks_already_done),
            })

            # Step 62.12: 更新进度条。
            pbar.update(1)

    # Step 63: 所有 session/chunk 处理结束，关闭进度条。
    pbar.close()

    # Step 64: 如果开启 debug，则写入 summary 并关闭 debug 文件。
    if debug_file:
        debug_file.write(f"\nSUMMARY: {total_processed}/{total_chunks} OK, {total_failed} failed\n")
        debug_file.close()

    # Step 65: 保存完整完成 checkpoint。
    checkpoint.save_complete(key, {
        "conversation_idx": conv_idx,
        "user_id": user_id,
        "run_id": run_id,
        "chunk_size": CHUNK_SIZE,
        "total_chunks_processed": total_processed,
        "total_chunks_failed": total_failed,
    })

    # Step 66: 返回 ingestion 状态。
    # total_failed == 0 表示整体成功。
    return total_failed == 0, user_id, total_processed


# ===============================================================================
# SEARCH + ANSWER + JUDGE
# ===============================================================================


async def process_question(
    qa: dict,
    qa_idx: int,
    conv_idx: int,
    user_id: str,
    mem0: Mem0Client,
    answerer: LLMClient,
    judge_llm: LLMClient,
    cutoffs: list[int],
    top_k: int,
    reference_date_human: str | None,
    user_profile: dict | None,
    evidence_lookup: dict | None,
    predict_only: bool,
    logger: Any,
    score_debug: bool = False,
) -> dict[str, Any]:
    """Process a single question: search + answer + judge at multiple cutoffs.

    Returns a result dict suitable for serialization.
    """

    # Step 67: 为当前问题构造唯一 question_id。
    question_id = f"conv{conv_idx}_q{qa_idx}"

    # Step 68: 读取问题字段。
    question = qa["question"]
    category = qa["category"]
    answer = str(qa["answer"])

    # Step 69: 执行 Mem0 search。
    # --- Search ---
    start = time.monotonic()
    search_results = await mem0.search(question, user_id, top_k=top_k, score_debug=score_debug)

    # Step 70: 计算 search latency，单位毫秒。
    search_latency = (time.monotonic() - start) * 1000

    # Step 71: 格式化 Mem0 search 结果。
    formatted, query_debug = format_search_results(search_results)

    # Step 72: 构造当前问题的基础 result。
    result: dict[str, Any] = {
        "question_id": question_id,
        "conversation_idx": conv_idx,
        "category": category,
        "category_name": CATEGORY_NAMES.get(category, "unknown"),
        "question": question,
        "ground_truth_answer": answer,
        "evidence": qa.get("evidence", []),
        "user_id": user_id,
        "reference_date": reference_date_human,
        "retrieval": {
            "search_query": question,
            "search_results": formatted,
            "search_latency_ms": round(search_latency, 1),
            "total_results": len(formatted),
        },
    }

    # Step 73: 如果有 query_debug，则加入 retrieval。
    if query_debug:
        result["retrieval"]["query_debug"] = query_debug

    # Step 74: 如果有 user_profile，则加入 result。
    if user_profile:
        result["user_profile"] = user_profile

    # Step 75: predict_only 模式只保存检索结果，不生成答案、不 judge。
    if predict_only:
        return result

    # Step 76: 初始化不同 cutoff 下的结果。
    # --- Answer + Judge at each cutoff ---
    cutoff_results: dict[str, dict] = {}

    # Step 77: 根据 LOCOMO category 对 ground truth answer 做预处理。
    processed_answer = preprocess_answer(category, answer)

    # Step 78: 如果有 evidence_lookup，则构造 evidence context。
    # Build evidence context if available
    ev_ctx = ""
    if evidence_lookup:
        for ref in qa.get("evidence", []):
            key = (conv_idx, ref)
            if key in evidence_lookup:
                ev_ctx += evidence_lookup[key] + "\n"
        ev_ctx = ev_ctx.strip()

    # Step 79: 遍历每个 cutoff，例如 top_10/top_20/top_50/top_200。
    for c in cutoffs:
        # Step 79.1: 只取前 c 条 retrieved memories。
        sliced = formatted[:c]

        # Step 79.2: 生成 cutoff label。
        label = cutoff_label(c)

        # Step 79.3: 构造 answer generation prompt。
        # Generate answer
        gen_prompt = get_answer_generation_prompt(question, sliced, reference_date=reference_date_human, user_profile=user_profile)

        # Step 79.4: 调用 answerer 生成答案。
        generated_answer = await answerer.generate(system="", user=gen_prompt)

        # Step 79.5: 如果模型输出中包含 ANSWER:，只保留后面的答案文本。
        if "ANSWER:" in generated_answer:
            generated_answer = generated_answer.rsplit("ANSWER:", 1)[-1].strip()

        # Step 79.6: 构造 judge prompt。
        # Judge
        if ev_ctx:
            # Step 79.6.1: 如果有 evidence context，则使用带 evidence 的 judge prompt。
            judge_prompt = get_judge_prompt_with_evidence(category, question, processed_answer, generated_answer, ev_ctx)
        else:
            # Step 79.6.2: 否则使用普通 judge prompt。
            judge_prompt = get_judge_prompt(category, question, processed_answer, generated_answer)

        # Step 79.7: 调用 judge LLM，要求结构化输出。
        raw = await judge_llm.generate_structured(
            system=JUDGE_SYSTEM_PROMPT,
            user=judge_prompt,
        )

        # Step 79.8: 解析 judge 返回。
        if isinstance(raw, dict):
            label_val = raw.get("label", "").upper()
            correct = label_val == "CORRECT"
        else:
            # Step 79.9: 如果返回不是 dict，则视为错误。
            correct = False

        # Step 79.10: 将 correct 转成数值分数和文本 judgment。
        score = 1.0 if correct else 0.0
        judgment = "CORRECT" if correct else "WRONG"

        # Step 79.11: 保存当前 cutoff 的评测结果。
        cutoff_results[label] = {
            "judgment": judgment,
            "score": score,
            "generated_answer": generated_answer,
            "memories_evaluated": len(sliced),
            "reason": raw.get("reasoning", "") if isinstance(raw, dict) else "",
        }

    # Step 80: 将 cutoff_results 写入 result。
    result["cutoff_results"] = cutoff_results

    # Step 81: 返回单题完整结果。
    return result


async def apply_locomo_judge_to_saved_result(
    result: dict,
    qa: dict,
    conv_idx: int,
    answerer: LLMClient,
    judge_llm: LLMClient,
    cutoffs: list[int],
    evidence_lookup: dict | None,
) -> None:
    """Fill ``cutoff_results`` using ``retrieval.search_results`` only (no Mem0)."""

    # Step 82: evaluate-only 模式下，直接读取已保存的 search_results。
    # 这个函数不会重新调用 Mem0。
    formatted = list(result["retrieval"]["search_results"])

    # Step 83: 读取问题和答案信息。
    question = result["question"]
    category = qa["category"]
    answer = str(qa["answer"])
    reference_date_human = result.get("reference_date")
    user_profile = result.get("user_profile")

    # Step 84: 初始化 cutoff_results。
    cutoff_results: dict[str, dict] = {}

    # Step 85: 对标准答案做 category 相关预处理。
    processed_answer = preprocess_answer(category, answer)

    # Step 86: 如果有 evidence_lookup，则构造 evidence context。
    ev_ctx = ""
    if evidence_lookup:
        for ref in qa.get("evidence", []):
            key = (conv_idx, ref)
            if key in evidence_lookup:
                ev_ctx += evidence_lookup[key] + "\n"
        ev_ctx = ev_ctx.strip()

    # Step 87: 遍历每个 cutoff，基于已保存检索结果重新 answer + judge。
    for c in cutoffs:
        # Step 87.1: 取前 c 条 retrieval results。
        sliced = formatted[:c]

        # Step 87.2: 构造 cutoff label。
        label = cutoff_label(c)

        # Step 87.3: 构造 answer generation prompt。
        gen_prompt = get_answer_generation_prompt(
            question, sliced, reference_date=reference_date_human, user_profile=user_profile,
        )

        # Step 87.4: 调用 answerer 生成答案。
        generated_answer = await answerer.generate(system="", user=gen_prompt)

        # Step 87.5: 如果输出包含 ANSWER:，只保留最终答案部分。
        if "ANSWER:" in generated_answer:
            generated_answer = generated_answer.rsplit("ANSWER:", 1)[-1].strip()

        # Step 87.6: 构造 judge prompt。
        if ev_ctx:
            judge_prompt = get_judge_prompt_with_evidence(
                category, question, processed_answer, generated_answer, ev_ctx,
            )
        else:
            judge_prompt = get_judge_prompt(category, question, processed_answer, generated_answer)

        # Step 87.7: 调用 judge LLM。
        raw = await judge_llm.generate_structured(
            system=JUDGE_SYSTEM_PROMPT,
            user=judge_prompt,
        )

        # Step 87.8: 解析结构化 judge 结果。
        if isinstance(raw, dict):
            label_val = raw.get("label", "").upper()
            correct = label_val == "CORRECT"
        else:
            correct = False

        # Step 87.9: 转成分数和 judgment。
        score = 1.0 if correct else 0.0
        judgment = "CORRECT" if correct else "WRONG"

        # Step 87.10: 保存当前 cutoff 结果。
        cutoff_results[label] = {
            "judgment": judgment,
            "score": score,
            "generated_answer": generated_answer,
            "memories_evaluated": len(sliced),
            "reason": raw.get("reasoning", "") if isinstance(raw, dict) else "",
        }

    # Step 88: 将重新生成的 cutoff_results 写回 result。
    result["cutoff_results"] = cutoff_results


def expected_locomo_question_items(
    dataset: list[dict],
    conv_indices: list[int],
    categories: list[int],
    max_questions: int | None,
) -> list[tuple[str, int, int, dict]]:
    """(question_id, conv_idx, qa_idx, qa_dict) for every question in scope."""

    # Step 89: 构造 evaluate-only 预期要处理的问题列表。
    items: list[tuple[str, int, int, dict]] = []

    # Step 90: 遍历用户指定的 conversation index。
    for conv_idx in conv_indices:
        # Step 90.1: 如果 conv_idx 超出数据集范围，则跳过。
        if conv_idx >= len(dataset):
            continue

        # Step 90.2: 取出当前 conversation entry。
        entry = dataset[conv_idx]

        # Step 90.3: LOCOMO 问题字段可能是 qa 或 qa_pairs。
        questions = entry.get("qa", entry.get("qa_pairs", []))

        # Step 90.4: 按 category 过滤问题。
        conv_questions = [
            (qi, qa) for qi, qa in enumerate(questions)
            if qa.get("category") in categories
        ]

        # Step 90.5: 如果指定 max_questions，则只取前 max_questions 个。
        if max_questions is not None:
            conv_questions = conv_questions[:max_questions]

        # Step 90.6: 为每个问题构造统一 item。
        for qi, qa in conv_questions:
            items.append((f"conv{conv_idx}_q{qi}", conv_idx, qi, qa))

    # Step 91: 返回所有预期问题。
    return items


def locomo_predict_outputs_complete(
    output_dir: str,
    expected_items: list[tuple[str, int, int, dict]],
) -> tuple[bool, list[str]]:
    """True if every expected question has JSON with retrieval.search_results."""

    # Step 92: 检查 predict-only 输出是否完整。
    missing: list[str] = []

    # Step 93: 遍历所有预期 question item。
    for qid, _, _, _ in expected_items:
        # Step 93.1: 构造该问题的 JSON 结果路径。
        path = os.path.join(output_dir, f"{qid}.json")

        # Step 93.2: 如果文件不存在，则记录 missing。
        if not os.path.isfile(path):
            missing.append(qid)
            continue

        try:
            # Step 93.3: 读取并解析 JSON。
            data = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            # Step 93.4: 如果文件不可读或 JSON 格式错误，也记录 missing。
            missing.append(f"{qid} (unreadable)")
            continue

        # Step 93.5: 检查 retrieval 字段中是否有 search_results。
        retr = data.get("retrieval") or {}
        if "search_results" not in retr:
            missing.append(f"{qid} (no search_results)")

    # Step 94: 如果 missing 为空，则说明输出完整。
    return len(missing) == 0, missing


# ===============================================================================
# METRICS + DISPLAY
# ===============================================================================


def compute_locomo_metrics(evaluations: list[dict], cutoffs: list[int]) -> dict:
    """Compute per-category and overall metrics at each cutoff."""

    # Step 95: 初始化按 cutoff 组织的 metrics。
    metrics_by_cutoff = {}

    # Step 96: 遍历每个 cutoff。
    for c in cutoffs:
        # Step 96.1: 生成 cutoff label。
        label = cutoff_label(c)

        # Step 96.2: 当前评测问题总数。
        total = len(evaluations)

        # Step 96.3: 提取当前 cutoff 下每个 evaluation 的 score。
        scores = [e.get("cutoff_results", {}).get(label, {}).get("score", 0.0) for e in evaluations]

        # Step 96.4: score >= 0.5 视为正确。
        correct = sum(1 for s in scores if s >= 0.5)

        # Step 97: 按 category_name 分组收集 score。
        by_category: dict[str, list] = defaultdict(list)
        for e in evaluations:
            cat_name = e.get("category_name", "unknown")
            by_category[cat_name].append(e.get("cutoff_results", {}).get(label, {}).get("score", 0.0))

        # Step 98: 计算每个 category 的指标。
        cat_metrics = {}
        for cat_name in sorted(by_category):
            cat_scores = by_category[cat_name]
            cat_correct = sum(1 for s in cat_scores if s >= 0.5)
            cat_metrics[cat_name] = {
                "total": len(cat_scores),
                "correct": cat_correct,
                "accuracy": cat_correct / len(cat_scores) * 100 if cat_scores else 0.0,
                "avg_score": statistics.mean(cat_scores) * 100 if cat_scores else 0.0,
            }

        # Step 99: 写入当前 cutoff 的 overall 和 by_category 指标。
        metrics_by_cutoff[label] = {
            "overall": {
                "total": total,
                "correct": correct,
                "accuracy": correct / total * 100 if total else 0.0,
                "avg_score": statistics.mean(scores) * 100 if scores else 0.0,
            },
            "by_category": cat_metrics,
        }

    # Step 100: 返回所有 cutoff 的 metrics。
    return metrics_by_cutoff


def display_results(metrics_by_cutoff: dict, cutoffs: list[int]) -> None:
    """Print metrics to console."""

    # Step 101: 按 cutoff 顺序打印 metrics。
    for c in cutoffs:
        # Step 101.1: 生成 cutoff label。
        label = cutoff_label(c)

        # Step 101.2: 取出当前 cutoff 的 metrics。
        m = metrics_by_cutoff.get(label, {})

        # Step 101.3: 取出 overall 指标。
        overall = m.get("overall", {})

        # Step 101.4: 打印 cutoff 标题。
        print(f"\n--- {label} ---")

        # Step 101.5: 打印整体正确率和平均分。
        print(f"  Overall: {overall.get('correct', 0)}/{overall.get('total', 0)} "
              f"({overall.get('accuracy', 0):.1f}%) avg={overall.get('avg_score', 0):.1f}%")

        # Step 101.6: 打印每个 category 的结果。
        for cat_name, cm in sorted(m.get("by_category", {}).items()):
            print(f"  {cat_name}: {cm['correct']}/{cm['total']} ({cm['accuracy']:.1f}%)")


# ===============================================================================
# CLI
# ===============================================================================


def parse_args() -> argparse.Namespace:
    # Step 102: 创建命令行参数解析器。
    parser = argparse.ArgumentParser(
        description="Run LOCOMO-10 benchmark: ingest + search + answer + judge",
    )

    # Step 103: 添加项目名参数，用于区分不同实验输出目录。
    parser.add_argument("--project-name", required=True, help="Name for this eval run")

    # Step 104: answerer 和 judge 模型配置。
    parser.add_argument("--answerer-model", default="gpt-5", help="Model for answer generation")
    parser.add_argument("--judge-model", default="gpt-5", help="Model for judging")

    # Step 105: LLM provider 配置。
    parser.add_argument("--provider", default="openai", help="LLM provider (openai, anthropic, azure)")
    parser.add_argument("--judge-provider", default=None, help="Judge provider (defaults to --provider)")

    # Step 106: conversation 范围配置。
    parser.add_argument("--conversations", default="0,1,2,3,4,5,6,7,8,9", help="Comma-separated conversation indices")

    # Step 107: 检索 top_k 和评估 cutoffs。
    parser.add_argument("--top-k", type=int, default=200, help="Number of search results to retrieve")
    parser.add_argument("--top-k-cutoffs", default="10,20,50,200", help="Comma-separated cutoffs for evaluation")

    # Step 108: 并发和输出目录配置。
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel workers")
    parser.add_argument("--output-dir", default="results/locomo", help="Output directory")

    # Step 109: predict-only 模式：只 ingest + search，不 answer/judge。
    parser.add_argument("--predict-only", action="store_true", help="Skip answer+judge, only ingest+search")

    # Step 110: evaluate-only 模式：不调用 Mem0，只读取已有 search_results 做 answer/judge。
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Judge only: requires all predict outputs on disk (use after --predict-only or full search). No Mem0.",
    )

    # Step 111: evaluate-only 下是否强制重新 judge。
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="With --evaluate-only: re-run answer+judge even if cutoff_results already exist",
    )

    # Step 112: 断点续跑、debug、score_debug 配置。
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--score-debug", action="store_true", help="Include score breakdowns in output")

    # Step 113: 数据集路径和 run_id 配置。
    parser.add_argument("--dataset-path", default=None, help="Path to local locomo10.json")
    parser.add_argument("--run-id", default=None, help="Reuse a specific run_id for resume")

    # Step 114: LOCOMO category、evidence、user_profile、max_questions 配置。
    parser.add_argument("--categories", default="1,2,3,4", help="Comma-separated categories")
    parser.add_argument("--with-evidence", action="store_true", help="Pass evidence to judge")
    parser.add_argument("--user-profile", action="store_true", help="Fetch user profiles")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions to process (for quick testing)")

    # Step 115: LLM RPM 限制。
    parser.add_argument("--rpm", type=int, default=200, help="Requests per minute for LLM")

    # Step 116: Mem0 backend 配置。
    parser.add_argument("--backend", default="oss", choices=["oss", "cloud"],
                        help="Mem0 backend: 'oss' for self-hosted server (default), 'cloud' for api.mem0.ai")

    # Step 117: Mem0 server host 配置。
    parser.add_argument("--mem0-host", default=None,
                        help="Mem0 server URL (default: http://localhost:8888 for oss, https://api.mem0.ai for cloud)")

    # Step 118: Mem0 cloud API key。
    parser.add_argument("--mem0-api-key", default=None,
                        help="Mem0 API key (cloud mode only)")

    # Step 119: 解析并返回命令行参数。
    return parser.parse_args()


# ===============================================================================
# MAIN
# ===============================================================================


async def async_main() -> None:
    # Step 120: 解析命令行参数。
    args = parse_args()

    # Step 121: 初始化日志。
    logger = setup_logging("locomo", debug=args.debug)

    # Step 122: 解析 top_k cutoffs、categories、conversation indices。
    cutoffs = parse_cutoffs(args.top_k_cutoffs)
    categories = [int(c) for c in args.categories.split(",")]
    conv_indices = [int(c) for c in args.conversations.split(",")]

    # Step 123: 生成或复用 run_id。
    run_id = args.run_id or uuid.uuid4().hex[:8]

    # Step 124: 创建当前 project 的输出目录。
    output_dir = os.path.join(args.output_dir, f"predicted_{args.project_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Step 125: 打印运行配置。
    print(f"LOCOMO Benchmark | project={args.project_name} run_id={run_id}")
    print(f"  Answerer: {args.answerer_model} ({args.provider})")
    print(f"  Judge: {args.judge_model} ({args.judge_provider or args.provider})")
    print(f"  Conversations: {args.conversations}")
    print(f"  Cutoffs: {args.top_k_cutoffs}")

    # Step 126: 加载数据集。
    # Load dataset
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = download_dataset(DEFAULT_DATASET_DIR, logger)
    dataset = load_dataset(dataset_path)

    # Step 127: 如果开启 with_evidence，则构建 evidence lookup。
    # Evidence
    evidence_lookup = None
    if args.with_evidence:
        evidence_lookup = load_evidence_lookup(dataset_path)
        print(f"  Evidence lookup: {len(evidence_lookup)} entries")

    # Step 128: 初始化 answerer LLM 和 judge LLM。
    answerer = LLMClient(model=args.answerer_model, provider=args.provider, rpm=args.rpm)
    judge_provider = args.judge_provider or args.provider
    judge_llm = LLMClient(model=args.judge_model, provider=judge_provider, rpm=args.rpm)

    # Step 129: evaluate-only 模式。
    # 不启动 Mem0，不执行 ingest/search，只基于已有 predict outputs 重新 answer/judge。
    if args.evaluate_only:
        # Step 129.1: 构造预期需要评测的问题列表。
        expected_items = expected_locomo_question_items(
            dataset, conv_indices, categories, args.max_questions,
        )

        # Step 129.2: 如果没有问题在范围内，则退出。
        if not expected_items:
            print("No questions in scope (check --conversations / --categories).")
            return

        # Step 129.3: 检查所有预期结果文件是否存在且包含 retrieval.search_results。
        complete, missing = locomo_predict_outputs_complete(output_dir, expected_items)

        # Step 129.4: 如果 predict outputs 不完整，则不能 evaluate-only。
        if not complete:
            print(
                "Evaluate-only aborted: not all predict outputs are on disk. "
                "Finish ingest+search for every in-scope question first "
                "(full run without --predict-only, or --predict-only until complete)."
            )
            print(f"  Missing or invalid: {len(missing)} (showing up to 25): {missing[:25]}")
            return

        # Step 129.5: 输出完整，开始 judge phase。
        print(f"  Predict complete ({len(expected_items)} questions). Running judge phase (no Mem0)...")

        # Step 129.6: 用 semaphore 限制并发 judge 数量。
        sem = asyncio.Semaphore(args.max_workers)

        async def judge_one(qid: str, conv_idx: int, qi: int, qa: dict) -> None:
            # Step 129.7: 读取单题已保存结果。
            path = os.path.join(output_dir, f"{qid}.json")
            data = json.loads(Path(path).read_text())

            # Step 129.8: 如果已经有 cutoff_results 且没有 rejudge，则跳过。
            if data.get("cutoff_results") and not args.rejudge:
                return

            # Step 129.9: 并发受控地执行 answer + judge。
            async with sem:
                await apply_locomo_judge_to_saved_result(
                    data, qa, conv_idx, answerer, judge_llm, cutoffs, evidence_lookup,
                )

                # Step 129.10: 保存更新后的结果。
                save_result_json(path, data)

        # Step 129.11: 并发处理所有 expected items。
        await asyncio.gather(*[
            judge_one(qid, conv_idx, qi, qa)
            for qid, conv_idx, qi, qa in expected_items
        ])

        # Step 129.12: 读取所有评测结果。
        all_evaluations = [
            json.loads(Path(os.path.join(output_dir, f"{qid}.json")).read_text())
            for qid, _, _, _ in expected_items
        ]

        # Step 129.13: 计算并展示 metrics。
        metrics = compute_locomo_metrics(all_evaluations, cutoffs)
        display_results(metrics, cutoffs)

        # Step 129.14: 记录 metadata 中的 run_id。
        run_id_meta = args.run_id or run_id

        # Step 129.15: 保存统一结果文件。
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unified_path = os.path.join(args.output_dir, f"locomo_results_{timestamp}.json")
        save_result_json(unified_path, {
            "metadata": {
                "benchmark": "locomo",
                "project_name": args.project_name,
                "run_id": run_id_meta,
                "timestamp": timestamp,
                "answerer_model": args.answerer_model,
                "judge_model": args.judge_model,
                "provider": args.provider,
                "top_k": args.top_k,
                "top_k_cutoffs": [cutoff_label(c) for c in cutoffs],
                "total_questions": len(all_evaluations),
                "categories": categories,
                "evaluate_only": True,
            },
            "metrics_by_cutoff": metrics,
            "evaluations": all_evaluations,
        })

        # Step 129.16: 打印保存路径和评测数量。
        print(f"\nResults saved to: {unified_path}")
        print(f"\nTotal questions evaluated: {len(all_evaluations)}")
        return

    # Step 130: 非 evaluate-only 模式下初始化 Mem0。
    # Init Mem0 (not used for --evaluate-only)
    backend = os.getenv("MEM0_BACKEND", args.backend)

    # Step 131: 创建 Mem0Client。
    # cloud 模式传 api_key，oss 模式不传。
    mem0 = Mem0Client(
        mode=backend,
        host=args.mem0_host,
        api_key=args.mem0_api_key if backend == "cloud" else None,
        rpm=args.rpm,
    )

    # Step 132: 初始化优雅退出控制器。
    shutdown = GracefulShutdown()

    # Step 133: 初始化 checkpoint。
    # 注意：这里创建了 checkpoint，但当前片段后续没有直接使用这个变量。
    checkpoint = Checkpoint(output_dir)

    # Step 134: 初始化所有评测结果。
    all_evaluations: list[dict] = []

    # Step 135: 如果开启 resume，则加载已有结果。
    if args.resume:
        for p in sorted(Path(output_dir).glob("*.json")):
            # Step 135.1: 跳过内部文件。
            if p.name.startswith("_"):
                continue

            try:
                # Step 135.2: 读取已有 JSON。
                data = json.loads(p.read_text())

                # Step 135.3: 只保留当前 categories 范围内的结果。
                if data.get("category") in categories:
                    all_evaluations.append(data)
            except (json.JSONDecodeError, KeyError):
                # Step 135.4: JSON 损坏或字段缺失时跳过。
                continue

        # Step 135.5: 打印恢复出的已有结果数。
        print(f"  Loaded {len(all_evaluations)} existing results")

    # Step 136: 构造已完成 question_id 集合，用于避免重复处理。
    existing_ids = {e["question_id"] for e in all_evaluations}

    # Step 137: 创建异步锁，保护 all_evaluations 和 existing_ids 的并发写入。
    results_lock = asyncio.Lock()

    # Step 138: 创建 conversation 级 semaphore，限制并发 conversation 数量。
    conv_semaphore = asyncio.Semaphore(args.max_workers)

    async def process_conversation(conv_idx: int):
        """Process a single conversation: ingest → answer questions."""

        # Step 139: 单个 conversation 的完整处理流程。
        async with conv_semaphore:
            # Step 139.1: 如果收到退出信号，直接返回。
            if shutdown.requested:
                return

            # Step 139.2: 如果 conv_idx 超出数据集范围，记录 warning 并跳过。
            if conv_idx >= len(dataset):
                logger.warning("Conversation %d out of range (dataset has %d)", conv_idx, len(dataset))
                return

            # Step 139.3: 取出当前 conversation entry。
            entry = dataset[conv_idx]
            conversation = entry["conversation"]

            # Step 139.4: 先 ingest 当前 conversation 的所有 sessions。
            # --- Ingest ---
            success, user_id, chunks = await ingest_conversation(
                conv_idx, entry, mem0, logger, run_id, args.project_name,
                output_dir, shutdown, debug=args.debug,
            )

            # Step 139.5: 如果 ingestion 失败，记录错误，但仍继续后续流程。
            if not success:
                logger.error("Ingestion failed for conversation %d", conv_idx)

            # Step 139.6: ingestion 后如果收到退出信号，则返回。
            if shutdown.requested:
                return

            # Step 139.7: 如果需要 user_profile，则从 Mem0 获取。
            # Fetch user profile if requested
            user_profile = None
            if args.user_profile:
                user_profile = await mem0.get_user_profile(user_id)

            # Step 139.8: 获取 reference date。
            # 当前实现取排序后最后一个 session 的日期作为 reference date。
            # Get reference date from first session
            sorted_sessions = get_sorted_sessions(conversation)
            ref_date_human = None
            if sorted_sessions:
                ref_date_human = sorted_sessions[-1][1]

            # Step 139.9: 读取当前 conversation 的问题列表。
            # --- Process questions ---
            questions = entry.get("qa", entry.get("qa_pairs", []))

            # Step 139.10: 按 category 过滤问题。
            conv_questions = [
                (qi, qa) for qi, qa in enumerate(questions)
                if qa.get("category") in categories
            ]

            # Step 139.11: 如果指定 max_questions，则截断问题列表。
            if args.max_questions is not None:
                conv_questions = conv_questions[:args.max_questions]

            # Step 139.12: 创建当前 conversation 的问题进度条。
            search_pbar = tqdm(conv_questions, desc=f"Questions conv {conv_idx}", leave=True)

            # Step 139.13: 遍历当前 conversation 的每个问题。
            for qi, qa in search_pbar:
                # Step 139.14: 构造 question_id。
                qid = f"conv{conv_idx}_q{qi}"

                # Step 139.15: 如果收到退出信号，停止当前 conversation。
                if shutdown.requested:
                    break

                # Step 139.16: 如果该问题已经处理过，则跳过。
                # Skip if already done
                async with results_lock:
                    if qid in existing_ids:
                        continue

                # Step 139.17: 执行单题 search + answer + judge。
                result = await process_question(
                    qa=qa,
                    qa_idx=qi,
                    conv_idx=conv_idx,
                    user_id=user_id,
                    mem0=mem0,
                    answerer=answerer,
                    judge_llm=judge_llm,
                    cutoffs=cutoffs,
                    top_k=args.top_k,
                    reference_date_human=ref_date_human,
                    user_profile=user_profile,
                    evidence_lookup=evidence_lookup,
                    predict_only=args.predict_only,
                    logger=logger,
                    score_debug=args.score_debug,
                )

                # Step 139.18: 保存单题结果 JSON。
                # Save per-question result
                result_path = os.path.join(output_dir, f"{qid}.json")
                save_result_json(result_path, result)

                # Step 139.19: 加锁更新全局结果列表和已完成 ID 集合。
                async with results_lock:
                    all_evaluations.append(result)
                    existing_ids.add(qid)

    # Step 140: 进入 Mem0 异步上下文和优雅退出上下文。
    async with mem0:
        with shutdown:
            # Step 140.1: 为每个 conversation 创建异步任务。
            tasks = [process_conversation(idx) for idx in conv_indices]

            # Step 140.2: 并发执行所有 conversation。
            await asyncio.gather(*tasks)

    # Step 141: 如果不是 predict_only，并且有评测结果，则计算 metrics。
    # --- Metrics ---
    if not args.predict_only and all_evaluations:
        # Step 141.1: 检查是否有 cutoff_results。
        has_cutoffs = any("cutoff_results" in e for e in all_evaluations)

        if has_cutoffs:
            # Step 141.2: 计算 LOCOMO metrics。
            metrics = compute_locomo_metrics(all_evaluations, cutoffs)

            # Step 141.3: 打印结果。
            display_results(metrics, cutoffs)

            # Step 141.4: 保存统一结果。
            # Save unified result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unified_path = os.path.join(args.output_dir, f"locomo_results_{timestamp}.json")
            save_result_json(unified_path, {
                "metadata": {
                    "benchmark": "locomo",
                    "project_name": args.project_name,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "answerer_model": args.answerer_model,
                    "judge_model": args.judge_model,
                    "provider": args.provider,
                    "top_k": args.top_k,
                    "top_k_cutoffs": [cutoff_label(c) for c in cutoffs],
                    "total_questions": len(all_evaluations),
                    "categories": categories,
                },
                "metrics_by_cutoff": metrics,
                "evaluations": all_evaluations,
            })

            # Step 141.5: 打印统一结果保存路径。
            print(f"\nResults saved to: {unified_path}")

    # Step 142: 打印最终处理的问题数量。
    print(f"\nTotal questions processed: {len(all_evaluations)}")


def main() -> None:
    # Step 143: 同步入口函数。
    # 使用 asyncio.run 启动 async_main。
    asyncio.run(async_main())


# Step 144: 当该文件作为脚本直接运行时，调用 main()。
if __name__ == "__main__":
    main()
