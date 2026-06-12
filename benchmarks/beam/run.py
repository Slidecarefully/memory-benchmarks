# Source: :contentReference[oaicite:0]{index=0}

"""
BEAM Benchmark Runner
=====================

Combined ingest + search + answer + judge pipeline for the BEAM benchmark
(ICLR 2026).  100 conversations x 4 size buckets (100K--10M tokens),
20 probing questions each spanning 10 memory ability types.

Flow:
    1. Download dataset from HuggingFace (auto-cached locally)
    2. For each conversation:
        a. Parse chat into batches, ingest via Mem0 (chunked)
        b. For each probing question:
            - Search Mem0 -> retrieved memories
            - Generate answer (answerer model)
            - Judge with rubric-based nugget scoring (0/0.5/1.0 per nugget)
            - Optional Kendall tau-b for event_ordering questions
            - Save per-question checkpoint
    3. Compute metrics (by question type, by cutoff)
    4. Write unified result JSON

Usage:
    python -m benchmarks.beam.run --project-name test
    python -m benchmarks.beam.run --project-name full --chat-sizes 100K,500K
    python -m benchmarks.beam.run --project-name test --predict-only
    python -m benchmarks.beam.run --project-name test --evaluate-only
"""

# Step 1: 启用延迟类型注解解析。
# 这样 Python 不会在函数定义时立刻解析所有类型注解，
# 可以减少前向引用、循环导入带来的运行时问题。
from __future__ import annotations

# Step 2: 导入命令行参数解析库。
import argparse

# Step 3: 导入 asyncio，用于异步执行 Mem0 add/search、LLM answer/judge 等 IO 密集型任务。
import asyncio

# Step 4: 导入 ast，用于安全解析 HuggingFace 数据集中可能以字符串形式存储的 Python 字面量。
import ast

# Step 5: 导入 json，用于读写数据集缓存、单题结果、统一评测结果。
import json

# Step 6: 导入 os，用于路径拼接、创建目录、读取环境变量。
import os

# Step 7: 导入 statistics，用于计算平均分。
import statistics

# Step 8: 导入 sys，用于遇到非法 chat size 时退出程序。
import sys

# Step 9: 导入 time，用于统计 search latency。
import time

# Step 10: 导入 uuid，用于生成 run_id。
import uuid

# Step 11: 导入 defaultdict，用于按 question_type 聚合 metrics。
from collections import defaultdict

# Step 12: 导入 datetime/timezone，用于记录结果时间戳、处理时间锚点。
from datetime import datetime, timezone

# Step 13: 导入 Path，用于遍历/读取结果 JSON 文件。
from pathlib import Path

# Step 14: 导入 Any，用于宽泛类型标注，例如 logger、conversation_meta。
from typing import Any

# Step 15: 导入 load_dotenv，用于加载 .env 中的 API key、Mem0 配置等。
from dotenv import load_dotenv

# Step 16: 导入 tqdm，用于显示 ingestion/question processing 进度条。
from tqdm import tqdm

# Step 17: 导入统一 LLMClient。
# answerer 和 judge_llm 都通过这个 client 调用不同 provider/model。
from benchmarks.common.llm_client import LLMClient

# Step 18: 导入 Mem0Client 和搜索结果格式化函数。
# Mem0Client 负责 add/search；format_search_results 负责把检索结果转成统一结构。
from benchmarks.common.mem0_client import Mem0Client, format_search_results

# Step 19: 导入通用 metrics 工具。
# compute_kendall_tau_b 用于 event_ordering 类型问题；
# compute_overall_metrics 在当前片段中被导入，但没有直接使用。
from benchmarks.common.metrics import compute_kendall_tau_b, compute_overall_metrics

# Step 20: 导入通用 schema。
# 这些 schema 主要用于统一评测结果结构，当前代码中更多使用 dict 直接组织结果。
from benchmarks.common.schema import (
    CutoffResult,
    EvalItem,
    NuggetScore,
    Metadata,
    Metrics,
    UnifiedResult,
)

# Step 21: 导入 benchmark 通用工具函数。
# 包括 checkpoint、优雅退出、cutoff label、保存 JSON、日志初始化等。
from benchmarks.common.utils import (
    Checkpoint,
    GracefulShutdown,
    IngestionCheckpoint,
    cutoff_label,
    parse_cutoffs,
    save_result_json,
    setup_logging,
)

# Step 22: 导入 BEAM 专用 prompts 和 question type 常量。
# 这些函数负责构造 answer prompt、nugget judge prompt、事件抽取/对齐 prompt。
from .prompts import (
    BEAM_JUDGE_SYSTEM_PROMPT,
    BEAM_QUESTION_TYPES,
    get_beam_answer_generation_prompt,
    get_beam_event_alignment_prompt,
    get_beam_fact_extraction_prompt,
    get_beam_nugget_judge_prompt,
)

# Step 23: 加载 .env 文件，并允许覆盖已有环境变量。
load_dotenv(override=True)

# ===============================================================================
# CONSTANTS
# ===============================================================================

# Step 24: BEAM 100K/500K/1M 数据集在 HuggingFace 上的名称。
HF_DATASET_NAME = "Mohammadta/BEAM"

# Step 25: BEAM 10M 数据集单独放在另一个 HuggingFace dataset。
HF_DATASET_10M = "Mohammadta/BEAM-10M"

# Step 26: chat size 到 HuggingFace split 名称的映射。
HF_SPLIT_MAP: dict[str, str] = {"100K": "100K", "500K": "500K", "1M": "1M", "10M": "10M"}

# Step 27: 当前 benchmark 支持的 chat size。
VALID_CHAT_SIZES = ["100K", "500K", "1M", "10M"]

# Step 28: 默认数据集缓存目录。
DEFAULT_DATASET_DIR = "datasets/beam"

# Step 29: 每个 ingestion chunk 包含的 turn 数量。
# 这里是 2，通常对应一组 user/assistant 对话片段。
CHUNK_SIZE = 2  # turns per ingestion chunk


# ===============================================================================
# DATASET
# ===============================================================================


def download_dataset(
    chat_sizes: list[str],
    cache_dir: str,
    logger: Any,
) -> dict[str, list[dict]]:
    """Download BEAM dataset from HuggingFace, cache locally.

    Returns:
        Dict mapping chat_size -> list of conversation dicts.
    """

    # Step 30: 确保数据集缓存目录存在。
    os.makedirs(cache_dir, exist_ok=True)

    # Step 31: 初始化总数据集容器。
    # 结构为：
    # {
    #   "100K": [conversation, ...],
    #   "500K": [conversation, ...],
    # }
    dataset: dict[str, list[dict]] = {}

    # Step 32: 遍历用户指定的每个 chat size，逐个下载或读取缓存。
    for size in chat_sizes:
        # Step 32.1: 构造当前 size 的本地缓存文件路径。
        cache_path = os.path.join(cache_dir, f"beam_{size}.json")

        # Step 32.2: 如果本地缓存已存在，直接读取缓存。
        if os.path.exists(cache_path):
            logger.info("Loading cached %s dataset: %s", size, cache_path)
            with open(cache_path, "r", encoding="utf-8") as f:
                dataset[size] = json.load(f)
            continue

        # Step 32.3: 如果没有缓存，则从 HuggingFace 下载。
        logger.info("Downloading BEAM %s dataset from HuggingFace...", size)

        try:
            # Step 32.4: 延迟导入 datasets.load_dataset。
            # 这样只有真的需要下载 HF 数据集时才依赖 datasets 包。
            from datasets import load_dataset as hf_load

            # Step 32.5: 10M 数据集使用单独的 dataset name 和 split。
            if size == "10M":
                ds = hf_load(HF_DATASET_10M, split="10M")
            else:
                # Step 32.6: 其他 size 使用主 BEAM dataset 和对应 split。
                ds = hf_load(HF_DATASET_NAME, split=HF_SPLIT_MAP[size])

            # Step 32.7: 初始化当前 size 的 conversation 列表。
            conversations: list[dict] = []

            # Step 32.8: 遍历 HuggingFace dataset 中的每条 conversation。
            for idx, item in enumerate(ds):
                # Step 32.9: 统一整理 conversation 字段。
                conv: dict[str, Any] = {
                    "conversation_id": item.get("conversation_id", f"{size}_{idx}"),
                    "conversation_seed": item.get("conversation_seed", {}),
                    "user_profile": item.get("user_profile", {}),
                    "chat": item.get("chat", []),
                }

                # probing_questions may be stored as a string repr in HF
                # Step 32.10: 读取 probing_questions。
                # HF 中该字段有时是 dict，有时是字符串形式的 Python repr/JSON。
                pq_raw = item.get("probing_questions", "{}")

                # Step 32.11: 如果 probing_questions 是字符串，尝试解析。
                if isinstance(pq_raw, str):
                    try:
                        # Step 32.12: 优先用 ast.literal_eval 解析 Python 字面量字符串。
                        conv["probing_questions"] = ast.literal_eval(pq_raw)
                    except (ValueError, SyntaxError):
                        try:
                            # Step 32.13: 如果 literal_eval 失败，再尝试按 JSON 解析。
                            conv["probing_questions"] = json.loads(pq_raw)
                        except json.JSONDecodeError:
                            # Step 32.14: 如果两种解析都失败，则记录 warning，并使用空 dict。
                            logger.warning(
                                "Could not parse probing_questions for %s[%d]",
                                size,
                                idx,
                            )
                            conv["probing_questions"] = {}
                else:
                    # Step 32.15: 如果 pq_raw 本身是 dict，则直接使用；否则退化为空 dict。
                    conv["probing_questions"] = pq_raw if isinstance(pq_raw, dict) else {}

                # Step 32.16: 将整理后的 conversation 加入列表。
                conversations.append(conv)

            # Step 32.17: 将下载并整理后的 conversations 写入本地缓存。
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False)

            # Step 32.18: 记录下载成功日志。
            logger.info("Downloaded and cached %s: %d conversations", size, len(conversations))

            # Step 32.19: 写入总 dataset。
            dataset[size] = conversations

        except Exception as exc:
            # Step 32.20: 下载失败时抛出带安装提示的 RuntimeError。
            raise RuntimeError(
                f"Failed to download BEAM {size} dataset: {exc}\n"
                f"Install datasets: pip install datasets\n"
                f"Or manually download and place in {cache_dir}"
            ) from exc

    # Step 33: 返回所有 chat size 对应的数据。
    return dataset


# ===============================================================================
# CHAT PARSING
# ===============================================================================


def _unwrap_batch_dicts(batch_dicts: list[dict]) -> list[list[dict]]:
    """Unwrap a list of batch dicts (with ``turns`` key) into flat turn lists."""

    # Step 34: 将 [{"turns": [...]}, ...] 这种 batch-dict 格式转换成 list[list[turn]]。
    batches: list[list[dict]] = []

    # Step 35: 遍历每个 batch dict。
    for batch in batch_dicts:
        # Step 35.1: 取出 turns 字段，不存在则为空列表。
        turns = batch.get("turns", [])

        # Step 35.2: 初始化当前 batch 的扁平 turn 列表。
        flat_turns: list[dict] = []

        # Step 35.3: turns 里有时可能嵌套 list，有时是单个 dict。
        for item in turns:
            if isinstance(item, list):
                # Step 35.4: 如果 item 是 list，则展开加入 flat_turns。
                flat_turns.extend(item)
            elif isinstance(item, dict):
                # Step 35.5: 如果 item 是 dict，则直接加入。
                flat_turns.append(item)

        # Step 35.6: 将当前 batch 的 flat_turns 加入 batches。
        batches.append(flat_turns)

    # Step 36: 返回所有 batch。
    return batches


def parse_beam_chat(chat_data: Any) -> list[list[dict]]:
    """Parse BEAM chat data into list of batches, each a list of turn dicts.

    Handles three HuggingFace storage formats:
    - 1M and smaller: chat is a 2D list ``[[turn, ...], ...]``
    - 10M: chat is a list of session dicts mapping plan keys to batch lists
    - Batch-dict format: chat is a list of dicts with ``"turns"`` key
    """

    # Step 37: 如果 chat_data 为空，直接返回空 batch 列表。
    if not chat_data:
        return []

    # List of dicts with "turns" key -> unwrap
    # Step 38: 处理 batch-dict 格式：
    # chat_data = [{"turns": [...]}, {"turns": [...]}]
    if (
        isinstance(chat_data, list)
        and chat_data
        and isinstance(chat_data[0], dict)
        and "turns" in chat_data[0]
    ):
        return _unwrap_batch_dicts(chat_data)

    # 10M plan-based format: list of session dicts
    # Step 39: 处理 10M 的 plan-based 格式。
    # 这种格式是 list[session_dict]，每个 session_dict 下面还有 plan key。
    if (
        isinstance(chat_data, list)
        and chat_data
        and isinstance(chat_data[0], dict)
        and "turns" not in chat_data[0]
    ):
        # Step 39.1: 取第一个 session，用来判断是否是 plan format。
        first_session = chat_data[0]

        # Step 39.2: 取该 session 的任意一个 value 作为样本。
        sample_val = next(iter(first_session.values()), None)

        # Step 39.3: 判断样本是否像 plan format：
        # value 是 list，list 的元素是 dict，且 dict 中有 turns 字段。
        is_plan_format = (
            isinstance(sample_val, list)
            and sample_val
            and isinstance(sample_val[0], dict)
            and "turns" in sample_val[0]
        )

        # Step 39.4: 如果是 plan format，则需要遍历 session -> plan -> batch。
        if is_plan_format:
            batches: list[list[dict]] = []

            # Step 39.5: 遍历每个 session。
            for session in chat_data:
                if not isinstance(session, dict):
                    continue

                # Step 39.6: 对 plan keys 排序。
                # 如果 key 形如 xxx-12，则按最后的数字排序。
                plan_keys = sorted(
                    session.keys(),
                    key=lambda k: int(k.split("-")[-1]) if k.split("-")[-1].isdigit() else 0,
                )

                # Step 39.7: 遍历排序后的 plan。
                for plan_key in plan_keys:
                    plan_batches = session[plan_key]

                    # Step 39.8: 空 plan 跳过。
                    if plan_batches is None:
                        continue

                    # Step 39.9: 解包 plan 内的 batch dicts，并追加到 batches。
                    batches.extend(_unwrap_batch_dicts(plan_batches))

            # Step 39.10: 返回从 10M plan format 中解析出的 batches。
            return batches

        # Single flat list of turn dicts
        # Step 40: 如果不是 plan format，但第一个元素看起来像 turn dict，
        # 则把整个 chat_data 当作一个 batch。
        if "role" in first_session or "content" in first_session:
            return [chat_data]

        # Step 41: 无法识别的 dict-list 格式，返回空。
        return []

    # Already a 2D list
    # Step 42: 如果 chat_data 已经是二维列表，即 [[turn, ...], [turn, ...]]，直接返回。
    if isinstance(chat_data, list) and chat_data and isinstance(chat_data[0], list):
        return chat_data

    # Step 43: 其他未知格式返回空。
    return []


def batch_to_chunks(turns: list[dict], chunk_size: int = CHUNK_SIZE) -> list[list[dict]]:
    """Convert turns in a batch to message chunks for ingestion."""

    # Step 44: 将一个 batch 的 turn 列表转换成 Mem0 add() 可接收的 message chunks。
    messages: list[dict] = []

    # Step 45: 遍历 batch 中的每个 turn。
    for turn in turns:
        # Step 45.1: 读取 role，缺省为 user。
        role = turn.get("role", "user")

        # Step 45.2: 读取 content，缺省为空字符串。
        content = turn.get("content", "")

        # Step 45.3: 空 content 不进入 Mem0。
        if not content:
            continue

        # Step 45.4: 标准化 role。
        # Mem0 期望 role 是 user 或 assistant；
        # 如果原始 role 是 human/user，则映射为 user，否则映射为 assistant。
        if role not in ("user", "assistant"):
            role = "user" if role.lower() in ("human", "user") else "assistant"

        # Step 45.5: 加入标准 message。
        messages.append({"role": role, "content": content})

    # Step 46: 按 chunk_size 切分 messages。
    chunks: list[list[dict]] = []

    # Step 47: 每 chunk_size 条 message 组成一个 chunk。
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i : i + chunk_size]

        # Step 47.1: 非空 chunk 才保留。
        if chunk:
            chunks.append(chunk)

    # Step 48: 返回 chunks，后续每个 chunk 会调用一次 mem0.add。
    return chunks


def get_time_anchor_epoch(turns: list[dict]) -> int | None:
    """Extract the earliest time_anchor from a batch and convert to epoch."""

    # Step 49: 从 batch turns 中找到第一个 time_anchor 并转换为 epoch 秒。
    for turn in turns:
        # Step 49.1: 读取当前 turn 的 time_anchor。
        anchor = turn.get("time_anchor")

        # Step 49.2: 如果存在 time_anchor，则尝试解析。
        if anchor:
            try:
                # Step 49.3: 延迟导入 dateutil.parser.parse。
                from dateutil.parser import parse as dateparse

                # Step 49.4: 将 anchor 中的 "-" 替换为空格后解析。
                dt = dateparse(anchor.replace("-", " "))

                # Step 49.5: 转成 Unix timestamp。
                return int(dt.timestamp())
            except Exception:
                # Step 49.6: 解析失败则继续找下一个 turn。
                pass

    # Step 50: 如果没有任何可解析 time_anchor，则返回 None。
    return None


# ===============================================================================
# PROBING QUESTIONS
# ===============================================================================


def extract_probing_questions(conversation: dict) -> list[dict]:
    """Extract all probing questions from a BEAM conversation.

    BEAM has 10 question types, 2 questions per type = 20 per conversation.
    The ``probing_questions`` field is a dict keyed by question_type.
    """

    # Step 51: 从 conversation 中读取 probing_questions。
    pq = conversation.get("probing_questions", {})

    # Step 52: 如果没有 probing_questions，则返回空列表。
    if not pq:
        return []

    # Step 53: 初始化问题列表。
    questions: list[dict] = []

    # Step 54: 按 BEAM 固定 question type 顺序遍历。
    for q_type in BEAM_QUESTION_TYPES:
        # Step 54.1: 读取当前类型下的问题。
        type_questions = pq.get(q_type, [])

        # Step 54.2: 如果当前类型的问题是 list，则逐个处理。
        if isinstance(type_questions, list):
            for q in type_questions:
                if isinstance(q, dict):
                    # Step 54.3: 如果问题本身是 dict，则补充 question_type 字段后加入。
                    q["question_type"] = q_type
                    questions.append(q)
                elif isinstance(q, str):
                    # Step 54.4: 如果问题是字符串，则包装成标准 dict。
                    questions.append({"question_type": q_type, "question_text": q, "rubric": []})

        # Step 54.5: 如果当前类型的问题是单个 dict，也补充 question_type 后加入。
        elif isinstance(type_questions, dict):
            type_questions["question_type"] = q_type
            questions.append(type_questions)

    # Step 55: 返回当前 conversation 的所有 probing questions。
    return questions


def extract_rubric_nuggets(question_data: dict) -> list[str]:
    """Extract rubric nugget descriptions from a question dict."""

    # Step 56: 从 question_data 中读取 rubric。
    rubric_raw = question_data.get("rubric", {})

    # Step 57: 如果 rubric 是 dict，则通常包含 nuggets 字段。
    if isinstance(rubric_raw, dict):
        nuggets = rubric_raw.get("nuggets", [])

        # Step 57.1: 对每个 nugget，优先取 description；否则转成字符串。
        return [
            n.get("description", str(n)) if isinstance(n, dict) else str(n)
            for n in nuggets
        ]

    # Step 58: 如果 rubric 已经是 list，则每个元素转为字符串。
    if isinstance(rubric_raw, list):
        return [str(n) for n in rubric_raw]

    # Step 59: 如果 rubric 是其他非空值，则包装成单元素列表。
    if rubric_raw:
        return [str(rubric_raw)]

    # Step 60: 没有 rubric 时返回空列表。
    return []


# ===============================================================================
# INGESTION
# ===============================================================================


async def ingest_conversation(
    chat_size: str,
    conv_idx: int,
    conversation: dict,
    mem0: Mem0Client,
    logger: Any,
    run_id: str,
    output_dir: str,
    shutdown: GracefulShutdown,
    debug: bool = True,
) -> tuple[bool, str, int]:
    """Ingest all batches of a BEAM conversation into Mem0.

    Returns:
        (success, user_id, total_chunks_processed)
    """

    # Step 61: 为当前 BEAM conversation 构造独立 user_id。
    # 这样不同 chat_size / conversation / run 的 memories 不会互相污染。
    user_id = f"beam_{chat_size}_{conv_idx}_{run_id}"

    # Step 62: 读取原始 chat 数据。
    chat_data = conversation.get("chat", [])

    # Step 63: 将各种 HF 存储格式解析成 batches。
    batches = parse_beam_chat(chat_data)

    # Step 64: 初始化 ingestion checkpoint。
    checkpoint = IngestionCheckpoint(output_dir)

    # Step 65: checkpoint key 使用 chat_size + conv_idx，确保不同 size/conversation 独立。
    key = f"{chat_size}_{conv_idx}"

    # Check if already complete
    # Step 66: 检查该 conversation 是否已经完整 ingest。
    is_done, cp_data = checkpoint.is_complete(key, CHUNK_SIZE)

    # Step 67: 如果已经完整 ingest，则直接返回 checkpoint 中记录的信息。
    if is_done and cp_data:
        chunks_done = cp_data.get("total_chunks_processed", 0)
        user_id = cp_data.get("user_id", user_id)
        logger.info(
            "[%s][%d] Already ingested (user_id=%s, %d chunks)",
            chat_size,
            conv_idx,
            user_id,
            chunks_done,
        )
        return True, user_id, chunks_done

    # Check for partial progress
    # Step 68: 如果没有完整完成，则检查是否有部分进度。
    chunks_already_done, resumed_uid = checkpoint.load_progress(key, CHUNK_SIZE)

    # Step 69: 如果 checkpoint 中有 user_id 和已完成 chunks，则使用之前的 user_id 续跑。
    if resumed_uid and chunks_already_done:
        user_id = resumed_uid
        logger.info(
            "[%s][%d] Resuming from %d completed chunks",
            chat_size,
            conv_idx,
            len(chunks_already_done),
        )

    # Step 70: 统计总 chunk 数，用于进度条。
    total_chunks = sum(len(batch_to_chunks(b)) for b in batches)

    # Step 71: 读取 conversation_seed 中的 category，用于日志展示。
    conv_seed = conversation.get("conversation_seed", {})
    category = conv_seed.get("category", "unknown") if isinstance(conv_seed, dict) else "unknown"

    # Step 72: 打印 ingestion 开始日志。
    logger.info(
        "[%s][%d] Ingesting: %d batches, %d chunks (category=%s)",
        chat_size,
        conv_idx,
        len(batches),
        total_chunks,
        category,
    )

    # Debug log file
    # Step 73: 如果开启 debug，则创建 ingestion debug 文件。
    debug_file = None
    if debug:
        # Step 73.1: 创建 debug 目录。
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Step 73.2: 构造当前 conversation 的 debug 文件路径。
        debug_path = os.path.join(debug_dir, f"beam_{chat_size}_{conv_idx}_ingestion.txt")

        # Step 73.3: 如果是续跑，则追加写；否则覆盖写。
        debug_mode = "a" if chunks_already_done else "w"

        # Step 73.4: buffering=1 表示行缓冲，方便实时查看日志。
        debug_file = open(debug_path, debug_mode, encoding="utf-8", buffering=1)

        # Step 73.5: 非续跑时写入文件头。
        if not chunks_already_done:
            debug_file.write(f"{'=' * 80}\n")
            debug_file.write(f"CONVERSATION {chat_size}/{conv_idx}: {category}\n")
            debug_file.write(f"Batches: {len(batches)}, Chunks: {total_chunks} (size={CHUNK_SIZE})\n")
            debug_file.write(f"User ID: {user_id}\n")
            debug_file.write(f"{'=' * 80}\n\n")

    # Step 74: 初始化 ingestion 进度条。
    pbar = tqdm(
        total=total_chunks,
        desc=f"Ingest {chat_size}/{conv_idx}",
        initial=len(chunks_already_done),
        leave=True,
    )

    # Step 75: 初始化处理计数。
    total_processed = len(chunks_already_done)
    total_failed = 0

    # Step 76: 遍历每个 batch。
    for batch_idx, batch_turns in enumerate(batches):
        # Step 76.1: 将当前 batch 切成 Mem0 chunks。
        chunks = batch_to_chunks(batch_turns)

        # Step 76.2: 当前 batch 没有有效 chunks 时跳过。
        if not chunks:
            continue

        # Step 76.3: 从当前 batch 中提取 time_anchor epoch。
        time_epoch = get_time_anchor_epoch(batch_turns)

        # Step 76.4: 如果开启 debug 且 batch header 没写过，则写入 batch 头部。
        if debug_file and f"batch_{batch_idx}_header" not in chunks_already_done:
            time_anchor_str = None

            # Step 76.5: 找到第一个 time_anchor，用于 debug 展示。
            for t in batch_turns:
                if t.get("time_anchor"):
                    time_anchor_str = t["time_anchor"]
                    break

            # Step 76.6: 写入 batch header。
            debug_file.write(f"\n{'---' * 27}\n")
            debug_file.write(
                f"SESSION: batch_{batch_idx}  |  Date: {time_anchor_str or 'N/A'}  "
                f"|  Epoch: {time_epoch}  |  Chunks: {len(chunks)}\n"
            )
            debug_file.write(f"{'---' * 27}\n\n")

        # Step 77: 遍历当前 batch 的每个 chunk。
        for chunk_idx, messages in enumerate(chunks):
            # Step 77.1: 构造 chunk_key，用于 checkpoint。
            chunk_key = f"batch_{batch_idx}_c{chunk_idx}"

            # Step 77.2: 如果该 chunk 已处理过，直接跳过。
            if chunk_key in chunks_already_done:
                continue

            # Step 77.3: 如果收到优雅退出请求，关闭资源并返回当前进度。
            if shutdown.requested:
                logger.info(
                    "Shutdown at %s/%d, chunk %s", chat_size, conv_idx, chunk_key
                )
                pbar.close()
                if debug_file:
                    debug_file.close()
                return True, user_id, total_processed

            # Skip empty messages
            # Step 77.4: 如果 chunk 里有空 content，跳过并标记为已处理。
            if any(not msg.get("content", "").strip() for msg in messages):
                chunks_already_done.add(chunk_key)
                total_processed += 1
                pbar.update(1)
                continue

            # Step 77.5: 如果开启 debug，则写入当前 chunk 原始 messages。
            if debug_file:
                debug_file.write(f"--- Chunk {chunk_idx} ({len(messages)} messages) ---\n")
                for msg in messages:
                    debug_file.write(f"  {msg['role']}: {msg['content']}\n")
                debug_file.write("\n")

            # Step 77.6: 调用 Mem0 add 写入当前 chunk。
            # 这里会触发 memory extraction、embedding、vector store insert 等流程。
            response = await mem0.add(messages, user_id, timestamp=time_epoch)

            # Step 77.7: response 非 None 认为当前 chunk 成功。
            if response is not None:
                total_processed += 1

                # Step 77.8: 如果 debug 开启，则记录本次抽取出的 memories。
                if debug_file:
                    results = response.get("results", [])

                    if results:
                        debug_file.write(f"--- Chunk {chunk_idx} (extracted) ---\n")

                        for mem_item in results:
                            # Step 77.9: 优先从 memory 字段取文本。
                            mem_text = mem_item.get("memory", "")

                            # Step 77.10: 某些返回结构可能把 memory 放在 data 里，这里做兼容。
                            if not mem_text:
                                data = mem_item.get("data", {})
                                if isinstance(data, dict):
                                    mem_text = data.get("memory", "")

                            # Step 77.11: 取 event 类型，例如 ADD。
                            event_type = mem_item.get("event", "")

                            # Step 77.12: 写入 debug 文件。
                            debug_file.write(f"  [{event_type}] {mem_text}\n")

                        debug_file.write("\n")
            else:
                # Step 77.13: response 为 None 表示 ingestion 失败。
                total_failed += 1
                logger.warning(
                    "Ingestion failed: %s/%d batch_%d chunk %d",
                    chat_size,
                    conv_idx,
                    batch_idx,
                    chunk_idx,
                )

            # Step 77.14: 无论成功失败，都标记当前 chunk 已完成。
            chunks_already_done.add(chunk_key)

            # Step 77.15: 保存部分 checkpoint，支持断点续跑。
            checkpoint.save_progress(
                key,
                {
                    "chat_size": chat_size,
                    "conversation_idx": conv_idx,
                    "user_id": user_id,
                    "run_id": run_id,
                    "chunk_size": CHUNK_SIZE,
                    "completed_chunks": list(chunks_already_done),
                },
            )

            # Step 77.16: 更新进度条。
            pbar.update(1)

    # Step 78: 所有 batch/chunk 处理完，关闭进度条。
    pbar.close()

    # Step 79: 如果 debug 文件打开，则写入 summary 并关闭。
    if debug_file:
        debug_file.write(
            f"\nSUMMARY: {total_processed}/{total_chunks} OK, {total_failed} failed\n"
        )
        debug_file.close()

    # Step 80: 保存 complete checkpoint。
    checkpoint.save_complete(
        key,
        {
            "chat_size": chat_size,
            "conversation_idx": conv_idx,
            "conversation_id": conversation.get("conversation_id", ""),
            "user_id": user_id,
            "run_id": run_id,
            "chunk_size": CHUNK_SIZE,
            "total_chunks_processed": total_processed,
            "total_chunks_failed": total_failed,
        },
    )

    # Step 81: 返回 ingestion 结果。
    return total_failed == 0, user_id, total_processed


# ===============================================================================
# NUGGET JUDGING
# ===============================================================================


def _clamp_nugget_score(raw_score: float) -> float:
    """Clamp a raw score to 0.0 / 0.5 / 1.0."""

    # Step 82: 将 judge LLM 给出的任意浮点分数离散化到 BEAM 规定的 0 / 0.5 / 1。
    if raw_score >= 0.75:
        return 1.0

    # Step 83: 中等支持度映射为 0.5。
    if raw_score >= 0.25:
        return 0.5

    # Step 84: 低于 0.25 映射为 0。
    return 0.0


async def judge_single_nugget(
    question: str,
    nugget: str,
    generated_answer: str,
    judge_llm: LLMClient,
) -> dict[str, Any]:
    """Judge a single rubric nugget.

    Returns:
        ``{"score": 0.0|0.5|1.0, "reason": "..."}``
    """

    # Step 85: 为单个 rubric nugget 构造 judge prompt。
    # 判断 generated_answer 是否覆盖这个 nugget。
    prompt = get_beam_nugget_judge_prompt(question, nugget, generated_answer)

    # Step 86: 调用 judge LLM，要求结构化输出。
    raw = await judge_llm.generate_structured(
        system=BEAM_JUDGE_SYSTEM_PROMPT,
        user=prompt,
    )

    # Step 87: 如果 judge 返回 dict，尝试读取 score/reason。
    if isinstance(raw, dict):
        try:
            # Step 87.1: 将 raw["score"] 转成 float，并离散化为 0/0.5/1。
            score = _clamp_nugget_score(float(raw.get("score", 0.0)))
        except (ValueError, TypeError):
            # Step 87.2: score 解析失败时按 0 处理。
            score = 0.0

        # Step 87.3: 返回该 nugget 的分数和理由。
        return {"score": score, "reason": raw.get("reason", "")}

    # Fallback: look for score in text
    # Step 88: 如果 judge 返回不是 dict，则尝试从文本中粗略识别分数。
    raw_str = str(raw)

    # Step 89: 文本中包含 1.0，则认为满分。
    if "1.0" in raw_str:
        return {"score": 1.0, "reason": raw_str[:200]}

    # Step 90: 文本中包含 0.5，则认为部分分。
    if "0.5" in raw_str:
        return {"score": 0.5, "reason": raw_str[:200]}

    # Step 91: 否则视为解析失败，分数为 0。
    return {"score": 0.0, "reason": f"Parse error: {raw_str[:200]}"}


# ===============================================================================
# EVENT ORDERING (Kendall tau-b)
# ===============================================================================


async def compute_event_ordering_score(
    question: str,
    rubric_nuggets: list[str],
    generated_answer: str,
    judge_llm: LLMClient,
    logger: Any,
) -> dict[str, Any]:
    """Compute Kendall tau-b score for event_ordering questions.

    Steps:
        1. Extract ordered facts from the LLM response.
        2. Align each extracted fact to a rubric event.
        3. Compute Kendall tau-b between predicted and reference orderings.

    Returns:
        Dict with ``tau_b``, ``predicted_order``, ``reference_order`` keys.
    """

    # Step 92: event_ordering 类型需要额外评估事件顺序。
    # 先从 generated_answer 中抽取模型回答里的有序事件。

    # Step 1: extract ordered events
    # Step 92.1: 构造事件抽取 prompt。
    extract_prompt = get_beam_fact_extraction_prompt(generated_answer)

    # Step 92.2: 调用 judge LLM 抽取事件列表。
    extract_raw = await judge_llm.generate_structured(
        system="Extract events as a JSON array of strings.",
        user=extract_prompt,
    )

    # Step 93: 初始化 extracted_events。
    extracted_events: list[str] = []

    # Step 93.1: 如果返回 dict，尝试从常见字段中取事件列表。
    if isinstance(extract_raw, dict):
        # Some models wrap in a key
        for key in ("events", "facts", "result"):
            if key in extract_raw and isinstance(extract_raw[key], list):
                extracted_events = extract_raw[key]
                break

    # Step 93.2: 如果返回本身就是 list，则直接使用。
    elif isinstance(extract_raw, list):
        extracted_events = extract_raw

    # Step 94: 如果没有抽取到事件，或者没有 rubric nuggets，则 tau_b 为 0。
    if not extracted_events or not rubric_nuggets:
        return {"tau_b": 0.0, "predicted_order": [], "reference_order": []}

    # Step 2: align each extracted event to a rubric event
    # Step 95: 将每个模型抽取出的事件对齐到 rubric_nuggets 的某个索引。
    predicted_indices: list[int] = []

    # Step 96: 遍历模型回答里的事件。
    for event in extracted_events:
        # Step 96.1: 构造事件对齐 prompt。
        align_prompt = get_beam_event_alignment_prompt(event, rubric_nuggets)

        # Step 96.2: 调用 judge LLM，返回该事件对应的 reference event index。
        align_raw = await judge_llm.generate_structured(
            system="Align the event to a reference event index. Return JSON.",
            user=align_prompt,
        )

        # Step 96.3: 默认 index 为 -1，表示无法对齐。
        idx = -1

        # Step 96.4: 如果返回 dict，尝试读取 index 字段。
        if isinstance(align_raw, dict):
            try:
                idx = int(align_raw.get("index", -1))
            except (ValueError, TypeError):
                idx = -1

        # Step 96.5: 只有合法 index 才加入 predicted_indices。
        if 0 <= idx < len(rubric_nuggets):
            predicted_indices.append(idx)

    # Step 3: Kendall tau-b
    # Step 97: reference_order 是 rubric_nuggets 的原始顺序。
    reference_order = list(range(len(rubric_nuggets)))

    # Step 98: 计算预测顺序和参考顺序之间的 Kendall tau-b。
    tau_b = compute_kendall_tau_b(predicted_indices, reference_order)

    # Step 99: 返回事件顺序评估结果。
    return {
        "tau_b": round(tau_b, 4),
        "predicted_order": predicted_indices,
        "reference_order": reference_order,
    }


# ===============================================================================
# SEARCH + ANSWER + JUDGE
# ===============================================================================


async def process_question(
    question_data: dict,
    qi: int,
    chat_size: str,
    conv_idx: int,
    user_id: str,
    mem0: Mem0Client,
    answerer: LLMClient,
    judge_llm: LLMClient,
    cutoffs: list[int],
    top_k: int,
    predict_only: bool,
    logger: Any,
    score_debug: bool = False,
    conversation_meta: dict | None = None,
) -> dict[str, Any]:
    """Process a single question: search + answer + rubric judge at multiple cutoffs.

    Returns:
        Result dict suitable for JSON serialization and checkpointing.
    """

    # Step 100: 读取 question_type，缺省为 unknown。
    question_type = question_data.get("question_type", "unknown")

    # Step 101: 构造唯一 question_id。
    # 格式包含 chat_size、conversation index、question index、question type。
    question_id = f"{chat_size}_{conv_idx}_q{qi}_{question_type}"

    # Step 102: 读取问题文本。
    # BEAM 字段可能叫 question_text，也可能叫 question。
    question_text = question_data.get("question_text", question_data.get("question", ""))

    # Step 103: 提取 rubric nuggets。
    rubric = extract_rubric_nuggets(question_data)

    # --- Search ---
    # Step 104: 执行 Mem0 search。
    start = time.monotonic()
    search_results = await mem0.search(
        question_text, user_id, top_k=top_k, score_debug=score_debug
    )

    # Step 105: 统计 search latency，单位毫秒。
    search_latency = (time.monotonic() - start) * 1000

    # Step 106: 格式化搜索结果，并提取 query_debug。
    formatted, query_debug = format_search_results(search_results)

    # Step 107: 构造基础 result。
    result: dict[str, Any] = {
        "question_id": question_id,
        "chat_size": chat_size,
        "conversation_idx": conv_idx,
        "conversation_id": (conversation_meta or {}).get("conversation_id", ""),
        "question_type": question_type,
        "question_type_idx": qi,
        "difficulty": question_data.get("difficulty", "unknown"),
        "question": question_text,
        "rubric": rubric,
        "ground_truth_answer": " | ".join(rubric),
        "source_chat_ids": question_data.get("source_chat_ids", []),
        "user_id": user_id,
        "retrieval": {
            "search_query": question_text,
            "search_results": formatted,
            "search_latency_ms": round(search_latency, 1),
            "total_results": len(formatted),
        },
    }

    # Step 108: 如果有 query_debug，则写入 retrieval。
    if query_debug:
        result["retrieval"]["query_debug"] = query_debug

    # Step 109: predict_only 模式只返回检索结果，不做 answer/judge。
    if predict_only:
        return result

    # --- Answer + Judge at each cutoff ---
    # Step 110: 初始化每个 cutoff 的评测结果。
    cutoff_results: dict[str, dict[str, Any]] = {}

    # Step 111: 遍历每个 cutoff，例如 top_100。
    for c in cutoffs:
        # Step 111.1: 截取前 c 条搜索结果。
        sliced = formatted[:c]

        # Step 111.2: 生成 cutoff label。
        label = cutoff_label(c)

        # Sort memories chronologically (oldest first) before answer generation
        # Step 111.3: 定义排序函数，用 created_at 将 memories 按时间从旧到新排序。
        def _sort_key(m):
            if isinstance(m, dict):
                return m.get("created_at", "") or ""
            return ""

        # Step 111.4: 将 sliced memories 按时间排序，帮助 answerer 看到自然时间线。
        sliced = sorted(sliced, key=_sort_key)

        # Generate answer
        # Step 112: 构造 BEAM answer generation prompt。
        gen_prompt = get_beam_answer_generation_prompt(question_text, sliced, top_k=c)

        # Step 113: 调用 answerer LLM 生成答案。
        generated_answer = await answerer.generate(system="", user=gen_prompt)

        # Step 114: 如果输出包含 ANSWER:，只保留后面的最终答案。
        if "ANSWER:" in generated_answer:
            generated_answer = generated_answer.rsplit("ANSWER:", 1)[-1].strip()

        # Step 115: 如果没有 rubric nuggets，无法 judge，记录 ERROR 并进入下一个 cutoff。
        if not rubric:
            cutoff_results[label] = {
                "judgment": "ERROR",
                "score": 0.0,
                "generated_answer": generated_answer,
                "memories_evaluated": len(sliced),
                "nugget_scores": [],
                "error": "No rubric nuggets found",
            }
            continue

        # Judge each nugget independently
        # Step 116: 逐个 rubric nugget 独立评分。
        nugget_scores: list[dict[str, Any]] = []

        # Step 117: 遍历每个 nugget。
        for nugget in rubric:
            # Step 117.1: 调用 judge_single_nugget 判断答案是否覆盖该 nugget。
            ns = await judge_single_nugget(question_text, nugget, generated_answer, judge_llm)

            # Step 117.2: 保存当前 nugget 的分数和理由。
            nugget_scores.append({
                "nugget": nugget,
                "score": ns["score"],
                "reason": ns["reason"],
            })

        # Question score = mean of nugget scores
        # Step 118: 当前问题总分 = 所有 nugget 分数的平均值。
        avg_score = (
            statistics.mean(ns["score"] for ns in nugget_scores)
            if nugget_scores
            else 0.0
        )

        # Step 119: 构造当前 cutoff 的结果。
        cr: dict[str, Any] = {
            "judgment": "PASS" if avg_score >= 0.5 else "FAIL",
            "score": round(avg_score, 4),
            "generated_answer": generated_answer,
            "memories_evaluated": len(sliced),
            "nugget_scores": nugget_scores,
        }

        # Event ordering: additionally compute Kendall tau-b
        # Step 120: 如果是 event_ordering 问题，额外计算 Kendall tau-b。
        if question_type == "event_ordering":
            try:
                # Step 120.1: 计算事件顺序得分。
                tau_result = await compute_event_ordering_score(
                    question_text,
                    rubric,
                    generated_answer,
                    judge_llm,
                    logger,
                )

                # Step 120.2: 将 event_ordering 结果写入 cutoff result。
                cr["event_ordering"] = tau_result

                # Combine: average of nugget score and normalized tau-b (mapped to 0-1)
                # Step 120.3: 将 tau_b 从 [-1, 1] 映射到 [0, 1]。
                tau_normalized = (tau_result["tau_b"] + 1.0) / 2.0  # map [-1,1] to [0,1]

                # Step 120.4: 将 nugget 平均分和 tau_normalized 再平均，得到 score_with_tau。
                combined = (avg_score + tau_normalized) / 2.0
                cr["score_with_tau"] = round(combined, 4)

            except Exception as exc:
                # Step 120.5: event ordering 失败不影响主评测结果。
                logger.warning(
                    "Event ordering tau-b failed for %s: %s", question_id, exc
                )

        # Step 121: 保存当前 cutoff 的结果。
        cutoff_results[label] = cr

    # Step 122: 将所有 cutoff_results 写入 result。
    result["cutoff_results"] = cutoff_results

    # Step 123: 返回单题完整结果。
    return result


# ===============================================================================
# METRICS + DISPLAY
# ===============================================================================


def compute_beam_metrics(
    evaluations: list[dict],
    cutoffs: list[int],
) -> dict[str, Any]:
    """Compute per-question-type and overall metrics at each cutoff."""

    # Step 124: 初始化按 cutoff 组织的 metrics。
    metrics_by_cutoff: dict[str, Any] = {}

    # Step 125: 定义 PASS 阈值。
    pass_threshold = 0.5

    # Step 126: 遍历每个 cutoff。
    for c in cutoffs:
        # Step 126.1: 生成 cutoff label。
        label = cutoff_label(c)

        # Step 126.2: 收集当前 cutoff 下所有 evaluation 的 score。
        scores: list[float] = []

        # Step 126.3: 遍历每条 evaluation。
        for e in evaluations:
            cr = e.get("cutoff_results", {}).get(label, {})
            scores.append(cr.get("score", 0.0))

        # Step 126.4: 当前 cutoff 下总题数。
        total = len(scores)

        # Step 126.5: score >= pass_threshold 视为正确。
        correct = sum(1 for s in scores if s >= pass_threshold)

        # Step 126.6: 统计当前 cutoff 下有 error 字段的结果数。
        errors = sum(
            1
            for e in evaluations
            if e.get("cutoff_results", {}).get(label, {}).get("error")
        )

        # Step 127: 按 question_type 分组。
        by_type: dict[str, list[dict]] = defaultdict(list)

        # Step 127.1: 遍历 evaluations，放入对应 question_type 组。
        for e in evaluations:
            by_type[e.get("question_type", "unknown")].append(e)

        # Step 128: 计算每个 question_type 的指标。
        type_metrics: dict[str, dict[str, Any]] = {}

        # Step 128.1: 按 question_type 排序，保证输出稳定。
        for qt in sorted(by_type):
            items = by_type[qt]

            # Step 128.2: 当前 question type 下所有分数。
            qt_scores = [
                i.get("cutoff_results", {}).get(label, {}).get("score", 0.0)
                for i in items
            ]

            # Step 128.3: 当前 question type 下通过数量。
            qt_correct = sum(1 for s in qt_scores if s >= pass_threshold)

            # Step 128.4: 写入当前 question type 的 metrics。
            type_metrics[qt] = {
                "total": len(items),
                "correct": qt_correct,
                "accuracy": qt_correct / len(items) * 100 if items else 0.0,
                "avg_score": statistics.mean(qt_scores) if qt_scores else 0.0,
            }

        # Step 129: 写入当前 cutoff 的 overall 和 by_question_type。
        metrics_by_cutoff[label] = {
            "overall": {
                "total": total,
                "correct": correct,
                "errors": errors,
                "accuracy": correct / total * 100 if total > 0 else 0.0,
                "avg_score": statistics.mean(scores) if scores else 0.0,
            },
            "by_question_type": type_metrics,
        }

    # Step 130: 返回 metrics。
    return metrics_by_cutoff


def display_results(
    metrics_by_cutoff: dict[str, Any],
    cutoffs: list[int],
) -> None:
    """Print metrics to console."""

    # Step 131: 先将 cutoffs 转换成 labels。
    labels = [cutoff_label(c) for c in cutoffs]

    # Step 132: 遍历每个 label 并打印 metrics。
    for label in labels:
        # Step 132.1: 读取当前 cutoff metrics。
        m = metrics_by_cutoff.get(label, {})

        # Step 132.2: 读取 overall metrics。
        overall = m.get("overall", {})

        # Step 132.3: 打印 cutoff 标题。
        print(f"\n--- {label} ---")

        # Step 132.4: 打印整体结果。
        print(
            f"  Overall: {overall.get('correct', 0)}/{overall.get('total', 0)} "
            f"pass (>= 0.5)  |  avg_score={overall.get('avg_score', 0):.3f}  "
            f"|  errors={overall.get('errors', 0)}"
        )

        # Step 132.5: 打印每个 question_type 的结果。
        for qt, tm in sorted(m.get("by_question_type", {}).items()):
            print(
                f"  {qt}: {tm['correct']}/{tm['total']} "
                f"({tm['accuracy']:.1f}%)  avg={tm['avg_score']:.3f}"
            )


# ===============================================================================
# CLI
# ===============================================================================


def parse_args() -> argparse.Namespace:
    # Step 133: 创建 argparse 参数解析器。
    parser = argparse.ArgumentParser(
        description="Run BEAM benchmark: ingest + search + answer + rubric judge",
    )

    # Step 134: 项目名，用于输出目录命名。
    parser.add_argument("--project-name", required=True, help="Name for this eval run")

    # Step 135: answerer 模型。
    parser.add_argument(
        "--answerer-model", default="gpt-5", help="Model for answer generation"
    )

    # Step 136: judge 模型。
    parser.add_argument("--judge-model", default="gpt-5", help="Model for rubric judging")

    # Step 137: answerer provider。
    parser.add_argument(
        "--provider", default="openai", help="LLM provider (openai, anthropic, azure)"
    )

    # Step 138: judge provider；默认跟 provider 一致。
    parser.add_argument(
        "--judge-provider", default=None, help="Judge provider (defaults to --provider)"
    )

    # Step 139: 指定要跑哪些 BEAM chat sizes。
    parser.add_argument(
        "--chat-sizes",
        default="100K",
        help="Comma-separated chat sizes: 100K,500K,1M,10M (default: 100K)",
    )

    # Step 140: 指定 conversation index 范围或列表。
    parser.add_argument(
        "--conversations",
        default="0-99",
        help="Conversation indices: 0-99 or 0,1,5 (default: 0-99)",
    )

    # Step 141: Mem0 search 获取的最大结果数。
    parser.add_argument(
        "--top-k", type=int, default=200, help="Number of search results to retrieve"
    )

    # Step 142: 评估 cutoff，例如只看 top_100。
    parser.add_argument(
        "--top-k-cutoffs",
        default="100",
        help="Comma-separated cutoffs for evaluation (default: 100)",
    )

    # Step 143: 最大并发 worker 数。
    parser.add_argument(
        "--max-workers", type=int, default=10, help="Max parallel workers"
    )

    # Step 144: 输出目录。
    parser.add_argument("--output-dir", default="results/beam", help="Output directory")

    # Step 145: predict-only 模式，只做 ingest/search，不做 answer/judge。
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip answer+judge, only ingest+search",
    )

    # Step 146: evaluate-only 模式，读取已有结果评估，不重新 ingest/search。
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip ingest+search, evaluate existing predict results",
    )

    # Step 147: resume 模式，从已有 checkpoint / result 续跑。
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # Step 148: debug 模式，输出更详细日志和 debug 文件。
    parser.add_argument("--debug", action="store_true", help="Verbose logging + debug files")

    # Step 149: 是否在 search 输出中包含 score breakdown。
    parser.add_argument(
        "--score-debug",
        action="store_true",
        help="Include score breakdowns in search output",
    )

    # Step 150: 指定 run_id，用于续跑同一次实验。
    parser.add_argument("--run-id", default=None, help="Reuse a specific run_id for resume")

    # Step 151: 指定 HF 数据集本地缓存目录。
    parser.add_argument(
        "--dataset-cache-dir", default=None, help="Local cache for HF dataset"
    )

    # Step 152: 指定要评估的问题类型。
    parser.add_argument(
        "--question-types",
        default=None,
        help="Comma-separated question types to evaluate (default: all)",
    )

    # Step 153: LLM 请求速率限制。
    parser.add_argument("--rpm", type=int, default=200, help="Requests per minute for LLM")

    # Step 154: Mem0 backend，oss 或 cloud。
    parser.add_argument("--backend", default="oss", choices=["oss", "cloud"],
                        help="Mem0 backend: 'oss' for self-hosted server (default), 'cloud' for api.mem0.ai")

    # Step 155: Mem0 server URL。
    parser.add_argument("--mem0-host", default=None,
                        help="Mem0 server URL")

    # Step 156: Mem0 cloud API key。
    parser.add_argument("--mem0-api-key", default=None,
                        help="Mem0 API key (cloud mode only)")

    # Step 157: 解析并返回命令行参数。
    return parser.parse_args()


def _parse_conversation_indices(spec: str) -> list[int]:
    """Parse conversation index specification.

    Supports:
        - Range:  ``"0-99"``
        - List:   ``"0,1,5,10"``
        - Mixed:  ``"0-9,50,90-99"``
    """

    # Step 158: 解析 conversation index 表达式。
    indices: list[int] = []

    # Step 159: 按逗号拆分表达式。
    for part in spec.split(","):
        # Step 159.1: 去掉前后空格。
        part = part.strip()

        # Step 159.2: 如果包含 "-"，按区间处理。
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            # Step 159.3: 否则按单个 index 处理。
            indices.append(int(part))

    # Step 160: 去重并排序后返回。
    return sorted(set(indices))


# ===============================================================================
# MAIN
# ===============================================================================


async def async_main() -> None:
    # Step 161: 解析命令行参数。
    args = parse_args()

    # Step 162: 初始化日志。
    logger = setup_logging("beam", debug=args.debug)

    # Step 163: 解析 top_k cutoffs。
    cutoffs = parse_cutoffs(args.top_k_cutoffs)

    # Step 164: 解析 chat sizes。
    chat_sizes = [s.strip() for s in args.chat_sizes.split(",")]

    # Step 165: 校验 chat size 是否合法。
    for s in chat_sizes:
        if s not in VALID_CHAT_SIZES:
            print(f"ERROR: Invalid chat size '{s}'. Valid: {VALID_CHAT_SIZES}")
            sys.exit(1)

    # Step 166: 解析 conversation indices。
    conv_indices = _parse_conversation_indices(args.conversations)

    # Step 167: 解析 question type filter。
    # 如果用户指定 --question-types，则只评估这些类型。
    q_type_filter = (
        set(args.question_types.split(",")) if args.question_types else None
    )

    # Step 168: 生成或复用 run_id。
    run_id = args.run_id or uuid.uuid4().hex[:8]

    # Step 169: 创建输出目录。
    output_dir = os.path.join(args.output_dir, f"predicted_{args.project_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Step 170: 确定数据集缓存目录。
    cache_dir = args.dataset_cache_dir or DEFAULT_DATASET_DIR

    # Step 171: 打印本次运行配置。
    print(f"BEAM Benchmark | project={args.project_name} run_id={run_id}")
    print(f"  Answerer: {args.answerer_model} ({args.provider})")
    print(f"  Judge: {args.judge_model} ({args.judge_provider or args.provider})")
    print(f"  Chat sizes: {args.chat_sizes}")
    print(f"  Conversations: {args.conversations}")
    print(f"  Cutoffs: {args.top_k_cutoffs}")

    # Download dataset
    # Step 172: 下载或读取缓存数据集。
    dataset = download_dataset(chat_sizes, cache_dir, logger)

    # Init clients
    # Step 173: 初始化 Mem0 backend。
    backend = os.getenv("MEM0_BACKEND", args.backend)

    # Step 174: 创建 Mem0Client。
    mem0 = Mem0Client(
        mode=backend,
        host=args.mem0_host,
        api_key=args.mem0_api_key if backend == "cloud" else None,
        rpm=args.rpm,
    )

    # Step 175: 创建 answerer LLM client。
    answerer = LLMClient(
        model=args.answerer_model, provider=args.provider, rpm=args.rpm
    )

    # Step 176: 创建 judge LLM client。
    judge_provider = args.judge_provider or args.provider
    judge_llm = LLMClient(
        model=args.judge_model, provider=judge_provider, rpm=args.rpm
    )

    # Step 177: 初始化优雅退出控制器。
    shutdown = GracefulShutdown()

    # Step 178: 初始化结果列表。
    all_evaluations: list[dict] = []

    # Load existing results for resume / evaluate-only
    # Step 179: 如果 resume 或 evaluate_only，则加载已有单题结果 JSON。
    if args.resume or args.evaluate_only:
        for p in sorted(Path(output_dir).glob("*.json")):
            # Step 179.1: 跳过内部文件。
            if p.name.startswith("_"):
                continue

            try:
                # Step 179.2: 读取已有结果。
                data = json.loads(p.read_text())

                # Step 179.3: 只要有 question_id，就认为是单题结果。
                if data.get("question_id"):
                    all_evaluations.append(data)
            except (json.JSONDecodeError, KeyError):
                # Step 179.4: JSON 损坏或字段缺失则跳过。
                continue

        # Step 179.5: 打印加载到的已有结果数量。
        print(f"  Loaded {len(all_evaluations)} existing results")

    # Step 180: evaluate-only 模式。
    # 当前实现只基于已有 cutoff_results 计算 metrics，不会重新调用 answerer/judge。
    if args.evaluate_only:
        # Step 180.1: 如果没有已有结果，直接结束。
        if not all_evaluations:
            print("No results found for evaluate-only mode.")
            return

        # Step 180.2: 检查已有结果中是否已经包含 cutoff_results。
        has_cutoffs = any("cutoff_results" in e for e in all_evaluations)

        # Step 180.3: 如果有 cutoff_results，则计算并展示指标。
        if has_cutoffs:
            metrics = compute_beam_metrics(all_evaluations, cutoffs)
            display_results(metrics, cutoffs)
        else:
            # Step 180.4: 如果没有 cutoff_results，说明之前只跑了 predict-only 或 search 结果不完整。
            print("Results don't have cutoff_results. Run without --evaluate-only first.")

        # Step 180.5: evaluate-only 到此结束。
        return

    # Step 181: 构造已完成 question_id 集合，用于 resume 时跳过已处理问题。
    existing_ids = {e["question_id"] for e in all_evaluations}

    # Track user_ids: (chat_size, conv_idx) -> user_id
    # Step 182: 记录每个 conversation 对应的 user_id。
    conv_user_ids: dict[tuple[str, int], str] = {}

    # Step 183: 进入 Mem0 异步上下文，并注册优雅退出处理。
    async with mem0:
        with shutdown:
            # === Phase 1: Ingestion ===
            # Step 184: Phase 1，先对所有指定 conversations 做 ingestion。
            for size in chat_sizes:
                # Step 184.1: 取出当前 chat size 的 conversations。
                convs = dataset[size]

                # Step 184.2: 过滤掉超出当前数据集长度的 conversation index。
                indices = [i for i in conv_indices if i < len(convs)]

                # Step 184.3: 如果没有合法 conversation index，则跳过当前 size。
                if not indices:
                    logger.warning(
                        "No valid conversation indices for %s (dataset has %d)",
                        size,
                        len(convs),
                    )
                    continue

                # Step 184.4: 打印 ingestion 阶段标题。
                print(f"\n=== Ingesting {len(indices)} {size} conversations ===")

                # Step 184.5: 逐个 conversation ingestion。
                for ci in indices:
                    # Step 184.6: 如果收到退出信号，停止当前 size。
                    if shutdown.requested:
                        break

                    # Step 184.7: 调用 ingest_conversation 写入 Mem0。
                    success, user_id, chunks = await ingest_conversation(
                        chat_size=size,
                        conv_idx=ci,
                        conversation=convs[ci],
                        mem0=mem0,
                        logger=logger,
                        run_id=run_id,
                        output_dir=output_dir,
                        shutdown=shutdown,
                        debug=args.debug,
                    )

                    # Step 184.8: 记录该 conversation 的 user_id。
                    conv_user_ids[(size, ci)] = user_id

                    # Step 184.9: 如果 ingestion 有失败，记录 warning。
                    if not success:
                        logger.warning("[%s][%d] Had failures during ingestion", size, ci)

            # Step 185: 如果 ingestion 阶段收到退出信号，提示用户可续跑并退出。
            if shutdown.requested:
                print("Shutdown requested -- progress saved. Re-run to resume.")
                return

            # === Phase 2: Search + Answer + Judge ===
            # Step 186: Phase 2，构造所有需要处理的问题任务。
            all_questions: list[tuple] = []

            # Step 187: 遍历每个 chat size。
            for size in chat_sizes:
                convs = dataset[size]
                indices = [i for i in conv_indices if i < len(convs)]

                # Step 188: 遍历当前 size 下每个 conversation。
                for ci in indices:
                    key = (size, ci)

                    # Step 188.1: 如果 ingestion 阶段没有记录 user_id，尝试从 checkpoint 恢复。
                    if key not in conv_user_ids:
                        # Try to recover user_id from ingestion checkpoint
                        cp = IngestionCheckpoint(output_dir)
                        is_done, cp_data = cp.is_complete(f"{size}_{ci}", CHUNK_SIZE)

                        # Step 188.2: 如果 checkpoint 完整，则使用 checkpoint 中的 user_id。
                        if is_done and cp_data:
                            conv_user_ids[key] = cp_data["user_id"]
                        else:
                            # Step 188.3: 否则使用默认规则构造 user_id。
                            conv_user_ids[key] = f"beam_{size}_{ci}_{run_id}"

                    # Step 188.4: 取出 user_id 和 conversation。
                    user_id = conv_user_ids[key]
                    conv = convs[ci]

                    # Step 188.5: 从 conversation 中抽取 probing questions。
                    questions = extract_probing_questions(conv)

                    # Step 188.6: 如果设置了 question type filter，则过滤问题。
                    if q_type_filter:
                        questions = [
                            q for q in questions if q.get("question_type") in q_type_filter
                        ]

                    # Step 188.7: 构造 conversation metadata。
                    conv_meta = {
                        "conversation_id": conv.get("conversation_id", ""),
                        "conversation_seed": conv.get("conversation_seed", {}),
                    }

                    # Step 188.8: 将每个问题加入 all_questions。
                    for qi, q in enumerate(questions):
                        all_questions.append((q, qi, size, ci, user_id, conv_meta))

            # Count already done
            # Step 189: 统计已经处理过的问题数量。
            already_done = sum(
                1
                for q, qi, size, ci, _, _ in all_questions
                if f"{size}_{ci}_q{qi}_{q.get('question_type', 'unknown')}" in existing_ids
            )

            # Step 190: 计算剩余问题数。
            remaining = len(all_questions) - already_done

            # Step 191: 打印 question processing 概况。
            print(
                f"\n=== Processing {len(all_questions)} questions "
                f"({already_done} done, {remaining} remaining) ==="
            )

            # Step 192: 如果还有剩余问题，则开始处理。
            if remaining > 0:
                # Step 192.1: 创建问题处理进度条。
                pbar = tqdm(
                    total=len(all_questions),
                    initial=already_done,
                    desc="Questions",
                )

                # Step 192.2: 逐个处理问题。
                for q_data, qi, size, ci, uid, meta in all_questions:
                    # Step 192.3: 构造 question_id。
                    qid = f"{size}_{ci}_q{qi}_{q_data.get('question_type', 'unknown')}"

                    # Step 192.4: 如果收到退出信号，停止处理。
                    if shutdown.requested:
                        break

                    # Step 192.5: 如果该问题已经完成，跳过。
                    if qid in existing_ids:
                        continue

                    # Step 192.6: 执行单题 search + answer + judge。
                    result = await process_question(
                        question_data=q_data,
                        qi=qi,
                        chat_size=size,
                        conv_idx=ci,
                        user_id=uid,
                        mem0=mem0,
                        answerer=answerer,
                        judge_llm=judge_llm,
                        cutoffs=cutoffs,
                        top_k=args.top_k,
                        predict_only=args.predict_only,
                        logger=logger,
                        score_debug=args.score_debug,
                        conversation_meta=meta,
                    )

                    # Save per-question checkpoint
                    # Step 192.7: 保存单题 checkpoint/result。
                    result_path = os.path.join(output_dir, f"{qid}.json")
                    save_result_json(result_path, result)

                    # Step 192.8: 更新内存中的结果列表和已完成 ID 集合。
                    all_evaluations.append(result)
                    existing_ids.add(qid)

                    # Step 192.9: 更新进度条。
                    pbar.update(1)

                # Step 192.10: 关闭进度条。
                pbar.close()

    # === Metrics ===
    # Step 193: 如果不是 predict_only，并且有评测结果，则计算 metrics。
    if not args.predict_only and all_evaluations:
        # Step 193.1: 检查是否已经有 cutoff_results。
        has_cutoffs = any("cutoff_results" in e for e in all_evaluations)

        if has_cutoffs:
            # Step 193.2: 计算 BEAM metrics。
            metrics_by_cutoff = compute_beam_metrics(all_evaluations, cutoffs)

            # Step 193.3: 打印 metrics。
            display_results(metrics_by_cutoff, cutoffs)

            # Save unified result JSON
            # Step 193.4: 保存统一结果 JSON。
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unified_path = os.path.join(
                args.output_dir, f"beam_results_{timestamp}.json"
            )

            # Step 193.5: 写入统一结果文件。
            save_result_json(
                unified_path,
                {
                    "metadata": {
                        "benchmark": "beam",
                        "project_name": args.project_name,
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "answerer_model": args.answerer_model,
                        "judge_model": args.judge_model,
                        "provider": args.provider,
                        "judge_provider": judge_provider,
                        "top_k": args.top_k,
                        "top_k_cutoffs": [cutoff_label(c) for c in cutoffs],
                        "chat_sizes": chat_sizes,
                        "conversations": args.conversations,
                        "total_questions": len(all_evaluations),
                        "question_types": q_type_filter or BEAM_QUESTION_TYPES,
                    },
                    "metrics_by_cutoff": metrics_by_cutoff,
                    "evaluations": all_evaluations,
                },
            )

            # Step 193.6: 打印统一结果路径。
            print(f"\nResults saved to: {unified_path}")

    # Step 194: 打印本次处理的问题总数。
    print(f"\nTotal questions processed: {len(all_evaluations)}")


def main() -> None:
    # Step 195: 同步入口函数，用 asyncio.run 启动 async_main。
    asyncio.run(async_main())


# Step 196: 当该文件作为脚本直接执行时，调用 main()。
if __name__ == "__main__":
    main()
