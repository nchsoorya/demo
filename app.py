import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import os
import ast
import json
import base64
import requests
import tempfile
import shutil
import time
import threading
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from difflib import SequenceMatcher

from config import (
    SCHEMA,
    FINAL_OUTPUT_SCHEMA,
    VLM_URL,
    TOKEN_URL,
    LLM_URL,
    CATALYST_ORG,
    MODEL_NAME,
    CONSOLIDATION_MODEL_NAME,
    CLIENT_ID,
    CLIENT_SECRET,
    REFRESH_TOKEN,
    OCR_MAX_WORKERS,
    CONSOLIDATION_MAX_WORKERS,
    CONSOLIDATION_BATCH_SIZE,
    FINAL_PROMPT_CHAR_LIMIT,
    SHOW_EXTRACTION_TIME,
    LOCAL_UPLOAD_DIR,
)
from prompts import (
    OCR_SYSTEM_PROMPT,
    OCR_USER_PROMPT,
    OCR_SYSTEM_PROMPT_PART_1,
    OCR_USER_PROMPT_PART_1,
    OCR_SCHEMA_PART_1,
    OCR_SYSTEM_PROMPT_PART_2,
    OCR_USER_PROMPT_PART_2,
    OCR_SCHEMA_PART_2,
    CONSOLIDATION_SYSTEM_PROMPT,
    FINAL_CONSOLIDATION_SYSTEM_PROMPT,
)
from education_dedupe_helper import clean_education_duplicates_semantically

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="DOC Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# HELPER PARSING & TOKEN UTILITIES
# =====================================================
def parse_model_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        return {}

    snippet = text[start:].strip()
    stack = []
    clean_index = 0
    in_string = False
    escape = False

    for i, char in enumerate(snippet):
        if char == '"' and not escape:
            in_string = not in_string
        elif escape:
            escape = False
            continue
        elif char == '\\' and in_string:
            escape = True
            continue

        if not in_string:
            if char in ['{', '[']:
                stack.append(char)
            elif char in ['}', ']']:
                if not stack:
                    break
                if (char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '['):
                    stack.pop()
                    if not stack:
                        clean_index = i + 1
                        break
        if not stack:
            clean_index = i + 1

    if stack and clean_index == 0:
        fallback_snippet = snippet
        for reverse_idx in range(len(snippet) - 1, -1, -1):
            if snippet[reverse_idx] in [',', '{', '['] and not in_string:
                fallback_snippet = snippet[:reverse_idx].strip()
                break

        repair_stack = []
        for char in fallback_snippet:
            if char in ['{', '[']:
                repair_stack.append(char)
            elif char in ['}', ']'] and repair_stack:
                repair_stack.pop()

        closure = "".join(['}' if token == '{' else ']' for token in reversed(repair_stack)])
        snippet = fallback_snippet + closure
    elif clean_index > 0:
        snippet = snippet[:clean_index]

    with suppress(Exception):
        return json.loads(snippet)

    with suppress(Exception):
        parsed = ast.literal_eval(snippet)
        if isinstance(parsed, dict):
            return parsed

    return {}


def get_zoho_access_token() -> str:
    """
    Returns a cached Zoho OAuth access token, refreshing it only when expired.
    Tokens are valid for 60 minutes; we refresh ~5 minutes early as a safety buffer.
    Retries with exponential backoff on Zoho rate-limit (HTTP 400 / Access Denied).
    """
    cache = _get_token_cache()
    with cache["lock"]:
        now = time.time()
        if cache["token"] and now < cache["expires_at"]:
            print(f"Using cached Zoho access token, expires in {int(cache['expires_at'] - now)} seconds.")
            return cache["token"]
        print("Cache expired..")
        payload = {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": REFRESH_TOKEN
        }
        max_retries = 4
        backoff_seconds = [5, 15, 30, 60]
        for attempt in range(max_retries):
            try:
                response = requests.post(TOKEN_URL, data=payload, timeout=30)
                if not response.ok:
                    try:
                        err_body = response.json()
                    except Exception:
                        err_body = response.text

                    # Zoho rate-limit: retry with backoff
                    err_str = str(err_body).lower()
                    is_rate_limit = (
                        "too many requests" in err_str
                        or "access denied" in err_str
                        or response.status_code == 429
                    )
                    if is_rate_limit and attempt < max_retries - 1:
                        wait = backoff_seconds[attempt]
                        print(f"Zoho token rate-limited. Waiting {wait}s before retry (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait)
                        continue

                    raise RuntimeError(
                        f"Zoho token endpoint returned HTTP {response.status_code} "
                        f"from {TOKEN_URL}. Response: {err_body}. "
                        f"Common causes: "
                        f"(0) RATE LIMIT — too many token requests in a short window; wait and retry; "
                        f"(1) wrong data center domain "
                        f"(set ZOHO_ACCOUNTS_DOMAIN to accounts.zoho.eu/.in/.com.au/.jp "
                        f"to match where the refresh token was issued); "
                        f"(2) refresh token revoked or regenerated; "
                        f"(3) CLIENT_ID/CLIENT_SECRET don't match the refresh token."
                    )

                token_data = response.json()
                if "access_token" not in token_data:
                    raise RuntimeError(f"Failed to generate token layout: {token_data}")

                # Zoho returns expires_in in seconds (typically 3600). Refresh 5 min early.
                expires_in = int(token_data.get("expires_in", 3600))
                cache["token"] = token_data["access_token"]
                cache["expires_at"] = now + max(60, expires_in - 300)
                return cache["token"]

            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Zoho Authorization Gateway connection error: {str(e)}")

# Token cache backed by st.cache_resource — survives Streamlit reruns and
# is shared across all threads. Initialized only once per server process.
@st.cache_resource
def _get_token_cache() -> dict:
    return {"token": None, "expires_at": 0.0, "lock": threading.Lock()}


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _call_page_extraction_vlm(
        image_b64: str,
        user_prompt: str,
        system_prompt: str,
        guided_schema: dict,
        access_token: str,
) -> dict:
    """Call VLM for page extraction with guided JSON schema."""
    payload = {
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "model": MODEL_NAME,
        "images": [image_b64],
        "top_k": 1,
        "top_p": 0.01,
        "temperature": 0.0,
        "max_tokens": 4096,
        "guided_json": guided_schema,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "CATALYST-ORG": CATALYST_ORG
    }
    response = requests.post(VLM_URL, json=payload, headers=headers, timeout=300,
                             proxies={"http": None, "https": None})
    result = response.json()
    return parse_model_json(result.get("response", ""))


def extract_from_image(image_path: str, access_token: str) -> dict:
    image_b64 = encode_image(image_path)
    page_ref = os.path.basename(image_path)

    # Extract PART 1 (Personal/Identity information)
    part_1_json = _call_page_extraction_vlm(
        image_b64=image_b64,
        user_prompt=OCR_USER_PROMPT_PART_1,
        system_prompt=OCR_SYSTEM_PROMPT_PART_1,
        guided_schema=OCR_SCHEMA_PART_1,
        access_token=access_token,
    )

    # Extract PART 2 (Education/Religious information)
    part_2_json = _call_page_extraction_vlm(
        image_b64=image_b64,
        user_prompt=OCR_USER_PROMPT_PART_2,
        system_prompt=OCR_SYSTEM_PROMPT_PART_2,
        guided_schema=OCR_SCHEMA_PART_2,
        access_token=access_token,
    )

    print(
        f"[OCR_PAGE_PART_1] page={page_ref} "
        f"{json.dumps(part_1_json, ensure_ascii=False, default=str)}"
    )
    print(
        f"[OCR_PAGE_PART_2] page={page_ref} "
        f"{json.dumps(part_2_json, ensure_ascii=False, default=str)}"
    )

    # Merge the two parts
    merged_json = {}
    if isinstance(part_1_json, dict):
        merged_json.update(part_1_json)
    if isinstance(part_2_json, dict):
        for key, value in part_2_json.items():
            if key == "Document_Metadata" and key in merged_json:
                continue
            merged_json[key] = value

    _ensure_page_confidence_scores(merged_json)

    # --- TERMINAL PRINT ADDITION ---
    page_name = os.path.basename(image_path)
    print(f"\n{'=' * 20} TERMINAL OUTPUT: {page_name} {'=' * 20}")
    print(json.dumps(merged_json, indent=2, ensure_ascii=False))
    print(f"{'=' * 60}\n")
    # -------------------------------

    return merged_json


def _call_consolidation_llm(prompt_text: str, system_prompt: str, access_token: str) -> dict:
    """
    Single LLM call to the Zoho QuickML consolidation endpoint.
    Returns the parsed JSON dict from the model response.
    """
    payload = {
        "prompt": prompt_text,
        "model": CONSOLIDATION_MODEL_NAME,
        "system_prompt": system_prompt,
        "top_p": 1.0,
        "top_k": 1,
        "temperature": 0.0,
        "max_tokens": 5000,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "CATALYST-ORG": CATALYST_ORG
    }
    response = requests.post(LLM_URL, json=payload, headers=headers, timeout=600, proxies={"http": None, "https": None})
    result = response.json()
    # print(f"{result = }")
    return parse_model_json(result.get("response", ""))


def _to_compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_empty_value(value: object) -> bool:
    if value in (None, "", "None", "null"):
        return True
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _is_numeric_grade_value(value: object) -> bool:
    if _is_empty_value(value):
        return False
    normalized = str(value).strip().replace(",", ".")
    return bool(re.fullmatch(r"\d{1,2}(?:\.\d{1,2})?", normalized))


def _is_invalid_numeric_degree_entry(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    return _is_numeric_grade_value(entry.get("Degree"))


def _prune_empty_fields(value: object) -> object:
    if isinstance(value, dict):
        pruned = {}
        for key, child in value.items():
            cleaned_child = _prune_empty_fields(child)
            if not _is_empty_value(cleaned_child):
                pruned[key] = cleaned_child
        return pruned
    if isinstance(value, list):
        pruned_list = []
        for child in value:
            cleaned_child = _prune_empty_fields(child)
            if not _is_empty_value(cleaned_child):
                pruned_list.append(cleaned_child)
        return pruned_list
    return value


def _compact_batch_for_final_prompt(batch_result: dict) -> dict:
    """
    Shrink batch-level JSON before final merge to avoid model max-length errors.
    We keep core values + minimal metadata needed for doc-type/year logic.
    """
    compacted = _prune_empty_fields(batch_result)
    if not isinstance(compacted, dict):
        return {}

    metadata_entries = compacted.get("Document_Metadata")
    if isinstance(metadata_entries, list):
        reduced_metadata = []
        for entry in metadata_entries:
            if not isinstance(entry, dict):
                continue

            # Keep only minimal metadata required by final-stage rules.
            slim_entry = {}
            for key in ("Document_Type", "Document_Date", "Document_Year"):
                value = entry.get(key)
                if not _is_empty_value(value):
                    slim_entry[key] = value

            # Confidence_Scores are useful in batch stage but very expensive in final payload.
            # Dropping them here keeps behavior stable while preventing max-length failures.
            if slim_entry:
                reduced_metadata.append(slim_entry)

        if reduced_metadata:
            compacted["Document_Metadata"] = reduced_metadata
        else:
            compacted.pop("Document_Metadata", None)

    return compacted


def _normalize_scalar_for_compare(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip().lower()


def _get_values_by_path(payload: object, path: str) -> list[object]:
    parts = path.split(".")

    def _walk(node: object, remaining: list[str]) -> list[object]:
        if node is None:
            return []
        if not remaining:
            return [node]
        if isinstance(node, list):
            values: list[object] = []
            for item in node:
                values.extend(_walk(item, remaining))
            return values
        if not isinstance(node, dict):
            return []
        return _walk(node.get(remaining[0]), remaining[1:])

    return _walk(payload, parts)


def _collect_scalar_field_paths(payload: dict, prefix: str = "") -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in payload.items():
        if key in ("Document_Metadata", "Confidence", "Confidence_Scores"):
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            fields.update(_collect_scalar_field_paths(value, full_key))
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    fields.update(_collect_scalar_field_paths(item, full_key))
                    continue
                if _is_empty_value(item):
                    continue
                fields[full_key] = item
            continue
        if _is_empty_value(value):
            continue
        fields[full_key] = value
    return fields


MIN_EXTRACTED_CONFIDENCE = 0.2


def _apply_min_confidence(score: float) -> float:
    bounded = max(0.0, min(1.0, score))
    return round(max(MIN_EXTRACTED_CONFIDENCE, bounded), 2)


def _to_confidence_value(raw_score: object):
    with suppress(Exception):
        score = float(raw_score)
        return _apply_min_confidence(score)
    return None


def _extract_education_history_entries(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    education_history = payload.get("Education_History")
    if not isinstance(education_history, list):
        return []
    return [entry for entry in education_history if isinstance(entry, dict)]


def _build_education_level_confidence(final_payload: dict, source_pages: list[dict]):
    final_entries = _extract_education_history_entries(final_payload)
    final_degree_values = {
        _normalize_scalar_for_compare(entry.get("Degree"))
        for entry in final_entries
        if not _is_empty_value(entry.get("Degree"))
    }
    if not final_degree_values:
        return None

    matched_scores: list[float] = []
    fallback_scores: list[float] = []

    for page in source_pages:
        if not isinstance(page, dict):
            continue
        metadata = page.get("Document_Metadata", {})
        if not isinstance(metadata, dict):
            continue

        confidence_scores = metadata.get("Confidence_Scores", {})
        if not isinstance(confidence_scores, dict):
            continue

        page_entries = _extract_education_history_entries(page)
        page_has_matching_degree = False
        for entry in page_entries:
            degree_value = _normalize_scalar_for_compare(entry.get("Degree"))
            if degree_value and degree_value in final_degree_values:
                page_has_matching_degree = True
                break

        degree_score = _to_confidence_value(
            confidence_scores.get("Education_History.Degree", confidence_scores.get("Degree"))
        )
        school_score = _to_confidence_value(
            confidence_scores.get("Education_History.School_Name", confidence_scores.get("School_Name"))
        )
        candidate_scores = [s for s in (degree_score, school_score) if s is not None]
        if not candidate_scores:
            continue

        best_page_score = max(candidate_scores)
        if page_has_matching_degree:
            matched_scores.append(best_page_score)
        else:
            fallback_scores.append(best_page_score)

    if matched_scores:
        return _apply_min_confidence(max(matched_scores))
    if fallback_scores:
        return _apply_min_confidence(max(fallback_scores))
    return MIN_EXTRACTED_CONFIDENCE


def _ensure_page_confidence_scores(page_payload: dict) -> None:
    """
    Normalize OCR output so every non-null scalar extracted field has confidence.
    """
    if not isinstance(page_payload, dict):
        return

    metadata = page_payload.get("Document_Metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    existing_scores = metadata.get("Confidence_Scores")
    if not isinstance(existing_scores, dict):
        existing_scores = {}

    normalized_scores: dict[str, float] = {}
    scalar_paths = set(_collect_scalar_field_paths(page_payload).keys())
    scalar_leaf_keys = {path.split(".")[-1] for path in scalar_paths}

    # Keep any valid model-provided confidence keys.
    for k, v in existing_scores.items():
        score = _to_confidence_value(v)
        if score is not None:
            confidence_key = str(k)
            if confidence_key in scalar_paths or confidence_key in scalar_leaf_keys:
                normalized_scores[confidence_key] = score

    # Enforce confidence coverage for all extracted non-null scalar fields.
    for path in scalar_paths:
        leaf_key = path.split(".")[-1]
        score = _to_confidence_value(existing_scores.get(path, existing_scores.get(leaf_key)))
        if score is None:
            score = MIN_EXTRACTED_CONFIDENCE
        prev = normalized_scores.get(path)
        normalized_scores[path] = score if prev is None else max(prev, score)

    metadata["Confidence_Scores"] = normalized_scores
    page_payload["Document_Metadata"] = metadata


def _schema_default_value(schema_node: object) -> object:
    if isinstance(schema_node, dict):
        if not schema_node:
            return {}
        return {k: _schema_default_value(v) for k, v in schema_node.items()}
    if isinstance(schema_node, list):
        return []
    return None


def _apply_schema_shape(value: object, schema_node: object) -> object:
    if isinstance(schema_node, dict):
        if not schema_node:
            return value if isinstance(value, dict) else {}

        source = value if isinstance(value, dict) else {}
        shaped = {}
        for key, child_schema in schema_node.items():
            shaped[key] = _apply_schema_shape(source.get(key), child_schema)
        return shaped

    if isinstance(schema_node, list):
        if not isinstance(value, list):
            return []
        if not schema_node:
            return value
        item_schema = schema_node[0]
        return [_apply_schema_shape(item, item_schema) for item in value]

    if value in ("None", "null"):
        return None
    return value if value is not None else None


def _enforce_mandatory_degree_rule(final_payload: dict) -> dict:
    if not isinstance(final_payload, dict):
        return final_payload

    education_history = final_payload.get("Education_History")
    if not isinstance(education_history, list):
        return final_payload

    filtered_history = []
    for entry in education_history:
        if not isinstance(entry, dict):
            continue
        degree_value = entry.get("Degree")
        if _is_empty_value(degree_value):
            print(f"Removed education entry due to empty degree: {entry}")
            continue
        if _is_invalid_numeric_degree_entry(entry):
            print(f"Removed education entry due to numeric degree value: {entry}")
            continue
        filtered_history.append(entry)

    final_payload["Education_History"] = filtered_history
    t1 = time.time()
    final_payload["Education_History"] = clean_education_duplicates_semantically(final_payload["Education_History"])
    t2 = time.time()
    print(f"Time taken for education deduplication {t2 - t1} seconds")
    return final_payload


def _normalize_text_for_similarity(value: object) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _string_similarity(left: object, right: object) -> float:
    left_norm = _normalize_text_for_similarity(left)
    right_norm = _normalize_text_for_similarity(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _pick_more_descriptive_education_entry(first: dict, second: dict) -> dict:
    first_degree = str(first.get("Degree") or "")
    second_degree = str(second.get("Degree") or "")
    if len(second_degree.strip()) > len(first_degree.strip()):
        return second
    return first


def _deduplicate_education_by_similarity(education_history: list[dict]) -> list[dict]:
    deduped: list[dict] = []

    for entry in education_history:
        if not isinstance(entry, dict):
            continue

        school_name = entry.get("School_Name")
        degree_name = entry.get("Degree")
        if _is_empty_value(degree_name):
            print(f"Removed education entry due to empty degree: {entry}")
            continue

        matched_index = None
        for idx, existing in enumerate(deduped):
            existing_school = existing.get("School_Name")
            existing_degree = existing.get("Degree")

            school_similarity = _string_similarity(school_name, existing_school)
            degree_similarity = _string_similarity(degree_name, existing_degree)
            print(f"{existing}, {entry}, {school_similarity}, {degree_similarity}")

            # Merge if the school is likely the same and degree wording is equivalent/near-equivalent.
            if school_similarity >= 0.8 and degree_similarity >= 0.7:
                matched_index = idx
                break

        if matched_index is None:
            deduped.append(entry)
            continue

        existing_entry = deduped[matched_index]
        selected_entry = _pick_more_descriptive_education_entry(existing_entry, entry)
        removed_entry = entry if selected_entry is existing_entry else existing_entry
        print(f"Removed duplicate education entry by similarity: {removed_entry}")
        deduped[matched_index] = selected_entry

    return deduped


def _enforce_explicit_seminary_rule(final_payload: dict) -> dict:
    if not isinstance(final_payload, dict):
        return final_payload

    seminary_name = final_payload.get("Seminary_Name")
    seminary_address = final_payload.get("Seminary_Address")

    comparison_candidates = [
        final_payload.get("Diocese"),
        final_payload.get("Bishop_Name"),
        final_payload.get("Religious_Order"),
        final_payload.get("Seminary_Email"),
        final_payload.get("Street_Address"),
        final_payload.get("City"),
        final_payload.get("State_or_Province"),
        final_payload.get("Zip_or_Postal_Code"),
        final_payload.get("Country"),
    ]
    comparison_pool = [value for value in comparison_candidates if not _is_empty_value(value)]

    if not _is_empty_value(seminary_name):
        for candidate in comparison_pool:
            if _string_similarity(seminary_name, candidate) >= 0.95:
                final_payload["Seminary_Name"] = None
                break

    seminary_name = final_payload.get("Seminary_Name")
    if not _is_empty_value(seminary_address):
        # Drop seminary address when it appears to be copied from unrelated fields.
        for candidate in comparison_pool:
            if _string_similarity(seminary_address, candidate) >= 0.95:
                final_payload["Seminary_Address"] = None
                break

        # If address is effectively the same as seminary name, treat as unlabeled/ambiguous.
        if final_payload.get("Seminary_Address") is not None and _string_similarity(seminary_address,
                                                                                    seminary_name) >= 0.9:
            final_payload["Seminary_Address"] = None

    return final_payload


def _nullify_overlong_values(value: object, max_length: int = 100) -> object:
    if isinstance(value, dict):
        return {key: _nullify_overlong_values(child, max_length) for key, child in value.items()}

    if isinstance(value, list):
        return [_nullify_overlong_values(item, max_length) for item in value]

    if isinstance(value, str) and len(value) > max_length:
        return None

    return value


def _normalize_final_output_schema(final_payload: dict) -> dict:
    if not isinstance(final_payload, dict):
        return _schema_default_value(FINAL_OUTPUT_SCHEMA)
    normalized_payload = _apply_schema_shape(final_payload, FINAL_OUTPUT_SCHEMA)
    normalized_payload = _enforce_mandatory_degree_rule(normalized_payload)
    normalized_payload = _enforce_explicit_seminary_rule(normalized_payload)
    return _nullify_overlong_values(normalized_payload)


def _consolidate_batch(batch_index: int, batch_pages: list[dict], access_token: str) -> dict:
    """
    Worker: consolidate a single batch (<= CONSOLIDATION_BATCH_SIZE pages).
    Returns batch-level JSON where Document_Metadata is a list of per-page metadata.
    """
    combined_prompt = (
        f"INPUT DATA FRAGMENTS TO MERGE (Array of Page-Level JSONs for batch #{batch_index + 1}, "
        f"containing {len(batch_pages)} page(s)):\n"
        f"{_to_compact_json(batch_pages)}"
    )
    print(f"  -> Consolidating batch #{batch_index + 1} ({len(batch_pages)} pages)...")
    result = _call_consolidation_llm(combined_prompt, CONSOLIDATION_SYSTEM_PROMPT, access_token)

    # --- Log per-batch consolidation output to terminal ---
    print(f"\n{'=' * 20} BATCH #{batch_index + 1} CONSOLIDATION OUTPUT {'=' * 20}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{'=' * 70}\n")

    return result


def consolidate_jsons(page_jsons: list[dict], uploaded_filename: str) -> dict:
    """
    Two-stage parallel consolidation:
      1) Split page JSONs into batches of CONSOLIDATION_BATCH_SIZE and consolidate
         each batch in parallel via the QuickML LLM. Each batch response keeps
         `Document_Metadata` as a LIST of per-page metadata entries (logged to stdout).
      2) Make ONE final LLM call that merges all batch-level responses into the
         single final JSON (with Document_Metadata removed).
    """
    pure_data_payload = [p["data"] for p in page_jsons if "data" in p and "error" not in p["data"]]

    if not pure_data_payload:
        print("⚠️ No valid page data found to consolidate.")
        return _normalize_final_output_schema({})

    access_token = get_zoho_access_token()

    # ---- Stage 1: build batches of CONSOLIDATION_BATCH_SIZE pages ----
    batches = [
        pure_data_payload[i:i + CONSOLIDATION_BATCH_SIZE]
        for i in range(0, len(pure_data_payload), CONSOLIDATION_BATCH_SIZE)
    ]
    print(
        f"Running batched consolidation via Zoho QuickML "
        f"({CONSOLIDATION_MODEL_NAME}): {len(batches)} batch(es) of up to "
        f"{CONSOLIDATION_BATCH_SIZE} pages each..."
    )

    # Fast path: only one batch — no need for a second final call.
    if len(batches) == 1:
        batch_result = _consolidate_batch(0, batches[0], access_token)
        if isinstance(batch_result, dict):
            batch_result.pop("Document_Metadata", None)
            return _normalize_final_output_schema(batch_result)
        return _normalize_final_output_schema({})

    # Run batches in parallel (HTTP I/O bound → thread pool is appropriate).
    batch_results: list[dict] = [None] * len(batches)  # type: ignore[list-item]
    workers_count = max(1, min(len(batches), CONSOLIDATION_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=workers_count) as executor:
        future_to_idx = {
            executor.submit(_consolidate_batch, idx, batch, access_token): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                batch_results[idx] = future.result()
            except Exception as e:
                print(f"  !! Batch #{idx + 1} consolidation failed: {e}")
                batch_results[idx] = {}

    # Filter out empty failures but keep order.
    valid_batch_results = [b for b in batch_results if isinstance(b, dict) and b]

    if not valid_batch_results:
        return _normalize_final_output_schema({})

    # ---- Stage 2: single final merge call ----
    final_merge_inputs = [_compact_batch_for_final_prompt(b) for b in valid_batch_results]

    final_prompt = (
        f'UPLOADED_FILENAME: "{uploaded_filename or ""}"\n'
        "Use this only for First_Name/Last_Name spelling correction.\n"
        f"INPUT BATCH-LEVEL JSONs TO MERGE (Array of {len(final_merge_inputs)} batch result(s); "
        f"each may contain a Document_Metadata LIST of per-page metadata):\n"
        f"{_to_compact_json(final_merge_inputs)}"
    )

    print(f"Running FINAL consolidation merge over {len(final_merge_inputs)} batch result(s)...")

    # Token may have expired during long batch runs — re-fetch via cache.
    access_token = get_zoho_access_token()
    final_result = _call_consolidation_llm(final_prompt, FINAL_CONSOLIDATION_SYSTEM_PROMPT, access_token)

    # Defensive: strip Document_Metadata if model leaked it through.
    if isinstance(final_result, dict):
        final_result.pop("Document_Metadata", None)
        final_result = _normalize_final_output_schema(final_result)

        # --- Log final consolidation output to terminal ---
        print(f"\n{'=' * 20} FINAL CONSOLIDATION OUTPUT {'=' * 20}")
        print(json.dumps(final_result, indent=2, ensure_ascii=False))
        print(f"{'=' * 65}\n")
    else:
        final_result = _normalize_final_output_schema({})

    return final_result


# =====================================================
# PDF TO IMAGES CONVERSION
# =====================================================
def pdf_to_images(pdf_path, output_folder="output_images", zoom=2):
    os.makedirs(output_folder, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    image_paths = []
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    pdf_document.close()
    return image_paths


# =====================================================
# PROFESSIONAL UI CSS INJECTIONS
# =====================================================
st.markdown("""
<style>
    .stApp { background-color: #0B0F17; }
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1600px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .header-bar {
        background: #111827;
        padding: 18px 24px;
        border-radius: 16px;
        border: 1px solid #1F2937;
        margin-bottom: 24px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.2);
    }
    .header-flex { display: flex; align-items: center; gap: 14px; }
    .header-icon { font-size: 50px; line-height: 1; }
    .header-title { font-size: 22px; font-weight: 700; color: #F8FAFC; margin: 0; line-height: 1.2; }
    .header-subtitle { font-size: 13px; color: #9CA3AF; margin: 0; }

    .section-title { font-size: 1.45rem; font-weight: 650; color: #F8FAFC; margin-bottom: 1.3rem; }

    .success-box {
        background-color: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        padding: 1.2rem;
        border-radius: 14px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    .status-box {
        background-color: #111827;
        border: 1px solid #1F2937;
        padding: 1rem;
        border-radius: 14px;
        color: #9CA3AF;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }
    section[data-testid="stFileUploader"] {
        border: 2px dashed #374151;
        border-radius: 16px;
        padding: 1rem;
        background: #111827;
    }
    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #1F2937;
        padding: 1rem;
        border-radius: 14px;
    }
    .metric-container-card [data-testid="stMetricLabel"] > div {
        color: white; font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important; font-weight: 500 !important;
        text-transform: uppercase !important; letter-spacing: 0.5px !important;
    }
    .metric-container-card [data-testid="stMetricValue"] > div {
        color: #FFFFFF; font-family: 'Courier New', monospace !important;
        font-size: 1.6rem !important; font-weight: 700 !important;
    }
    .file-details-heading { font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin-top: 1.5rem; margin-bottom: 1rem; }

    .stButton > button {
        width: 100%; height: 3.2rem; border-radius: 14px; border: none;
        background: #3B82F6; color: #FFFFFF; font-size: 1rem; font-weight: 600;
        transition: all 0.2s ease; margin-top: 1rem;
    }
    .stButton > button:hover { background: #2563EB; box-shadow: 0 0 12px rgba(59, 130, 246, 0.4); }

    .stDownloadButton > button {
        width: 100%; height: 3rem; border-radius: 14px; font-weight: 600;
        background: transparent; color: #3B82F6; border: 1px solid #3B82F6;
    }
    .stDownloadButton > button:hover { background: rgba(59, 130, 246, 0.1); color: #3B82F6; }

    .dashboard-card-container ::-webkit-scrollbar { width: 6px !important; height: 6px !important; }
    .dashboard-card-container ::-webkit-scrollbar-track { background: #111827 !important; border-radius: 10px !important; }
    .dashboard-card-container ::-webkit-scrollbar-thumb { background: #374151 !important; border-radius: 10px !important; }
    .dashboard-card-container ::-webkit-scrollbar-thumb:hover { background: #60A5FA !important; }

    .dashboard-card-container [data-testid="stHorizontalBlock"] { margin-bottom: 20px !important; }
    .dashboard-card-container [data-testid="column"] { display: flex; flex-direction: column; justify-content: center; }
</style>
""", unsafe_allow_html=True)

# Render Heading Banner
st.markdown(
    '<div class="header-bar">'
    '    <div class="header-flex">'
    '        <div class="header-icon">📄</div>'
    '        <div>'
    '            <div class="header-title">DOC Extractor</div>'
    '            <div class="header-subtitle">Intelligent document processing and structured JSON extraction</div>'
    '        </div>'
    '    </div>'
    '</div>',
    unsafe_allow_html=True
)

# =====================================================
# STATE CACHE MANAGEMENT
# =====================================================
for key, initial_val in [
    ("temp_dir", None), ("pdf_path", None), ("total_pages", 0),
    ("pdf_loaded", False), ("page_jsons", None), ("page_number", 0),
    ("extracted_json", None), ("preview_ready", False),
    ("extraction_start_time", None), ("extraction_elapsed", None)
]:
    if key not in st.session_state:
        st.session_state[key] = initial_val

# =====================================================
# LAYOUT STRUCTURE
# =====================================================
left_col, right_col = st.columns([1, 1], gap="large")

# =====================================================
# LEFT PANEL: DOCUMENT MANAGEMENT
# =====================================================
with left_col:
    st.markdown('<div class="dashboard-card-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Document Preview</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    # Reset session state when the uploader is cleared (file removed by user)
    if uploaded_file is None and st.session_state.get("pdf_loaded", False):
        with suppress(Exception):
            if st.session_state.temp_dir and os.path.isdir(st.session_state.temp_dir):
                shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
        st.session_state.temp_dir = None
        st.session_state.pdf_path = None
        st.session_state.total_pages = 0
        st.session_state.pdf_loaded = False
        st.session_state.page_jsons = None
        st.session_state.page_number = 0
        st.session_state.extracted_json = None
        st.session_state.preview_ready = False
        st.session_state.extraction_start_time = None
        st.session_state.extraction_elapsed = None
        st.rerun()

    if uploaded_file:
        if st.session_state.temp_dir is None:
            st.session_state.temp_dir = tempfile.mkdtemp()

        if st.session_state.pdf_path is None:
            st.session_state.pdf_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            file_bytes = uploaded_file.read()
            with open(st.session_state.pdf_path, "wb") as f:
                f.write(file_bytes)

            # Also persist a copy into the local project folder so uploaded
            # PDFs are downloadable/inspectable outside the temp directory.
            try:
                local_save_path = os.path.join(LOCAL_UPLOAD_DIR, uploaded_file.name)
                with open(local_save_path, "wb") as lf:
                    lf.write(file_bytes)
                print(f"📥 Saved uploaded file locally to: {local_save_path}")
            except Exception as e:
                print(f"⚠️ Failed to save uploaded file locally: {e}")

            st.session_state.page_jsons = None
            st.session_state.extracted_json = None
            st.session_state.page_number = 0
            st.session_state.preview_ready = False
            st.session_state.extraction_start_time = None
            st.session_state.extraction_elapsed = None

            pdf_doc = fitz.open(st.session_state.pdf_path)
            st.session_state.total_pages = len(pdf_doc)
            pdf_doc.close()

            st.session_state.pdf_loaded = True
            st.rerun()

        # ------------------------------------------------------------------
        # 10-second loader gate: show a spinner before revealing the preview.
        # The right panel also suppresses its status until preview_ready=True.
        # ------------------------------------------------------------------
        if not st.session_state.preview_ready:
            st.markdown(
                """
                <div style="display:flex; flex-direction:column; align-items:center;
                            justify-content:center; height:520px; gap:18px;">
                    <div style="
                        width:64px; height:64px; border-radius:50%;
                        border:6px solid #1F2937; border-top-color:#3B82F6;
                        animation: docspin 1s linear infinite;"></div>
                    <div style="color:#9CA3AF; font-size:0.95rem;">uploading...</div>
                </div>
                <style>
                @keyframes docspin { to { transform: rotate(360deg); } }
                </style>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(5)
            st.session_state.preview_ready = True
            st.rerun()

        # Render Page Navigation View
        pdf_doc = fitz.open(st.session_state.pdf_path)
        page = pdf_doc.load_page(st.session_state.page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        pdf_doc.close()

        preview_img = os.path.join(st.session_state.temp_dir, "preview.png")
        pix.save(preview_img)

        st.image(preview_img, use_container_width=True)

        nav_col1, nav_col2, nav_col3 = st.columns([2, 3, 2])
        with nav_col1:
            if st.button("⬅ Previous"):
                if st.session_state.page_number > 0:
                    st.session_state.page_number -= 1
                    st.rerun()
        with nav_col2:
            st.markdown(
                f"<div style='display: flex; align-items: center; justify-content: center; height: 2.8rem; font-weight: 600; color: #F8FAFC; font-size: 0.95rem;'>"
                f"Page {st.session_state.page_number + 1} of {st.session_state.total_pages}</div>",
                unsafe_allow_html=True
            )
        with nav_col3:
            if st.button("Next ➡"):
                if st.session_state.page_number < st.session_state.total_pages - 1:
                    st.session_state.page_number += 1
                    st.rerun()

        st.markdown(
            f'<div class="success-box"><span style="font-weight: 600; font-size: 1rem;">✓ File Uploaded Successfully</span>'
            f'<div style="color: #9CA3AF; margin-top: 0.5rem; font-family: monospace; font-size: 0.85rem;">{uploaded_file.name}</div></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="file-details-heading">File Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-container-card">', unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("File Type", "PDF")
        with m_col2:
            st.metric("Size", f"{uploaded_file.size / (1024 * 1024):.2f} MB")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FIELD LABEL MAPPING & DYNAMIC FORM RECONCILIATION
# =====================================================
FIELD_LABELS = {
    "First_Name": "First Name",
    "Last_Name": "Last Name",
    "Salutation": "Salutation",
    "Gender": "Gender",
    "Date_Of_Birth": "Date of Birth",
    "Birth_City": "Birth City",
    "Birth_Province_or_State": "Birth State / Province",
    "Birth_Country": "Birth Country",
    "Tax_Code": "Tax Code",
    "Citizenship_Country": "Citizenship Country",
    "Marital_Status": "Marital Status",
    "ID_Info": "ID Information",
    "Passport_Number": "Passport Number",
    "ID_Card_Number": "ID Card Number",
    "Document_Issue_Date": "Issue Date",
    "Document_Expiry_Date": "Expiry Date",
    "Street_Address": "Street Address",
    "City": "City",
    "State_or_Province": "State / Province",
    "Zip_or_Postal_Code": "ZIP / Postal Code",
    "Country": "Country",
    "Phone": "Phone",
    "Mobile": "Mobile",
    "Fax": "Fax",
    "Primary_Email": "Primary Email",
    "Personal_Email": "Personal Email",
    "Education_History": "Education History",
    "School_Name": "School Name", "Degree": "Degree",
    "Primary_Language": "Primary Language",
    "Languages": "Languages",
    "Education_Level": "Education Level",
    "Diocese": "Diocese", "Bishop_Email": "Bishop Email", "Bishop_Name": "Bishop Name",
    "Seminary_Name": "Seminary Name", "Seminary_Address": "Seminary Address", "Seminary_Email": "Seminary Email",
    "Religious_Status": "Religious Status", "Religious_Order": "Religious Order", "Ordination_Date": "Ordination Date"
}

SECTION_GROUPS = {
    "Personal Details": ["First_Name", "Last_Name", "Salutation", "Gender", "Date_Of_Birth"],
    "Birth Details": ["Birth_City", "Birth_Province_or_State", "Birth_Country"],
    "Identity": ["Tax_Code", "Citizenship_Country", "Marital_Status"],
    "ID / Document": ["ID_Info", "Passport_Number", "ID_Card_Number", "Document_Issue_Date", "Document_Expiry_Date"],
    "Address & Contact": ["Street_Address", "City", "State_or_Province", "Zip_or_Postal_Code", "Country", "Phone",
                          "Mobile", "Fax", "Primary_Email"],
    "Education": ["Education_History", "School_Name", "Degree", "Education_Level"],
    "Languages": ["Primary_Language", "Languages"],
    "Religious Information": ["Diocese", "Bishop_Email", "Bishop_Name", "Seminary_Name", "Seminary_Address",
                              "Seminary_Email", "Religious_Status", "Religious_Order", "Ordination_Date"]
}


def get_section_name(key):
    for section, fields in SECTION_GROUPS.items():
        if key in fields: return section
    return "Other Details"


def has_visible_data(val):
    """Recursively checks if a field contains any non-empty, non-null values."""
    if val in [None, "", [], {}, "None", [""]]:
        return False
    if isinstance(val, dict):
        return any(has_visible_data(v) for v in val.values())
    if isinstance(val, list):
        return any(has_visible_data(item) for item in val)
    return True


def _render_value_box(label: str, value: object) -> None:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f'<div style="padding-top: 10px; font-weight: 600; color: #E5E7EB;">{label}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="padding: 10px 12px; border: 1px solid #1F2937; border-radius: 10px; background: #F8FAFC; color: black; font-size: 0.95rem;">{str(value)}</div>',
            unsafe_allow_html=True,
        )


def _render_structured_list_item(item: dict) -> None:
    with st.container():
        for sub_key, sub_value in item.items():
            if not has_visible_data(sub_value):
                continue

            sub_label = FIELD_LABELS.get(sub_key, sub_key.replace("_", " "))
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(
                    f'<div style="padding-top: 10px; font-weight: 600; color: #E5E7EB; font-size: 0.9rem;">{sub_label}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div style="padding: 10px 12px; border: 1px solid #1F2937; border-radius: 10px; background: #F8FAFC; color: black; font-size: 0.95rem; margin-bottom: 8px;">{str(sub_value)}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<hr style="border: 0; border-top: 1px dashed #374151; margin: 15px 0;">',
            unsafe_allow_html=True,
        )


def _render_simple_list_item(item: object) -> None:
    st.markdown(
        f'<div style="padding: 10px 12px; margin-bottom: 6px; border: 1px solid #1F2937; border-radius: 12px; background: #F8FAFC; color: black;">{item}</div>',
        unsafe_allow_html=True,
    )


def render_dynamic_form(data, parent_key=""):
    if not isinstance(data, dict):
        return

    # Group fields by section, but ONLY if they contain actual data
    grouped = {}
    for k, v in data.items():
        if k in ("Confidence", "Confidence_Scores"):
            continue
        if not has_visible_data(v):
            continue
        sec = get_section_name(k)
        grouped.setdefault(sec, []).append((k, v))

    ordered_section_names = [
                                section_name for section_name in SECTION_GROUPS.keys() if section_name in grouped
                            ] + [
                                section_name for section_name in grouped.keys() if section_name not in SECTION_GROUPS
                            ]

    def _order_fields_for_section(section_name: str, fields: list[tuple[str, object]]) -> list[tuple[str, object]]:
        preferred_order = SECTION_GROUPS.get(section_name, [])
        preferred_set = set(preferred_order)
        field_map = {field_key: field_value for field_key, field_value in fields}

        ordered_fields = [
            (field_key, field_map[field_key])
            for field_key in preferred_order
            if field_key in field_map
        ]

        for field_key, field_value in fields:
            if field_key not in preferred_set:
                ordered_fields.append((field_key, field_value))

        return ordered_fields

    # Render the valid groups
    for section_name in ordered_section_names:
        fields = _order_fields_for_section(section_name, grouped.get(section_name, []))
        st.markdown(
            f'<div style="margin-top: 25px; margin-bottom: 10px; font-size: 1.2rem; font-weight: 800; color: #60A5FA; border-bottom: 1px solid #2d3748; padding-bottom: 6px;">{section_name}</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):
            for key, value in fields:
                display_label = FIELD_LABELS.get(key, key.replace("_", " "))
                field_id = f"{parent_key}_{key}" if parent_key else key

                if isinstance(value, dict):
                    st.markdown(
                        f'<div style="margin-top: 15px; margin-bottom: 10px; font-size: 1.05rem; font-weight: 700; color: #60A5FA;">{display_label}</div>',
                        unsafe_allow_html=True,
                    )
                    with st.container(border=True):
                        render_dynamic_form(value, parent_key=field_id)

                elif isinstance(value, list):
                    st.markdown(
                        f'<div style="margin-top: 15px; margin-bottom: 10px; font-size: 1.05rem; font-weight: 700; color: #60A5FA;">{display_label}</div>',
                        unsafe_allow_html=True,
                    )
                    with st.container(border=True):
                        for item in value:
                            if not has_visible_data(item):
                                continue

                            if isinstance(item, dict):
                                _render_structured_list_item(item)
                            else:
                                _render_simple_list_item(item)
                else:
                    _render_value_box(display_label, value)


def remap_json_keys(data, mapping):
    """
    Recursively remaps dictionary keys based on a provided mapping configuration.
    """

    def _remap_key(raw_key: str) -> str:
        if raw_key in mapping:
            return mapping[raw_key]
        if "." in raw_key:
            return ".".join(mapping.get(part, part) for part in raw_key.split("."))
        return raw_key

    if isinstance(data, list):
        return [remap_json_keys(item, mapping) for item in data]

    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            # Determine the target key from the mapping dictionary
            new_key = _remap_key(k)

            # If the value is a dictionary or list, recurse into it
            if isinstance(v, (dict, list)):
                new_dict[new_key] = remap_json_keys(v, mapping)
            else:
                new_dict[new_key] = v
        return new_dict

    return data


def prepare_final_output_json(data: dict, mapping: dict) -> dict:
    remapped_json = remap_json_keys(data, mapping)

    if isinstance(remapped_json, dict):
        remapped_json.pop("Confidence", None)
        remapped_json.pop("Confidence_Scores", None)

        if "ID_Info" in remapped_json:
            id_data = remapped_json.pop("ID_Info", {})
            if isinstance(id_data, dict):
                for id_k, id_v in id_data.items():
                    mapped_id_k = mapping.get(id_k, id_k)
                    remapped_json[mapped_id_k] = id_v

        remapped_json = order_output_json_keys(remapped_json, mapping)

    return remapped_json


def order_output_json_keys(data: dict, mapping: dict) -> dict:
    if not isinstance(data, dict):
        return data

    ordered = {}

    # First, follow OUTPUT_JSON_MAPPING order.
    for _, mapped_key in mapping.items():
        if mapped_key in data and mapped_key not in ordered:
            ordered[mapped_key] = data[mapped_key]

    # Then append any remaining keys.
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value

    return ordered


# =====================================================
# RIGHT PANEL: PARALLEL AGGREGATION PIPELINE
# =====================================================
with right_col:
    st.markdown('<div class="dashboard-card-container">', unsafe_allow_html=True)

    # All right-panel work is gated by `preview_ready` so the left-side loader
    # is the only visible status during the initial 10-second delay.
    panel_active = st.session_state.get("pdf_loaded", False) and st.session_state.get("preview_ready", False)

    # ----------------------------------------------------------------------
    # AUTO-RUN STAGE 1: Per-page OCR extraction (after the 10s preview delay).
    # ----------------------------------------------------------------------
    if panel_active and st.session_state.page_jsons is None:
        try:
            # Start the extraction timer at the very beginning of stage 1.
            if st.session_state.extraction_start_time is None:
                st.session_state.extraction_start_time = time.time()
                st.session_state.extraction_elapsed = None
            ocr_progress = st.progress(0)
            ocr_status = st.empty()

            image_folder = os.path.join(st.session_state.temp_dir, "output_images")

            # Page Splitting Phase
            if not os.path.exists(image_folder) or len(os.listdir(image_folder)) == 0:
                ocr_status.markdown('<div class="status-box">Converting PDF into pages...</div>',
                                    unsafe_allow_html=True)
                pdf_to_images(st.session_state.pdf_path, image_folder)

            ocr_progress.progress(15)

            # Zoho Catalyst Cloud OCR Extraction Phase
            ocr_status.markdown('<div class="status-box">Running OCR extraction...</div>', unsafe_allow_html=True)

            image_files = sorted(
                [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            access_token = get_zoho_access_token()

            results_map = {}


            def _ocr_thread_worker(index, file_name):
                path = os.path.join(image_folder, file_name)
                extracted_data = extract_from_image(path, access_token)
                return {"page": index + 1, "file": file_name, "data": extracted_data}


            workers_count = max(1, min(len(image_files), OCR_MAX_WORKERS))
            with ThreadPoolExecutor(max_workers=workers_count) as executor:
                future_to_page = {
                    executor.submit(_ocr_thread_worker, i, file): i + 1
                    for i, file in enumerate(image_files)
                }

                completed_count = 0
                for future in as_completed(future_to_page):
                    p_num = future_to_page[future]
                    results_map[p_num] = future.result()
                    completed_count += 1

                    current_pct = 15 + int((completed_count / len(image_files)) * 80)
                    ocr_progress.progress(min(95, current_pct))
                    ocr_status.markdown(
                        f'<div class="status-box">Extracted Page {p_num} of {len(image_files)}...</div>',
                        unsafe_allow_html=True)

            st.session_state.page_jsons = [results_map[i + 1] for i in range(len(image_files))]
            ocr_progress.progress(100)
            ocr_status.empty()
            ocr_progress.empty()
            st.rerun()

        except Exception as e:
            st.error(f"OCR Extraction Aborted: {str(e)}")

    st.markdown('<div class="section-title">Candidate Details</div>', unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # AUTO-RUN STAGE 2: Consolidation runs automatically as soon as page JSONs
    # are ready and no final result exists yet. No button click required.
    # ----------------------------------------------------------------------
    if panel_active and st.session_state.page_jsons is not None and (
            st.session_state.extracted_json is None or "error" in (st.session_state.extracted_json or {})):
        try:
            progress_bar = st.progress(0)
            status_box = st.empty()

            progress_bar.progress(25)
            status_box.markdown('<div class="status-box">Running layout consolidation...</div>',
                                unsafe_allow_html=True)

            uploaded_filename = (
                os.path.basename(st.session_state.pdf_path)
                if st.session_state.get("pdf_path")
                else None
            )
            st.session_state.extracted_json = consolidate_jsons(
                st.session_state.page_jsons,
                uploaded_filename=uploaded_filename,
            )

            progress_bar.progress(100)

            # Stop the extraction timer once the final JSON has been produced.
            if (
                    st.session_state.extraction_start_time is not None
                    and isinstance(st.session_state.extracted_json, dict)
                    and "error" not in st.session_state.extracted_json
            ):
                st.session_state.extraction_elapsed = (
                        time.time() - st.session_state.extraction_start_time
                )

            if "error" in st.session_state.extracted_json:
                status_box.markdown(
                    f'<div class="status-box" style="color: #EF4444;">Pipeline Execution Error: {st.session_state.extracted_json["error"]}</div>',
                    unsafe_allow_html=True)
            else:
                status_box.markdown('<div class="success-box">Document processed successfully </div>',
                                    unsafe_allow_html=True)
                st.rerun()

        except Exception as e:
            st.error(f"Extraction Pipeline Aborted: {str(e)}")

    # ====================================================================
    # RENDER SECTION: UI Forms, Dynamic Mapping, and View Toggle
    # ====================================================================
    if st.session_state.extracted_json and "error" not in st.session_state.extracted_json:

        # Display the total extraction time (OCR + consolidation pipeline).
        elapsed = st.session_state.get("extraction_elapsed")
        if SHOW_EXTRACTION_TIME and elapsed is not None:
            mins, secs = divmod(elapsed, 60)
            if mins >= 1:
                time_str = f"{int(mins)}m {secs:.1f}s"
            else:
                time_str = f"{secs:.2f}s"
            st.markdown(
                f'<div class="success-box" style="display:flex; align-items:center; gap:10px;">'
                f'<span style="font-size:1.1rem;">⏱️</span>'
                f'<span><b>Total Extraction Time:</b> '
                f'<span style="font-family: \'Courier New\', monospace; color:#F8FAFC;">{time_str}</span></span>'
                f'</div>',
                unsafe_allow_html=True
            )

        OUTPUT_JSON_MAPPING = {
            "First_Name": "First_Name", "Last_Name": "Last_Name", "Salutation": "Salutation", "Gender": "Genere",
            "Date_Of_Birth": "Date_of_Birth", "Birth_City": "Citt_di_nascita",
            "Birth_Province_or_State": "Regione_di_nascitta", "Birth_Country": "Nazione_di_nascitta",
            "Tax_Code": "CF", "Citizenship_Country": "Cittadinanza", "Marital_Status": "Stato_civile",
            "Passport_Number": "Passaporto", "ID_Card_Number": "Carta_identit",
            "Document_Issue_Date": "Data_di_emissione", "Document_Expiry_Date": "Data_di_scadenza",
            "Street_Address": "Mailing_Street", "City": "Mailing_City", "State_or_Province": "Mailing_State",
            "Zip_or_Postal_Code": "Mailing_Zip", "Country": "Mailing_Country",
            "Phone": "Phone", "Mobile": "Mobile", "Fax": "Fax", "Primary_Email": "Email",
            "Personal_Email": "Email_personale",
            "Education_History": "Storia_istruzione", "School_Name": "School_Name", "Degree": "Degree",
            "Primary_Language": "Lingua", "Languages": "Lingue", "Education_Level": "Livello",
            "Diocese": "Diocesi", "Bishop_Email": "Email_del_Vescovo", "Bishop_Name": "S_E_R_Mons",
            "Seminary_Name": "Nome_del_seminario", "Seminary_Address": "Indirizzo_del_seminario",
            "Seminary_Email": "E_mail_del_seminario",
            "Religious_Status": "Stato_religioso", "Religious_Order": "Congregazione",
            "Ordination_Date": "Data_ordinazione_sacerdotale"
        }

        remapped_json = prepare_final_output_json(st.session_state.extracted_json, OUTPUT_JSON_MAPPING)

        output_json_str = json.dumps(remapped_json, indent=2, ensure_ascii=False)

        # View Mode Toggle Switch
        view_mode_json = st.toggle("View Raw Output JSON Schema", value=False)
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

        # Dynamic Switcher View Display Frame Container
        form_container = st.container(height=650, border=False)
        with form_container:
            if view_mode_json:
                st.code(output_json_str, language="json")
            else:
                render_dynamic_form(st.session_state.extracted_json)

        # Target Action Output Stream Download Button
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="Download Final JSON",
            data=output_json_str,
            file_name="verified_output.json",
            mime="application/json"
        )
    else:
        # Context-aware guide text for the right panel.
        if not st.session_state.get("pdf_loaded", False):
            # Initial state — no file uploaded yet.
            st.markdown(
                '<div class="status-box" style="margin-top: 10px;">Upload a PDF document to preview pages and extract structured fields into a form.</div>',
                unsafe_allow_html=True)
        elif not panel_active:
            # File uploaded but still inside the 10-second preview loader window.
            # Keep the right panel intentionally quiet — the left-side spinner is the only status.
            pass
