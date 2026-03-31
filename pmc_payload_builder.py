import os
import json
import re
from typing import Dict, Any, List, Optional
import pandas as pd


# -------------------------------------------------------------------
# Column Mapping (MUST match preprocessing output)
# -------------------------------------------------------------------
COLUMN_MAP = {
    "cluster_id": "PMC_CLUSTER",
    "ir_number": "INCIDENT_NUMBER",
    "title": "translated_title",
    "description": "CLEAN_DESCRIPTION",
    "extracted_error_message": "CLEANED_ERROR_MESSAGE",
    "category_function": "CATEGORY&FUNCTION",
    "generation": "GENERATION",
    "affected_ci_name": "AFFECTED_CI_NAME",
    "brand": "BRAND",
    "location": "LOCATION",
    "outage_datetime": "Outage DateTime",
    "input_channel": "TAG3"
}

MAX_TICKETS_PER_CLUSTER = None   # None = include all tickets


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _safe_get(row: pd.Series, col: Optional[str]) -> str:
    if not col or col not in row.index:
        return ""
    val = row.get(col, "")
    return "" if pd.isna(val) else str(val).strip()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _unique_list(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for v in values:
        v = (v or "").strip()
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _col(df: pd.DataFrame, key: str) -> Optional[str]:
    col_name = COLUMN_MAP.get(key)
    return col_name if col_name in df.columns else None


def _slugify_filename(value: str, max_len: int = 80) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-z0-9._-]", "", value)
    value = re.sub(r"[-_]{2,}", "-", value)
    return value[:max_len] or "cluster"


# -------------------------------------------------------------------
# Build Single Cluster Payload
# -------------------------------------------------------------------

def build_cluster_payload(df_cluster: pd.DataFrame) -> Dict[str, Any]:

    if df_cluster.empty:
        raise ValueError("df_cluster is empty")

    # Resolve columns
    c_cluster = _col(df_cluster, "cluster_id")
    c_ir = _col(df_cluster, "ir_number")
    c_title = _col(df_cluster, "title")
    c_desc = _col(df_cluster, "description")
    c_err = _col(df_cluster, "extracted_error_message")
    c_cat = _col(df_cluster, "category_function")
    c_gen = _col(df_cluster, "generation")
    c_ci = _col(df_cluster, "affected_ci_name")
    c_brand = _col(df_cluster, "brand")
    c_loc = _col(df_cluster, "location")
    c_out = _col(df_cluster, "outage_datetime")
    c_chan = _col(df_cluster, "input_channel")

    cluster_id = str(df_cluster[c_cluster].iloc[0]) if c_cluster else "UNKNOWN"

    # Sort + optional limiting
    if c_out:
        df_sorted = df_cluster.sort_values(c_out)
    else:
        df_sorted = df_cluster.copy()

    if MAX_TICKETS_PER_CLUSTER is None:
        df_take = df_sorted
    else:
        df_take = df_sorted.head(MAX_TICKETS_PER_CLUSTER)

    tickets = []
    irs, brands, locs, errs = [], [], [], []

    for _, row in df_take.iterrows():

        ticket = {
            "ir_number": _safe_get(row, c_ir),
            "title": _normalize_space(_safe_get(row, c_title)),
            "description": _normalize_space(_safe_get(row, c_desc)),
            "extracted_error_message": _normalize_space(_safe_get(row, c_err)),
            "category_function": _safe_get(row, c_cat),
            "generation": _safe_get(row, c_gen),
            "affected_ci_name": _safe_get(row, c_ci),
            "brand": _safe_get(row, c_brand),
            "location": _safe_get(row, c_loc),
            "outage_datetime": _safe_get(row, c_out),
            "input_channel": _safe_get(row, c_chan)
        }

        tickets.append(ticket)

        if ticket["ir_number"]: irs.append(ticket["ir_number"])
        if ticket["brand"]: brands.append(ticket["brand"])
        if ticket["location"]: locs.append(ticket["location"])
        if ticket["extracted_error_message"]: errs.append(ticket["extracted_error_message"])

    return {
        "cluster_id": cluster_id,
        "ticket_count": len(df_cluster),
        "tickets": tickets,
        "ir_numbers": _unique_list(irs),
        "brands": _unique_list(brands),
        "locations": _unique_list(locs),
        "error_messages": _unique_list(errs)
    }


# -------------------------------------------------------------------
# Build Payloads For ALL PMCs
# -------------------------------------------------------------------

def build_all_cluster_payloads(pmc_df: pd.DataFrame) -> List[Dict[str, Any]]:
    c_cluster = _col(pmc_df, "cluster_id")
    if not c_cluster:
        raise ValueError("PMC_CLUSTER column missing.")

    payloads = []
    for cluster_id, df_cluster in pmc_df.groupby(c_cluster):
        payloads.append(build_cluster_payload(df_cluster))

    return payloads


# -------------------------------------------------------------------
# Save Payloads to Files
# -------------------------------------------------------------------

def save_cluster_payloads_as_files(
    pmc_df: pd.DataFrame,
    out_dir: str = "payload",
    include_master_index: bool = True,
    write_ndjson: bool = True
) -> None:

    payloads = build_all_cluster_payloads(pmc_df)
    os.makedirs(out_dir, exist_ok=True)

    index_records = []
    ndjson_path = os.path.join(out_dir, "all_clusters.ndjson")

    if write_ndjson:
        open(ndjson_path, "w", encoding="utf-8").close()

    for payload in payloads:

        cluster_id = payload.get("cluster_id", "UNKNOWN")
        safe_name = _slugify_filename(cluster_id)
        filename = f"{safe_name}.json"
        filepath = os.path.join(out_dir, filename)

        # Avoid filename collision
        counter = 2
        while os.path.exists(filepath):
            filename = f"{safe_name}-{counter}.json"
            filepath = os.path.join(out_dir, filename)
            counter += 1

        # Write JSON file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        index_records.append({
            "cluster_id": cluster_id,
            "filename": filename
        })

        if write_ndjson:
            with open(ndjson_path, "a", encoding="utf-8") as nd:
                nd.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if include_master_index:
        with open(os.path.join(out_dir, "master_index.json"), "w", encoding="utf-8") as f:
            json.dump(index_records, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(payloads)} payloads to {out_dir}")