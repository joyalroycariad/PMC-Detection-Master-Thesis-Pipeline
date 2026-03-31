import os
import json
import requests
from typing import Optional, List, Dict

from pmc.pmc_payload_builder import (
    build_all_cluster_payloads,
    _slugify_filename
)


# -------------------------------------------------------------------
# 1. AUTHENTICATION
# -------------------------------------------------------------------

def get_access_token(client_id: str, client_secret: str) -> str:
    url = "https://idp.cloud.vwgroup.com/auth/realms/kums-mfa/protocol/openid-connect/token"

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }

    resp = requests.post(url, data=data)
    if resp.status_code != 200:
        raise Exception(f"Token Error: {resp.status_code} - {resp.text}")

    return resp.json()["access_token"]


# -------------------------------------------------------------------
# 2. LLM API CALL
# -------------------------------------------------------------------

def call_llm_api(access_token: str, virtual_key: str, model: str, prompt: str) -> str:
    url = "https://llmapi.ai.vwgroup.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-LLM-API-CLIENT-ID": f"Bearer {virtual_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    resp = requests.post(url, headers=headers, json=payload)

    if resp.status_code != 200:
        raise Exception(f"LLM Error: {resp.status_code} - {resp.text}")

    return resp.json()["choices"][0]["message"]["content"]


# -------------------------------------------------------------------
# 3. PROMPT LOADING
# -------------------------------------------------------------------

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_final_prompt(base_prompt: str, cluster_data: dict) -> str:
    cluster_json = json.dumps(cluster_data, indent=2, ensure_ascii=False)
    return f"{base_prompt}\n\n=== CLUSTER DATA ===\n{cluster_json}"


# -------------------------------------------------------------------
# 4. SUMMARISE ONE CLUSTER
# -------------------------------------------------------------------

def summarise_one_cluster(
    cluster_payload: dict,
    prompt_text: str,
    access_token: str,
    virtual_key: str,
    model_name: str
) -> str:
    final_prompt = build_final_prompt(prompt_text, cluster_payload)
    return call_llm_api(access_token, virtual_key, model_name, final_prompt)


# -------------------------------------------------------------------
# 5. SUMMARISE MULTIPLE PMCs (LIMIT N)
# -------------------------------------------------------------------

def summarise_pmc_clusters(
    pmc_df,
    prompt_file: str,
    client_id: str,
    client_secret: str,
    virtual_key: str,
    model_name: str = "gpt-4.1-mini",
    limit: Optional[int] = None,
    output_dir: str = "outputs/summaries"
):
    os.makedirs(output_dir, exist_ok=True)

    # AUTH
    token = get_access_token(client_id, client_secret)

    # PROMPT
    base_prompt = load_prompt(prompt_file)

    # BUILD PAYLOADS
    all_payloads = build_all_cluster_payloads(pmc_df)

    if limit:
        payloads = all_payloads[:limit]
    else:
        payloads = all_payloads

    summaries = []

    for payload in payloads:
        cluster_id = payload["cluster_id"]
        print(f"📝 Summarising PMC: {cluster_id}")

        result = summarise_one_cluster(
            payload,
            base_prompt,
            token,
            virtual_key,
            model_name
        )

        safe_name = _slugify_filename(cluster_id)
        out_path = os.path.join(output_dir, f"{safe_name}_summary.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)

        summaries.append({
            "cluster_id": cluster_id,
            "summary_file": out_path
        })

        print(f"✔ Saved summary: {out_path}")

    print(f"🎉 Total summaries generated: {len(summaries)}")
    return summaries