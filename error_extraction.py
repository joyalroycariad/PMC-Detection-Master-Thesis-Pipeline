import re
import pandas as pd
from typing import Optional

# ---------------------------
# Normalization
# ---------------------------

def normalize_text(text: Optional[str]) -> str:
    if pd.isnull(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------
# Placeholder phrases
# ---------------------------

PLACEHOLDER_PHRASES = [
    "n/a", "na", "", "none", "attached", "screenshot", "screenshot attached",
    "see image", "see screenshot", "see attached", "kindly see attachments", "see attachment",
    "see attachments", "refer to screenshot", "refer to image", "see attached 01. and 02.",
    "see attached 01", "see attached 02.", "please find attached",
    "please find the attached screenshot", "please find the attached", "see attached images",
    "see attached screenshots", "screenshot attached 01", "image attached", "images attached",
    "refer to attached screenshot", "attached 01. and 02.", "attached 01.", "attached 02.",
    "screenshot attached 01 and 02.", "screenshot 01", "screenshot 02", "attached the screenshot",
    "n a", "not applicable", "screenshot 03", "screenshot 04",
    "screenshot 01,02,03,04", "see 01", "see 02", "see 03", "see 04",
    "screenshot 01,02,03,04,05", "please attachment", "please see the attachments",
    "please check attachments.", "please check the attachments", "please find attachments",
    "attachment", "attachments", "in attachment", "see the attachments", "see attachment",
    "screenshots 01", "screenshots 01 02 03", "screenshots 02"
]

PLACEHOLDER_PHRASES = [p.lower() for p in PLACEHOLDER_PHRASES]


# ---------------------------
# Error Keywords
# ---------------------------

ERROR_KEYWORDS = [
    "error", "not working", "failed", "cannot", "can't", "unable", "guest user",
    "went wrong", "problem", "issue", "crash", "bug", "fault", "glitch", "incorrect",
    "malfunction", "disconnect", "stuck", "missing", "oops", "wrong", "not", "fail",
    "can’t", "doesn’t work", "won’t start", "no response", "freeze", "hang", "timeout",
    "unexpected", "unavailable", "not responding", "not accessible"
]


# ---------------------------
# A#3 Block Extraction
# ---------------------------

def extract_from_a3(description: str) -> Optional[str]:
    if not isinstance(description, str):
        return None

    match = re.search(
        r"(?:q#\s*3.*?a#\s*3[:\-]?\s*)(.*?)(?=q#\s*4|$)",
        description,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        a3_content = match.group(1).strip().lower()
        if a3_content in PLACEHOLDER_PHRASES:
            return None
        return a3_content

    return None


# ---------------------------
# Extract error phrase from title
# ---------------------------

def extract_error_phrase_from_title(title: str) -> Optional[str]:
    if not isinstance(title, str):
        return None

    segments = re.split(r"[-:/]", title)
    for segment in segments:
        segment = segment.strip()
        if any(kw in segment for kw in ERROR_KEYWORDS):
            return segment
    return None


# ---------------------------
# Clean final error message
# ---------------------------

def clean_error_message(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    # Remove placeholder content
    for phrase in PLACEHOLDER_PHRASES:
        pattern = rf"\b{re.escape(phrase)}[:\-]?\b"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove leading punctuation
    text = re.sub(r"^[^\w]+", "", text)

    # Remove references like "01."
    text = re.sub(r"\b\d{1,2}\.\s*", "", text)

    # Remove unwanted special characters
    text = re.sub(r"[^a-z0-9\s\-':./_]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else None


# ---------------------------
# Pipeline for each row
# ---------------------------

def get_final_error(row) -> Optional[str]:
    # Step 1: A#3 content
    if row.get("ERROR_MESSAGE"):
        return row["ERROR_MESSAGE"]

    # Step 2: fallback to title
    title_segment = extract_error_phrase_from_title(row.get("NORMALIZED_TITLE", ""))
    if title_segment:
        return title_segment

    return None


# ---------------------------
# Full Error Extraction Pipeline
# ---------------------------

def extract_error_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full error extraction logic on a dataframe. 
    Produces CLEANED_ERROR_MESSAGE.
    """

    # Normalize text
    df["NORMALIZED_TITLE"] = df["translated_title"].apply(normalize_text)
    df["NORMALIZED_DESCRIPTION"] = df["translated_description"].apply(normalize_text)

    # Extract from A#3
    df["ERROR_MESSAGE"] = df["NORMALIZED_DESCRIPTION"].apply(extract_from_a3)

    # Fallback to title logic
    df["EXTRACTED_ERROR_MESSAGE"] = df.apply(get_final_error, axis=1)

    # Clean final error message
    df["CLEANED_ERROR_MESSAGE"] = df["EXTRACTED_ERROR_MESSAGE"].apply(clean_error_message)

    # Count BEFORE filtering
    extracted_count = df["EXTRACTED_ERROR_MESSAGE"].notnull().sum()
    cleaned_count = df["CLEANED_ERROR_MESSAGE"].notnull().sum()

    print(f"🔍 Extracted error messages: {extracted_count}")
    print(f"🧹 Cleaned final error messages: {cleaned_count}")

    # Filter by length
    df = df[df["CLEANED_ERROR_MESSAGE"].apply(lambda x: len(x) <= 150 if x else True)]

    # Remove empty cleaned errors
    df = df[
        df["CLEANED_ERROR_MESSAGE"].notnull()
        & df["CLEANED_ERROR_MESSAGE"].apply(lambda x: isinstance(x, str) and x.strip() != "")
    ].copy()

    # Count AFTER filtering
    final_count = len(df)
    print(f"✅ Final usable error messages for embeddings: {final_count}")

    return df