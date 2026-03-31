import re
from typing import Optional

# Detect NA-like answers
NA_PATTERN = re.compile(
    r"""^\s*(?:na|n/a|n\.a\.|not\s*applicable|none|null|nil|-\s*|--+)\s*$""",
    re.IGNORECASE
)

# Detect headers like: "Info ###", "Informations ###", "# Information #"
INFO_HEADING_PATTERN = re.compile(
    r"""(?im)^\s*#*\s*info(?:rmation|rmations|s)?\s*#*\s*$"""
)

def clean_description_keep_answers(
    text: Optional[str],
    drop_na_answers: bool = True,
    remove_solution_blocks: bool = True
) -> Optional[str]:
    """
    Cleans ITSM ticket descriptions for GenAI summarisation:
    
    - Removes heading lines: 'Informations ###', 'Information ##', 'Info ###'
    - Removes "Solution KRxxxxx:" blocks
    - Keeps narrative section before Q&A
    - Removes questions (Q#n:)
    - Extracts only answers (A#n:) without labels
    - Drops NA/None answers
    - Normalizes output spacing
    
    This cleaned description is used ONLY for GenAI and NOT for embeddings/clustering.
    """
    
    if text is None:
        return None

    # Normalize newlines
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # Remove lines like "Informations ###"
    s = re.sub(INFO_HEADING_PATTERN, "", s)

    # Remove Solution KRxxxxx blocks (multi-line)
    if remove_solution_blocks:
        s = re.sub(
            r"""(?ims)^\s*Solution\s+KR\d+\s*:\s*.*?(?=^\s*Solution\s+KR\d+\s*:|\Z)""",
            "",
            s
        )

    # Detect beginning of Q/A block
    q_start = re.search(r"(?im)^\s*Q\s*#?\s*\d+\s*:", s)

    # Narrative text before Q#n:
    narrative = s[:q_start.start()].strip() if q_start else s.strip()

    answers = []
    if q_start:
        qa_part = s[q_start.start():]

        # Extract answer blocks only (A#n: ...)
        a_blocks = re.findall(
            r"""(?ims)A\s*#?\s*\d+\s*:\s*(.*?)(?=^\s*Q\s*#?\s*\d+\s*:|\Z)""",
            qa_part
        )

        for a in a_blocks:
            a_clean = a.strip()

            # Remove separators like "-----"
            a_clean = re.sub(r"(?m)^\s*[-=]{2,}\s*$", "", a_clean).strip()

            # Drop NA answers
            if drop_na_answers and NA_PATTERN.match(a_clean):
                continue

            if a_clean:
                answers.append(a_clean)

    # Combine narrative + answers
    parts = []
    if narrative:
        parts.append(narrative)

    parts.extend(answers)

    out = "\n\n".join(parts)

    # Remove extra blank lines
    out = re.sub(r"\n{3,}", "\n\n", out).strip()

    return out

def add_clean_description_column(df):
    """
    Adds a cleaned description column for LLM summarisation.
    """
    df["CLEAN_DESCRIPTION"] = df["translated_description"].apply(
        lambda x: clean_description_keep_answers(
            x,
            drop_na_answers=True,
            remove_solution_blocks=True
        )
    )
    return df