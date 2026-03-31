import pandas as pd
from typing import List


def drop_irrelevant_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drop irrelevant columns from the dataset.
    """
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values for specific categorical columns.
    """
    df = df.fillna({
        'CATEGORY&FUNCTION': 'Unknown',
        'LOCATION': 'Unknown',
        'TAG3': 'Unknown'
    })
    return df


def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date/time columns to unified datetime format and create Outage DateTime column.
    """

    # Convert to datetime (safe parsing)
    df['Outage Time From (Date)'] = pd.to_datetime(
        df.get('Outage Time From (Date)'), 
        format='%A, %B %d, %Y', 
        errors='coerce'
    )

    df['OPEN_DATE'] = pd.to_datetime(
        df.get('OPEN_DATE'), 
        format='%A, %B %d, %Y', 
        errors='coerce'
    )

    # Fill missing Outage Date using OPEN_DATE
    if 'OPEN_DATE' in df.columns:
        df['Outage Time From (Date)'] = df['Outage Time From (Date)'].fillna(
            df['OPEN_DATE'].dt.date.astype(str)
        )

    # Fill missing time with "00:00:00"
    if 'Outage Time (Time)' in df.columns:
        df['Outage Time (Time)'] = df['Outage Time (Time)'].fillna("00:00:00")

    # Combine into Outage DateTime
    df['Outage DateTime'] = pd.to_datetime(
        df['Outage Time From (Date)'].astype(str) + " " + df['Outage Time (Time)'].astype(str),
        errors='coerce'
    )

    # Drop old columns
    drop_cols = ['Outage Time (Time)', 'OPEN_DATE', 'Outage Time From (Date)']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline: drop irrelevant columns, fill missing values,
    convert date columns, and create Outage DateTime.
    """

    columns_to_drop = [
        "TAG1", "TAG2", "TAG6", "SW Version",
        'Last_Activity_Description', 'Next steps', 'Result / Summary of analysis',
        'RESOLVED_GROUP_NAME', 'OPEN_GROUP_NAME', 'SOLUTION', 'UPDATE_DATE',
        'ASSIGNMENT_GROUP_NAME', 'ROUTING_CI_NAME', 'LASTFRYANALYSISDESCRIPTION',
        'LATEST_ANALYSIS/RESEARCH_DESCRIPTION_OF_CC', 'Scope of IR',
        'AGE (in days)', 'CLOSE_DATE', 'PRIORITY', 'STATUS', 'RESOLVED_TIME',
        'REPORTED_TIME', 'LATEST_DEPEND_PROBLEM_TITLE', 'REPORTED_CI_NAME',
        'LATEST_DEPEND_PROBLEM_DESCRIPTION', 'EXTERNAL', 'REFERENCE_NO',
        'LATEST_DEPEND_PROBLEM_NUMBER'
    ]

    df = drop_irrelevant_columns(df, columns_to_drop)
    df = fill_missing_values(df)
    df = convert_datetime_columns(df)

    return df