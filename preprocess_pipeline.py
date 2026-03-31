import pandas as pd

from translation.translator import Translator
from preprocessing.cleaning import clean_dataset
from preprocessing.error_extraction import extract_error_messages
from preprocessing.llm_description_cleaner import add_clean_description_column


class PreprocessPipeline:
    def __init__(self, config):
        self.config = config

        # Translation ON/OFF
        self.translator = Translator(
            key=config["translation"]["api_key"],
            region=config["translation"]["region"],
            endpoint=config["translation"]["endpoint"],
            enabled=config["translation"]["enabled"]
        )

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        # 1. Translate (optional)
        if self.config["translation"]["enabled"]:
            df = self.translator.translate_columns(
                df,
                title_col="TITLE",
                desc_col="DESCRIPTION"
            )

        # 2. Cleaning
        df = clean_dataset(df)

        # 3. Error extraction → CLEANED_ERROR_MESSAGE
        df = extract_error_messages(df)

        # 4. Cleaned LLM description → CLEAN_DESCRIPTION
        df = add_clean_description_column(df)

        return df