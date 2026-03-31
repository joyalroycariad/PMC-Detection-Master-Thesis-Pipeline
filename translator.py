import requests
import uuid
import logging
from typing import Optional

class Translator:
    """
    Translator class that wraps Azure Cognitive Translator API
    and provides auto language detection translation.
    """

    def __init__(self, key: str, region: str, endpoint: str, to_lang: str = "en", enabled: bool = True):
        self.key = key
        self.region = region
        self.endpoint = endpoint
        self.to_lang = to_lang
        self.enabled = enabled  # 🔥 Translation ON/OFF switch

        self.constructed_url = f"{self.endpoint}/translate"
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Ocp-Apim-Subscription-Region": self.region,
            "Content-type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4())
        }

    def translate_text(self, text: Optional[str]) -> str:
        """
        Translate text to target language using Azure auto-detection.
        If translation is disabled → returns original text.
        """

        # Skip translation if disabled
        if not self.enabled:
            return text if isinstance(text, str) else ""

        if not isinstance(text, str) or not text.strip():
            return ""

        body = [{"text": text}]
        params = {
            "api-version": "3.0",
            "to": [self.to_lang]
        }

        try:
            response = requests.post(
                self.constructed_url,
                params=params,
                headers=self.headers,
                json=body
            )
            response.raise_for_status()
            result = response.json()
            return result[0]["translations"][0]["text"]

        except Exception as e:
            logging.error(f"Translation error: {e}")
            return text  # fallback: return original text

    def translate_columns(self, df, title_col="TITLE", desc_col="DESCRIPTION"):
        """
        Translate 2 columns of a dataframe.
        Returns df with translated_title and translated_description.
        """

        translated_titles = []
        translated_descriptions = []

        for _, row in df.iterrows():
            title = row.get(title_col, "")
            desc = row.get(desc_col, "")

            translated_titles.append(self.translate_text(title))
            translated_descriptions.append(self.translate_text(desc))

        df["translated_title"] = translated_titles
        df["translated_description"] = translated_descriptions

        return df