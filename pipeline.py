import pandas as pd
import os

from preprocess_pipeline import PreprocessPipeline
from embeddings.text_embeddings import TextEmbedder
from clustering.clusterer import Clusterer
from pmc.pmc_creation import PMCCreator
from pmc.pmc_payload_builder import save_cluster_payloads_as_files
from pmc.pmc_summarizer import summarise_pmc_clusters


class PMCPipeline:
    def __init__(self, config):
        self.config = config
        self.preprocessor = PreprocessPipeline(config)

    def run(self):

        # Load data
        df = pd.read_excel(self.config["data"]["input_path"])

        # PREPROCESSING
        df = self.preprocessor.run(df)

        # EMBEDDINGS
        embedder = TextEmbedder(
            model_name=self.config["embeddings"]["model"],
            batch_size=self.config["embeddings"]["batch_size"]
        )
        df, embeddings = embedder.generate_embeddings(
            df,
            text_column="CLEANED_ERROR_MESSAGE"
        )

        # CLUSTERING
        clusterer = Clusterer(
            eps=self.config["clustering"]["eps"],
            min_samples=self.config["clustering"]["min_samples"]
        )
        df = clusterer.cluster_embeddings(df, embeddings)

        # PMC CREATION
        pmc_creator = PMCCreator(
            min_tickets=self.config["pmc"]["min_tickets"],
            window_days=self.config["pmc"]["window_days"]
        )
        pmc_df = pmc_creator.create_pmc_clusters(
            df,
            export_excel=self.config["pmc"]["export_path"]
        )

        # PAYLOAD BUILDER
        save_cluster_payloads_as_files(
            pmc_df,
            out_dir=self.config["payload"]["output_dir"]
        )

        # SUMMARISATION (optional)
        if self.config["summarisation"]["enabled"]:
            summarise_pmc_clusters(
                pmc_df,
                prompt_file=self.config["summarisation"]["prompt_file"],
                client_id=self.config["summarisation"]["client_id"],
                client_secret=self.config["summarisation"]["client_secret"],
                virtual_key=self.config["summarisation"]["virtual_key"],
                model_name=self.config["summarisation"]["model"],
                limit=self.config["summarisation"]["limit"],
                output_dir=self.config["summarisation"]["output_dir"]
            )