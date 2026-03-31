import pandas as pd
from typing import List, Optional


class PMCCreator:
    """
    PMC creation logic:
    - Use DBSCAN clusters (CLUSTER_ID)
    - Group by (cluster, generation, category/function)
    - Check 21-day backward window
    - If >= 5 tickets within window → create PMC cluster
    """

    def __init__(
        self,
        cluster_col: str = "CLUSTER_ID",
        gen_col: str = "GENERATION",
        category_col: str = "CATEGORY&FUNCTION",
        outage_col: str = "Outage DateTime",
        id_col: str = "INCIDENT_NUMBER",
        title_col: str = "translated_title",
        desc_col: str = "translated_description",
        cleaned_desc_col: str = "CLEAN_DESCRIPTION",
        error_col: str = "CLEANED_ERROR_MESSAGE",
        min_tickets: int = 5,
        window_days: int = 21
    ):
        self.cluster_col = cluster_col
        self.gen_col = gen_col
        self.category_col = category_col
        self.outage_col = outage_col
        self.id_col = id_col
        self.title_col = title_col
        self.desc_col = desc_col
        self.cleaned_desc_col = cleaned_desc_col
        self.error_col = error_col
        self.min_tickets = min_tickets
        self.window_days = window_days

    def create_pmc_clusters(
        self,
        df: pd.DataFrame,
        export_excel: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create PMC clusters using rule-based logic + DBSCAN clusters.
        Returns a dataframe containing PMC candidate tickets.
        """

        # Validate outage column
        df[self.outage_col] = pd.to_datetime(df[self.outage_col], errors="coerce")

        # Drop noise clusters
        df_valid = df[df[self.cluster_col] != -1].copy()

        if df_valid.empty:
            print("⚠️ No valid (non-noise) clusters found. No PMCs created.")
            return pd.DataFrame()

        pmc_entries: List[pd.DataFrame] = []

        group_cols = [self.cluster_col, self.gen_col, self.category_col]
        groups = df_valid.groupby(group_cols)

        print(f"🔍 Evaluating {len(groups)} candidate cluster groups...")

        for group_key, subgroup in groups:

            cluster_id, gen, cat_func = group_key

            if len(subgroup) < self.min_tickets:
                continue

            subgroup = subgroup.sort_values(self.outage_col)

            latest_date = subgroup[self.outage_col].max()
            window_start = latest_date - pd.Timedelta(days=self.window_days)

            # Tickets within rolling 21-day window
            window_data = subgroup[
                (subgroup[self.outage_col] >= window_start) &
                (subgroup[self.outage_col] <= latest_date)
            ]

            if len(window_data) >= self.min_tickets:
                pmc_key = f"{cluster_id}_{cat_func}_{gen}_{latest_date.date()}"

                window_data = window_data.assign(PMC_CLUSTER=pmc_key)
                pmc_entries.append(window_data)

        if not pmc_entries:
            print("⚠️ No PMC clusters met the criteria.")
            return pd.DataFrame()

        pmc_df = pd.concat(pmc_entries, ignore_index=True)

        print(f"✅ Total PMC clusters created: {pmc_df['PMC_CLUSTER'].nunique()}")

        if export_excel:
            export_cols = [
                self.cluster_col, "PMC_CLUSTER",
                self.id_col, self.title_col, self.desc_col,
                self.error_col, self.cleaned_desc_col,
                self.category_col, self.gen_col,
                self.outage_col
            ]

            pmc_df.to_excel(export_excel, index=False)
            print(f"📂 PMC supervisor view exported to: {export_excel}")

        return pmc_df
