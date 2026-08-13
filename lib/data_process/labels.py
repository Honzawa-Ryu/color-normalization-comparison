from pathlib import Path
from typing import Union

import pandas as pd


def load_batch_labels(pathological_image_csv: Union[str, Path]) -> pd.DataFrame:
    """
    Open TG-GATEsの画像テーブルから、スライドID(.svsファイル名)ごとのバッチラベルを
    まとめたテーブルを返す。

    Open TG-GATEsには本当の意味での「施設(病院)」列は存在しないため、EXP_ID
    (化合物投与実験のバッチID)を「施設間差」相当のグループラベルとして使う
    (色/スキャナ由来ではなく生物学的な投与実験単位のバッチだが、このプロジェクトでは
    ユーザーの選択によりこの定義を採用している)。「スライド間差」はslide_id自体を
    使えばよいため、ここでは返さない。

    open_tggates_pathological_image.csv の FILE_LOCATION 末尾 "{slide_id}.svs" から
    slide_idを取り出す。1スライド(.svsファイル)につき1行のみ存在する
    (EXP_ID/ORGANの組み合わせごとに別ファイルのため重複はない)。

    Returns
    -------
    pd.DataFrame
        slide_idをindexとし、以下の列を持つ:
        exp_id, organ, compound_name, dose, sacrifice_period
    """
    image_df = pd.read_csv(pathological_image_csv, encoding="cp932", encoding_errors="replace")
    image_df["slide_id"] = image_df["FILE_LOCATION"].str.extract(r"/(\d+)\.svs$", expand=False)
    image_df = image_df.dropna(subset=["slide_id"]).set_index("slide_id")

    return image_df[["EXP_ID", "ORGAN", "COMPOUND_NAME", "DOSE", "SACRIFICE_PERIOD"]].rename(
        columns={
            "EXP_ID": "exp_id",
            "ORGAN": "organ",
            "COMPOUND_NAME": "compound_name",
            "DOSE": "dose",
            "SACRIFICE_PERIOD": "sacrifice_period",
        }
    )


def load_finding_labels(
    pathological_image_csv: Union[str, Path],
    pathology_csv: Union[str, Path],
) -> pd.DataFrame:
    """
    Open TG-GATEsの画像テーブルと病理所見テーブルを突合し、スライドID(.svsファイル名)
    ごとに病理所見の有無(has_finding)をまとめたテーブルを返す(concept-erasing-toxpathoの
    同名関数を移植)。

    open_tggates_pathological_image.csv の FILE_LOCATION 末尾 "{slide_id}.svs" から
    slide_idを取り出し、(EXP_ID, GROUP_ID, INDIVIDUAL_ID, ORGAN)をキーに
    open_tggates_pathology.csv(実際の病理所見)と突合する。一致する所見行が1件も無い
    スライドは「所見なし(正常)」(has_finding=False)として扱う(TG-GATEsの病理テーブルは
    異常所見のみを記録する形式のため)。

    正規化手法によって「バッチ効果だけ消えたのか、生物学的シグナルごと潰れていないか」を
    確認するための、eta-squared/KNN(batch leakage側)と対になるタスク有用性チェックに使う。

    Returns
    -------
    pd.DataFrame
        slide_idをindexとし、以下の列を持つ:
        has_finding, n_findings, compound_name, organ, dose, sacrifice_period
    """
    image_df = pd.read_csv(pathological_image_csv, encoding="cp932", encoding_errors="replace")
    image_df["slide_id"] = image_df["FILE_LOCATION"].str.extract(r"/(\d+)\.svs$", expand=False)
    image_df = image_df.dropna(subset=["slide_id"])

    pathology_df = pd.read_csv(pathology_csv, encoding="cp932", encoding_errors="replace")

    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "ORGAN"]
    finding_counts = pathology_df.groupby(key).size().rename("n_findings")

    merged = image_df.join(finding_counts, on=key)
    merged["n_findings"] = merged["n_findings"].fillna(0).astype(int)
    merged["has_finding"] = merged["n_findings"] > 0

    merged = merged.set_index("slide_id")
    return merged[
        ["has_finding", "n_findings", "COMPOUND_NAME", "ORGAN", "DOSE", "SACRIFICE_PERIOD"]
    ].rename(
        columns={
            "COMPOUND_NAME": "compound_name",
            "ORGAN": "organ",
            "DOSE": "dose",
            "SACRIFICE_PERIOD": "sacrifice_period",
        }
    )
