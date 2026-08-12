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
