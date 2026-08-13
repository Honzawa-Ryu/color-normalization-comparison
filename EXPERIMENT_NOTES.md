# EXPERIMENT_NOTES.md

実験固有の注意事項をIDごとに記録する。plan-next-experimentでplan.mdを書く際、
review-expで結果を集約する際、debug-experimentで調査する際は、対象実験のIDに
該当する節があればまず読むこと。

## 全体構成

4実験で1本のパイプラインを構成する(1実験=1フェーズ原則)。0002以降は必ず0001の
完了後に、0003/0004は0002の4手法すべて完了後に実行すること(0003と0004はどちらも
0002の出力だけを読むので、互いの前後関係は無い)。

```
0001 extract_tissue_coords                  (TRIDENT seg+coords、正規化非依存、単発)
        ↓ data/trident_processed/{coords_dir}_sub{N}/patches/*.h5
0002 extract_uni_features_by_normalization  (UNI特徴量抽出、array 4タスク: none/macenko/reinhard/vahadane)
        ↓ data/trident_processed/{coords_dir}_sub{N}/features_uni_v1_{method}/*.h5
        ├─ 0003 compare_stain_normalization_batch_effect     (eta-squared/KNNでバッチ効果を比較、単発)
        └─ 0004 evaluate_finding_utility_and_visualize        (has_finding AUROC + t-SNE可視化、単発)
```

`data/trident_processed/` はTRIDENTの共有ジョブディレクトリで、`outputs/`ではなく
`data/`直下に置いている(get_run_dir()のcompletedガード・scratch待避の対象外。
0001が作り、0002の4タスクが読み書きし、複数回のジョブ投入をまたいで永続する必要が
あるため)。`data/raw_table`はconcept-erasing-toxpathoの`data/raw_table`への
シンボリックリンク(同じOpen TG-GATEsスライド由来のEXP_ID/ORGANラベルを参照するため、
コピーはしない)。

**2026-08-12追記(既存資産の再利用)**: `data/trident_processed/20x_224px_0px_overlap/`
配下に、同じ998スライドに対して別プロジェクトで抽出済みだった
セグメンテーション(`contours/`, `contours_geojson/`, `_config_segmentation.json`)・
パッチ座標(`patches/*.h5`)・無正規化のUNI特徴量(`features_uni_v1/*.h5`、全パッチ、
998スライド分)一式をユーザーが複製・配置した。998枚全てで一致することを確認済み
(`raw_slide`のsvsファイル名 = `features_uni_v1`のh5ファイル名、差分なし)。これを
最大限再利用するため、**patch_sizeをTRIDENTの表が挙げる256ではなく既存資産に合わせて
224にした**(UNI自体のネイティブ入力解像度も224であり、256にしてResizeで224へ
落とすより224px@20xをそのまま使う方が素直、という判断も込み)。
これにより:
- 0001は998スライド分のセグメンテーション・座標抽出をほぼ全てスキップできる
  (TRIDENTは`contours/<slide>.jpg`や`patches/<slide>_patches.h5`が既に存在する
  スライドを自動的にスキップする設計のため、mag/patch_size/overlapを既存資産に
  一致させるだけでよい)。GPU/HFアクセスも不要になったため`segmenter: otsu`にし、
  `run_slurm.sh`もCPU・小さいリソースに縮小した。
- 0002の`method=none`タスクは、UNI推論を一切走らせず、既存`features_uni_v1/`から
  0001がサブサンプルした座標と完全一致する行をlookupして再利用する
  (`lib/trident_pipeline.extract_features_by_coords_lookup`)。macenko/reinhard/
  vahadaneの3タスクだけが実際にGPU推論を行う。

「施設間差」は、Open TG-GATEsに本当の意味での施設(病院)列が存在しないため、
EXP_ID(化合物投与実験のバッチID、998スライド中261種)をユーザーの選択により
代理指標として採用している(wsi-adプロジェクトの「試験間差」と同じ考え方)。
「スライド間差」はslide_id(.svsファイル名)そのもの。

## 0001_20260812_extract_tissue_coords

- TRIDENTでdata/raw_slideの998枚を組織セグメンテーション + パッチ座標抽出
  (mag=20, patch_size=224, overlap=0)し、続けて`n_patches_per_slide`(既定500)
  枚/スライドへ決定的にサブサンプルする(`lib/trident_pipeline.subsample_coords`、
  seed固定)。既存資産(上記「全体構成」の追記参照)により998枚全てスキップされる
  想定で、`segmenter: otsu`にしGPU不要にしている。
- もし既存資産が無い/一部欠けている状態でこの実験を使う場合は、`config.yml`の
  `segmenter`を`hest`に戻し、`run_slurm.sh`に`--gres=gpu:1`を追加、`--time`も
  数時間〜規模に応じて伸ばすこと(998枚のセグメンテーションから行う場合、
  hestはGPU/HFアクセスが必要)。
- `run_slurm.sh`の`--time=1:00:00`(small-andre01)は「ほぼ全スキップ」前提の
  見積り。既存資産無しでhestから走らせる場合はこの限りではない。
- `USE_LOCAL_SSD_INPUT`/`USE_LOCAL_SSD_OUTPUT`は両方0にしている
  (data/raw_slideの丸ごとrsyncが無駄なうえ、書き込み先はscratch非対応の永続共有
  キャッシュのため)。

## 0002_20260812_extract_uni_features_by_normalization

- TRIDENT自体には染色正規化のフックが無いため、`trident.patch_encoder_models`の
  `BasePatchEncoder.eval_transforms`(生PILパッチに適用される前処理Callable、
  `WSIPatcherDataset.__getitem__`内でTRIDENT自身のDataLoader worker内で呼ばれる)を
  `lib/stain_normalization.make_patch_transform`でラップすることで、TRIDENT自身の
  `Processor.run_patch_feature_extraction_job`を無改造のまま4バリアントに使い回している
  (`lib/trident_pipeline.get_stain_normalized_encoder`)。詳細は各libファイルの
  docstring参照。
- 染色正規化(Macenko/Reinhard/Vahadane)の実装はtiatoolboxパッケージを直接依存に
  せず、`lib/_vendor_stainnorm.py`にコア部分だけを移植している。
  **理由: tiatoolboxは`timm>=1.0.3,<1.0.28`を要求するが、TRIDENTは`timm==0.9.16`
  固定を要求しており両立しない**(実際に`pip install tiatoolbox`を試して確認済み)。
  正規化のfit対象は自前データの1枚ではなく、tiatoolbox由来の標準参照画像
  (`lib/assets/stain_norm_target.png`、BSD-3-Clause)を使っている。
- `encoder.enc_name`を`uni_v1_{method}`に上書きしている点に注意
  (`lib/trident_pipeline.get_stain_normalized_encoder`のdocstring参照)。
  これをしないと、4手法が同じcoords_dir配下の`_logs_feats_uni_v1.txt`等に
  同時書き込みで競合する。
- `run_slurm.sh`は`RUN_MODE="array"`、`GRID_ARGS=("--method")`、
  `GRID_VALUES=("none macenko reinhard vahadane")`で4タスクに展開(`--array=0-3`)。
  `--time=8:00:00`はmacenko/vahadaneのパッチごとのstain matrix推定(CPU律速)を
  見込んだ初期見積りで未検証。実測を見て調整すること。
- `method=none`は既存`features_uni_v1/`(上記「全体構成」の追記参照)からの座標lookup
  で完結するため実質GPU推論なし(数分オーダーの想定)。ただしarrayの4タスクは同じ
  `#SBATCH`ヘッダーを共有するため`--gres=gpu:1`はnoneタスクでも予約される
  (単体で流したい場合は`experiment.py --method none`をGPU無しで直接実行しても動く)。
- 0001が同じseedでサブサンプルした座標を`--coords_dir`として共有するため、
  4手法は常に同一のパッチ集合(スライド内の同じ座標)を比較している。none以外の
  3手法(macenko/reinhard/vahadane)だけが実際にUNI推論を行う。

## 0003_20260812_compare_stain_normalization_batch_effect

- eta-squared(`lib/validate/batch_effect.compute_eta_squared`、
  concept-erasing-toxpathoの実装をそのまま移植)を主指標として、slide_id/exp_idの
  2グルーピング×4手法で比較。値が低いほどそのグルーピングに沿ったバッチ効果が
  小さい(=正規化が効いている)ことを意味する。
- KNN分類精度(`compute_knn_accuracy`)を副次指標として併記。全パッチ(手法あたり
  約50万)でKNN CVを行うと計算コストが大きいため、`knn_max_per_group`
  (既定20/グループ)でグループ単位に均等サブサンプルしてから評価している
  (グローバルにランダム間引きすると、パッチ数の少ないグループが
  StratifiedKFoldの分割数を下回りクラッシュしうるため、グループ単位の
  キャップにしている — `lib/validate/batch_effect.stratified_subsample_indices`)。
  重い場合は`config.yml`の`compute_knn: false`で無効化できる。
- 出力: `comparison_table.csv`(手法×指標の一覧)、
  `eta_squared_comparison.png`(eta_sq_meanの棒グラフ)、`results.json`(全詳細)。
- **結果(2026-08-13実行、job 8456)**: eta_sq_mean(exp_id/slide)・KNN精度とも
  macenkoが4手法中最も低い(=バッチ効果が最も小さい)。reinhardはnoneより悪化。
  KNN精度の方が手法間の差がはっきり出る(none: slide 0.51/exp_id 0.50 →
  macenko: slide 0.28/exp_id 0.26)。ただし点推定のみで信頼区間は未算出、
  かつ生物学的シグナルを保持できているかは未検証だったため、0004を追加した。

## 0004_20260813_evaluate_finding_utility_and_visualize

ユーザーからのフィードバック(2026-08-13): eta-squared/KNNだけでは「バッチ情報が
消えた」のか「情報ごと潰れた」のか区別できない、という指摘を受けて追加。

- **has_finding利用性プローブ**: `lib/data_process/labels.load_finding_labels`
  (concept-erasing-toxpathoから移植、`open_tggates_pathology.csv`との突合)で
  病理所見の有無を取得し、パッチ特徴量をスライド単位で平均プーリング
  (`lib/trident_pipeline.pool_slide_mean_features`)した上で
  `lib/validate/batch_effect.compute_logreg_probe`(L2ロジスティック回帰、
  balanced_accuracy/ROC-AUC)で4手法を比較する。998スライド中has_finding=True 193/
  False 805(comparison-ad-toxpathoの既知の内訳と一致確認済み)。
- **t-SNE可視化**: 4手法とも同一の座標を共有している性質を利用し、最初に読み込んだ
  手法のslide_idsから`stratified_subsample_indices`で選んだ行インデックス
  (既定: スライドあたり8パッチ、998スライド分で約8000点)を4手法すべてに使い回し、
  同じパッチ集合でt-SNE 2次元embeddingを計算・比較する。スライドIDで色分け
  (`tab20`カラーマップを20周期で使い回す設計、998スライド分の凡例は出さない —
  「どの色が何スライドか」ではなく「塊が残っているか混ざっているか」という
  見た目のパターンを見るためのもの)。出力は`tsne_by_slide.png`(2x2グリッド、
  1手法1パネル)と`tsne_embeddings.npz`(埋め込み生データ、再プロット用)。
- CPUのみ(GPU不要)。`--time=1:00:00`はt-SNE(約8000点×1024次元、4手法分)の
  実測に基づかない初期見積り。

## 既知の問題(2026-08-13発生、対応済み)

0002(job 8427, SIF_PATH未export状態でagentが手動`sbatch`)・0003(job 8456)の
`logs/.../run_metadata.yaml`は`status: FAILED`(`fail_reason`に
`NONZERO_EXIT_⚠️ Apptainer not found or SIF_PATH not set...`)と記録されているが、
**実際には両方とも正常に完了しており(各`outputs/.../completion.json`のstatusは
"completed"、0003の`comparison_table.csv`も正しい値)、FAILED表示は誤り**。

原因は2つの重なり:
1. agentが`runx`ではなく手動`sbatch`で投入した際に`SIF_PATH`をexportし忘れた
   ([[template-daily-experiments-sif-path]]と同じ罠)。これ自体は
   `scripts/slurm_entry.sh`側で非fatalにhost実行へフォールバックするだけで、
   計算自体は成功する(実際にvahadaneのUNI推論も正しく完走している)。
2. **`scripts/slurm_entry.sh`の`_run_single()`のバグ**: SIF_PATH未設定時の警告
   `echo "⚠️ Apptainer not found..."`が標準エラーではなく標準出力に出ていたため、
   `EXIT_CODE=$(_run_single ...)`のコマンド置換がこの警告文と本来の終了コード(0)を
   まとめて`EXIT_CODE`変数に取り込んでしまい、`_handle_final_state`の
   `[ "${exit_code}" -eq 0 ]`が非数値比較で失敗して`FAILED`扱いになっていた。
   `>&2`を付けて標準エラーに逃がすよう修正済み(以後の投入では正しく`COMPLETED`と
   記録される)。

**教訓**: `runx`を使わず手動で`sbatch`する場合は`export SIF_PATH="./env.sif"`を
忘れないこと。ステータス確認は`run_metadata.yaml`の`status`だけでなく、
`outputs/.../completion.json`の`status`(実験スクリプト自身が書く、より信頼できる
記録)も合わせて見ること。

## 依存関係まわりの補足

- `pyproject.toml`に`trident`(GitHub直接依存、00-utils/wsi_preprocessと同じ
  コミットpin)・`timm==0.9.16`・`h5py`・`openslide-python`・
  `opencv-python-headless`・`scikit-image`(vendored stainnormのcontrast_enhancer用)
  を追加済み。`tiatoolbox`自体は上記のtimm競合のため**意図的に**依存に含めていない。
- `runx`/`make uv_sync`実行前は`export SIF_PATH="./env.sif"`を同じBash呼び出し内で
  必ず行うこと(このテンプレート系プロジェクト共通の既知の罠)。
