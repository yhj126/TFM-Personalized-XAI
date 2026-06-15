
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = PROJECT_ROOT / "base_de_casos/cbr_case_base_outputs/case_base_double_full.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "base_de_casos/cbr_similarity_outputs"
DEFAULT_IMAGE_SIMILARITY = DEFAULT_OUTPUT_DIR / "image_ssim_by_id.csv"
GROUP_COL = "image_id"

BLOCK_WEIGHTS = {
    "text": 0.40,
    "profile": 0.40,
    "image": 0.20,
}

# ------------------------------------------------------------
# Campos del problema: información conocida antes de recomendar
# ------------------------------------------------------------

# Bloque de perfil: datos del usuario disponibles antes de recomendar.
PROBLEM_PROFILE_NUMERIC = ["ai_knowledge_level", "domain_knowledge_level"]

# Bloque de perfil: variables categoricas del usuario.
PROBLEM_PROFILE_CATEGORICAL = [
    "age_range",
    "education_level",
    "occupation_raw",
]

# Bloque de imagen: metadatos disponibles antes de recomendar.
PROBLEM_IMAGE_CATEGORICAL = ["image_id", "domain", "model_predicted_class"]

# Bloque textual: descripcion inicial y soporte VQA del problema.
PROBLEM_TEXT_COLUMNS = ["initial_description", "vqa_support_description"]
PROBLEM_TEXT = "problem_text"

PROBLEM_FEATURES = [
    PROBLEM_TEXT,
    *PROBLEM_PROFILE_NUMERIC,
    *PROBLEM_PROFILE_CATEGORICAL,
    *PROBLEM_IMAGE_CATEGORICAL,
]

# ------------------------------------------------------------
# Campos de solución: lo que el CBR debe recuperar o predecir
# ------------------------------------------------------------

SOLUTION_FIELDS = [
    "selected_option",
    "preferred_response_length",
    "preferred_technical_level",
    "preferred_format",
    "perceived_error_impact",
    "preferred_explanation_types_raw",
    "main_goals_raw",
]

# Campos multivalor: no conviene evaluarlos solo con accuracy exacta.
MULTIVALUE_FIELDS = [
    "preferred_explanation_types_raw",
    "main_goals_raw",
]


def build_dataset(path: Path) -> pd.DataFrame:
    """Carga la base de casos y construye el texto del problema."""

    df = pd.read_csv(path)

    required_columns = [
        GROUP_COL,
        "case_id",
        "user_id",
        *PROBLEM_PROFILE_NUMERIC,
        *PROBLEM_PROFILE_CATEGORICAL,
        *PROBLEM_IMAGE_CATEGORICAL,
        *SOLUTION_FIELDS,
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    for col in PROBLEM_TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df[PROBLEM_TEXT] = (
        df["initial_description"].fillna("").astype(str)
        + " "
        + df["vqa_support_description"].fillna("").astype(str)
    ).str.strip()

    return df


def build_profile_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador del bloque de perfil."""

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                PROBLEM_PROFILE_NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                PROBLEM_PROFILE_CATEGORICAL,
            ),
        ],
        sparse_threshold=1.0,
    )


def build_image_metadata_preprocessor(columns: list[str] | None = None) -> ColumnTransformer:
    """Construye el preprocesador de metadatos de imagen si no hay SSIM."""

    selected_columns = columns or PROBLEM_IMAGE_CATEGORICAL
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                selected_columns,
            ),
        ],
        sparse_threshold=1.0,
    )


def normalize_image_id(value: Any) -> Any:
    """Normaliza IDs de imagen para alinear CSVs y casos."""

    if pd.isna(value):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def load_image_similarity_matrix(path: Path | None) -> pd.DataFrame | None:
    """Carga una matriz visual precalculada, por ejemplo SSIM por image_id."""

    if path is None or not path.exists():
        return None

    matrix = pd.read_csv(path, index_col=0)
    matrix.index = [normalize_image_id(value) for value in matrix.index]
    matrix.columns = [normalize_image_id(value) for value in matrix.columns]
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0, upper=1)


def image_similarity_matrix_query_vs_base(
    query_image_ids: pd.Series,
    base_image_ids: pd.Series,
    image_similarity: pd.DataFrame,
) -> np.ndarray:
    """Devuelve similitudes visuales query-base usando la matriz precalculada."""

    sim = np.zeros((len(query_image_ids), len(base_image_ids)), dtype=float)
    for query_pos, query_image_id in enumerate(query_image_ids):
        query_id = normalize_image_id(query_image_id)
        for base_pos, base_image_id in enumerate(base_image_ids):
            candidate_id = normalize_image_id(base_image_id)
            if query_id in image_similarity.index and candidate_id in image_similarity.columns:
                sim[query_pos, base_pos] = float(image_similarity.loc[query_id, candidate_id])
            elif query_id == candidate_id:
                sim[query_pos, base_pos] = 1.0
    return sim


def _safe_cosine_similarity(query_x: Any, base_x: Any) -> np.ndarray | None:
    """Calcula coseno si el bloque tiene columnas útiles."""

    if getattr(base_x, "shape", (0, 0))[1] == 0:
        return None
    return cosine_similarity(query_x, base_x)


def _text_similarity(train_df: pd.DataFrame, query_df: pd.DataFrame) -> np.ndarray | None:
    """Similitud coseno TF-IDF para el bloque textual."""

    train_text = train_df[PROBLEM_TEXT].fillna("").astype(str)
    query_text = query_df[PROBLEM_TEXT].fillna("").astype(str)
    if not train_text.str.strip().any():
        return None

    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=3000,
        min_df=1,
    )
    try:
        train_x = vectorizer.fit_transform(train_text)
        query_x = vectorizer.transform(query_text)
    except ValueError:
        return None
    return _safe_cosine_similarity(query_x, train_x)


def _profile_similarity(train_df: pd.DataFrame, query_df: pd.DataFrame) -> np.ndarray | None:
    """Similitud coseno para perfil de usuario con imputacion, escalado y OHE."""

    columns = PROBLEM_PROFILE_NUMERIC + PROBLEM_PROFILE_CATEGORICAL
    preprocessor = build_profile_preprocessor()
    train_x = preprocessor.fit_transform(train_df[columns])
    query_x = preprocessor.transform(query_df[columns])
    return _safe_cosine_similarity(query_x, train_x)


def _image_similarity(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    image_similarity: pd.DataFrame | None,
) -> np.ndarray | None:
    """Similitud del bloque de imagen: SSIM si existe, metadatos OHE si no."""

    if image_similarity is not None:
        return image_similarity_matrix_query_vs_base(
            query_df[GROUP_COL],
            train_df[GROUP_COL],
            image_similarity,
        )

    available_columns = [
        col
        for col in PROBLEM_IMAGE_CATEGORICAL
        if col in train_df.columns and train_df[col].notna().any()
    ]
    if not available_columns:
        return None

    preprocessor = build_image_metadata_preprocessor(available_columns)
    train_x = preprocessor.fit_transform(train_df[available_columns])
    query_x = preprocessor.transform(query_df[available_columns])
    return _safe_cosine_similarity(query_x, train_x)


def compute_weighted_similarity(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    block_weights: dict[str, float] | None = None,
    image_similarity: pd.DataFrame | None = None,
    return_block_similarities: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Calcula similitud CBR por bloques y combina con suma ponderada.

    Solo usa campos de problema. Si un bloque no esta disponible, su peso se
    ignora y el total se normaliza por la suma de pesos activos.
    """

    weights = block_weights or BLOCK_WEIGHTS
    block_similarities: dict[str, np.ndarray] = {}

    text_sim = _text_similarity(train_df, query_df)
    if text_sim is not None:
        block_similarities["text"] = text_sim

    profile_sim = _profile_similarity(train_df, query_df)
    if profile_sim is not None:
        block_similarities["profile"] = profile_sim

    image_sim = _image_similarity(train_df, query_df, image_similarity)
    if image_sim is not None:
        block_similarities["image"] = image_sim

    total = np.zeros((len(query_df), len(train_df)), dtype=float)
    active_weight = 0.0
    for block_name, block_similarity in block_similarities.items():
        weight = float(weights.get(block_name, 0.0))
        if weight <= 0:
            continue
        total += weight * np.nan_to_num(block_similarity, nan=0.0)
        active_weight += weight

    if active_weight > 0:
        total = total / active_weight

    if return_block_similarities:
        return total, block_similarities
    return total


def weighted_vote(neighbors: pd.DataFrame, target: str) -> Any:
    """
    Predice un campo de solución por voto ponderado.

    El peso usado es la similitud. No se usa información del caso de validación.
    """

    valid = neighbors[[target, "similarity"]].dropna()
    if valid.empty:
        return np.nan

    weights = valid.groupby(target, dropna=False)["similarity"].sum()
    return weights.sort_values(ascending=False).index[0]


def majority_vote(candidates: pd.DataFrame, target: str) -> Any:
    """
    Baseline de clase mayoritaria.

    Para cada consulta, predice el valor más frecuente del campo en los candidatos
    disponibles. Este baseline se calcula sin usar el caso de validación.
    """

    valid = candidates[target].dropna()
    if valid.empty:
        return np.nan

    return valid.mode(dropna=True).iloc[0]


def split_multivalue(value: Any) -> set[str]:
    """
    Convierte un campo multivalor textual en un conjunto de etiquetas normalizadas.

    Se separa por coma, punto y coma o barra vertical. Esto permite evaluar
    coincidencia parcial en campos como objetivos o tipos de explicación.
    """

    if pd.isna(value):
        return set()

    text = str(value).strip().lower()
    if not text:
        return set()

    parts = re.split(r"[,;|]+", text)
    return {
        re.sub(r"\s+", " ", part).strip()
        for part in parts
        if part and part.strip()
    }


def jaccard_score(true_value: Any, pred_value: Any) -> float:
    """Calcula similitud Jaccard entre dos campos multivalor."""

    true_set = split_multivalue(true_value)
    pred_set = split_multivalue(pred_value)

    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0

    return len(true_set & pred_set) / len(true_set | pred_set)


def leave_one_out_clean(
    df: pd.DataFrame,
    k: int = 7,
    exclude_same_user: bool = True,
    limit_folds: int | None = None,
    image_similarity_path: Path | None = DEFAULT_IMAGE_SIMILARITY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta validación Leave-One-Out limpia.

    La similitud se calcula por bloques usando únicamente PROBLEM_FEATURES.
    Los campos de SOLUTION_FIELDS se usan solo como solución a predecir/evaluar.
    """

    if set(PROBLEM_FEATURES) & set(SOLUTION_FIELDS):
        raise RuntimeError("Data leakage: PROBLEM_FEATURES contiene campos de solución.")

    image_similarity = load_image_similarity_matrix(image_similarity_path)

    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []

    image_ids = sorted(df[GROUP_COL].dropna().unique())
    if limit_folds is not None:
        image_ids = image_ids[:limit_folds]

    for image_id in image_ids:
        train_df = df[df[GROUP_COL] != image_id].copy()
        val_df = df[df[GROUP_COL] == image_id].copy()

        if train_df.empty or val_df.empty:
            continue

        sim_matrix, block_similarities = compute_weighted_similarity(
            train_df=train_df,
            query_df=val_df,
            image_similarity=image_similarity,
            return_block_similarities=True,
        )

        fold_predictions = {field: [] for field in SOLUTION_FIELDS}
        fold_baselines = {field: [] for field in SOLUTION_FIELDS}

        for val_pos, (_, query_row) in enumerate(val_df.iterrows()):
            candidates = train_df.copy()
            candidates["similarity"] = sim_matrix[val_pos]
            for block_name, block_similarity in block_similarities.items():
                candidates[f"sim_{block_name}"] = block_similarity[val_pos]

            if exclude_same_user:
                candidates = candidates[candidates["user_id"] != query_row["user_id"]]

            candidates = candidates.sort_values("similarity", ascending=False)
            neighbors = candidates.head(k).copy()

            row: dict[str, Any] = {
                "case_id": query_row["case_id"],
                "user_id": query_row["user_id"],
                "image_id": query_row["image_id"],
                "n_candidates": len(candidates),
                "n_neighbors": len(neighbors),
            }

            for field in SOLUTION_FIELDS:
                true_value = query_row[field]
                pred_value = weighted_vote(neighbors, field)
                baseline_value = majority_vote(candidates, field)

                row[f"true_{field}"] = true_value
                row[f"pred_{field}"] = pred_value
                row[f"baseline_{field}"] = baseline_value

                row[f"correct_{field}"] = true_value == pred_value
                row[f"baseline_correct_{field}"] = true_value == baseline_value

                if field in MULTIVALUE_FIELDS:
                    row[f"jaccard_{field}"] = jaccard_score(true_value, pred_value)
                    row[f"baseline_jaccard_{field}"] = jaccard_score(true_value, baseline_value)

                fold_predictions[field].append(pred_value)
                fold_baselines[field].append(baseline_value)

            pred_rows.append(row)

            for rank, (_, neighbor) in enumerate(neighbors.iterrows(), start=1):
                neighbor_row: dict[str, Any] = {
                    "query_case_id": query_row["case_id"],
                    "query_user_id": query_row["user_id"],
                    "query_image_id": query_row["image_id"],
                    "rank": rank,
                    "case_id": neighbor["case_id"],
                    "user_id": neighbor["user_id"],
                    "image_id": neighbor["image_id"],
                    "similarity": float(neighbor["similarity"]),
                }
                for block_name in BLOCK_WEIGHTS:
                    debug_col = f"sim_{block_name}"
                    if debug_col in neighbor:
                        neighbor_row[debug_col] = float(neighbor[debug_col])

                for field in SOLUTION_FIELDS:
                    neighbor_row[field] = neighbor[field]

                neighbor_rows.append(neighbor_row)

        fold: dict[str, Any] = {
            "image_id": image_id,
            "n_val_samples": len(val_df),
        }

        for field in SOLUTION_FIELDS:
            y_true = val_df[field].astype(str)
            y_pred = pd.Series(fold_predictions[field], index=val_df.index).astype(str)
            y_base = pd.Series(fold_baselines[field], index=val_df.index).astype(str)

            fold[f"accuracy_{field}"] = accuracy_score(y_true, y_pred)
            fold[f"baseline_accuracy_{field}"] = accuracy_score(y_true, y_base)
            fold[f"f1_macro_{field}"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

            if field in MULTIVALUE_FIELDS:
                fold[f"jaccard_{field}"] = np.mean(
                    [
                        jaccard_score(t, p)
                        for t, p in zip(y_true, y_pred)
                    ]
                )
                fold[f"baseline_jaccard_{field}"] = np.mean(
                    [
                        jaccard_score(t, p)
                        for t, p in zip(y_true, y_base)
                    ]
                )

        fold_rows.append(fold)

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.DataFrame(pred_rows)
    neighbors_df = pd.DataFrame(neighbor_rows)

    metric_rows: list[dict[str, Any]] = []

    for field in SOLUTION_FIELDS:
        y_true = preds_df[f"true_{field}"].astype(str)
        y_pred = preds_df[f"pred_{field}"].astype(str)
        y_base = preds_df[f"baseline_{field}"].astype(str)

        row = {
            "field": field,
            "accuracy": accuracy_score(y_true, y_pred),
            "baseline_accuracy": accuracy_score(y_true, y_base),
            "delta_accuracy_vs_baseline": accuracy_score(y_true, y_pred)
            - accuracy_score(y_true, y_base),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        if field in MULTIVALUE_FIELDS:
            row["jaccard_mean"] = preds_df[f"jaccard_{field}"].mean()
            row["baseline_jaccard_mean"] = preds_df[f"baseline_jaccard_{field}"].mean()
            row["delta_jaccard_vs_baseline"] = (
                row["jaccard_mean"] - row["baseline_jaccard_mean"]
            )
        else:
            row["jaccard_mean"] = np.nan
            row["baseline_jaccard_mean"] = np.nan
            row["delta_jaccard_vs_baseline"] = np.nan

        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)

    return folds_df, preds_df, neighbors_df, metrics_df


def run_k_sensitivity(
    df: pd.DataFrame,
    k_values: list[int],
    exclude_same_user: bool = True,
    limit_folds: int | None = None,
    image_similarity_path: Path | None = DEFAULT_IMAGE_SIMILARITY,
) -> pd.DataFrame:
    """Ejecuta análisis de sensibilidad para varios valores de k."""

    rows: list[dict[str, Any]] = []

    for k in k_values:
        _, _, _, metrics_df = leave_one_out_clean(
            df,
            k=k,
            exclude_same_user=exclude_same_user,
            limit_folds=limit_folds,
            image_similarity_path=image_similarity_path,
        )

        row: dict[str, Any] = {
            "k": k,
            "mean_accuracy_across_solution_fields": metrics_df["accuracy"].mean(),
            "mean_baseline_accuracy_across_solution_fields": metrics_df[
                "baseline_accuracy"
            ].mean(),
            "mean_delta_accuracy_vs_baseline": metrics_df[
                "delta_accuracy_vs_baseline"
            ].mean(),
            "mean_f1_macro_across_solution_fields": metrics_df["f1_macro"].mean(),
        }

        for field in SOLUTION_FIELDS:
            field_row = metrics_df[metrics_df["field"] == field].iloc[0]
            row[f"accuracy_{field}"] = field_row["accuracy"]
            row[f"baseline_accuracy_{field}"] = field_row["baseline_accuracy"]
            row[f"f1_macro_{field}"] = field_row["f1_macro"]

            if field in MULTIVALUE_FIELDS:
                row[f"jaccard_{field}"] = field_row["jaccard_mean"]
                row[f"baseline_jaccard_{field}"] = field_row["baseline_jaccard_mean"]

        rows.append(row)

    return pd.DataFrame(rows)


def parse_k_grid(raw: str) -> list[int]:
    """Convierte una cadena tipo '1,3,5,7' en lista de enteros."""

    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluación CBR Leave-One-Out sin usar columnas de solución "
            "en la similitud."
        )
    )

    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--image-similarity",
        type=Path,
        default=DEFAULT_IMAGE_SIMILARITY,
        help=(
            "CSV opcional con similitud visual precalculada por image_id "
            "(por ejemplo SSIM). Si no existe, se usan metadatos de imagen."
        ),
    )
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--include-same-user", action="store_true")
    parser.add_argument("--limit-folds", type=int, default=None)
    parser.add_argument(
        "--k-grid",
        type=str,
        default="1,3,5,7,9,11,15",
        help="Valores de k para análisis de sensibilidad, separados por comas.",
    )
    parser.add_argument(
        "--no-k-sensitivity",
        action="store_true",
        help="Desactiva el análisis de sensibilidad por k.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = build_dataset(args.dataset)

    exclude_same_user = not args.include_same_user

    folds_df, preds_df, neighbors_df, metrics_df = leave_one_out_clean(
        df,
        k=args.k,
        exclude_same_user=exclude_same_user,
        limit_folds=args.limit_folds,
        image_similarity_path=args.image_similarity,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    folds_path = args.output_dir / "cbr_loo_clean_folds.csv"
    preds_path = args.output_dir / "cbr_loo_clean_predictions.csv"
    neighbors_path = args.output_dir / "cbr_loo_clean_neighbors.csv"
    metrics_path = args.output_dir / "cbr_loo_clean_metrics_by_field.csv"
    summary_path = args.output_dir / "cbr_loo_clean_summary.csv"
    k_sensitivity_path = args.output_dir / "cbr_loo_clean_k_sensitivity.csv"

    folds_df.to_csv(folds_path, index=False)
    preds_df.to_csv(preds_path, index=False)
    neighbors_df.to_csv(neighbors_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "dataset": str(args.dataset),
                "k": args.k,
                "exclude_same_user": exclude_same_user,
                "n_folds": len(folds_df),
                "n_predictions": len(preds_df),
                "mean_accuracy_across_solution_fields": metrics_df["accuracy"].mean(),
                "mean_baseline_accuracy_across_solution_fields": metrics_df[
                    "baseline_accuracy"
                ].mean(),
                "mean_delta_accuracy_vs_baseline": metrics_df[
                    "delta_accuracy_vs_baseline"
                ].mean(),
                "mean_f1_macro_across_solution_fields": metrics_df["f1_macro"].mean(),
                "block_weights": str(BLOCK_WEIGHTS),
                "image_similarity_path": str(args.image_similarity),
                "problem_features": ", ".join(PROBLEM_FEATURES),
                "solution_fields": ", ".join(SOLUTION_FIELDS),
                "multivalue_fields": ", ".join(MULTIVALUE_FIELDS),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    if not args.no_k_sensitivity:
        k_values = parse_k_grid(args.k_grid)
        k_sensitivity_df = run_k_sensitivity(
            df,
            k_values=k_values,
            exclude_same_user=exclude_same_user,
            limit_folds=args.limit_folds,
            image_similarity_path=args.image_similarity,
        )
        k_sensitivity_df.to_csv(k_sensitivity_path, index=False)
    else:
        k_sensitivity_df = pd.DataFrame()

    print("=== Leave-One-Out limpio ===")
    print(f"Dataset: {args.dataset}")
    print(f"Pesos por bloque: {BLOCK_WEIGHTS}")
    print(f"Similitud visual: {args.image_similarity}")
    print(f"K: {args.k}")
    print(f"Folds: {len(folds_df)}")
    print(f"Exclude same user: {exclude_same_user}")
    print("\n=== Métricas por campo ===")
    print(metrics_df.to_string(index=False))

    print("\n=== Resumen ===")
    print(summary.to_string(index=False))

    if not k_sensitivity_df.empty:
        print("\n=== Sensibilidad por k ===")
        display_columns = [
            "k",
            "mean_accuracy_across_solution_fields",
            "mean_baseline_accuracy_across_solution_fields",
            "mean_delta_accuracy_vs_baseline",
            "mean_f1_macro_across_solution_fields",
        ]
        print(k_sensitivity_df[display_columns].to_string(index=False))

    print(f"\nArchivos guardados en: {args.output_dir}")
    print(f"- {folds_path.name}")
    print(f"- {preds_path.name}")
    print(f"- {neighbors_path.name}")
    print(f"- {metrics_path.name}")
    print(f"- {summary_path.name}")
    if not args.no_k_sensitivity:
        print(f"- {k_sensitivity_path.name}")


if __name__ == "__main__":
    main()
