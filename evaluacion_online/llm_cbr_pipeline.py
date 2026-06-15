from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluacion_offline.cbr_loo_eval import BLOCK_WEIGHTS as DEFAULT_BLOCK_WEIGHTS
from evaluacion_offline.cbr_loo_eval import (
    compute_weighted_similarity as compute_weighted_similarity_matrix,
)
from evaluacion_offline.cbr_loo_eval import (
    load_image_similarity_matrix as load_weighted_image_similarity_matrix,
)


DEFAULT_CASE_BASE = PROJECT_ROOT / "base_de_casos/cbr_case_base_outputs/case_base_double_full.csv"
DEFAULT_DESCRIPTIONS = Path(
    PROJECT_ROOT
    / "generacion_descripcion_XAI/resultados_descripciones_xai/descripciones_por_imagen_xai.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluacion_online/resultados"
DEFAULT_IMAGE_SIMILARITY = PROJECT_ROOT / "base_de_casos/cbr_similarity_outputs/image_ssim_by_id.csv"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "imagenes"

CONTENT_FIELDS = [
    "image_id",
    "image_label",
    "original_image_path",
    "domain",
    "model_predicted_class",
    "model_confidence",
    "initial_description",
    "vqa_support_description",
]

USER_PROFILE_FIELDS = [
    "age_range",
    "education_level",
    "occupation_raw",
    "ai_knowledge_level",
    "domain_knowledge_level",
]

PROBLEM_TEXT_COLUMNS = ["initial_description", "vqa_support_description"]
PROBLEM_TEXT = "problem_text"
SIMILARITY_IMAGE_FIELDS = ["image_id", "domain", "model_predicted_class"]

PROBLEM_FEATURES = CONTENT_FIELDS + USER_PROFILE_FIELDS

SOLUTION_PREFERENCE_FIELDS = [
    "preferred_response_length",
    "preferred_technical_level",
    "preferred_format",
    "preferred_explanation_types_raw",
    "main_goals_raw",
    "perceived_error_impact",
]

SOLUTION_FIELDS = [
    "selected_option",
    "recommended_method",
    *SOLUTION_PREFERENCE_FIELDS,
    "explanation_components",
    "final_output",
    "k_instances",
    "k_counterexamples",
]

OPTIONAL_CONTEXT_FIELDS = [
    "satisfaction",
    "confidence",
    "understanding",
    "mean_helpfulness",
    "free_comment",
]

FEEDBACK_FIELDS = [
    "satisfaction",
    "confidence",
    "understanding",
    "mean_helpfulness",
    "free_comment",
]

BLOCK_WEIGHTS = DEFAULT_BLOCK_WEIGHTS
WEIGHTS = BLOCK_WEIGHTS

OPTION_TO_METHOD = {
    "Opcion A": "anchor",
    "Opcion B": "gradcam",
    "Opcion C": "integrated_gradients",
    "Opcion D": "lime",
    "Opcion E": "saliency",
    "Opcion F (Ninguna)": "none",
    "Opción A": "anchor",
    "Opción B": "gradcam",
    "Opción C": "integrated_gradients",
    "Opción D": "lime",
    "Opción E": "saliency",
    "Opción F (Ninguna)": "none",
}

METHOD_LABEL = {
    "anchor": "Anchor",
    "gradcam": "Grad-CAM",
    "integrated_gradients": "Integrated Gradients",
    "lime": "LIME",
    "saliency": "Saliency",
    "none": "Sin metodo XAI visual",
}

VISUAL_XAI_METHODS = {"anchor", "gradcam", "integrated_gradients", "lime", "saliency"}

METHOD_EXPLANATION_MODE = {
    "anchor": "Basado en reglas/superpíxeles: regiones suficientes que mantienen la predicción.",
    "gradcam": "Basado en features: mapa de calor con regiones relevantes para la clase predicha.",
    "integrated_gradients": (
        "Basado en features: atribución píxel a píxel respecto a una referencia."
    ),
    "lime": "Basado en features/regiones: superpíxeles que apoyan la predicción local.",
    "saliency": "Basado en features: sensibilidad del modelo ante cambios en los píxeles.",
    "none": "Sin explicación XAI visual: se prioriza una explicación textual con limitaciones.",
}

AGE_MAP = {
    "18-24": 0,
    "18–24": 0,
    "25-34": 1,
    "25–34": 1,
    "35-44": 2,
    "35–44": 2,
    "45-54": 3,
    "45–54": 3,
    "55-64": 4,
    "55–64": 4,
    "65 o mas": 5,
    "65 o más": 5,
}

EDUCATION_MAP = {
    "bachillerato/fp": 0,
    "grado": 1,
    "master": 2,
    "máster": 2,
    "doctorado": 3,
    "prefiero no contestar": np.nan,
}

RESPONSE_LENGTH_MAP = {
    "corta: solo lo esencial": 0,
    "media: explicacion breve con algo de detalle": 1,
    "media: explicación breve con algo de detalle": 1,
    "larga: explicacion completa y detallada": 2,
    "larga: explicación completa y detallada": 2,
}

TECH_LEVEL_MAP = {
    "simple: lenguaje claro y sin tecnicismos": 0,
    "intermedio: algunos terminos tecnicos, pero faciles de seguir": 1,
    "intermedio: algunos términos técnicos, pero fáciles de seguir": 1,
    "tecnico: explicacion mas especializada y precisa": 2,
    "técnico: explicación más especializada y precisa": 2,
}

ERROR_IMPACT_MAP = {
    "bajo: el error apenas tendria consecuencias": 0,
    "bajo: el error apenas tendría consecuencias": 0,
    "medio: el error podria causar cierta confusion o problema": 1,
    "medio: el error podría causar cierta confusión o problema": 1,
    "alto: el error podria tener consecuencias importantes": 2,
    "alto: el error podría tener consecuencias importantes": 2,
}

OCCUPATION_PATTERNS = {
    "ocup_estudiante": ["estudiante"],
    "ocup_investigacion": ["investigador", "investigadora"],
    "ocup_docencia": ["docente", "profesor", "profesora"],
    "ocup_dev": ["desarrollador", "desarrolladora"],
    "ocup_prof_tech": ["sector tecnologico", "sector tecnológico", "/ ia / datos", "ia / datos"],
    "ocup_prof_otro": ["profesional de otro sector"],
    "ocup_prefiere_no": ["prefiero no contestar"],
}

EXPLANATION_TYPE_PATTERNS = {
    "pref_ejemplos": ["ejemplos similares"],
    "pref_contraejemplos": ["comparaciones o contraejemplos"],
    "pref_atributos": ["atributos o caracteristicas", "atributos o características"],
    "pref_reglas": ["reglas o razonamientos paso a paso"],
    "pref_visual": ["zonas destacadas", "visuales"],
    "pref_sin_preferencia": ["no tengo preferencia", "me da igual"],
}

XAI_GOAL_PATTERNS = {
    "goal_transparencia": ["transparencia"],
    "goal_eficiencia": ["eficiencia"],
    "goal_efectividad": ["efectividad"],
    "goal_confianza": ["confianza"],
    "goal_persuasion": ["persuasion", "persuasión"],
    "goal_satisfaccion": ["satisfaccion", "satisfacción"],
    "goal_educacion": ["educacion", "educación"],
    "goal_debugging": ["deteccion de errores", "detección de errores", "debugging"],
    "goal_escrutinio": ["escrutinio"],
}


@dataclass
class PipelineResult:
    timestamp: str
    query: dict[str, Any]
    recommended_option: str
    recommended_method: str
    neighbor_case_id: str
    neighbor_similarity: float
    xai_description: str
    prompt: str
    explanation: str
    problem: dict[str, Any] | None = None
    solution: dict[str, Any] | None = None
    counterexamples: list[dict[str, Any]] | None = None
    convinced: bool | None = None
    rating: int | None = None
    acceptance: int | None = None
    satisfaction: int | None = None
    confidence: int | None = None
    understanding: int | None = None
    requested_change: str | None = None
    alternatives_exhausted: bool = False
    iteration: int = 1


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def original_image_path_for_image_id(image_id: Any) -> Path | None:
    if is_missing(image_id):
        return None
    try:
        return DEFAULT_IMAGES_DIR / "original" / f"image{int(image_id):02d}.jpg"
    except (TypeError, ValueError):
        return None


def first_present(*values: Any, default: Any = "no indicado") -> Any:
    for value in values:
        if not is_missing(value) and str(value).strip() != "":
            return value
    return default


def minmax_series(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    filled = series.fillna(series.median())
    smin = filled.min()
    smax = filled.max()
    if smax == smin:
        return pd.Series(np.zeros(len(filled)), index=series.index)
    return (filled - smin) / (smax - smin)


def map_with_fallback(series: pd.Series, mapping: dict[str, Any]) -> pd.Series:
    return series.apply(lambda value: mapping.get(normalize_text(value), np.nan))


def extract_flags_from_text(raw_text: Any, patterns_dict: dict[str, list[str]]) -> dict[str, int]:
    text = normalize_text(raw_text)
    return {
        label: int(any(pattern in text for pattern in patterns))
        for label, patterns in patterns_dict.items()
    }


def align_columns(block: pd.DataFrame, template_columns: pd.Index) -> pd.DataFrame:
    return block.reindex(columns=template_columns, fill_value=0)


def safe_cosine_query_vs_base(query_x: np.ndarray, base_x: np.ndarray) -> np.ndarray:
    if base_x.shape[1] == 0:
        return np.zeros(base_x.shape[0])
    return cosine_similarity(query_x, base_x).ravel()


def normalize_image_id(value: Any) -> Any:
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
    if path is None or not path.exists():
        return None
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = [normalize_image_id(value) for value in matrix.index]
    matrix.columns = [normalize_image_id(value) for value in matrix.columns]
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0, upper=1)


def image_similarity_query_vs_base(
    query_image_id: Any,
    base_image_ids: pd.Series,
    image_similarity: pd.DataFrame,
) -> np.ndarray:
    query_id = normalize_image_id(query_image_id)
    sims = []
    for base_image_id in base_image_ids:
        candidate_id = normalize_image_id(base_image_id)
        if query_id in image_similarity.index and candidate_id in image_similarity.columns:
            sims.append(float(image_similarity.loc[query_id, candidate_id]))
        elif query_id == candidate_id:
            sims.append(1.0)
        else:
            sims.append(0.0)
    return np.asarray(sims, dtype=float)


def option_type_from_case(case_data: dict[str, Any] | pd.Series, selected_option: str) -> Any:
    option_letter = selected_option.replace("Opción ", "").replace("Opcion ", "").split()[0]
    column = f"option_{option_letter}_type"
    if isinstance(case_data, pd.Series):
        return case_data.get(column, "no indicado")
    return case_data.get(column, "no indicado")


def contains_any(value: Any, patterns: list[str]) -> bool:
    text = normalize_text(value)
    return any(pattern in text for pattern in patterns)


def normalize_response_length(value: Any) -> str:
    text = normalize_text(value)
    if "corta" in text:
        return "corta"
    if "larga" in text:
        return "larga"
    return "media"


def normalize_technical_level(value: Any) -> str:
    text = normalize_text(value)
    if "simple" in text:
        return "simple"
    if "tecnico" in text or "técnico" in text:
        return "tecnico"
    return "intermedio"


def infer_output_structure(data: dict[str, Any] | pd.Series) -> str:
    explanation_types = data.get("preferred_explanation_types_raw", "")
    goals = data.get("main_goals_raw", "")
    length = normalize_response_length(data.get("preferred_response_length", ""))
    if contains_any(explanation_types, ["contraejemplos"]) or contains_any(
        goals, ["escrutinio"]
    ):
        return "tabla"
    if contains_any(explanation_types, ["reglas", "paso a paso"]) or contains_any(
        goals, ["debugging", "deteccion de errores", "detección de errores"]
    ):
        return "pasos"
    if length == "larga":
        return "narrativa"
    return "lista"


def should_include_counterexamples(data: dict[str, Any] | pd.Series) -> bool:
    explanation_types = data.get("preferred_explanation_types_raw", "")
    goals = data.get("main_goals_raw", "")
    impact = data.get("perceived_error_impact", "")
    return (
        contains_any(explanation_types, ["contraejemplos"])
        or contains_any(goals, ["escrutinio", "debugging", "deteccion de errores", "detección de errores"])
        or contains_any(impact, ["alto"])
    )


def build_explanation_components(
    data: dict[str, Any] | pd.Series,
    method: str,
) -> dict[str, dict[str, Any]]:
    confidence = data.get("model_confidence", None)
    confidence_note = "confianza no disponible"
    if not is_missing(confidence):
        confidence_note = f"confianza del modelo: {confidence}"
    return {
        "based_on_features": {
            "include": method in VISUAL_XAI_METHODS,
            "description": (
                "Heatmaps, regiones o superpixeles que justifican la prediccion."
            ),
            "method": METHOD_LABEL.get(method, method),
        },
        "based_on_instances": {
            "include": True,
            "description": (
                "Ejemplos de casos o imagenes similares recuperados por el CBR."
            ),
        },
        "counterexamples": {
            "include": should_include_counterexamples(data),
            "description": (
                "Imagenes o casos claramente distintos para contrastar la decision y evitar sesgos."
            ),
        },
        "limitations": {
            "include": True,
            "description": (
                "Incertidumbres detectadas: calidad de evidencia, posible confusion entre clases, "
                f"falta de evidencia o {confidence_note}."
            ),
        },
    }


def build_problem_representation(data: dict[str, Any] | pd.Series) -> dict[str, dict[str, Any]]:
    getter = data.get
    return {
        "content_layer": {
            "image_id": getter("image_id", None),
            "image_label": getter("image_label", None),
            "original_image": getter("original_image_path", None),
            "initial_description": getter("initial_description", None),
            "domain": getter("domain", None),
            "model_predicted_class": getter("model_predicted_class", None),
            "model_confidence": getter("model_confidence", None),
            "vqa_support_description": getter("vqa_support_description", None),
        },
        "user_preference_layer": {
            "role": getter("occupation_raw", None),
            "education_level": getter("education_level", None),
            "domain_knowledge": getter("domain_knowledge_level", None),
            "ai_knowledge": getter("ai_knowledge_level", None),
        },
    }


def build_solution_representation(
    data: dict[str, Any] | pd.Series,
    selected_option: str,
    method: str,
) -> dict[str, Any]:
    getter = data.get
    k_counterexamples = getter("k_counterexamples", None)
    if is_missing(k_counterexamples):
        k_counterexamples = 2 if should_include_counterexamples(data) else 0
    k_instances = getter("k_instances", None)
    if is_missing(k_instances):
        k_instances = 3
    return {
        "selected_option": selected_option,
        "recommended_method": method,
        "option_type": option_type_from_case(data, selected_option),
        "explanation_components": build_explanation_components(data, method),
        "final_output": {
            "length": normalize_response_length(getter("preferred_response_length", None)),
            "technical_level": normalize_technical_level(getter("preferred_technical_level", None)),
            "structure": infer_output_structure(data),
            "preferred_format": getter("preferred_format", None),
        },
        "k_instances": k_instances,
        "k_counterexamples": k_counterexamples,
    }


def build_feedback_representation(data: dict[str, Any] | pd.Series) -> dict[str, Any]:
    getter = data.get
    return {
        "satisfaction": getter("satisfaction", None),
        "confidence": getter("confidence", None),
        "understanding": getter("understanding", None),
        "mean_helpfulness": getter("mean_helpfulness", None),
        "free_comment": getter("free_comment", None),
    }


def build_case_representation(
    data: dict[str, Any] | pd.Series,
    selected_option: str | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    option = selected_option or str(data.get("selected_option", "no indicado"))
    recommended_method = method or OPTION_TO_METHOD.get(option, "none")
    return {
        "problem": build_problem_representation(data),
        "solution": build_solution_representation(data, option, recommended_method),
        "feedback": build_feedback_representation(data),
    }


def build_similarity_problem_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    """Prepara solo los campos de problema usados por la similitud CBR."""

    df = df_input.copy()
    for col in USER_PROFILE_FIELDS + SIMILARITY_IMAGE_FIELDS + PROBLEM_TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df[PROBLEM_TEXT] = (
        df["initial_description"].fillna("").astype(str)
        + " "
        + df["vqa_support_description"].fillna("").astype(str)
    ).str.strip()
    return df


def prepare_case_dataframe(df_input: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Helper legado para bloques no textuales; la recuperación usa compute_weighted_similarity."""

    df = df_input.copy()
    expected_cols = PROBLEM_FEATURES
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    profile = pd.DataFrame(index=df.index)
    profile["age_code"] = map_with_fallback(df["age_range"], AGE_MAP)
    profile["education_code"] = map_with_fallback(df["education_level"], EDUCATION_MAP)
    profile["ai_knowledge"] = pd.to_numeric(df["ai_knowledge_level"], errors="coerce")
    profile["domain_knowledge"] = pd.to_numeric(df["domain_knowledge_level"], errors="coerce")
    for col in profile.columns:
        profile[col] = minmax_series(profile[col])
    occupation_flags = df["occupation_raw"].fillna("").apply(
        lambda text: pd.Series(extract_flags_from_text(text, OCCUPATION_PATTERNS))
    )
    profile = pd.concat([profile, occupation_flags], axis=1).fillna(0)

    image_block = pd.get_dummies(df["image_id"].astype("Int64").astype(str), prefix="img")
    if "domain" in df.columns and df["domain"].notna().sum() > 0:
        image_block = pd.concat(
            [image_block, pd.get_dummies(df["domain"].fillna("desconocido"), prefix="domain")],
            axis=1,
        )
    if "model_predicted_class" in df.columns and df["model_predicted_class"].notna().sum() > 0:
        image_block = pd.concat(
            [
                image_block,
                pd.get_dummies(df["model_predicted_class"].fillna("desconocido"), prefix="pred"),
            ],
            axis=1,
        )
    return {
        "profile": profile.fillna(0).astype(float),
        "image": image_block.fillna(0).astype(float),
    }


def image_id_from_description_case_id(description_case_id: str | None) -> int:
    if not description_case_id:
        return 1

    text = str(description_case_id).strip().lower()
    if text.startswith("image"):
        suffix = text.removeprefix("image")
        if suffix.isdigit():
            return int(suffix)

    return 1


def load_query(
    path: Path | None,
    inline_json: str | None,
    description_case_id: str | None = None,
) -> dict[str, Any]:
    if inline_json:
        return json.loads(inline_json)
    if path:
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "image_id": image_id_from_description_case_id(description_case_id),
        "age_range": "25-34",
        "education_level": "Master",
        "occupation_raw": "Investigador/a (academico)",
        "ai_knowledge_level": 4,
        "domain_knowledge_level": 3,
    }


class CBRRetriever:
    def __init__(
        self,
        case_base_path: Path,
        weights: dict[str, float] | None = None,
        image_similarity_path: Path | None = DEFAULT_IMAGE_SIMILARITY,
    ) -> None:
        self.case_base_path = case_base_path
        self.df = build_similarity_problem_dataframe(pd.read_csv(case_base_path))
        self.weights = weights or WEIGHTS
        self.blocks = {
            "text": [PROBLEM_TEXT],
            "profile": USER_PROFILE_FIELDS,
            "image": SIMILARITY_IMAGE_FIELDS,
        }
        self.image_similarity = load_weighted_image_similarity_matrix(image_similarity_path)

    def enrich_query_with_image_metadata(self, query: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(query)
        if "image_id" not in enriched or "image_id" not in self.df.columns:
            return enriched
        image_rows = self.df[self.df["image_id"] == enriched["image_id"]]
        if image_rows.empty:
            return enriched
        image_row = image_rows.iloc[0]
        for field in CONTENT_FIELDS:
            if field not in enriched or is_missing(enriched.get(field)):
                enriched[field] = image_row.get(field, enriched.get(field))
        project_original_path = original_image_path_for_image_id(enriched.get("image_id"))
        if project_original_path is not None:
            enriched["original_image_path"] = str(project_original_path)
        for option in ["A", "B", "C", "D", "E", "F"]:
            field = f"option_{option}_type"
            if field not in enriched or is_missing(enriched.get(field)):
                enriched[field] = image_row.get(field, enriched.get(field))
        return enriched

    def get_neighbors(
        self,
        query: dict[str, Any],
        k: int = 5,
        same_image_only: bool = True,
        min_similarity: float | None = None,
    ) -> pd.DataFrame:
        query = self.enrich_query_with_image_metadata(query)
        query_df = build_similarity_problem_dataframe(pd.DataFrame([query]))
        total, block_similarities = compute_weighted_similarity_matrix(
            train_df=self.df,
            query_df=query_df,
            block_weights=self.weights,
            image_similarity=self.image_similarity,
            return_block_similarities=True,
        )

        candidates = self.df.copy()
        candidates["similarity"] = total.ravel()
        for block_name, block_similarity in block_similarities.items():
            candidates[f"sim_{block_name}"] = block_similarity.ravel()
        if same_image_only and "image_id" in query:
            candidates = candidates[candidates["image_id"] == query["image_id"]]
        if min_similarity is not None:
            candidates = candidates[candidates["similarity"] >= min_similarity]
        return candidates.sort_values("similarity", ascending=False).head(k).reset_index(drop=True)

    @staticmethod
    def recommend_explanation(neighbors: pd.DataFrame) -> pd.DataFrame:
        tmp = neighbors.copy()
        for col in ["mean_helpfulness", "satisfaction", "confidence", "understanding", "similarity"]:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)
        tmp["utility_score"] = (
            0.40 * tmp["mean_helpfulness"]
            + 0.20 * tmp["satisfaction"]
            + 0.20 * tmp["confidence"]
            + 0.20 * tmp["understanding"]
        )
        tmp["weighted_vote"] = tmp["utility_score"] * tmp["similarity"]
        return (
            tmp.groupby("selected_option", dropna=False)
            .agg(
                n_neighbors=("case_id", "count"),
                mean_similarity=("similarity", "mean"),
                mean_utility=("utility_score", "mean"),
                total_weighted_vote=("weighted_vote", "sum"),
            )
            .sort_values(["total_weighted_vote", "mean_similarity"], ascending=False)
            .reset_index()
        )

    @staticmethod
    def best_neighbor_for_option(neighbors: pd.DataFrame, selected_option: str) -> pd.Series:
        tmp = neighbors[neighbors["selected_option"] == selected_option].copy()
        if tmp.empty:
            tmp = neighbors.copy()
        for col in ["mean_helpfulness", "satisfaction", "confidence", "understanding", "similarity"]:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0)
        tmp["utility_score"] = (
            0.40 * tmp["mean_helpfulness"]
            + 0.20 * tmp["satisfaction"]
            + 0.20 * tmp["confidence"]
            + 0.20 * tmp["understanding"]
        )
        tmp["weighted_vote"] = tmp["utility_score"] * tmp["similarity"]
        return tmp.sort_values(["weighted_vote", "similarity"], ascending=False).iloc[0]


def get_xai_description(
    descriptions_path: Path,
    method: str,
    description_case_id: str | None = None,
    fallback_text: str | None = None,
) -> str:
    if fallback_text:
        return fallback_text
    if not descriptions_path.exists():
        return ""
    descriptions = pd.read_csv(descriptions_path)
    if description_case_id:
        descriptions = descriptions[descriptions["case_id"].astype(str) == str(description_case_id)]
    match = descriptions[descriptions["method"] == method]
    if match.empty:
        return ""
    return str(match.iloc[0]["descripcion"])


def build_prompt(
    query: dict[str, Any],
    recommended_option: str,
    recommended_method: str,
    neighbor: pd.Series,
    xai_description: str,
    counterexamples: list[dict[str, Any]] | None = None,
    requested_change: str | None = None,
) -> str:
    change_block = ""
    if requested_change:
        change_block = (
            "\nEl usuario no quedo convencido con la explicacion anterior. "
            f"Ahora necesita: {requested_change}\n"
        )

    problem = build_problem_representation(query)
    solution = build_solution_representation(query, recommended_option, recommended_method)
    neighbor_case = build_case_representation(neighbor)
    content = problem["content_layer"]
    user_preferences = problem["user_preference_layer"]
    neighbor_solution = neighbor_case["solution"]
    neighbor_feedback = neighbor_case["feedback"]
    components = solution["explanation_components"]
    final_output = solution["final_output"]
    counterexample_block = "No se han seleccionado contraejemplos reales."
    if counterexamples:
        counterexample_lines = []
        for idx, item in enumerate(counterexamples, start=1):
            counterexample_lines.append(
                "- Contraejemplo "
                f"{idx}: case_id={first_present(item.get('case_id'))}, "
                f"image_id={first_present(item.get('image_id'))}, "
                f"clase={first_present(item.get('model_predicted_class'))}, "
                f"similitud={first_present(item.get('similarity'))}, "
                f"descripcion={first_present(item.get('initial_description'))}"
            )
        counterexample_block = "\n".join(counterexample_lines)

    return f"""Eres un modulo LLM para explicar una clasificacion de imagen con XAI.
Genera una explicacion personalizada en espanol.

Problema CBR - capa de contenido:
- Imagen original: {first_present(content.get("original_image"))}
- Descripcion inicial: {first_present(content.get("initial_description"))}
- Dominio de la imagen: {first_present(content.get("domain"))}
- Clase predicha / respuesta principal del modelo: {first_present(content.get("model_predicted_class"))}
- Confianza del modelo: {first_present(content.get("model_confidence"))}
- Descripcion VQA de apoyo: {first_present(content.get("vqa_support_description"))}

Problema CBR - capa de preferencias del usuario:
- Rol / ocupacion: {first_present(user_preferences.get("role"))}
- Nivel educativo: {first_present(user_preferences.get("education_level"))}
- Conocimiento del dominio: {first_present(user_preferences.get("domain_knowledge"))}/5
- Conocimiento de IA: {first_present(user_preferences.get("ai_knowledge"))}/5

Solucion recuperada por CBR:
- Opcion recomendada: {recommended_option}
- Metodo XAI recomendado: {METHOD_LABEL.get(recommended_method, recommended_method)}
- Tipo de opcion en la base: {first_present(solution.get("option_type"))}
- Basado en features: {components["based_on_features"]["include"]} - {components["based_on_features"]["description"]}
- Basado en instancias: {components["based_on_instances"]["include"]} - {components["based_on_instances"]["description"]}
- Contraejemplos: {components["counterexamples"]["include"]} - {components["counterexamples"]["description"]}
- Limitaciones / dudas: {components["limitations"]["include"]} - {components["limitations"]["description"]}
- K_instancias: {solution.get("k_instances", "no indicado")}
- K_contraejemplos: {solution.get("k_counterexamples", "no indicado")}
- Longitud final recuperada: {final_output.get("length", "no indicado")}
- Nivel tecnico final recuperado: {final_output.get("technical_level", "no indicado")}
- Estructura final recuperada: {final_output.get("structure", "no indicado")}
- Formato final recuperado: {first_present(final_output.get("preferred_format"))}
- Caso vecino usado como referencia: {neighbor.get("case_id", "no indicado")}
- Similitud: {float(neighbor.get("similarity", 0.0)):.3f}
- Solucion elegida por el vecino: {neighbor.get("selected_option", "no indicado")}
- Metodo asociado al vecino: {METHOD_LABEL.get(neighbor_solution.get("recommended_method"), neighbor_solution.get("recommended_method"))}
- Utilidad media del vecino: {first_present(neighbor_feedback.get("mean_helpfulness"))}
- Satisfaccion/confianza/comprension del vecino: {first_present(neighbor_feedback.get("satisfaction"))}/{first_present(neighbor_feedback.get("confidence"))}/{first_present(neighbor_feedback.get("understanding"))}
- Comentario libre del vecino: {first_present(neighbor_feedback.get("free_comment"))}

Descripcion de la imagen y explicacion XAI disponible:
{xai_description or "No hay descripcion XAI precalculada; explica el metodo recomendado de forma prudente."}

Contraejemplos reales seleccionados por el CBR:
{counterexample_block}
{change_block}
Instrucciones:
- Ajusta el nivel tecnico al perfil.
- Genera la respuesta siguiendo la solucion CBR: features, instancias, contraejemplos y limitaciones solo cuando esten marcados como True.
- Si hablas de contraejemplos, usa solo los contraejemplos reales listados arriba. No inventes otros.
- Respeta longitud, nivel tecnico y estructura final.
- No inventes detalles visuales que no aparezcan en la descripcion disponible.
- Si falta informacion, dilo con cautela.
- Termina con una frase breve preguntando si la explicacion le convence.
"""


def select_recommended_solution(ranking: pd.DataFrame) -> tuple[str, str, str | None]:
    """Return an option/method pair, preferring visual XAI methods over 'Ninguna'."""
    if ranking.empty:
        raise RuntimeError("El ranking CBR esta vacio.")

    top_option = str(ranking.iloc[0]["selected_option"])
    for option in ranking["selected_option"]:
        option_text = str(option)
        method = OPTION_TO_METHOD.get(option_text)
        if method in VISUAL_XAI_METHODS:
            note = None
            if option_text != top_option:
                note = (
                    f"La opcion mejor puntuada por CBR fue '{top_option}', pero no "
                    "corresponde a un metodo XAI visual. Se usa la siguiente opcion visual."
                )
            return option_text, method, note

    fallback_method = OPTION_TO_METHOD.get(top_option, "none")
    return top_option, fallback_method, None


def select_next_distinct_solution(
    ranking: pd.DataFrame,
    rejected_methods: set[str] | list[str] | tuple[str, ...] | None = None,
) -> tuple[str | None, str | None]:
    """Return the next ranked option whose XAI method has not already been rejected.

    This helper is intentionally solution-side only: it consumes a ranking already
    computed from problem attributes and controls the feedback loop without
    recalculating similarity with user feedback text.
    """
    if ranking.empty:
        return None, None

    rejected = set(rejected_methods or [])
    for option in ranking["selected_option"]:
        option_text = str(option)
        method = OPTION_TO_METHOD.get(option_text, "none")
        if method in rejected:
            continue
        if method in VISUAL_XAI_METHODS:
            return option_text, method

    for option in ranking["selected_option"]:
        option_text = str(option)
        method = OPTION_TO_METHOD.get(option_text, "none")
        if method not in rejected:
            return option_text, method

    return None, None


def generate_with_ollama(prompt: str, model: str) -> str:
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("No esta instalado ollama. Ejecuta con --dry-run o instala la dependencia.") from exc
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096},
    )
    return response["message"]["content"].strip()


def save_result(result: PipelineResult, output_dir: Path) -> Path:
    def make_json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): make_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [make_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [make_json_safe(item) for item in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
        return value

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "llm_cbr_feedback_log.jsonl"
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(make_json_safe(asdict(result)), ensure_ascii=False) + "\n")
    return output_path


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    query = load_query(args.query_file, args.query_json, args.description_case_id)
    retriever = CBRRetriever(args.case_base, image_similarity_path=args.image_similarity)
    query = retriever.enrich_query_with_image_metadata(query)
    neighbors = retriever.get_neighbors(
        query,
        k=args.k,
        same_image_only=not args.disable_same_image,
        min_similarity=args.min_similarity,
    )
    if neighbors.empty:
        raise RuntimeError("No se encontraron vecinos para la query.")

    ranking = retriever.recommend_explanation(neighbors)
    recommended_option, recommended_method, _ = select_recommended_solution(ranking)
    neighbor = retriever.best_neighbor_for_option(neighbors, recommended_option)
    xai_description = get_xai_description(
        args.descriptions,
        method=recommended_method,
        description_case_id=args.description_case_id,
        fallback_text=args.xai_description,
    )
    prompt = build_prompt(query, recommended_option, recommended_method, neighbor, xai_description)
    explanation = prompt if args.dry_run else generate_with_ollama(prompt, args.model)
    problem = build_problem_representation(query)
    solution = build_solution_representation(query, recommended_option, recommended_method)

    return PipelineResult(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        query=query,
        recommended_option=recommended_option,
        recommended_method=recommended_method,
        neighbor_case_id=str(neighbor.get("case_id", "")),
        neighbor_similarity=float(neighbor.get("similarity", 0.0)),
        xai_description=xai_description,
        prompt=prompt,
        explanation=explanation,
        problem=problem,
        solution=solution,
    )


def interactive_feedback_loop(result: PipelineResult, args: argparse.Namespace) -> PipelineResult:
    current = result
    retriever = CBRRetriever(args.case_base, image_similarity_path=args.image_similarity)
    rejected_methods: list[str] = []
    for iteration in range(1, args.max_iterations + 1):
        print("\n=== Explicacion generada ===\n")
        print(current.explanation)
        answer = input("\nTe ha convencido la explicacion? [s/n]: ").strip().lower()
        current.iteration = iteration
        if answer.startswith("s"):
            while True:
                raw_rating = input("Califica la explicacion (1-5): ").strip()
                try:
                    rating = int(raw_rating)
                except ValueError:
                    print("Introduce un numero entero entre 1 y 5.")
                    continue
                if 1 <= rating <= 5:
                    current.convinced = True
                    current.rating = rating
                    save_result(current, args.output_dir)
                    return current
                print("Introduce un numero entero entre 1 y 5.")

        current.convinced = False
        current.requested_change = None
        save_result(current, args.output_dir)
        if current.recommended_method not in rejected_methods:
            rejected_methods.append(current.recommended_method)

        neighbors = retriever.get_neighbors(
            current.query,
            k=args.k,
            same_image_only=not args.disable_same_image,
            min_similarity=args.min_similarity,
        )
        ranking = retriever.recommend_explanation(neighbors) if not neighbors.empty else pd.DataFrame()
        next_option, next_method = select_next_distinct_solution(ranking, rejected_methods)

        requested_change = None
        if next_option is None or next_method is None:
            requested_change = input(
                "Se agotaron las soluciones XAI alternativas. "
                "Que necesitas saber o en que formato lo quieres?: "
            ).strip()
            current.requested_change = requested_change
            current.alternatives_exhausted = True
            save_result(current, args.output_dir)
            return current

        neighbor = retriever.best_neighbor_for_option(neighbors, next_option)
        query = dict(current.query)
        for field in SOLUTION_PREFERENCE_FIELDS:
            if field not in query or is_missing(query.get(field)):
                query[field] = neighbor.get(field)
        xai_description = get_xai_description(
            args.descriptions,
            method=next_method,
            description_case_id=args.description_case_id,
            fallback_text=args.xai_description,
        )

        prompt = build_prompt(
            query,
            next_option,
            next_method,
            neighbor,
            xai_description,
            requested_change=requested_change,
        )
        explanation = prompt if args.dry_run else generate_with_ollama(prompt, args.model)
        current = PipelineResult(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            query=query,
            recommended_option=next_option,
            recommended_method=next_method,
            neighbor_case_id=str(neighbor.get("case_id", "")),
            neighbor_similarity=float(neighbor.get("similarity", 0.0)),
            xai_description=xai_description,
            prompt=prompt,
            explanation=explanation,
            problem=build_problem_representation(query),
            solution=build_solution_representation(query, next_option, next_method),
            requested_change=requested_change,
            iteration=iteration + 1,
        )
    save_result(current, args.output_dir)
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline CBR + LLM con feedback iterativo del usuario."
    )
    parser.add_argument("--case-base", type=Path, default=DEFAULT_CASE_BASE)
    parser.add_argument("--descriptions", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--image-similarity", type=Path, default=DEFAULT_IMAGE_SIMILARITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query-file", type=Path, default=None)
    parser.add_argument("--query-json", type=str, default=None)
    parser.add_argument("--description-case-id", type=str, default=None)
    parser.add_argument("--xai-description", type=str, default=None)
    parser.add_argument("--model", type=str, default="qwen2.5vl:32b")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--disable-same-image", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args)
    if args.interactive:
        result = interactive_feedback_loop(result, args)
    else:
        save_result(result, args.output_dir)
        print(result.explanation)
        print(f"\nLog guardado en: {args.output_dir / 'llm_cbr_feedback_log.jsonl'}")


if __name__ == "__main__":
    main()
