from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from config.config import (
    UPLOAD_DIR,
    create_directories,
    validate_paths,
)
from models.attention_explain import (
    explain_image as run_attention_rollout,
)
from models.gradcam_explain import (
    explain_image as run_gradcam,
)
from models.inference import load_inference_model
from models.retfound_classifier import RETFoundClassifier
from utils.comparison_manager import (
    build_per_class_recall_dataframe,
    build_summary_dataframe,
    get_best_experiment,
    load_experiment_metrics,
)
from utils.history_manager import (
    clear_history,
    load_history,
    save_analysis,
)
from utils.pdf_report import generate_pdf_report


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="RetinaGuard AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(30, 64, 175, 0.20),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #020617 0%,
                #0f172a 50%,
                #111827 100%
            );
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f8fafc;
    }

    p, label, .stMarkdown {
        color: #dbeafe;
    }

    .hero-card {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 24px;
        padding: 28px 34px;
        margin-bottom: 24px;
        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.96),
                rgba(30, 41, 59, 0.90)
            );
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.30);
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: #bfdbfe;
        line-height: 1.7;
        max-width: 900px;
    }

    .status-card {
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 18px;
        padding: 20px;
        background: rgba(15, 23, 42, 0.88);
        min-height: 145px;
    }

    .status-label {
        color: #93c5fd;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    .status-value {
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 750;
    }

    .status-description {
        color: #cbd5e1;
        margin-top: 8px;
        line-height: 1.5;
    }

    .disclaimer {
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 15px 18px;
        background: rgba(120, 53, 15, 0.22);
        color: #fde68a;
        margin-top: 22px;
    }

    div.stButton > button {
        width: 100%;
        height: 3.1rem;
        border-radius: 12px;
        border: none;
        font-weight: 700;
        font-size: 1rem;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(96, 165, 250, 0.65);
        border-radius: 18px;
        padding: 14px;
        background: rgba(15, 23, 42, 0.60);
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 15px;
        padding: 16px;
        background: rgba(15, 23, 42, 0.80);
    }

    [data-testid="stTabs"] button {
        font-size: 1rem;
        font-weight: 650;
    }
    .workflow-container {
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 10px;
    margin: 20px 0 30px 0;
}

.workflow-step {
    flex: 1;
    position: relative;
    min-height: 130px;
    border: 1px solid rgba(96, 165, 250, 0.28);
    border-radius: 18px;
    padding: 18px 14px;
    background: rgba(15, 23, 42, 0.82);
    text-align: center;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);
}

.workflow-step:not(:last-child)::after {
    content: "→";
    position: absolute;
    right: -17px;
    top: 43%;
    color: #60a5fa;
    font-size: 1.7rem;
    font-weight: 800;
    z-index: 2;
}

.workflow-number {
    width: 34px;
    height: 34px;
    margin: 0 auto 10px auto;
    border-radius: 50%;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.workflow-title {
    color: #f8fafc;
    font-size: 1rem;
    font-weight: 750;
    margin-bottom: 6px;
}

.workflow-description {
    color: #cbd5e1;
    font-size: 0.82rem;
    line-height: 1.45;
}

@media (max-width: 950px) {
    .workflow-container {
        flex-direction: column;
    }

    .workflow-step:not(:last-child)::after {
        content: "↓";
        right: 49%;
        top: auto;
        bottom: -24px;
    }
}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Project helpers
# ---------------------------------------------------------
def initialize_project() -> None:
    """Validate paths and create local application folders."""

    create_directories()
    validate_paths()

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


@st.cache_resource(show_spinner=False)
def load_retfound_model() -> RETFoundClassifier:
    """Download, load, and cache the deployment model."""

    return load_inference_model(
        device=torch.device("cpu")
    )
   


def save_uploaded_image(uploaded_file) -> Path:
    """
    Save the uploaded image under a deterministic unique filename.

    A content hash avoids collisions when different files share a name.
    """

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()[:12]

    original_suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if original_suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
    }:
        original_suffix = ".png"

    safe_stem = "".join(
        character
        for character in Path(uploaded_file.name).stem
        if character.isalnum()
        or character in {"-", "_"}
    )

    if not safe_stem:
        safe_stem = "retinal_image"

    output_path = (
        UPLOAD_DIR
        / f"{safe_stem}_{file_hash}{original_suffix}"
    )

    output_path.write_bytes(file_bytes)

    return output_path


def get_severity_details(
    predicted_class: str,
) -> tuple[str, str]:
    """Return neutral screening guidance for each predicted class."""

    guidance = {
        "No DR": (
            "No apparent diabetic retinopathy",
            (
                "The model did not identify strong image-level evidence "
                "of diabetic retinopathy. Routine retinal assessment "
                "should still follow professional medical guidance."
            ),
        ),
        "Mild DR": (
            "Mild screening indication",
            (
                "Subtle retinal changes may be present. A qualified "
                "eye-care professional should review the image, "
                "particularly because mild disease can be difficult "
                "to distinguish from normal variation."
            ),
        ),
        "Moderate DR": (
            "Moderate screening indication",
            (
                "The model identified patterns associated with moderate "
                "diabetic retinopathy. Timely ophthalmic assessment is "
                "recommended for clinical confirmation."
            ),
        ),
        "Severe DR": (
            "Severe screening indication",
            (
                "The image contains patterns associated with severe "
                "diabetic retinopathy. Prompt specialist assessment "
                "is recommended."
            ),
        ),
        "Proliferative DR": (
            "Advanced screening indication",
            (
                "The model identified patterns associated with "
                "proliferative diabetic retinopathy. Urgent specialist "
                "review is recommended for clinical confirmation."
            ),
        ),
    }

    return guidance[predicted_class]


def run_complete_analysis(
    model: RETFoundClassifier,
    image_path: Path,
) -> dict:
    """Run both explainability pipelines on the same image."""

    attention_result = run_attention_rollout(
        model=model,
        image_path=image_path,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gradcam_result = run_gradcam(
        model=model,
        image_path=image_path,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "image_path": image_path,
        "predicted_class": attention_result["predicted_class"],
        "confidence": attention_result["confidence"],
        "probabilities": attention_result["probabilities"],
        "attention_path": attention_result["output_path"],
        "gradcam_path": gradcam_result["output_path"],
        "attention_layers": attention_result["attention_layers"],
    }


def probability_dataframe(
    probabilities: dict[str, float],
) -> pd.DataFrame:
    """Prepare class probabilities for Streamlit charts."""

    dataframe = pd.DataFrame(
        {
            "DR grade": list(probabilities.keys()),
            "Probability": list(probabilities.values()),
        }
    )

    dataframe["Probability (%)"] = (
        dataframe["Probability"] * 100
    )

    return dataframe


# ---------------------------------------------------------
# UI sections
# ---------------------------------------------------------
def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">
                👁️ RetinaGuard AI
            </div>
            <div class="hero-subtitle">
                Explainable diabetic retinopathy severity screening
                powered by the RETFound retinal foundation model.
                Upload a colour fundus photograph to generate a
                five-grade prediction, confidence analysis,
                Attention Rollout and ViT Grad-CAM visualization.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_prediction_timeline() -> None:
    st.markdown("## 🔬 How RetinaGuard Works")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.info("### ①\n📤 Upload\n\nRetinal Image")

    with col2:
        st.info("### ②\n🖼️ Preprocess\n\nResize + Normalize")

    with col3:
        st.info("### ③\n🧠 RETFound\n\nFeature Extraction")

    with col4:
        st.info("### ④\n📊 Prediction\n\n5 DR Classes")

    with col5:
        st.info("### ⑤\n🔥 Explainability\n\nAttention + GradCAM")

    with col6:
        st.info("### ⑥\n📋 Report\n\nClinical Guidance")
        
def render_model_information() -> None:
    with st.expander(
        "Model and project information",
        expanded=False,
    ):
        left, middle, right = st.columns(3)

        with left:
            st.markdown("### Foundation model")
            st.write("RETFound CFP")
            st.write("ViT-Large/16 retinal encoder")

        with middle:
            st.markdown("### Dataset")
            st.write("APTOS 2019")
            st.write("Five diabetic-retinopathy grades")

        with right:
            st.markdown("### Best evaluation")
            st.write("Accuracy: 80.05%")
            st.write("Macro AUROC: 94.26%")
            st.write("Quadratic kappa: 87.48%")


def render_upload_section():
    st.markdown("## Retinal image screening")

    uploaded_file = st.file_uploader(
        "Upload a colour fundus retinal image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help=(
            "Use a clear colour fundus photograph. "
            "OCT scans are not supported by this CFP model."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to begin RETFound analysis."
        )
        return None

    try:
        image = Image.open(
            uploaded_file
        ).convert("RGB")
    except Exception as error:
        st.error(
            f"The uploaded file could not be read as an image: {error}"
        )
        return None

    image_column, details_column = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    with image_column:
        st.image(
            image,
            caption="Uploaded fundus image",
            width="stretch",
        )

    with details_column:
        st.markdown("### Image details")

        st.write(
            f"**Filename:** {uploaded_file.name}"
        )
        st.write(
            f"**Dimensions:** {image.width} × {image.height}"
        )
        st.write(
            f"**Mode:** {image.mode}"
        )

        st.markdown(
            """
            The image will be resized to 224 × 224 pixels and
            normalized using the same evaluation pipeline used
            during model testing.
            """
        )

    analyze_clicked = st.button(
        "Run complete RETFound analysis",
        type="primary",
        width="stretch",
    )

    if analyze_clicked:
        image_path = save_uploaded_image(
            uploaded_file
        )

        try:
            with st.spinner(
                "Loading RETFound and generating explanations..."
            ):
                model = load_retfound_model()

                result = run_complete_analysis(
                    model=model,
                    image_path=image_path,
                )

                analysis_id = save_analysis(result)

                result["analysis_id"] = analysis_id

            

                st.session_state["retinaguard_result"] = result

        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            st.error(
                "The available memory was insufficient to run "
                "RETFound. The ViT-Large model may exceed the "
                "deployment platform's memory limit."
            )

        except RuntimeError as error:
            st.error(
                "The RetinaGuard model could not be downloaded "
                "or loaded. Check the Hugging Face repository, "
                "HF_TOKEN, and available system memory."
            )
            st.exception(error)

        except Exception as error:
            st.error(
                "The analysis could not be completed."
            )
            st.exception(error)

    return uploaded_file


def render_results(result: dict) -> None:
    st.markdown("---")
    st.markdown("## Screening result")

    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]
    analysis_id = result.get(
        "analysis_id",
        "Not recorded",
    )
    severity_title, guidance = get_severity_details(
        predicted_class
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        st.metric(
            label="Predicted DR grade",
            value=predicted_class,
        )

    with metric_2:
        st.metric(
            label="Model confidence",
            value=f"{confidence:.2%}",
        )

    with metric_3:
        st.metric(
            label="Transformer layers analyzed",
            value=str(result["attention_layers"]),
        )
    with metric_4:
        st.metric(
            label="Analysis ID",
            value=analysis_id,
        )
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">
                Screening interpretation
            </div>
            <div class="status-value">
                {severity_title}
            </div>
            <div class="status-description">
                {guidance}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Class probabilities")

    probability_data = probability_dataframe(
        probabilities
    )

    st.bar_chart(
        probability_data,
        x="DR grade",
        y="Probability (%)",
        horizontal=True,
    )

    with st.expander(
        "View exact probability values",
        expanded=False,
    ):
        display_data = probability_data[
            ["DR grade", "Probability (%)"]
        ].copy()

        display_data["Probability (%)"] = (
            display_data["Probability (%)"]
            .map(lambda value: f"{value:.2f}%")
        )

        st.dataframe(
            display_data,
            width="stretch",
            hide_index=True,
        )

    st.markdown("## Explainability")

    attention_tab, gradcam_tab, comparison_tab = st.tabs(
        [
            "Attention Rollout",
            "ViT Grad-CAM",
            "Side-by-side comparison",
        ]
    )

    attention_path = Path(
        result["attention_path"]
    )

    gradcam_path = Path(
        result["gradcam_path"]
    )

    with attention_tab:
        st.markdown(
            """
            Attention Rollout combines attention flow across the
            transformer layers to estimate which image patches
            contributed to the final representation.
            """
        )

        if attention_path.exists():
            st.image(
                str(attention_path),
                width="stretch",
            )

    with gradcam_tab:
        st.markdown(
            """
            ViT Grad-CAM uses gradients for the predicted class to
            estimate class-specific influence across retinal patches.
            """
        )

        if gradcam_path.exists():
            st.image(
                str(gradcam_path),
                width="stretch",
            )
    
    with comparison_tab:
        left, right = st.columns(2)

        with left:
            st.markdown("#### Attention Rollout")

            if attention_path.exists():
                st.image(
                    str(attention_path),
                    width="stretch",
                )

        with right:
            st.markdown("#### ViT Grad-CAM")

            if gradcam_path.exists():
                st.image(
                    str(gradcam_path),
                    width="stretch",
                )

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Important:</strong>
            Highlighted regions indicate areas that influenced the
            model output. They are not confirmed lesions, segmentation
            masks or proof of disease. This research prototype does not
            replace examination by a qualified ophthalmologist.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Download explanation images")

    download_left, download_right = st.columns(2)

    if attention_path.exists():
        with download_left:
            st.download_button(
                label="Download Attention Rollout report",
                data=attention_path.read_bytes(),
                file_name=attention_path.name,
                mime="image/png",
                width="stretch",
                key="download_attention_report",
            )

    if gradcam_path.exists():
        with download_right:
            st.download_button(
                label="Download ViT Grad-CAM report",
                data=gradcam_path.read_bytes(),
                file_name=gradcam_path.name,
                mime="image/png",
                width="stretch",
                key="download_gradcam_report",
            )

    st.markdown("### Download complete clinical report")

    image_path = Path(
        result["image_path"]
    )

    if (
        image_path.exists()
        and attention_path.exists()
        and gradcam_path.exists()
    ):
        pdf_bytes = generate_pdf_report(
            image_path=image_path,
            attention_path=attention_path,
            gradcam_path=gradcam_path,
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            severity_title=severity_title,
            guidance=guidance,
        )

        st.download_button(
            label="Download RetinaGuard PDF Report",
            data=pdf_bytes,
            file_name=(
                f"retinaguard_{image_path.stem}_report.pdf"
            ),
            mime="application/pdf",
            width="stretch",
            key="download_retinaguard_pdf",
        )
    else:
        st.warning(
            "The PDF report cannot be generated because one or "
            "more analysis files are missing."
        )

def render_history_dashboard() -> None:
    """Display previous RetinaGuard analyses."""

    st.markdown("## Analysis history")

    history = load_history()

    if history.empty:
        st.info(
            "No completed screenings have been saved yet."
        )
        return

    history["confidence"] = pd.to_numeric(
        history["confidence"],
        errors="coerce",
    ).fillna(0.0)

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce",
    )

    total_analyses = len(history)

    average_confidence = (
        history["confidence"].mean()
    )

    most_common_prediction = (
        history["predicted_class"]
        .mode()
        .iloc[0]
    )

    latest_analysis = (
        history["timestamp"]
        .max()
    )

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    with metric_1:
        st.metric(
            "Total analyses",
            total_analyses,
        )

    with metric_2:
        st.metric(
            "Average confidence",
            f"{average_confidence:.2%}",
        )

    with metric_3:
        st.metric(
            "Most common prediction",
            most_common_prediction,
        )

    with metric_4:
        latest_text = (
            latest_analysis.strftime(
                "%d %b %Y, %I:%M %p"
            )
            if pd.notna(latest_analysis)
            else "Unavailable"
        )

        st.metric(
            "Latest analysis",
            latest_text,
        )

    st.markdown("### Prediction distribution")

    prediction_counts = (
        history["predicted_class"]
        .value_counts()
        .rename_axis("DR grade")
        .reset_index(name="Count")
    )

    st.bar_chart(
        prediction_counts,
        x="DR grade",
        y="Count",
    )

    st.markdown("### Recent screenings")

    display_history = history.copy()

    display_history["confidence"] = (
        display_history["confidence"]
        .map(lambda value: f"{value:.2%}")
    )
    display_history = display_history.sort_values(
        by="timestamp",
        ascending=False,
    )
    
    display_history["timestamp"] = (
        display_history["timestamp"]
        .dt.strftime("%d %b %Y, %I:%M %p")
    )
    display_columns = [
    "analysis_id",
    "timestamp",
    "filename",
    "predicted_class",
    "confidence",
     ]
    st.dataframe(
        display_history[display_columns],
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        label="Download complete analysis history",
        data=history.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="retinaguard_analysis_history.csv",
        mime="text/csv",
        width="stretch",
        key="download_analysis_history",
    )

    with st.expander(
        "Clear analysis history",
        expanded=False,
    ):
        st.warning(
            "This action removes all saved history records."
        )

        confirm_clear = st.checkbox(
            "I understand that the history will be deleted",
            key="confirm_clear_history",
        )

        if st.button(
            "Clear all history",
            disabled=not confirm_clear,
            key="clear_history_button",
        ):
            clear_history()

            st.session_state.pop(
                "retinaguard_result",
                None,
            )

            st.success(
                "Analysis history cleared."
            )

            st.rerun()
            
def render_model_comparison() -> None:
    """Display experimental results side by side."""

    st.markdown("## Model comparison")

    st.markdown(
        """
        This section compares three RETFound adaptation strategies.
        Macro F1 is used as the primary selection metric because the
        APTOS dataset is strongly imbalanced.
        """
    )

    experiments = load_experiment_metrics()

    if not experiments:
        st.warning(
            "No experiment metric files were found. "
            "Run the evaluation scripts first."
        )
        return

    summary = build_summary_dataframe(
        experiments
    )

    best_name, best_macro_f1 = (
        get_best_experiment(
            summary,
            metric="Macro F1",
        )
    )

    best_accuracy_name, best_accuracy = (
        get_best_experiment(
            summary,
            metric="Accuracy",
        )
    )

    best_auc_name, best_auc = (
        get_best_experiment(
            summary,
            metric="Macro AUROC",
        )
    )

    metric_1, metric_2, metric_3 = (
        st.columns(3)
    )

    with metric_1:
        st.metric(
            "Best Macro F1",
            f"{best_macro_f1:.4f}",
        )
        st.caption(best_name)

    with metric_2:
        st.metric(
            "Best Accuracy",
            f"{best_accuracy:.2%}",
        )
        st.caption(best_accuracy_name)

    with metric_3:
        st.metric(
            "Best Macro AUROC",
            f"{best_auc:.4f}",
        )
        st.caption(best_auc_name)

    st.success(
        f"Recommended deployment model: **{best_name}**. "
        "It achieved the strongest macro F1 across the evaluated "
        "fine-tuning strategies."
    )

    st.markdown("### Overall metric comparison")

    display_summary = summary.copy()

    metric_columns = [
        column
        for column in display_summary.columns
        if column != "Experiment"
    ]

    for column in metric_columns:
        display_summary[column] = (
            display_summary[column]
            .map(
                lambda value: (
                    f"{value:.4f}"
                    if pd.notna(value)
                    else "N/A"
                )
            )
        )

    st.dataframe(
        display_summary,
        width="stretch",
        hide_index=True,
    )

    chart_metrics = [
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Quadratic Kappa",
        "Macro AUROC",
    ]

    chart_data = summary[
        ["Experiment"] + chart_metrics
    ].melt(
        id_vars="Experiment",
        var_name="Metric",
        value_name="Score",
    )

    st.bar_chart(
        chart_data,
        x="Metric",
        y="Score",
        color="Experiment",
        stack=False,
    )

    st.markdown("### Per-class recall comparison")

    per_class = (
        build_per_class_recall_dataframe(
            experiments
        )
    )

    st.bar_chart(
        per_class,
        x="DR grade",
        y="Recall",
        color="Experiment",
        stack=False,
    )

    with st.expander(
        "View exact per-class values",
        expanded=False,
    ):
        exact_values = per_class.copy()

        for column in [
            "Recall",
            "Precision",
            "F1",
        ]:
            exact_values[column] = (
                exact_values[column]
                .map(lambda value: f"{value:.4f}")
            )

        st.dataframe(
            exact_values,
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Confusion matrices")

    available_names = list(
        experiments.keys()
    )

    selected_experiment = st.selectbox(
        "Choose an experiment",
        options=available_names,
        key="comparison_experiment_selector",
    )

    selected_details = experiments[
        selected_experiment
    ]

    st.info(
        selected_details["description"]
    )

    matrix_path = Path(
        selected_details["confusion_matrix"]
    )

    if matrix_path.exists():
        st.image(
            str(matrix_path),
            caption=(
                f"{selected_experiment} confusion matrix"
            ),
            width="stretch",
        )
    else:
        st.warning(
            f"Confusion-matrix image not found: {matrix_path}"
        )

    st.markdown("### Experimental conclusion")

    st.markdown(
        f"""
        The **{best_name}** configuration produced the strongest
        macro F1 score of **{best_macro_f1:.4f}**.

        In your experiments, unfreezing the final transformer block
        increased training performance but did not improve validation
        or test generalization. This suggests that the frozen
        RETFound encoder provided stronger regularization for the
        available APTOS training set.
        """
    )

    comparison_csv = summary.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download model-comparison results",
        data=comparison_csv,
        file_name="retinaguard_model_comparison.csv",
        mime="text/csv",
        width="stretch",
        key="download_model_comparison",
    )            
def main() -> None:
    initialize_project()
    render_header()
    render_prediction_timeline()
    render_model_information()

    screening_tab, history_tab, comparison_tab = (
        st.tabs(
            [
                "Retinal Screening",
                "Analysis History",
                "Model Comparison",
            ]
        )
    )
    
    with screening_tab:
        render_upload_section()

        result = st.session_state.get(
            "retinaguard_result"
        )

        if result is not None:
            render_results(result)

    with history_tab:
        render_history_dashboard()
    with comparison_tab:
        render_model_comparison()

if __name__ == "__main__":
    main()
