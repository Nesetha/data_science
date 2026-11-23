
import streamlit as st
from pathlib import Path
from PIL import Image
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.retrieval import answer_question, index_research_folder
from src.safety import check_response
from src.storage import save_case, list_cases
from src.models.predict import predict

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="PrenatalAI | Research Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: #101820;
    color: #F0F4F8;
}
[data-testid="stSidebar"] {
    background: #172632;
}
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #EAF6F6;
}
.subtitle {
    color: #AABBC5;
    font-size: 1rem;
}
.status-card {
    background: #1B303B;
    border: 1px solid #34505B;
    padding: 18px;
    border-radius: 14px;
    min-height: 110px;
}
.status-label {
    color: #AABBC5;
    font-size: 0.85rem;
}
.status-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #72D6BD;
    margin-top: 8px;
}
.section-heading {
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 1rem;
}
.result-card {
    background: #1B303B;
    border: 1px solid #34505B;
    border-radius: 16px;
    padding: 22px;
}
.small-note {
    color: #AABBC5;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🧬 PrenatalAI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Prenatal Ultrasound Research & Evidence Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.warning(
    "Research prototype only — not a medical device. "
    "The anatomical classifier cannot diagnose, detect, or rule out Spina Bifida."
)

# ---------------- MODEL STATUS ----------------

MODEL_PATH = ROOT / "models" / "anatomical_plane_mobilenetv2.pth"
model_available = MODEL_PATH.exists()

# ---------------- NAVIGATION ----------------

st.sidebar.title("🧬 PrenatalAI")
st.sidebar.caption("Research workspace")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Ultrasound workspace",
        "Medical research assistant",
        "Research library",
        "Case history",
        "Evaluation"
    ]
)

st.sidebar.divider()
st.sidebar.caption("PrenatalAI Research Prototype")

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.markdown("## Research workspace")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""<div class="status-card">
            <div class="status-label">ANATOMICAL CLASSIFIER</div>
            <div class="status-value">
            {"Enabled" if model_available else "Not available"}
            </div></div>""",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """<div class="status-card">
            <div class="status-label">LLM PROVIDER</div>
            <div class="status-value">Optional</div>
            </div>""",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """<div class="status-card">
            <div class="status-label">DATABASE</div>
            <div class="status-value">Local demo</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("### Platform capabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧠 AI image classification")
        st.write("MobileNetV2-based six-class anatomical-plane classification.")

        st.markdown("#### 📚 Research library")
        st.write("Upload and index research PDFs for educational retrieval.")

    with col2:
        st.markdown("#### 💬 Research assistant")
        st.write("Ask educational questions using available research content.")

        st.markdown("#### 🛡️ Safety and evaluation")
        st.write("Basic response checks, evaluation tests and research case records.")

# ---------------- ULTRASOUND WORKSPACE ----------------

elif page == "Ultrasound workspace":

    st.markdown("## 🧠 Ultrasound research workspace")
    st.caption("Upload a de-identified ultrasound image for anatomical-plane classification.")

    uploaded = st.file_uploader(
        "Choose ultrasound image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:

        try:
            image = Image.open(uploaded).convert("RGB")

            left, right = st.columns([1.15, 0.85], gap="large")

            with left:
                st.markdown("### Uploaded image")
                st.image(image, use_container_width=True)

                st.caption(
                    f"Filename: {uploaded.name} | "
                    f"Dimensions: {image.width} × {image.height}"
                )

            with right:
                st.markdown("### Model analysis")

                if not model_available:
                    st.error("Trained model not found in models/.")

                else:
                    with st.spinner("Running MobileNetV2 inference..."):
                        result = predict(image)

                    if result.get("available"):

                        predicted = result["predicted_class"]
                        confidence = result["confidence"]
                        probabilities = result["probabilities"]

                        
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="small-note">MODEL PREDICTION</div>
                                <h1 style="color:#72D6BD; margin-bottom:5px;">
                                    {predicted}
                                </h1>
                                <div class="small-note">TOP CLASS SCORE</div>
                                <h2 style="margin-top:5px;">{confidence:.1%}</h2>
                                <div class="small-note">
                                    MobileNetV2 · Six-class anatomical classification
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                        # Confidence interpretation is descriptive,
                        # not a clinical certainty assessment.
                        if confidence < 0.60:
                            st.warning(
                                "Low model confidence. Treat this output "
                                "as uncertain and avoid interpreting it "
                                "as a reliable classification."
                            )
                        elif confidence < 0.80:
                            st.info(
                                "Moderate model confidence. "
                                "Independent review would be needed."
                            )
                        else:
                            st.success(
                                "High model confidence score. "
                                "This does not establish clinical correctness."
                            )


                            st.markdown("### Class probability distribution")

                            # Sort probabilities from highest to lowest
                            sorted_probabilities = sorted(
                                probabilities.items(),
                                key=lambda item: item[1],
                                reverse=True
                            )

                            for class_name, probability in sorted_probabilities:
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.write(class_name)

                                with col2:
                                    st.write(f"**{probability:.1%}**")

                                st.progress(probability)

                            st.divider()

                            st.info(
                                "Probabilities represent the model's relative output scores. "
                                "They are not clinical probabilities or diagnostic certainty."
                            )

                            st.warning(result["clinical_limit"])


                        st.caption(
                            "Probabilities are model outputs, not clinical "
                            "probabilities or diagnostic certainty."
                        )

                        if st.button(
                            "Save research record",
                            type="primary",
                            use_container_width=True
                        ):
                            case_id = save_case({
                                "filename": uploaded.name,
                                "image_size": list(image.size),
                                "task": result["task"],
                                "predicted_class": predicted,
                                "confidence": confidence,
                                "probabilities": probabilities,
                                "model": "MobileNetV2",
                                "timestamp": datetime.now().isoformat(),
                                "status": "research_classification"
                            })

                            st.success(
                                f"Research record saved: {case_id}"
                            )

                    else:
                        st.error(result.get("message", "Prediction unavailable."))

        except Exception as exc:
            st.error(f"Unable to process image: {exc}")

# ---------------- RESEARCH ASSISTANT ----------------

elif page == "Medical research assistant":

    st.markdown("## 💬 Evidence-grounded research assistant")

    question = st.text_area(
        "Enter your research question",
        placeholder="Ask an educational question about prenatal ultrasound research..."
    )

    if st.button(
        "Retrieve information",
        type="primary",
        disabled=not question.strip()
    ):

        with st.spinner("Retrieving research information..."):

            result = answer_question(question)

        verdict = check_response(result["answer"])

        st.markdown("### Response")
        st.write(result["answer"])

        st.caption(
            f"Mode: {result['mode']} | "
            f"Basic safety check: "
            f"{'Passed' if verdict['approved'] else 'Flagged'}"
        )

        if result["sources"]:
            st.markdown("### Retrieved sources")
            for source in result["sources"]:
                st.write(f"- {source}")

        st.info(
            "Educational research information only. "
            "Not patient-specific advice or diagnosis."
        )

# ---------------- RESEARCH LIBRARY ----------------

elif page == "Research library":

    st.markdown("## 📚 Research library")

    st.write(
        "Place trusted, legally obtained research PDFs in the research/ folder."
    )

    if st.button("Index research PDFs", type="primary"):

        with st.spinner("Indexing documents..."):
            count, message = index_research_folder()

        if count >= 0:
            st.success(message)
        else:
            st.error(message)

# ---------------- CASE HISTORY ----------------


elif page == "Case history":

    import pandas as pd

    st.markdown("## 🗂️ Research case history")
    st.caption("Review previously saved anatomical-plane classification records.")

    records = list_cases()

    if records:

        df = pd.DataFrame(records)

        # Summary metrics
        c1, c2, c3 = st.columns(3)

        c1.metric("Total research records", len(df))

        if "confidence" in df.columns:
            c2.metric(
                "Average model score",
                f"{df['confidence'].astype(float).mean():.1%}"
            )

        if "predicted_class" in df.columns:
            c3.metric(
                "Unique predicted classes",
                df["predicted_class"].nunique()
            )

        st.divider()

        # Filter
        if "predicted_class" in df.columns:
            classes = ["All"] + sorted(
                df["predicted_class"].dropna().unique().tolist()
            )

            selected_class = st.selectbox(
                "Filter by predicted class",
                classes
            )

            if selected_class != "All":
                df = df[
                    df["predicted_class"] == selected_class
                ]

        # Compact table
        preferred_columns = [
            "case_id",
            "filename",
            "predicted_class",
            "confidence",
            "model",
            "timestamp",
            "status"
        ]

        visible_columns = [
            c for c in preferred_columns if c in df.columns
        ]

        display_df = df[visible_columns].copy()

        if "confidence" in display_df.columns:
            display_df["confidence"] = (
                display_df["confidence"].astype(float) * 100
            ).round(2).astype(str) + "%"

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=min(60 + len(display_df) * 35, 500),
            column_config={
                "case_id": st.column_config.TextColumn("Case ID"),
                "filename": st.column_config.TextColumn("Image"),
                "predicted_class": st.column_config.TextColumn("Predicted class"),
                "confidence": st.column_config.TextColumn("Model score"),
                "model": st.column_config.TextColumn("Model"),
                "timestamp": st.column_config.TextColumn("Date and time"),
                "status": st.column_config.TextColumn("Status")
            }
        )

        # CSV export
        st.download_button(
            "Download research records as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="prenatalai_research_records.csv",
            mime="text/csv"
        )

        # Detailed record
        st.markdown("### Inspect individual record")

        if "case_id" in df.columns:
            selected_id = st.selectbox(
                "Select case ID",
                df["case_id"].astype(str).tolist()
            )

            selected_record = df[
                df["case_id"].astype(str) == selected_id
            ].iloc[0]

            with st.expander("View complete record"):
                for key, value in selected_record.items():
                    if key == "probabilities":
                        st.markdown("**Class probabilities**")
                        try:
                            import json
                            probs = (
                                json.loads(value)
                                if isinstance(value, str)
                                else value
                            )
                            st.bar_chart(
                                probs,
                                y_min=0,
                                y_max=1
                            )
                        except Exception:
                            st.write(value)
                    else:
                        st.write(f"**{key}:** {value}")

    else:
        st.info("No research records available yet.")


# ---------------- EVALUATION ----------------

elif page == "Evaluation":

    st.markdown("## 🛡️ Evaluation and safety")

    from src.evaluation import run_smoke_tests

    results = run_smoke_tests()

    passed = sum(x["passed"] for x in results)

    st.metric(
        "Basic software checks passed",
        f"{passed}/{len(results)}"
    )

    st.dataframe(results, use_container_width=True)

    st.caption(
        "These are software smoke tests, not a validated medical "
        "safety evaluation or clinical benchmark."
    )
