"""
Cynbit LeadIQ - AI-Powered Lead Intelligence, Qualification and Conversion Prediction System
Enhanced Production-Style Streamlit Application.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    BASE_DIR,
    SERVICE_CATEGORIES,
    BUDGET_TIERS,
    TIMELINE_OPTIONS,
    COMPANY_SIZES,
    LEAD_STATUS_OPTIONS,
    METRICS_FILE_PATH,
    MODEL_FILE_PATH,
    SAMPLE_LEADS_PATH
)
from src.database import (
    init_db,
    insert_lead,
    get_lead_by_id,
    get_all_leads_df,
    update_lead_status,
    delete_lead,
    clear_and_load_demo_leads
)
from src.preprocessing import clean_text, sanitize_input_record
from src.extraction import detect_service_categories, extract_keywords_tf
from src.scoring import compute_qualification_score
from src.prediction import ConversionPredictor, compare_qualification_and_ml
from src.response_generator import generate_lead_response, DISCLAIMER_HEADER
from src.demo_data import generate_processed_demo_leads
from src.similarity import find_similar_leads
from src.simulator import simulate_lead_modifications

# Streamlit Page Config
st.set_page_config(
    page_title="Cynbit LeadIQ | Lead Intelligence & Conversion Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Modern SaaS Polish)
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.2s ease;
    }
    .metric-title {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.95rem;
        font-weight: 800;
        margin-top: 4px;
    }
    .metric-sub {
        color: #2563eb;
        font-size: 0.82rem;
        margin-top: 4px;
        font-weight: 500;
    }
    /* Status Badges */
    .badge-hot {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-warm {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fde68a;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-cold {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #e2e8f0;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 700;
        display: inline-block;
    }
    /* Email Mockup Client */
    .email-client-box {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .email-header-row {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 8px;
        margin-bottom: 12px;
        font-size: 0.9rem;
        color: #475569;
    }
    /* Notice Banners */
    .disclaimer-banner {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #1e293b;
        margin: 12px 0;
    }
    .system-badge {
        font-size: 0.72rem;
        background: #e2e8f0;
        color: #0f172a;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Caches the ML Conversion Predictor instance."""
    return ConversionPredictor(MODEL_FILE_PATH)


def ensure_initial_data(predictor: ConversionPredictor):
    """Initializes SQLite schema and seeds demo leads if empty."""
    init_db()
    df = get_all_leads_df()
    if df.empty:
        demo_records = generate_processed_demo_leads(predictor)
        clear_and_load_demo_leads(demo_records)


predictor = get_predictor()
ensure_initial_data(predictor)

# ---------------------------------------------------------
# Sidebar Navigation & System Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Cynbit LeadIQ")
    st.markdown("<span class='system-badge'>OFFLINE PRO • LOCAL ML</span>", unsafe_allow_html=True)
    st.caption("Intelligent Qualification & Conversion Prediction System")
    st.markdown("---")

    selected_page = st.radio(
        "Navigation",
        [
            "Overview",
            "Add Lead",
            "Lead Intelligence",
            "Conversion Prediction",
            "Response Generator",
            "Lead Management",
            "Model Evaluation",
            "About & Methodology"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 🛠️ Demo Operations")
    col_reset, col_clear = st.columns(2)
    with col_reset:
        if st.button("Reset Demo", use_container_width=True, help="Reload 12 pre-scored synthetic benchmark leads"):
            demo_records = generate_processed_demo_leads(predictor)
            cnt = clear_and_load_demo_leads(demo_records)
            st.success(f"Loaded {cnt} demo leads!")
            st.rerun()

    with col_clear:
        if st.button("Clear Leads", use_container_width=True, help="Wipe all lead entries in SQLite database"):
            clear_and_load_demo_leads([])
            st.warning("All leads cleared.")
            st.rerun()

    st.markdown("---")
    st.caption("🛡️ **Zero Cloud Inference**: Runs 100% locally. No external APIs or proprietary telemetry.")


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if selected_page == "Overview":
    st.title("Executive Dashboard Overview")
    st.markdown("Live operational pipeline metrics, qualification distribution, and predictive conversion health.")

    df = get_all_leads_df()

    if df.empty:
        st.info("No leads currently recorded in database. Click **Reset Demo** in the sidebar to populate synthetic benchmark leads.")
    else:
        # KPI Row
        total_leads = len(df)
        hot_leads = len(df[df["qualification_tier"] == "Hot"])
        warm_leads = len(df[df["qualification_tier"] == "Warm"])
        cold_leads = len(df[df["qualification_tier"] == "Cold"])
        avg_score = round(df["qualification_score"].mean(), 1)
        avg_conversion_prob = round(df["conversion_probability"].mean() * 100, 1)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Total Inbound Leads</div>
                <div class='metric-value'>{total_leads}</div>
                <div class='metric-sub'>Across all acquisition channels</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Hot Qualified Leads</div>
                <div class='metric-value'>{hot_leads} <span style='font-size:1.1rem;color:#10b981;'>({round(hot_leads/total_leads*100)}%)</span></div>
                <div class='metric-sub'>Score ≥ 70/100 (Immediate routing)</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Mean Qualification Score</div>
                <div class='metric-value'>{avg_score} <span style='font-size:1.1rem;color:#64748b;'>/ 100</span></div>
                <div class='metric-sub'>Rule-based policy evaluation</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Mean Conversion Likelihood</div>
                <div class='metric-value'>{avg_conversion_prob}%</div>
                <div class='metric-sub'>Statistical ML prediction</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Strategic 2x2 Matrix & Service Distribution
        c_mat, c_pie = st.columns([1.4, 1])
        with c_mat:
            st.subheader("🎯 Strategic 2x2 Action Matrix (Score vs ML Likelihood)")
            st.caption("Categorizes leads into 4 strategic routing quadrants.")

            # Create quadrant scatter
            fig_matrix = px.scatter(
                df,
                x="qualification_score",
                y="conversion_probability",
                color="qualification_tier",
                size=[12] * len(df),
                hover_data=["name", "company", "primary_category", "lead_status"],
                labels={"qualification_score": "Rule Qualification Score (0-100)", "conversion_probability": "ML Conversion Probability"},
                color_discrete_map={"Hot": "#10B981", "Warm": "#F59E0B", "Cold": "#64748B"}
            )
            # Add quadrant dividing lines
            fig_matrix.add_vline(x=70, line_dash="dash", line_color="#94a3b8", annotation_text="Hot Policy Threshold (70)", annotation_position="top left")
            fig_matrix.add_hline(y=0.50, line_dash="dash", line_color="#94a3b8", annotation_text="ML Win Threshold (50%)", annotation_position="bottom right")

            # Add quadrant backdrop labels
            fig_matrix.add_annotation(x=85, y=0.88, text="🚀 Priority Alpha (Partner Route)", showarrow=False, font=dict(size=11, color="#059669"))
            fig_matrix.add_annotation(x=35, y=0.88, text="⚡ High Intent Niche (Fast Track)", showarrow=False, font=dict(size=11, color="#2563eb"))
            fig_matrix.add_annotation(x=85, y=0.15, text="🎯 Strategic Account (Qualify Scope)", showarrow=False, font=dict(size=11, color="#d97706"))
            fig_matrix.add_annotation(x=35, y=0.15, text="❄️ Nurture / Self-Serve", showarrow=False, font=dict(size=11, color="#64748b"))

            fig_matrix.update_layout(margin=dict(t=20, b=10, l=10, r=10), height=380, yaxis_tickformat=".0%")
            st.plotly_chart(fig_matrix, use_container_width=True)

        with c_pie:
            st.subheader("Leads by Service Domain")
            st.caption("Automated NLP categorization distribution.")
            cat_counts = df["primary_category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_cat = px.pie(
                cat_counts,
                values="Count",
                names="Category",
                hole=0.48,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_cat.update_layout(margin=dict(t=20, b=10, l=10, r=10), height=380)
            st.plotly_chart(fig_cat, use_container_width=True)

        # Charts Row 2
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Budget & Urgency Distribution")
            fig_bar = px.histogram(
                df,
                x="budget_tier",
                color="timeline",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, xaxis_tickangle=-20)
            st.plotly_chart(fig_bar, use_container_width=True)

        with c4:
            st.subheader("Pipeline Workflow Funnel")
            status_counts = df["lead_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_status = px.bar(
                status_counts,
                x="Status",
                y="Count",
                text="Count",
                color="Status",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_status.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, showlegend=False)
            st.plotly_chart(fig_status, use_container_width=True)

        st.markdown("---")
        st.subheader("Recent Inbound Enquiries")
        display_cols = ["id", "name", "company", "primary_category", "qualification_score", "qualification_tier", "conversion_probability", "lead_status"]
        st.dataframe(
            df[display_cols].head(8).style.format({
                "conversion_probability": "{:.1%}"
            }),
            use_container_width=True
        )


# =========================================================
# PAGE 2: ADD LEAD (MANUAL & BATCH CSV)
# =========================================================
elif selected_page == "Add Lead":
    st.title("Inbound Lead Intake")
    st.markdown("Enter customer inquiries manually or upload bulk CSV files for automated NLP evaluation and scoring.")

    tab_manual, tab_batch = st.tabs(["📝 Manual Lead Entry", "📁 Batch CSV Upload"])

    with tab_manual:
        st.subheader("Manual Lead Intake Form")
        with st.form("manual_lead_form", clear_on_submit=False):
            col_a, col_b = st.columns(2)
            with col_a:
                lead_name = st.text_input("Contact Name *", placeholder="e.g. Dr. Jennifer Cole")
                lead_email = st.text_input("Business Email *", placeholder="e.g. j.cole@enterprise-biotech.com")
                lead_company = st.text_input("Organization Name", placeholder="e.g. Enterprise Biotech Ltd")
                lead_role = st.text_input("Contact Role / Title", placeholder="e.g. Head of Data Science")

            with col_b:
                budget_tier = st.selectbox("Budget Availability Tier", BUDGET_TIERS, index=1)
                timeline = st.selectbox("Urgency / Timeline Requirement", TIMELINE_OPTIONS, index=0)
                company_size = st.selectbox("Organization Size", COMPANY_SIZES, index=1)
                source_channel = st.selectbox("Inquiry Channel", ["Website Form", "Direct Inbound", "Referral / Partner", "Organic Search", "Marketing Campaign"])

            inquiry_text = st.text_area(
                "Customer Inquiry / Requirement Details *",
                height=160,
                placeholder="Describe project objectives, tech stack, data availability, and deliverables..."
            )

            col_submit, col_opt = st.columns([1, 3])
            with col_submit:
                submitted = st.form_submit_button("Analyze & Save Lead", use_container_width=True)

        if submitted:
            raw_input = {
                "name": lead_name,
                "email": lead_email,
                "company": lead_company,
                "role": lead_role,
                "inquiry_text": inquiry_text,
                "budget_tier": budget_tier,
                "timeline": timeline,
                "company_size": company_size,
                "source_channel": source_channel
            }
            cleaned_data, errors = sanitize_input_record(raw_input)

            if errors:
                for err in errors:
                    st.error(f"Validation Error: {err}")
            else:
                # Run NLP & Scoring Pipeline
                primary_cat, cat_scores, detected_entities = detect_service_categories(cleaned_data["inquiry_text"])
                kw_scores = extract_keywords_tf(cleaned_data["inquiry_text"], top_n=6)
                extracted_kws = ", ".join([f"{w} ({s})" for w, s in kw_scores])

                score_res = compute_qualification_score(
                    inquiry_text=cleaned_data["inquiry_text"],
                    budget_tier=cleaned_data["budget_tier"],
                    timeline=cleaned_data["timeline"],
                    company_size=cleaned_data["company_size"],
                    email=cleaned_data["email"],
                    role=cleaned_data["role"],
                    source_channel=cleaned_data["source_channel"],
                    detected_entities=detected_entities
                )

                pred_res = predictor.predict_lead(
                    inquiry_text=cleaned_data["inquiry_text"],
                    primary_category=primary_cat,
                    budget_tier=cleaned_data["budget_tier"],
                    timeline=cleaned_data["timeline"],
                    company_size=cleaned_data["company_size"],
                    source_channel=cleaned_data["source_channel"],
                    qualification_score=score_res.total_score
                )

                record = {
                    **cleaned_data,
                    "primary_category": primary_cat,
                    "extracted_keywords": extracted_kws,
                    "qualification_score": score_res.total_score,
                    "qualification_tier": score_res.qualification_tier,
                    "score_reasons": "\n".join(score_res.scoring_reasons),
                    "conversion_probability": pred_res.conversion_probability,
                    "ml_prediction_label": pred_res.likelihood_label,
                    "lead_status": "New",
                    "notes": ""
                }

                new_id = insert_lead(record)
                st.success(f"Lead successfully evaluated and saved to database with ID #{new_id}!")

                # Live Evaluation Card
                st.markdown("#### Live Analysis Summary")
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric("Detected Category", primary_cat)
                with res_col2:
                    st.metric("Qualification Score", f"{score_res.total_score}/100", f"Tier: {score_res.qualification_tier}")
                with res_col3:
                    st.metric("Predicted Conversion", f"{pred_res.conversion_probability:.1%}", pred_res.likelihood_label)

    with tab_batch:
        st.subheader("Batch CSV Lead Intake")
        st.markdown(
            "Upload a CSV file containing incoming inquiries. Required columns: "
            "`name`, `email`, `inquiry_text`. Optional: `company`, `role`, `budget_tier`, `timeline`, `company_size`, `source_channel`."
        )

        # Download Sample Template
        if SAMPLE_LEADS_PATH.exists():
            with open(SAMPLE_LEADS_PATH, "rb") as f:
                st.download_button(
                    "📥 Download Sample CSV Template",
                    data=f.read(),
                    file_name="sample_cynbit_leads.csv",
                    mime="text/csv"
                )

        uploaded_file = st.file_uploader("Select CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Previewing {len(batch_df)} uploaded records:")
                st.dataframe(batch_df.head(5), use_container_width=True)

                if st.button("Process & Import Batch Into LeadIQ", type="primary"):
                    imported_count = 0
                    skipped_count = 0

                    with st.spinner("Processing batch through NLP and ML pipelines..."):
                        for _, row in batch_df.iterrows():
                            raw_dict = row.to_dict()
                            cleaned_data, errors = sanitize_input_record(raw_dict)
                            if errors:
                                skipped_count += 1
                                continue

                            primary_cat, cat_scores, detected_entities = detect_service_categories(cleaned_data["inquiry_text"])
                            kw_scores = extract_keywords_tf(cleaned_data["inquiry_text"], top_n=6)
                            extracted_kws = ", ".join([f"{w} ({s})" for w, s in kw_scores])

                            score_res = compute_qualification_score(
                                inquiry_text=cleaned_data["inquiry_text"],
                                budget_tier=cleaned_data["budget_tier"],
                                timeline=cleaned_data["timeline"],
                                company_size=cleaned_data["company_size"],
                                email=cleaned_data["email"],
                                role=cleaned_data["role"],
                                source_channel=cleaned_data["source_channel"],
                                detected_entities=detected_entities
                            )

                            pred_res = predictor.predict_lead(
                                inquiry_text=cleaned_data["inquiry_text"],
                                primary_category=primary_cat,
                                budget_tier=cleaned_data["budget_tier"],
                                timeline=cleaned_data["timeline"],
                                company_size=cleaned_data["company_size"],
                                source_channel=cleaned_data["source_channel"],
                                qualification_score=score_res.total_score
                            )

                            record = {
                                **cleaned_data,
                                "primary_category": primary_cat,
                                "extracted_keywords": extracted_kws,
                                "qualification_score": score_res.total_score,
                                "qualification_tier": score_res.qualification_tier,
                                "score_reasons": "\n".join(score_res.scoring_reasons),
                                "conversion_probability": pred_res.conversion_probability,
                                "ml_prediction_label": pred_res.likelihood_label,
                                "lead_status": "New",
                                "notes": "Imported via CSV batch"
                            }
                            insert_lead(record)
                            imported_count += 1

                    st.success(f"Batch Processing Complete: {imported_count} leads imported successfully ({skipped_count} skipped due to invalid inputs).")
            except Exception as e:
                st.error(f"Error parsing CSV file: {str(e)}")


# =========================================================
# PAGE 3: LEAD INTELLIGENCE
# =========================================================
elif selected_page == "Lead Intelligence":
    st.title("Lead Intelligence & NLP Diagnostics")
    st.markdown("Deep dive into raw text, extracted keywords, service categorization, and explainable score breakdown.")

    df = get_all_leads_df()
    if df.empty:
        st.warning("No leads found. Please add or load demo leads first.")
    else:
        lead_options = {f"#{row['id']} - {row['name']} ({row['company']})": row['id'] for _, row in df.iterrows()}
        selected_label = st.selectbox("Select Lead to Inspect:", list(lead_options.keys()))
        selected_id = lead_options[selected_label]
        lead = get_lead_by_id(selected_id)

        if lead:
            # Header info
            c_head1, c_head2, c_head3 = st.columns([2, 1, 1])
            with c_head1:
                st.subheader(f"{lead['name']} • {lead['company']}")
                st.caption(f"Role: **{lead['role']}** | Email: **{lead['email']}** | Channel: **{lead['source_channel']}**")
            with c_head2:
                tier = lead["qualification_tier"]
                badge_class = "badge-hot" if tier == "Hot" else ("badge-warm" if tier == "Warm" else "badge-cold")
                st.markdown(f"Qualification Tier:<br><span class='{badge_class}'>{tier.upper()} ({lead['qualification_score']}/100)</span>", unsafe_allow_html=True)
            with c_head3:
                st.metric("Conversion Probability", f"{lead['conversion_probability']:.1%}", lead['ml_prediction_label'])

            st.markdown("---")

            # Section 1: Inquiry Text & NLP
            col_text, col_nlp = st.columns(2)
            with col_text:
                st.markdown("#### Inquiry Text (Raw & Cleaned)")
                st.info(f"**Raw Submission:**\n\n\"{lead['inquiry_text']}\"")
                with st.expander("View Sanitized / Normalized Text"):
                    st.code(clean_text(lead['inquiry_text']), language="text")

            with col_nlp:
                st.markdown("#### Extracted Keywords & Entities")
                primary_cat, cat_scores, entities = detect_service_categories(lead["inquiry_text"])
                st.markdown(f"**Primary Category:** `{primary_cat}`")

                if entities:
                    st.markdown("**Detected Domain Entities:** " + " ".join([f"`{e}`" for e in entities]))
                else:
                    st.caption("No specific tech stack keywords recognized.")

                st.markdown("**Top Term Frequency Keywords:**")
                st.write(lead["extracted_keywords"])

                # Category Confidence Bar
                cat_df = pd.DataFrame(list(cat_scores.items()), columns=["Category", "Confidence"])
                fig_conf = px.bar(cat_df, x="Confidence", y="Category", orientation="h", color="Confidence", color_continuous_scale="Blues")
                fig_conf.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220, showlegend=False)
                st.plotly_chart(fig_conf, use_container_width=True)

            st.markdown("---")

            # Section 2: Rule-Based Qualification Scoring Breakdown & Radar Chart
            st.markdown("#### 📊 Multi-Dimensional Deal Health & Qualification Scoring")
            st.caption("Detailed 6-dimension evaluation compared against Cynbit's benchmark ideal closed-won deal profile.")

            score_res = compute_qualification_score(
                inquiry_text=lead["inquiry_text"],
                budget_tier=lead["budget_tier"],
                timeline=lead["timeline"],
                company_size=lead["company_size"],
                email=lead["email"],
                role=lead["role"],
                source_channel=lead["source_channel"],
                detected_entities=entities
            )

            col_radar, col_bars = st.columns([1.2, 1.8])
            with col_radar:
                # Plotly Radar Chart
                categories = ["Clarity", "Urgency", "Budget", "Scale", "Tech Scope", "Stakeholder"]
                lead_pcts = [
                    (score_res.clarity_score / 20) * 100,
                    (score_res.urgency_score / 20) * 100,
                    (score_res.budget_score / 20) * 100,
                    (score_res.company_size_score / 15) * 100,
                    (score_res.scope_depth_score / 15) * 100,
                    (score_res.engagement_score / 10) * 100
                ]
                ideal_pcts = [85, 90, 85, 80, 85, 80]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=ideal_pcts,
                    theta=categories,
                    fill='toself',
                    name='Ideal Benchmark',
                    line_color='rgba(148, 163, 184, 0.6)',
                    fillcolor='rgba(226, 232, 240, 0.3)'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=lead_pcts,
                    theta=categories,
                    fill='toself',
                    name='This Lead',
                    line_color='#2563EB',
                    fillcolor='rgba(37, 99, 235, 0.25)'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    height=320,
                    margin=dict(t=25, b=25, l=35, r=35)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with col_bars:
                sb1, sb2 = st.columns(2)
                with sb1:
                    st.markdown(f"**Requirement Clarity:** {score_res.clarity_score} / 20")
                    st.progress(score_res.clarity_score / 20)
                    st.markdown(f"**Urgency / Timeline:** {score_res.urgency_score} / 20")
                    st.progress(score_res.urgency_score / 20)
                    st.markdown(f"**Budget Availability:** {score_res.budget_score} / 20")
                    st.progress(score_res.budget_score / 20)
                with sb2:
                    st.markdown(f"**Company Scale:** {score_res.company_size_score} / 15")
                    st.progress(score_res.company_size_score / 15)
                    st.markdown(f"**Technical Alignment:** {score_res.scope_depth_score} / 15")
                    st.progress(score_res.scope_depth_score / 15)
                    st.markdown(f"**Stakeholder Quality:** {score_res.engagement_score} / 10")
                    st.progress(score_res.engagement_score / 10)

                st.markdown("##### Scoring Evidence:")
                for reason in score_res.scoring_reasons[:4]:
                    st.markdown(f"- {reason}")

            st.markdown("---")

            # Section 3: Semantic Similarity Engine
            st.markdown("#### 🔍 Semantic Similarity: Analogous Historical Inquiries")
            st.caption("Matches this inquiry against historical leads using cosine similarity over TF-IDF embeddings.")

            similar_leads = find_similar_leads(
                query_text=lead["inquiry_text"],
                leads_df=df,
                current_lead_id=lead["id"],
                top_k=3
            )

            if similar_leads:
                s_cols = st.columns(len(similar_leads))
                for idx, sim in enumerate(similar_leads):
                    with s_cols[idx]:
                        st.markdown(f"""
                        <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;'>
                            <div style='display:flex;justify-content:space-between;'>
                                <span style='font-weight:700;color:#0f172a;'>#{sim['id']} {sim['name']}</span>
                                <span style='background:#dbeafe;color:#1e40af;font-size:0.75rem;padding:2px 8px;border-radius:12px;font-weight:700;'>{sim['similarity_score']}% Match</span>
                            </div>
                            <div style='font-size:0.8rem;color:#64748b;margin:4px 0;'>{sim['company']} • {sim['primary_category']}</div>
                            <div style='font-size:0.8rem;'><b>Status:</b> <code>{sim['lead_status']}</code></div>
                            <div style='font-size:0.78rem;color:#475569;margin-top:4px;'><b>Shared Terms:</b> {', '.join(sim['shared_keywords']) if sim['shared_keywords'] else 'Contextual overlap'}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.caption("No additional historical records available for similarity ranking.")


# =========================================================
# PAGE 4: CONVERSION PREDICTION & WHAT-IF SIMULATOR
# =========================================================
elif selected_page == "Conversion Prediction":
    st.title("ML Conversion Likelihood & What-If Simulator")
    st.markdown("Empirical conversion probability estimation and real-time requirement optimization sandbox.")

    df = get_all_leads_df()
    if df.empty:
        st.warning("No leads available.")
    else:
        lead_options = {f"#{row['id']} - {row['name']} ({row['company']})": row['id'] for _, row in df.iterrows()}
        selected_label = st.selectbox("Select Target Lead:", list(lead_options.keys()))
        selected_id = lead_options[selected_label]
        lead = get_lead_by_id(selected_id)

        if lead:
            tab_diag, tab_sim = st.tabs(["📈 ML Diagnostic Analysis", "🧪 Interactive What-If Scenario Simulator"])

            with tab_diag:
                pred_res = predictor.predict_lead(
                    inquiry_text=lead["inquiry_text"],
                    primary_category=lead["primary_category"],
                    budget_tier=lead["budget_tier"],
                    timeline=lead["timeline"],
                    company_size=lead["company_size"],
                    source_channel=lead["source_channel"],
                    qualification_score=lead["qualification_score"]
                )

                # Conversion Gauge & Prediction Cards
                p_col1, p_col2 = st.columns([1, 1])
                with p_col1:
                    prob = pred_res.conversion_probability
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%"},
                        title={"text": "Estimated Conversion Probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#2563EB"},
                            "steps": [
                                {"range": [0, 35], "color": "#f1f5f9"},
                                {"range": [35, 65], "color": "#fef3c7"},
                                {"range": [65, 100], "color": "#dcfce7"}
                            ],
                            "threshold": {
                                "line": {"color": "#ef4444", "width": 3},
                                "thickness": 0.75,
                                "value": 50
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with p_col2:
                    st.markdown("#### Classification & Influencing Factors")
                    st.markdown(f"**Predicted Outcome:** `{pred_res.likelihood_label}`")
                    st.markdown(f"**Model Artifact:** `{pred_res.model_version}`")

                    st.markdown("##### Key Positive Drivers:")
                    for pf in pred_res.top_positive_factors:
                        st.markdown(f"- 🟢 {pf}")

                    if pred_res.top_negative_factors and pred_res.top_negative_factors != ["None identified"]:
                        st.markdown("##### Key Negative Drivers:")
                        for nf in pred_res.top_negative_factors:
                            st.markdown(f"- 🔴 {nf}")

                st.markdown("---")

                # Score vs ML Comparison Matrix
                st.subheader("⚖️ Strategic Comparison: Rule Qualification vs ML Conversion")
                st.markdown(
                    "A foundational feature of Cynbit LeadIQ is distinguishing **rule-based strategic qualification** "
                    "(which evaluates policy fit, budget, and business size) from **statistical ML prediction** "
                    "(which detects empirical conversion patterns in inquiry text and intent signals)."
                )

                comparison = compare_qualification_and_ml(lead["qualification_score"], pred_res.conversion_probability)
                m1, m2, m3 = st.columns([1, 1, 2])
                with m1:
                    st.metric("Rule Score Tier", f"{comparison['rule_tier']} ({lead['qualification_score']}/100)")
                with m2:
                    st.metric("ML Conversion Tier", f"{comparison['ml_tier']} ({pred_res.conversion_probability:.1%})")
                with m3:
                    st.info(f"**Synthesis Analysis:**\n\n{comparison['analysis']}")

            with tab_sim:
                st.subheader("🧪 Live Lead Optimization Sandbox")
                st.markdown("Adjust parameters below to simulate how scope adjustments or budget discussions impact qualification and conversion probability in real time.")

                sim_col1, sim_col2 = st.columns(2)
                with sim_col1:
                    sim_budget = st.selectbox(
                        "Simulated Budget Tier",
                        BUDGET_TIERS,
                        index=BUDGET_TIERS.index(lead["budget_tier"]) if lead["budget_tier"] in BUDGET_TIERS else 0
                    )
                    sim_timeline = st.selectbox(
                        "Simulated Urgency / Timeline",
                        TIMELINE_OPTIONS,
                        index=TIMELINE_OPTIONS.index(lead["timeline"]) if lead["timeline"] in TIMELINE_OPTIONS else 0
                    )
                    sim_size = st.selectbox(
                        "Simulated Company Size",
                        COMPANY_SIZES,
                        index=COMPANY_SIZES.index(lead["company_size"]) if lead["company_size"] in COMPANY_SIZES else 0
                    )
                    sim_role = st.text_input("Simulated Decision-Maker Role", value=lead["role"])

                with sim_col2:
                    sim_text = st.text_area(
                        "Simulated Inquiry Requirements (Add architectural/tech specifications):",
                        value=lead["inquiry_text"],
                        height=195
                    )

                # Compute Simulation
                sim_res = simulate_lead_modifications(
                    predictor=predictor,
                    base_lead=lead,
                    new_inquiry_text=sim_text,
                    new_budget=sim_budget,
                    new_timeline=sim_timeline,
                    new_company_size=sim_size,
                    new_role=sim_role
                )

                st.markdown("#### ⚡ Simulation Results & Delta Impact")
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.metric(
                        "Baseline Score",
                        f"{sim_res['baseline']['score']}/100",
                        f"Tier: {sim_res['baseline']['tier']}"
                    )
                with r2:
                    score_delta = sim_res['deltas']['score_delta']
                    st.metric(
                        "Simulated Score",
                        f"{sim_res['simulated']['score']}/100",
                        f"{'+' if score_delta >= 0 else ''}{score_delta} pts"
                    )
                with r3:
                    st.metric(
                        "Baseline ML Probability",
                        f"{sim_res['baseline']['probability']:.1%}"
                    )
                with r4:
                    prob_delta = sim_res['deltas']['probability_delta']
                    st.metric(
                        "Simulated ML Probability",
                        f"{sim_res['simulated']['probability']:.1%}",
                        f"{'+' if prob_delta >= 0 else ''}{prob_delta:.1%}"
                    )

                st.markdown("##### 💡 Prescriptive Scoping Recommendations:")
                for presc in sim_res["prescriptions"]:
                    st.markdown(f"- 🚀 {presc}")


# =========================================================
# PAGE 5: RESPONSE GENERATOR
# =========================================================
elif selected_page == "Response Generator":
    st.title("Deterministic Response Generator")
    st.markdown("Generate tailored, professional outreach and discovery replies based on extracted requirements and urgency.")

    st.markdown("""
    <div class='disclaimer-banner'>
        ℹ️ <b>Transparency Notice:</b> Responses are generated via structured, deterministic templates.
        No generative LLM or third-party cloud API is used, eliminating hallucinations and latency while ensuring total compliance.
    </div>
    """, unsafe_allow_html=True)

    df = get_all_leads_df()
    if df.empty:
        st.warning("No leads found. Please load demo leads or create a new lead.")
    else:
        lead_options = {f"#{row['id']} - {row['name']} ({row['company']})": row['id'] for _, row in df.iterrows()}
        selected_label = st.selectbox("Select Target Lead:", list(lead_options.keys()))
        selected_id = lead_options[selected_label]
        lead = get_lead_by_id(selected_id)

        if lead:
            _, _, entities = detect_service_categories(lead["inquiry_text"])

            col_cfg, col_preview = st.columns([1, 2])
            with col_cfg:
                st.markdown("#### Template Settings")
                template_type = st.selectbox(
                    "Response Purpose / Tone:",
                    [
                        "Auto-Select",
                        "Standard Discovery",
                        "High-Urgency Fast-Track",
                        "Budget & Scope Clarification",
                        "Technical Deep Dive"
                    ]
                )
                sender_name = st.text_input("Sender Name", value="Cynbit Client Solutions")
                sender_title = st.text_input("Sender Title", value="Technical Solutions Director")
                meeting_link = st.text_input("Calendar / Scheduling URL", value="https://cynbit.com/discovery")

                st.markdown("##### Extracted Context:")
                st.markdown(f"- **Category:** `{lead['primary_category']}`")
                st.markdown(f"- **Timeline:** `{lead['timeline']}`")
                st.markdown(f"- **Budget:** `{lead['budget_tier']}`")

            with col_preview:
                st.markdown("#### Generated Client Email Draft")
                resp_output = generate_lead_response(
                    client_name=lead["name"],
                    company=lead["company"],
                    primary_category=lead["primary_category"],
                    timeline=lead["timeline"],
                    budget_tier=lead["budget_tier"],
                    detected_entities=entities,
                    response_type=template_type,
                    sender_name=sender_name,
                    sender_title=sender_title,
                    meeting_link=meeting_link
                )

                st.markdown(f"""
                <div class='email-client-box'>
                    <div class='email-header-row'>
                        <b>To:</b> {lead['name']} &lt;{lead['email']}&gt;<br>
                        <b>From:</b> {sender_name} &lt;solutions@cynbit.com&gt;<br>
                        <b>Subject:</b> {resp_output['subject']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                edited_body = st.text_area("Email Content (Editable)", value=resp_output["body"], height=280)

                st.download_button(
                    label="💾 Download Formatted Email (.txt)",
                    data=DISCLAIMER_HEADER + f"To: {lead['name']} <{lead['email']}>\nSubject: {resp_output['subject']}\n\n" + edited_body,
                    file_name=f"cynbit_reply_lead_{lead['id']}.txt",
                    mime="text/plain"
                )


# =========================================================
# PAGE 6: LEAD MANAGEMENT
# =========================================================
elif selected_page == "Lead Management":
    st.title("Lead Database & CRM Workflow")
    st.markdown("Search, filter, update sales stages, and export full lead intelligence records.")

    df_all = get_all_leads_df()

    # Filter Bar
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.5, 1, 1, 1])
    with f_col1:
        search_query = st.text_input("🔍 Search Leads", placeholder="Search by name, company, email, or inquiry...")
    with f_col2:
        status_filter = st.selectbox("Filter Status", ["All"] + LEAD_STATUS_OPTIONS)
    with f_col3:
        cat_filter = st.selectbox("Filter Category", ["All"] + SERVICE_CATEGORIES)
    with f_col4:
        tier_filter = st.selectbox("Filter Tier", ["All", "Hot", "Warm", "Cold"])

    filtered_df = get_all_leads_df(
        filter_status=status_filter if status_filter != "All" else None,
        filter_category=cat_filter if cat_filter != "All" else None,
        filter_tier=tier_filter if tier_filter != "All" else None,
        search_query=search_query if search_query else None
    )

    st.markdown(f"**Showing {len(filtered_df)} matching leads:**")

    if not filtered_df.empty:
        # Action Toolbar: CSV Export
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered Leads to CSV",
            data=csv_data,
            file_name="cynbit_processed_leads.csv",
            mime="text/csv"
        )

        display_cols = [
            "id", "name", "company", "role", "primary_category",
            "qualification_score", "qualification_tier", "conversion_probability",
            "lead_status", "created_at"
        ]
        st.dataframe(
            filtered_df[display_cols].style.format({"conversion_probability": "{:.1%}"}),
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("Update Lead Status & Sales Notes")
        u_col1, u_col2, u_col3 = st.columns([1, 1, 1])
        with u_col1:
            target_id = st.selectbox("Select Lead ID to Update:", filtered_df["id"].tolist())
        target_lead = get_lead_by_id(target_id)
        if target_lead:
            with u_col2:
                curr_status_idx = LEAD_STATUS_OPTIONS.index(target_lead["lead_status"]) if target_lead["lead_status"] in LEAD_STATUS_OPTIONS else 0
                new_status = st.selectbox("New Lead Status:", LEAD_STATUS_OPTIONS, index=curr_status_idx)
            with u_col3:
                new_notes = st.text_input("Internal CRM Notes:", value=target_lead["notes"] or "")

            col_up, col_del = st.columns([1, 4])
            with col_up:
                if st.button("Save Changes", type="primary"):
                    update_lead_status(target_id, new_status, new_notes)
                    st.success(f"Lead #{target_id} updated successfully.")
                    st.rerun()
            with col_del:
                if st.button("🗑️ Delete Lead"):
                    delete_lead(target_id)
                    st.warning(f"Lead #{target_id} removed.")
                    st.rerun()


# =========================================================
# PAGE 7: MODEL EVALUATION
# =========================================================
elif selected_page == "Model Evaluation":
    st.title("ML Pipeline Architecture & Evaluation Report")
    st.markdown("Rigorous benchmark evaluation metrics, confusion matrix, and feature attribution coefficients.")

    if METRICS_FILE_PATH.exists():
        with open(METRICS_FILE_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Overview Metrics Row
        e1, e2, e3, e4, e5 = st.columns(5)
        with e1:
            st.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
        with e2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with e3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with e4:
            st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
        with e5:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

        st.markdown("---")

        # Confusion Matrix & Dataset Split
        c_cm, c_split = st.columns([1, 1])
        with c_cm:
            st.subheader("Confusion Matrix (Test Split)")
            cm_dict = metrics["confusion_matrix"]
            cm_matrix = np.array([
                [cm_dict["true_negatives"], cm_dict["false_positives"]],
                [cm_dict["false_negatives"], cm_dict["true_positives"]]
            ])
            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Class", y="True Label", color="Count"),
                x=["Non-Converted (0)", "Converted (1)"],
                y=["Non-Converted (0)", "Converted (1)"],
                text_auto=True,
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_split:
            st.subheader("Dataset & Split Integrity")
            st.markdown(f"- **Model Architecture:** `{metrics['model_type']}`")
            st.markdown(f"- **Training Samples:** `{metrics['train_samples']}` (80% Stratified Split)")
            st.markdown(f"- **Test Samples:** `{metrics['test_samples']}` (20% Holdout)")
            st.markdown(f"- **Random Seed:** `{metrics['random_seed']}` (Fully Reproducible)")
            st.markdown(f"- **Artifact Generated At:** `{metrics['trained_at']}`")
            st.markdown(f"- **Positive Class Ratio (Train):** `{metrics['positive_class_ratio_train']:.1%}`")
            st.markdown(f"- **Positive Class Ratio (Test):** `{metrics['positive_class_ratio_test']:.1%}`")

        st.markdown("---")

        # Feature Attribution
        st.subheader("Top Logistic Regression Coefficients (Feature Impact)")
        c_pos, c_neg = st.columns(2)
        with c_pos:
            st.markdown("##### 🟢 Top Positive Predictive Features")
            pos_df = pd.DataFrame(metrics.get("top_positive_features", []))
            if not pos_df.empty:
                st.dataframe(pos_df[["feature", "coefficient", "odds_ratio"]], use_container_width=True)

        with c_neg:
            st.markdown("##### 🔴 Top Negative Predictive Features")
            neg_df = pd.DataFrame(metrics.get("top_negative_features", []))
            if not neg_df.empty:
                st.dataframe(neg_df[["feature", "coefficient", "odds_ratio"]], use_container_width=True)

        st.markdown("---")
        st.subheader("Data Leakage Prevention & Methodological Notes")
        st.markdown("""
        1. **Strict Train/Test Isolation**: The Scikit-Learn `ColumnTransformer` (TF-IDF vectorizer, OneHotEncoder, and StandardScaler) is fitted strictly on the 80% training partition. It is applied to the 20% test partition via transform-only semantics.
        2. **Synthetic Benchmark Disclaimer**: All benchmark samples were generated using reproducible parametric distributions. High test accuracy reflects the structured consistency of the generative logic rather than real-world commercial performance.
        3. **Cold Start & Retraining**: When proprietary CRM history is available, the pipeline can be retrained directly with zero architectural changes.
        """)
    else:
        st.warning("Model metadata artifact not found. Please train the model via `python src/training.py`.")


# =========================================================
# PAGE 8: ABOUT & METHODOLOGY
# =========================================================
elif selected_page == "About & Methodology":
    st.title("About Cynbit LeadIQ & Architectural Methodology")
    st.markdown("System vision, AI/ML engineering design principles, and local offline deployment guarantees.")

    st.markdown("### 🏢 Executive Vision for Cynbit")
    st.markdown("""
    For software and AI consulting firms, incoming client inquiries vary wildly in technical maturity, budget readiness, and delivery urgency.
    Sales and solutions engineering teams waste hours triaging unqualified requests or writing repetitive discovery replies.
    
    **Cynbit LeadIQ** solves this challenge through an intelligent three-tier processing engine:
    1. **NLP Text Understanding**: Extracts domain entities and classifies service offerings without expensive LLM latency or API costs.
    2. **Explainable Rule-Based Qualification**: Evaluates commercial feasibility, urgency, and budget against transparent corporate policy.
    3. **Empirical ML Conversion Prediction**: Estimates conversion likelihood based on linguistic intent cues and historical win patterns.
    """)

    st.markdown("---")
    st.markdown("### ⚙️ System Architecture")
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Intake
            A1[Manual Form] --> B[Data Sanitizer & Validation]
            A2[CSV Batch] --> B
        end

        subgraph NLP Engine
            B --> C1[Text Normalization]
            C1 --> C2[TF-IDF Keyword Extraction]
            C1 --> C3[Domain Entity Matcher]
            C3 --> C4[Category Classification]
        end

        subgraph Dual-Engine Decision
            C4 --> D1[Rule-Based Scorer 0-100]
            D1 --> D2[Hot / Warm / Cold Tier]
            C1 & C4 & D1 --> E1[ML ColumnTransformer Pipeline]
            E1 --> E2[Logistic Regression Classifier]
            E2 --> E3[Conversion Probability & Odds Ratios]
        end

        subgraph Persistence & CRM
            D2 & E3 --> F[SQLite cynbit_leads.db]
            F --> G1[Interactive Streamlit Dashboard]
            F --> G2[Template-Based Reply Generator]
            F --> G3[CSV Intelligence Export]
        end
    ```
    """)

    st.markdown("---")
    st.markdown("### 🔒 Privacy, Security & Local Execution")
    st.markdown("""
    - **100% Offline & Free**: Operates without external cloud endpoints, proprietary LLM APIs, or network requests.
    - **No Cloud Telemetry or Secret Leakage**: All data stays on the local workstation in SQLite.
    - **SQL Injection Prevention**: All queries utilize parameterized SQLite calls.
    - **Explainable by Design**: Every score and prediction provides auditable reasons and feature attributions.
    """)
