"""Streamlit presentation layer. Run with python -m streamlit run app.py."""
import logging
import pandas as pd
import streamlit as st
from risk_service import RiskService, extract_applicant_id
from explanations import FEATURE_LABELS, format_value, contribution_label, generate_explanation

logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="Credit Risk Explainer Agent", page_icon="💳")

@st.cache_resource
def get_service():
    return RiskService()


def display_applicant(data):
    features = data['features']
    st.subheader('Model Prediction')
    st.write(f"Applicant ID: {data['applicant_id']}")
    st.write(f"Predicted risk: {data['predicted_risk']}")
    st.write(f"Model-estimated probability of serious delinquency within two years: {data['predicted_probability']:.2%}")
    st.write(f"Classification threshold: {data['classification_threshold']:.2%}")
    if features.get('MonthlyIncome') is None:
        st.info('Monthly income is missing. Verify the recorded debt ratio before interpreting it as debt relative to income.')
    st.subheader('Observed Applicant Data')
    observed_table = pd.DataFrame([{'Feature': FEATURE_LABELS.get(name, name), 'Observed value': format_value(name, value)} for name, value in features.items()])
    st.table(observed_table)
    if data['missing_features']:
        st.caption('Unknown values are filled internally by the model using training-set medians.')
    st.subheader('Model Feature Contributions')
    st.write(f"SHAP baseline probability: {data['shap_baseline']:.2%}")
    ranked_factors = sorted(data['shap_factors'], key=lambda factor: abs(factor['contribution']), reverse=True)
    shap_table = pd.DataFrame([{'Feature': contribution_label(factor['feature']), 'Contribution (percentage points)': factor['contribution'] * 100} for factor in ranked_factors])
    st.dataframe(shap_table, column_config={'Contribution (percentage points)': st.column_config.NumberColumn('Contribution (percentage points)', format='%+.3f')}, hide_index=True)
    st.bar_chart(shap_table.set_index('Feature'), horizontal=True)
    st.caption('Positive contributions increase the prediction relative to the SHAP baseline; negative contributions decrease it. The baseline plus all contributions reconstructs the prediction. These describe model behavior, not causal effects.')
    st.subheader('Suggested Next Steps')
    if data['recommendations']:
        for recommendation in data['recommendations']:
            st.markdown(f'- {recommendation}')
    else:
        st.write('No specific suggestions from the supplementary rules.')
    st.caption('These suggestions come from manual rules. They do not guarantee a lower model prediction or loan approval.')

st.title("💳 Credit Risk Explainer Agent")
st.markdown("""This demo uses a trained Random Forest to estimate the probability of
serious delinquency within two years. It displays applicant data,
SHAP feature contributions, and supplementary rule-based suggestions.""")

query = st.text_input("Enter an Applicant ID or query:", placeholder="Why was applicant 1 flagged as high risk?")

if query:
    try:
        service = get_service()
        applicant_id = extract_applicant_id(query)
        if applicant_id is not None:
            with st.spinner(f"Analyzing Applicant {applicant_id}..."):
                data = service.lookup_applicant(applicant_id)
                if "error" in data:
                    st.error(data["error"])
                else:
                    display_applicant(data)
                    result = generate_explanation(data)
                    if result.source == "validated_llm":
                        st.caption("LLM structured output passed validation; wording rendered by Python.")
                    else:
                        st.caption("Showing a template explanation.")
                    st.subheader("Decision Explanation")
                    st.markdown(result.text)
        else:
            with st.spinner("Searching applicants..."):
                results = service.search_applicants(query)
                st.subheader("Retrieved Applicants")
                if isinstance(results, pd.DataFrame):
                    if results.empty:
                        st.info("No matching applicants found.")
                    else:
                        st.dataframe(results, column_config={
                            "predicted_probability": st.column_config.NumberColumn(
                                "Predicted probability (0–1)", format="%.4f")
                        }, hide_index=True)
                else:
                    st.write(results)
    except Exception as exc:
        st.error(f"Error: {exc}")

st.divider()
