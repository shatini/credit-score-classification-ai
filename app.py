"""
Gradio demo for Credit Score Classification.

Deployed on Hugging Face Spaces.
Upload lightgbm.joblib and scaler.joblib to the Space root.
"""

import gradio as gr
import joblib
import numpy as np

# ── Model & scaler ──────────────────────────────────────────
clf = joblib.load("lightgbm.joblib")
scaler = joblib.load("scaler.joblib")

CLASS_NAMES = ["Poor", "Standard", "Good"]
CLASS_COLORS = {"Poor": "red", "Standard": "orange", "Good": "green"}

# Label-encoder mappings (alphabetical order, matches LabelEncoder.fit_transform)
OCCUPATION_MAP = {
    "Accountant": 0, "Architect": 1, "Developer": 2, "Doctor": 3,
    "Engineer": 4, "Entrepreneur": 5, "Journalist": 6, "Lawyer": 7,
    "Manager": 8, "Mechanic": 9, "Media_Manager": 10, "Musician": 11,
    "Scientist": 12, "Teacher": 13, "Writer": 14,
}
CREDIT_MIX_MAP = {"Bad": 0, "Good": 1, "Standard": 2}
PAYMENT_MIN_MAP = {"NM": 0, "No": 1, "Yes": 2}
PAYMENT_BEHAVIOUR_MAP = {
    "High_spent_Large_value_payments": 0,
    "High_spent_Medium_value_payments": 1,
    "High_spent_Small_value_payments": 2,
    "Low_spent_Large_value_payments": 3,
    "Low_spent_Medium_value_payments": 4,
    "Low_spent_Small_value_payments": 5,
}

# Feature order must match training (config.NUMERIC_FEATURES + CATEGORICAL + Credit_History_Age)
FEATURE_ORDER = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries",
    "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Total_EMI_per_month", "Amount_invested_monthly",
    "Monthly_Balance",
    "Occupation", "Credit_Mix", "Payment_of_Min_Amount",
    "Payment_Behaviour",
    "Credit_History_Age",
]


def predict(
    age, annual_income, monthly_salary, num_bank_accounts, num_credit_card,
    interest_rate, num_loans, delay_days, num_delayed_payments,
    changed_credit_limit, num_inquiries, outstanding_debt,
    credit_utilization, total_emi, amount_invested, monthly_balance,
    occupation, credit_mix, payment_min, payment_behaviour,
    credit_history_years,
):
    features = np.array([[
        age, annual_income, monthly_salary, num_bank_accounts, num_credit_card,
        interest_rate, num_loans, delay_days, num_delayed_payments,
        changed_credit_limit, num_inquiries, outstanding_debt,
        credit_utilization, total_emi, amount_invested, monthly_balance,
        OCCUPATION_MAP.get(occupation, 0),
        CREDIT_MIX_MAP.get(credit_mix, 2),
        PAYMENT_MIN_MAP.get(payment_min, 2),
        PAYMENT_BEHAVIOUR_MAP.get(payment_behaviour, 4),
        credit_history_years * 12,  # convert years → months
    ]], dtype=np.float32)

    X_scaled = scaler.transform(features)
    proba = clf.predict_proba(X_scaled)[0]

    return {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}


# ── Gradio interface ────────────────────────────────────────
with gr.Blocks(title="Credit Score Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Credit Score Classification\n"
        "Enter financial profile data to predict credit score category "
        "(**Poor** / **Standard** / **Good**).\n\n"
        "Model: LightGBM | Accuracy: 81.2% | F1: 0.808"
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Personal & Account Info")
            age = gr.Slider(18, 70, value=35, step=1, label="Age")
            annual_income = gr.Number(value=50000, label="Annual Income ($)")
            monthly_salary = gr.Number(value=3500, label="Monthly In-Hand Salary ($)")
            num_bank_accounts = gr.Slider(0, 15, value=4, step=1, label="Number of Bank Accounts")
            num_credit_card = gr.Slider(0, 12, value=3, step=1, label="Number of Credit Cards")
            occupation = gr.Dropdown(
                list(OCCUPATION_MAP.keys()), value="Engineer", label="Occupation",
            )
            credit_history_years = gr.Slider(0, 35, value=12, step=1, label="Credit History (years)")

        with gr.Column():
            gr.Markdown("### Debt & Payment Behaviour")
            interest_rate = gr.Slider(0, 35, value=10, step=1, label="Interest Rate (%)")
            num_loans = gr.Slider(0, 10, value=2, step=1, label="Number of Loans")
            outstanding_debt = gr.Number(value=1500, label="Outstanding Debt ($)")
            credit_utilization = gr.Slider(0, 50, value=30, step=0.5, label="Credit Utilization (%)")
            total_emi = gr.Number(value=50, label="Total EMI per Month ($)")
            credit_mix = gr.Dropdown(
                list(CREDIT_MIX_MAP.keys()), value="Standard", label="Credit Mix",
            )
            payment_behaviour = gr.Dropdown(
                list(PAYMENT_BEHAVIOUR_MAP.keys()),
                value="Low_spent_Medium_value_payments",
                label="Payment Behaviour",
            )

        with gr.Column():
            gr.Markdown("### Payment History")
            delay_days = gr.Slider(0, 60, value=5, step=1, label="Delay from Due Date (days)")
            num_delayed_payments = gr.Slider(0, 25, value=3, step=1, label="Number of Delayed Payments")
            changed_credit_limit = gr.Number(value=5, label="Changed Credit Limit (%)")
            num_inquiries = gr.Slider(0, 15, value=3, step=1, label="Number of Credit Inquiries")
            amount_invested = gr.Number(value=200, label="Amount Invested Monthly ($)")
            monthly_balance = gr.Number(value=300, label="Monthly Balance ($)")
            payment_min = gr.Dropdown(
                list(PAYMENT_MIN_MAP.keys()), value="Yes", label="Payment of Min Amount",
            )

    btn = gr.Button("Predict Credit Score", variant="primary", size="lg")
    output = gr.Label(num_top_classes=3, label="Prediction")

    btn.click(
        fn=predict,
        inputs=[
            age, annual_income, monthly_salary, num_bank_accounts, num_credit_card,
            interest_rate, num_loans, delay_days, num_delayed_payments,
            changed_credit_limit, num_inquiries, outstanding_debt,
            credit_utilization, total_emi, amount_invested, monthly_balance,
            occupation, credit_mix, payment_min, payment_behaviour,
            credit_history_years,
        ],
        outputs=output,
    )

    gr.Markdown(
        "---\n"
        "Built by [Nikolai Shatikhin](https://github.com/shatini) "
        "| [Source Code](https://github.com/shatini/credit-score-classification-ai)"
    )

if __name__ == "__main__":
    demo.launch()
