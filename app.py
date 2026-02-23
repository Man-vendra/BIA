import gradio as gr
import pandas as pd
import pickle
import statsmodels.api as sm

# ─────────────────────────────────────────────────────────────
# 1. LOAD YOUR MODEL
#    Save in notebook first:
#       import pickle
#       with open("logit_model.pkl", "wb") as f:
#           pickle.dump(logit_model, f)
# ─────────────────────────────────────────────────────────────
with open("final_logit.pkl", "rb") as f:
    model = pickle.load(f)


# ─────────────────────────────────────────────────────────────
# 2. DROPDOWN OPTIONS
#    Add all possible values your data has for each column
# ─────────────────────────────────────────────────────────────
CHECKING_ACC_OPTIONS  = ["A11", "A12", "A13", "A14"]   # A13 → one-hot 1, else 0
CREDIT_HISTORY_OPTIONS = ["A30", "A31", "A32", "A33", "A34"]  # A34 → one-hot 1, else 0
SAVINGS_ACC_OPTIONS   = ["A61", "A62", "A63", "A64", "A65"]   # A65 → one-hot 1, else 0


# ─────────────────────────────────────────────────────────────
# 3. PREDICTION FUNCTION
#    UI collects all fields; model only receives what it needs
# ─────────────────────────────────────────────────────────────
def predict(
    checking_acc,
    duration,
    amount,
    inst_rate,
    age,
    credit_history,
    savings_acc,
):
    # One-hot encode exactly as model expects

    
    features = {
        "duration":            duration,
        "amount":              amount,
        "inst_rate":           inst_rate,
        "age":                 age,
        "checkin_acc_A13":     1 if checking_acc == "A13" else 0,
        "checkin_acc_A14":     1 if checking_acc == "A14" else 0,
        "credit_history_A34":  1 if credit_history == "A34" else 0,
        "savings_acc_A65":     1 if savings_acc == "A65" else 0,
    }

    # Exact column order the model was trained on
    col_order = [
        "duration", "amount", "inst_rate", "age",
        "checkin_acc_A13", "checkin_acc_A14",
        "credit_history_A34", "savings_acc_A65"
    ]

    X = pd.DataFrame([features])[col_order].astype(float)
    X = sm.add_constant(X, has_constant="add")  # statsmodels needs constant

    prob = model.predict(X)[0]
    label = "✅  Yes — Good Credit" if prob >= 0.5 else "❌  No — Bad Credit"
    confidence = f"{prob * 100:.1f}%"

    return label, confidence


# ─────────────────────────────────────────────────────────────
# 4. GRADIO UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="Credit Risk Predictor") as demo:

    gr.Markdown("## 🏦 Credit Risk Predictor\nFill in the applicant details and click **Predict**.")

    with gr.Row():

        with gr.Column():
            gr.Markdown("### 💳 Account Info")
            checking_acc   = gr.Dropdown(
                label="Checking Account Status",
                choices=CHECKING_ACC_OPTIONS,
                value="A11",
                info="A13 = ≥ 200 DM | A14 = No account | Others = lower balance"
            )
            savings_acc    = gr.Dropdown(
                label="Savings Account",
                choices=SAVINGS_ACC_OPTIONS,
                value="A61",
                info="A65 = Unknown / No savings account"
            )
            credit_history = gr.Dropdown(
                label="Credit History",
                choices=CREDIT_HISTORY_OPTIONS,
                value="A32",
                info="A34 = Critical / Other credits existing"
            )

        with gr.Column():
            gr.Markdown("### 👤 Applicant Details")
            age       = gr.Number(label="Age", value=30, minimum=18, maximum=100, step=1)
            duration  = gr.Number(label="Loan Duration (months)", value=12, minimum=1, maximum=120, step=1)
            amount    = gr.Number(label="Loan Amount (DM)", value=1000, minimum=0, step=100)
            inst_rate = gr.Number(label="Instalment Rate (% of income)", value=3, minimum=1, maximum=4, step=1)

    predict_btn = gr.Button("🔍  Predict", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("### 📊 Result")
    with gr.Row():
        output_label = gr.Textbox(label="Prediction",   interactive=False, scale=2)
        output_conf  = gr.Textbox(label="Probability",  interactive=False, scale=1)

    predict_btn.click(
        fn=predict,
        inputs=[
            checking_acc, duration, amount,
            inst_rate, age, credit_history, savings_acc
        ],
        outputs=[output_label, output_conf]
    )

    gr.Markdown(
        "> **Note:** The model internally converts your selections into one-hot encoded inputs. "
        "Only `duration`, `amount`, `inst_rate`, `age`, `checkin_acc_A13`, `checkin_acc_A14`, "
        "`credit_history_A34`, and `savings_acc_A65` are passed to the model."
    )

if __name__ == "__main__":
    demo.launch()
