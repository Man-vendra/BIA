import gradio as gr
import pandas as pd
import pickle
import statsmodels.api as sm
import os

# ─────────────────────────────────────────────────────────────
# 1. LOAD MODEL
# ─────────────────────────────────────────────────────────────
with open("renege_model.pkl", "rb") as f:
    model = pickle.load(f)

# ─────────────────────────────────────────────────────────────
# 2. DROPDOWN OPTIONS
# ─────────────────────────────────────────────────────────────
BAND_OPTIONS     = ["E1", "E2", "E3", "Other"]
GENDER_OPTIONS   = ["Male", "Female"]
SOURCE_OPTIONS   = ["Direct", "Employee Referral", "Others"]
LOB_OPTIONS      = ["BFSI", "CSMP", "EAS", "ERS", "ETS", "Healthcare", "INFRA", "MMS"]
LOCATION_OPTIONS = ["Bangalore", "Chennai", "Cochin", "Gurgaon", "Hyderabad",
                    "Kolkata", "Mumbai", "Noida", "Pune", "Others"]

# ─────────────────────────────────────────────────────────────
# 3. PREDICTION FUNCTION
#    Derived feature thresholds (from notebook cell 5):
#      short_acceptance        = Duration.to.accept.offer <= 3
#      long_acceptance         = Duration.to.accept.offer >= 30
#      long_notice             = Notice.period >= 60
#      hike_mismatch           = Percent.difference.CTC < -10
#      high_offer_hike         = Percent.hike.offered.in.CTC >= 100
#      exp_lt_3                = Rex.in.Yrs < 3
#      exp_gt_10               = Rex.in.Yrs > 10
#      hike_mismatch_long_notice = hike_mismatch AND long_notice
# ─────────────────────────────────────────────────────────────
def predict(
    duration, notice_period, hike_offered, pct_diff_ctc,
    experience, age, relocate, joining_bonus,
    offered_band, gender, candidate_source, lob, joining_location
):
    # --- Derived features (exact thresholds from notebook) ---
    short_acceptance          = 1 if duration <= 3 else 0
    long_acceptance           = 1 if duration >= 30 else 0
    long_notice               = 1 if notice_period >= 60 else 0
    hike_mismatch             = 1 if pct_diff_ctc < -10 else 0
    high_offer_hike           = 1 if hike_offered >= 100 else 0
    exp_lt_3                  = 1 if experience < 3 else 0
    exp_gt_10                 = 1 if experience > 10 else 0
    hike_mismatch_long_notice = 1 if (hike_mismatch == 1 and long_notice == 1) else 0

    # --- Build feature dict ---
    features = {
        # Numeric
        "Duration.to.accept.offer":    duration,
        "Notice.period":               notice_period,
        "Percent.hike.offered.in.CTC": hike_offered,
        "Percent.difference.CTC":      pct_diff_ctc,
        "Rex.in.Yrs":                  experience,
        "Age":                         age,

        # Binary
        "relocate":      1 if relocate == "Yes" else 0,
        "joining_bonus": 1 if joining_bonus == "Yes" else 0,

        # Derived
        "short_acceptance":          short_acceptance,
        "long_acceptance":           long_acceptance,
        "long_notice":               long_notice,
        "hike_mismatch":             hike_mismatch,
        "high_offer_hike":           high_offer_hike,
        "exp_lt_3":                  exp_lt_3,
        "exp_gt_10":                 exp_gt_10,
        "hike_mismatch_long_notice": hike_mismatch_long_notice,

        # Offered Band
        "Offered.band_E1": 1 if offered_band == "E1" else 0,
        "Offered.band_E2": 1 if offered_band == "E2" else 0,
        "Offered.band_E3": 1 if offered_band == "E3" else 0,

        # Gender
        "Gender_Male": 1 if gender == "Male" else 0,

        # Candidate Source
        "Candidate.Source_Direct":            1 if candidate_source == "Direct" else 0,
        "Candidate.Source_Employee Referral": 1 if candidate_source == "Employee Referral" else 0,

        # LOB
        "LOB_BFSI":       1 if lob == "BFSI" else 0,
        "LOB_CSMP":       1 if lob == "CSMP" else 0,
        "LOB_EAS":        1 if lob == "EAS" else 0,
        "LOB_ERS":        1 if lob == "ERS" else 0,
        "LOB_ETS":        1 if lob == "ETS" else 0,
        "LOB_Healthcare": 1 if lob == "Healthcare" else 0,
        "LOB_INFRA":      1 if lob == "INFRA" else 0,
        "LOB_MMS":        1 if lob == "MMS" else 0,

        # Joining Location
        "Joining Location_Bangalore": 1 if joining_location == "Bangalore" else 0,
        "Joining Location_Chennai":   1 if joining_location == "Chennai" else 0,
        "Joining Location_Cochin":    1 if joining_location == "Cochin" else 0,
        "Joining Location_Gurgaon":   1 if joining_location == "Gurgaon" else 0,
        "Joining Location_Hyderabad": 1 if joining_location == "Hyderabad" else 0,
        "Joining Location_Kolkata":   1 if joining_location == "Kolkata" else 0,
        "Joining Location_Mumbai":    1 if joining_location == "Mumbai" else 0,
        "Joining Location_Noida":     1 if joining_location == "Noida" else 0,
        "Joining Location_Others":    1 if joining_location == "Others" else 0,
        "Joining Location_Pune":      1 if joining_location == "Pune" else 0,
    }

    # Exact column order matching significant_vars from training
    col_order = [
        'Duration.to.accept.offer', 'Notice.period',
        'Percent.hike.offered.in.CTC', 'Percent.difference.CTC', 'relocate',
        'Rex.in.Yrs', 'Age', 'short_acceptance', 'long_acceptance',
        'long_notice', 'hike_mismatch', 'high_offer_hike', 'exp_lt_3',
        'exp_gt_10', 'joining_bonus', 'hike_mismatch_long_notice',
        'Offered.band_E1', 'Offered.band_E2', 'Offered.band_E3', 'Gender_Male',
        'Candidate.Source_Direct', 'Candidate.Source_Employee Referral',
        'LOB_BFSI', 'LOB_CSMP', 'LOB_EAS', 'LOB_ERS', 'LOB_ETS',
        'LOB_Healthcare', 'LOB_INFRA', 'LOB_MMS', 'Joining Location_Bangalore',
        'Joining Location_Chennai', 'Joining Location_Cochin',
        'Joining Location_Gurgaon', 'Joining Location_Hyderabad',
        'Joining Location_Kolkata', 'Joining Location_Mumbai',
        'Joining Location_Noida', 'Joining Location_Others',
        'Joining Location_Pune'
    ]

    X = pd.DataFrame([features])[col_order].astype(float)
    X = sm.add_constant(X, has_constant="add")

    prob = model.predict(X)[0]  # probability of reneging

    # Cost-based threshold = 0.35 (FN cost=5, FP cost=1)
    # Lower bar intentionally to catch more at-risk candidates
    if prob >= 0.35:
        label = "🔴  High Risk — Likely to Renege"
    else:
        label = "🟢  Low Risk — Likely to Join"

    # Risk flags for transparency
    flags = []
    if short_acceptance:          flags.append("⚡ Accepted very quickly (≤ 3 days)")
    if long_acceptance:           flags.append("🐢 Took long to accept (≥ 30 days)")
    if long_notice:               flags.append("📅 Long notice period (≥ 60 days)")
    if hike_mismatch:             flags.append("💸 Hike below expectation (CTC diff < −10%)")
    if high_offer_hike:           flags.append("🚀 Very high offer hike (≥ 100%)")
    if exp_lt_3:                  flags.append("🌱 Low experience (< 3 yrs)")
    if exp_gt_10:                 flags.append("🏆 High experience (> 10 yrs)")
    if hike_mismatch_long_notice: flags.append("⚠️  Hike mismatch + Long notice (high risk combo)")
    flags_str = "\n".join(flags) if flags else "✅ No significant risk flags detected"

    return label, f"{prob * 100:.1f}%", flags_str


# ─────────────────────────────────────────────────────────────
# 4. GRADIO UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Employee Renege Risk Predictor") as demo:

    gr.Markdown("## 👔 Employee Renege Risk Predictor\nPredict whether a candidate is likely to **renege** (not join) after accepting an offer.")

    with gr.Row():

        with gr.Column():
            gr.Markdown("### 📋 Offer Details")
            duration     = gr.Number(label="Duration to Accept Offer (days)", value=7, minimum=1, maximum=90, step=1,
                                     info="How many days the candidate took to accept")
            hike_offered = gr.Number(label="Hike Offered in CTC (%)", value=20, minimum=0, maximum=200, step=1,
                                     info="% hike offered over current CTC")
            pct_diff_ctc = gr.Number(label="Percent Difference in CTC (%)", value=0, minimum=-100, maximum=100, step=1,
                                     info="Offered vs Expected CTC. Negative = below expectation")
            offered_band  = gr.Dropdown(label="Offered Band", choices=BAND_OPTIONS, value="E2")
            joining_bonus = gr.Radio(label="Joining Bonus Offered?", choices=["Yes", "No"], value="No")

        with gr.Column():
            gr.Markdown("### 👤 Candidate Details")
            age           = gr.Number(label="Age", value=28, minimum=18, maximum=65, step=1)
            experience    = gr.Number(label="Relevant Experience (Years)", value=4, minimum=0, maximum=40, step=1)
            notice_period = gr.Number(label="Notice Period (days)", value=30, minimum=0, maximum=180, step=1)
            gender        = gr.Radio(label="Gender", choices=GENDER_OPTIONS, value="Male")
            relocate      = gr.Radio(label="Willing to Relocate?", choices=["Yes", "No"], value="No")

        with gr.Column():
            gr.Markdown("### 🏢 Organisation Details")
            candidate_source = gr.Dropdown(label="Candidate Source", choices=SOURCE_OPTIONS, value="Direct")
            lob              = gr.Dropdown(label="Line of Business (LOB)", choices=LOB_OPTIONS, value="ETS")
            joining_location = gr.Dropdown(label="Joining Location", choices=LOCATION_OPTIONS, value="Bangalore")

    predict_btn = gr.Button("🔍  Predict Renege Risk", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("### 📊 Prediction Result")
    with gr.Row():
        output_label = gr.Textbox(label="Risk Assessment",      interactive=False, scale=2)
        output_prob  = gr.Textbox(label="Renege Probability",   interactive=False, scale=1)
    output_flags = gr.Textbox(label="🚩 Risk Flags Detected", interactive=False, lines=5)

    predict_btn.click(
        fn=predict,
        inputs=[
            duration, notice_period, hike_offered, pct_diff_ctc,
            experience, age, relocate, joining_bonus,
            offered_band, gender, candidate_source, lob, joining_location
        ],
        outputs=[output_label, output_prob, output_flags]
    )

    gr.Markdown("""
    > **Risk flags are auto-computed from your inputs:**
    > Short acceptance ≤ 3 days &nbsp;|&nbsp; Long acceptance ≥ 30 days &nbsp;|&nbsp; Long notice ≥ 60 days
    > Hike mismatch: CTC diff < −10% &nbsp;|&nbsp; High offer hike ≥ 100%
    > Low exp < 3 yrs &nbsp;|&nbsp; High exp > 10 yrs
    """)

# ─────────────────────────────────────────────────────────────
# 5. LAUNCH
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft()
    )
