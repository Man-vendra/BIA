import gradio as gr
import pandas as pd
import pickle
import statsmodels.api as sm
import os
import requests

# ─────────────────────────────────────────────────────────────
# 1. DOWNLOAD MODEL FROM GOOGLE DRIVE AT STARTUP
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "renege_model.pkl"
FILE_ID    = "1fK8B0gQ96ZKB-2G0ucCeWV6zMPCPWmbY"

def download_model_from_drive(file_id, dest_path):
    print("Downloading model from Google Drive...")
    URL     = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print("Model downloaded successfully!")

if not os.path.exists(MODEL_PATH):
    download_model_from_drive(FILE_ID, MODEL_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully!")

# ─────────────────────────────────────────────────────────────
# 2. OPTIONS
# ─────────────────────────────────────────────────────────────
BAND_OPTIONS     = ["E1", "E2", "E3", "Other"]
GENDER_OPTIONS   = ["Male", "Female"]
SOURCE_OPTIONS   = ["Direct", "Employee Referral", "Others"]
LOB_OPTIONS      = ["BFSI", "CSMP", "EAS", "ERS", "ETS", "Healthcare", "INFRA", "MMS"]
LOCATION_OPTIONS = ["Bangalore", "Chennai", "Cochin", "Gurgaon", "Hyderabad",
                    "Kolkata", "Mumbai", "Noida", "Pune", "Others"]

# ─────────────────────────────────────────────────────────────
# 3. PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────
def predict(
    duration, notice_period, hike_offered, pct_diff_ctc,
    experience, age, relocate, joining_bonus,
    offered_band, gender, candidate_source, lob, joining_location
):
    short_acceptance          = 1 if duration <= 3 else 0
    long_acceptance           = 1 if duration >= 30 else 0
    long_notice               = 1 if notice_period >= 60 else 0
    hike_mismatch             = 1 if pct_diff_ctc < -10 else 0
    high_offer_hike           = 1 if hike_offered >= 100 else 0
    exp_lt_3                  = 1 if experience < 3 else 0
    exp_gt_10                 = 1 if experience > 10 else 0
    hike_mismatch_long_notice = 1 if (hike_mismatch == 1 and long_notice == 1) else 0

    features = {
        "Duration.to.accept.offer":    duration,
        "Notice.period":               notice_period,
        "Percent.hike.offered.in.CTC": hike_offered,
        "Percent.difference.CTC":      pct_diff_ctc,
        "Rex.in.Yrs":                  experience,
        "Age":                         age,
        "relocate":                    1 if relocate == "Yes" else 0,
        "joining_bonus":               1 if joining_bonus == "Yes" else 0,
        "short_acceptance":            short_acceptance,
        "long_acceptance":             long_acceptance,
        "long_notice":                 long_notice,
        "hike_mismatch":               hike_mismatch,
        "high_offer_hike":             high_offer_hike,
        "exp_lt_3":                    exp_lt_3,
        "exp_gt_10":                   exp_gt_10,
        "hike_mismatch_long_notice":   hike_mismatch_long_notice,
        "Offered.band_E1":             1 if offered_band == "E1" else 0,
        "Offered.band_E2":             1 if offered_band == "E2" else 0,
        "Offered.band_E3":             1 if offered_band == "E3" else 0,
        "Gender_Male":                 1 if gender == "Male" else 0,
        "Candidate.Source_Direct":            1 if candidate_source == "Direct" else 0,
        "Candidate.Source_Employee Referral": 1 if candidate_source == "Employee Referral" else 0,
        "LOB_BFSI":       1 if lob == "BFSI" else 0,
        "LOB_CSMP":       1 if lob == "CSMP" else 0,
        "LOB_EAS":        1 if lob == "EAS" else 0,
        "LOB_ERS":        1 if lob == "ERS" else 0,
        "LOB_ETS":        1 if lob == "ETS" else 0,
        "LOB_Healthcare": 1 if lob == "Healthcare" else 0,
        "LOB_INFRA":      1 if lob == "INFRA" else 0,
        "LOB_MMS":        1 if lob == "MMS" else 0,
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
    prob = model.predict(X)[0]

    if prob >= 0.35:
        verdict   = "HIGH RISK"
        icon      = "🔴"
        sub       = "Candidate likely to renege — recommend immediate intervention"
        bar_color = "#ef4444"
    else:
        verdict   = "LOW RISK"
        icon      = "🟢"
        sub       = "Candidate likely to join — standard follow-up recommended"
        bar_color = "#22c55e"

    flags = []
    if short_acceptance:          flags.append("⚡ Accepted very quickly (≤ 3 days)")
    if long_acceptance:           flags.append("🐢 Slow acceptance (≥ 30 days)")
    if long_notice:               flags.append("📅 Long notice period (≥ 60 days)")
    if hike_mismatch:             flags.append("💸 Hike below expectation (diff < −10%)")
    if high_offer_hike:           flags.append("🚀 Very high offer hike (≥ 100%)")
    if exp_lt_3:                  flags.append("🌱 Low experience (< 3 yrs)")
    if exp_gt_10:                 flags.append("🏆 High experience (> 10 yrs)")
    if hike_mismatch_long_notice: flags.append("⚠️ Hike mismatch + Long notice combo")
    flags_str = "  |  ".join(flags) if flags else "✅ No significant risk flags"

    result_html = f"""
    <div style="
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid {bar_color}40;
        border-left: 5px solid {bar_color};
        border-radius: 16px;
        padding: 28px 32px;
        font-family: 'DM Sans', sans-serif;
        box-shadow: 0 0 40px {bar_color}20;
    ">
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px;">
            <span style="font-size:2.8rem;">{icon}</span>
            <div>
                <div style="font-size:2rem; font-weight:800; color:{bar_color}; letter-spacing:2px;">{verdict}</div>
                <div style="font-size:0.95rem; color:#94a3b8; margin-top:2px;">{sub}</div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <div style="font-size:3rem; font-weight:900; color:white;">{prob*100:.1f}%</div>
                <div style="font-size:0.8rem; color:#64748b;">Renege Probability</div>
            </div>
        </div>
        <div style="background:#0f172a; border-radius:999px; height:10px; margin-bottom:20px;">
            <div style="background:{bar_color}; width:{prob*100:.1f}%; height:10px; border-radius:999px; transition:width 0.6s ease;"></div>
        </div>
        <div style="
            background:#1e293b;
            border-radius:10px;
            padding:14px 18px;
            font-size:0.88rem;
            color:#cbd5e1;
            line-height:1.8;
        ">
            <span style="color:#f59e0b; font-weight:700; margin-right:8px;">RISK FLAGS</span>{flags_str}
        </div>
        <div style="margin-top:12px; font-size:0.75rem; color:#475569; text-align:right;">
            Threshold: 0.35 · Cost-based (FN=5, FP=1)
        </div>
    </div>
    """
    return result_html


# ─────────────────────────────────────────────────────────────
# 4. CUSTOM CSS
# ─────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800;900&family=Space+Mono:wght@700&display=swap');

body, .gradio-container {
    background: #020817 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* BIG TITLE */
.big-title {
    text-align: center;
    padding: 32px 0 8px 0;
}
.big-title h1 {
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
    font-family: 'DM Sans', sans-serif !important;
}
.big-title p {
    color: #64748b;
    font-size: 1rem;
    margin-top: 6px;
}

/* SECTION HEADERS */
.section-head {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    color: #38bdf8 !important;
    text-transform: uppercase !important;
    padding: 0 0 8px 0 !important;
    border-bottom: 1px solid #1e293b !important;
    margin-bottom: 12px !important;
}

/* CARD COLUMNS */
.gr-block.gr-box, .gr-column {
    background: transparent !important;
}

/* INPUT CARDS */
.input-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 20px;
    height: 100%;
}

/* LABELS */
label span, .gr-form label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* INPUTS */
input[type=number], select, .gr-input, textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    padding: 10px 14px !important;
}
input[type=number]:focus, select:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px #38bdf820 !important;
}

/* RADIO BUTTONS */
.gr-radio label {
    color: #cbd5e1 !important;
    font-size: 0.9rem !important;
}

/* PREDICT BUTTON */
.predict-btn button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    padding: 16px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 24px #3b82f640 !important;
    transition: all 0.2s !important;
}
.predict-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px #3b82f660 !important;
}

/* OUTPUT HTML */
.output-html {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* DROPDOWN */
.gr-dropdown select {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
}

/* Hide gradio footer */
footer { display: none !important; }
.gr-prose { color: #64748b !important; }

/* Info text */
.gr-info { color: #475569 !important; font-size: 0.75rem !important; }
"""

# ─────────────────────────────────────────────────────────────
# 5. GRADIO UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Employee Renege Risk Predictor", css=css) as demo:

    # BIG TITLE
    gr.HTML("""
    <div class="big-title">
        <h1>👔 Employee Renege Risk Predictor</h1>
        <p>Predict whether a candidate is likely to <strong style="color:#f1f5f9">renege</strong> after accepting an offer · Powered by Random Forest Classifier · Threshold: 0.35</p>
    </div>
    """)

    # INPUT SECTION — 3 columns side by side
    with gr.Row(equal_height=True):

        # COL 1: Offer Details
        with gr.Column():
            gr.HTML('<div class="section-head">📋 &nbsp;Offer Details</div>')
            duration      = gr.Number(label="Duration to Accept Offer (days)", value=7, minimum=1, maximum=90, step=1,
                                      info="Days taken by candidate to accept")
            hike_offered  = gr.Number(label="Hike Offered in CTC (%)", value=20, minimum=0, maximum=200, step=1)
            pct_diff_ctc  = gr.Number(label="Percent Difference in CTC (%)", value=0, minimum=-100, maximum=100, step=1,
                                      info="Negative = offered below expectation")
            offered_band  = gr.Dropdown(label="Offered Band", choices=BAND_OPTIONS, value="E2")
            joining_bonus = gr.Radio(label="Joining Bonus Offered?", choices=["Yes", "No"], value="No")

        # COL 2: Candidate Details
        with gr.Column():
            gr.HTML('<div class="section-head">👤 &nbsp;Candidate Details</div>')
            age           = gr.Number(label="Age", value=28, minimum=18, maximum=65, step=1)
            experience    = gr.Number(label="Relevant Experience (Years)", value=4, minimum=0, maximum=40, step=1)
            notice_period = gr.Number(label="Notice Period (days)", value=30, minimum=0, maximum=180, step=1)
            gender        = gr.Radio(label="Gender", choices=GENDER_OPTIONS, value="Male")
            relocate      = gr.Radio(label="Willing to Relocate?", choices=["Yes", "No"], value="No")

        # COL 3: Org Details + Result
        with gr.Column():
            gr.HTML('<div class="section-head">🏢 &nbsp;Organisation Details</div>')
            candidate_source = gr.Dropdown(label="Candidate Source", choices=SOURCE_OPTIONS, value="Direct")
            lob              = gr.Dropdown(label="Line of Business (LOB)", choices=LOB_OPTIONS, value="ETS")
            joining_location = gr.Dropdown(label="Joining Location", choices=LOCATION_OPTIONS, value="Bangalore")

            gr.HTML('<div style="height:16px"></div>')

            # PREDICT BUTTON inside col 3
            with gr.Row(elem_classes="predict-btn"):
                predict_btn = gr.Button("🔍  Predict Renege Risk", variant="primary", size="lg")

    # RESULT — full width below, no scroll needed
    gr.HTML('<div style="height:16px"></div>')
    output_result = gr.HTML(
        value="""
        <div style="
            background:#0f172a;
            border:1px dashed #1e293b;
            border-radius:16px;
            padding:32px;
            text-align:center;
            color:#334155;
            font-family:'DM Sans',sans-serif;
            font-size:1rem;
        ">
            Fill in the candidate details above and click <strong style="color:#475569">Predict Renege Risk</strong> to see results here
        </div>
        """,
        elem_classes="output-html"
    )

    predict_btn.click(
        fn=predict,
        inputs=[
            duration, notice_period, hike_offered, pct_diff_ctc,
            experience, age, relocate, joining_bonus,
            offered_band, gender, candidate_source, lob, joining_location
        ],
        outputs=[output_result]
    )

# ─────────────────────────────────────────────────────────────
# 6. LAUNCH
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Base()
    )
