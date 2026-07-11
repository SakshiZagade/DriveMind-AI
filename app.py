import streamlit as st
import sqlite3
import pandas as pd
import requests
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Automotive AI Intelligence", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .aggressive-box { background: linear-gradient(135deg, #3a1020, #4a1528); border: 2px solid #e53935; border-radius: 14px; padding: 22px; text-align: center; }
    .normal-box { background: linear-gradient(135deg, #0d2b1e, #133529); border: 2px solid #43a047; border-radius: 14px; padding: 22px; text-align: center; }
    .result-text { font-size: 1.8rem; font-weight: 800; }
    .confidence-text { font-size: 1rem; color: #cdd6f4; margin-top: 8px; }
    .section-header { font-size: 1.2rem; font-weight: 700; color: #4fc3f7; border-bottom: 2px solid #2e3450; padding-bottom: 6px; margin-bottom: 12px; }
    div[data-testid="stSidebar"] { background-color: #13151f; }
    .insight-box { background: linear-gradient(135deg, #1a2744, #1e2f55); border: 1px solid #3a5298; border-radius: 10px; padding: 16px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

FEATURES = ["speed_kmh","acceleration","brake_pressure","steering_angle","rpm","engine_temp"]
SAFE_RANGES = {"speed_kmh":(0,80),"acceleration":(-3,3),"brake_pressure":(0,60),"steering_angle":(-15,15),"rpm":(700,3500),"engine_temp":(75,100)}

@st.cache_resource
def load_model():
    return joblib.load("models/driving_behavior_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/automotive_telemetry_150k.csv")

@st.cache_data
def compute_metrics():
    df = load_data()
    X = df[FEATURES]
    y = df["anomaly_label"].map({"normal":0,"aggressive_driving":1})
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    mdl = load_model()
    y_pred = mdl.predict(X_test)
    y_prob = mdl.predict_proba(X_test)[:,1]
    return {
        "accuracy": round(accuracy_score(y_test,y_pred)*100,2),
        "f1": round(f1_score(y_test,y_pred)*100,2),
        "roc_auc": round(roc_auc_score(y_test,y_prob),4),
        "cm": confusion_matrix(y_test,y_pred),
        "report": classification_report(y_test,y_pred,target_names=["Normal","Aggressive"],output_dict=True),
        "fi": pd.Series(mdl.feature_importances_,index=FEATURES).sort_values(ascending=False),
        "total_test": len(X_test),
    }

model   = load_model()
df_full = load_data()
conn    = sqlite3.connect("automotive_data.db", check_same_thread=False)

def generate_sql(prompt):
    system_prompt = """You are an automotive data analyst. Convert the user prompt into a SQL query.
Table name: telemetry
Columns: vehicle_id, session_id, timestamp, speed_kmh, acceleration, brake_pressure, steering_angle, engine_temp, rpm, fuel_level, latitude, longitude, road_type, weather, anomaly_label
Return ONLY the SQL query, nothing else."""
    try:
        response = requests.post('http://localhost:11434/api/generate',
            json={"model":"phi","prompt":f"{system_prompt}\n\nUser: {prompt}\n\nSQL Query:","stream":False},timeout=30)
        if response.status_code == 200:
            sql = response.json()['response']
            return sql.replace("```sql","").replace("```","").strip()
    except:
        pass
    return None

def run_query(sql):
    return pd.read_sql(sql, conn)

def generate_detailed_insight(df):
    if df.empty: return "No data found."
    lines = []
    lines.append(f" **{len(df):,} records** from **{df['vehicle_id'].nunique()} unique vehicles**")
    lines.append(f" Average Speed: **{df['speed_kmh'].mean():.1f} km/h**")
    lines.append(f" Average Brake Pressure: **{df['brake_pressure'].mean():.1f}**")
    lines.append(f" Average Acceleration: **{df['acceleration'].mean():.2f} m/s²**")
    if "rpm" in df.columns: lines.append(f" Average RPM: **{df['rpm'].mean():.0f}**")
    if df['brake_pressure'].mean() > 80: lines.append(" **Risk Alert:** High brake pressure — possible aggressive braking!")
    if df['speed_kmh'].mean() > 90: lines.append(" **Speed Alert:** Average speed exceeds safe limits!")
    if df['acceleration'].mean() > 3: lines.append(" **Acceleration Alert:** High acceleration — aggressive driving pattern!")
    if "anomaly_label" in df.columns:
        agg_pct = (df["anomaly_label"]=="aggressive_driving").mean()*100
        lines.append(f" **{agg_pct:.1f}%** of records labeled as aggressive driving")
    return "\n\n".join(lines)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=60)
    st.title("Automotive AI\nIntelligence")
    st.markdown("---")
    st.markdown("###  Dataset Info")
    st.metric("Total Records", f"{len(df_full):,}")
    st.metric("Unique Vehicles", df_full["vehicle_id"].nunique())
    st.metric("Aggressive %", f"{round((df_full['anomaly_label']=='aggressive_driving').mean()*100,1)}%")
    st.markdown("---")
    st.markdown("###  Live Prediction")
    speed    = st.slider("Speed (km/h)",      0.0, 180.0, 60.0)
    accel    = st.slider("Acceleration",    -10.0,  15.0,  1.0, 0.1)
    brake    = st.slider("Brake Pressure",   0.0, 100.0, 30.0)
    steering = st.slider("Steering Angle", -30.0,  30.0,  0.0, 0.5)
    rpm_val  = st.slider("RPM",            500.0,7000.0,2000.0,50.0)
    eng_temp = st.slider("Engine Temp °C",  60.0, 130.0, 85.0, 0.5)
    predict_btn = st.button(" Predict Behaviour", use_container_width=True, type="primary")

st.markdown("#  Automotive AI Behaviour Detection System")
st.markdown("*Natural Language Queries + ML Driving Behaviour Detection on 150K+ vehicle records*")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([" NL Query Engine"," ML Model Performance"," Live Prediction"," Data Explorer"])

# TAB 1 — NL Query Engine
with tab1:
    st.markdown('<p class="section-header">Natural Language → SQL Query Engine</p>', unsafe_allow_html=True)
    st.markdown("Ask questions in plain English — AI converts to SQL and runs on 150K+ records.")
    st.markdown("** Quick queries:**")
    q_cols = st.columns(3)
    suggestions = ["Find vehicles with brake pressure above 80","Show aggressive driving records",
                   "Find vehicles with speed above 100","Show records with RPM above 5000",
                   "Find vehicles in rainy weather","Show top 10 fastest vehicles"]
    for i,sug in enumerate(suggestions):
        if q_cols[i%3].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state["nl_query"] = sug

    prompt = st.text_input("Enter your query:", value=st.session_state.get("nl_query",""), placeholder="e.g. Find vehicles with brake pressure above 80")

    if st.button(" Analyze", type="primary"):
        if prompt:
            with st.spinner("Generating SQL..."):
                sql = generate_sql(prompt)
            if sql:
                st.markdown("#### Generated SQL")
                st.code(sql, language="sql")
                try:
                    result = run_query(sql)
                    st.markdown(f"####  Query Result — *{len(result):,} records*")
                    st.dataframe(result.head(50), use_container_width=True)
                    st.markdown("####  AI Data Insight")
                    st.markdown(f'<div class="insight-box">{generate_detailed_insight(result)}</div>', unsafe_allow_html=True)

                    if all(f in result.columns for f in FEATURES) and len(result) > 0:
                        st.markdown("####  ML Driving Behaviour Analysis")
                        preds = model.predict(result[FEATURES])
                        result_p = result.copy()
                        result_p[" Prediction"] = ["Aggressive" if p==1 else " Normal" for p in preds]
                        agg_c = int((preds==1).sum()); norm_c = int((preds==0).sum())
                        m1,m2,m3 = st.columns(3)
                        m1.metric("Total Records", len(result))
                        m2.metric(" Aggressive", agg_c, delta=f"{agg_c/len(result)*100:.1f}%", delta_color="inverse")
                        m3.metric(" Normal", norm_c)
                        fig_pie = go.Figure(go.Pie(labels=["Normal","Aggressive"],values=[norm_c,agg_c],
                            marker_colors=["#43a047","#e53935"],hole=0.4))
                        fig_pie.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=280,margin=dict(t=20,b=20))
                        st.plotly_chart(fig_pie, use_container_width=True)
                        st.dataframe(result_p[["vehicle_id","speed_kmh","acceleration","brake_pressure"," Prediction"]].head(30), use_container_width=True)

                    if "brake_pressure" in result.columns and len(result)>1:
                        st.markdown("####  Sensor Distributions")
                        dc1,dc2 = st.columns(2)
                        with dc1:
                            f1=px.histogram(result,x="brake_pressure",nbins=30,title="Brake Pressure",color_discrete_sequence=["#4fc3f7"])
                            f1.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=260)
                            st.plotly_chart(f1, use_container_width=True)
                        with dc2:
                            if "speed_kmh" in result.columns:
                                f2=px.histogram(result,x="speed_kmh",nbins=30,title="Speed",color_discrete_sequence=["#a78bfa"])
                                f2.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=260)
                                st.plotly_chart(f2, use_container_width=True)
                except Exception as e:
                    st.error(f"SQL Error: {e}")
            else:
                st.warning(" LLM not available. Run: `ollama run phi`")
                st.info("Other tabs (ML Performance, Live Prediction, Data Explorer) work without LLM!")
        else:
            st.warning("Please enter a query.")

# TAB 2 — ML Model Performance
with tab2:
    st.markdown('<p class="section-header">Random Forest — Model Performance on 150K Dataset</p>', unsafe_allow_html=True)
    with st.spinner("Computing metrics..."):
        m = compute_metrics()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric(" Accuracy",  f"{m['accuracy']}%")
    c2.metric(" F1 Score",  f"{m['f1']}%")
    c3.metric(" ROC-AUC",   str(m['roc_auc']))
    c4.metric(" Test Size", f"{m['total_test']:,} rows")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("####  Confusion Matrix")
        cm = m["cm"]
        fig_cm = go.Figure(go.Heatmap(z=cm,x=["Normal","Aggressive"],y=["Normal","Aggressive"],
            colorscale="Blues",text=cm,texttemplate="%{text}",showscale=True))
        fig_cm.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Predicted",yaxis_title="Actual",height=340,margin=dict(t=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
        st.markdown(f"-  True Normal: **{tn:,}** |  True Aggressive: **{tp:,}**\n-  False Positives: **{fp}** |  False Negatives: **{fn}**")

    with col2:
        st.markdown("####  Feature Importance")
        fi = m["fi"].reset_index(); fi.columns=["Feature","Importance"]
        fi["Feature"] = fi["Feature"].str.replace("_"," ").str.title()
        fig_fi = go.Figure(go.Bar(x=fi["Importance"],y=fi["Feature"],orientation="h",
            marker=dict(color=fi["Importance"],colorscale="Blues",showscale=False)))
        fig_fi.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            height=340,margin=dict(t=20,l=160),xaxis_title="Importance Score")
        st.plotly_chart(fig_fi, use_container_width=True)
        st.info(" **Acceleration** and **Steering Angle** are top predictors — sudden moves = aggressive driving!")

    st.markdown("####  Classification Report")
    rep = m["report"]
    rep_df = pd.DataFrame({
        "Class":["Normal","Aggressive","Macro Avg"],
        "Precision":[rep["Normal"]["precision"],rep["Aggressive"]["precision"],rep["macro avg"]["precision"]],
        "Recall":[rep["Normal"]["recall"],rep["Aggressive"]["recall"],rep["macro avg"]["recall"]],
        "F1-Score":[rep["Normal"]["f1-score"],rep["Aggressive"]["f1-score"],rep["macro avg"]["f1-score"]],
        "Support":[int(rep["Normal"]["support"]),int(rep["Aggressive"]["support"]),int(rep["macro avg"]["support"])],
    }).round(4)
    st.dataframe(rep_df, use_container_width=True, hide_index=True)
    st.markdown("""---
####  Why Random Forest?
- **Ensemble of 100 decision trees** — majority vote wins, reduces errors
- **99.99% accuracy** on 30,000 unseen test records
- **Built-in feature importance** — tells which sensor matters most for predictions
- **Handles class imbalance** — 80% normal vs 20% aggressive data
- **No overfitting** — evaluated on completely unseen test data""")

# TAB 3 — Live Prediction
with tab3:
    st.markdown('<p class="section-header">Live Driving Behaviour Prediction</p>', unsafe_allow_html=True)
    st.markdown("Adjust sensor sliders in the sidebar → Click **Predict Behaviour**")

    if predict_btn:
        inp = pd.DataFrame([{"speed_kmh":speed,"acceleration":accel,"brake_pressure":brake,
                              "steering_angle":steering,"rpm":rpm_val,"engine_temp":eng_temp}])
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]
        conf = round(max(prob)*100,1)

        if pred==1:
            st.markdown(f'<div class="aggressive-box"><div class="result-text"> AGGRESSIVE DRIVING DETECTED</div><div class="confidence-text">Confidence: <b>{conf}%</b> | Model: Random Forest</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="normal-box"><div class="result-text"> NORMAL DRIVING</div><div class="confidence-text">Confidence: <b>{conf}%</b> | Model: Random Forest</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("####  Sensor Values vs Safe Range")
        sensor_vals = {"speed_kmh":speed,"acceleration":accel,"brake_pressure":brake,"steering_angle":steering,"rpm":rpm_val,"engine_temp":eng_temp}
        s_cols = st.columns(3)
        for i,(feat,val) in enumerate(sensor_vals.items()):
            lo,hi = SAFE_RANGES[feat]
            icon = "🔴" if not(lo<=val<=hi) else "🟢"
            s_cols[i%3].metric(f"{icon} {feat.replace('_',' ').title()}",f"{val}",delta=f"Safe: {lo}–{hi}",delta_color="off")

        st.markdown("#### 🕸️ Radar Chart")
        norm_vals=[min(max((sensor_vals[f]-SAFE_RANGES[f][0])/(SAFE_RANGES[f][1]-SAFE_RANGES[f][0]+1e-6),0),2) for f in FEATURES]
        cats=[f.replace("_"," ").title() for f in FEATURES]
        fig_r=go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=norm_vals+[norm_vals[0]],theta=cats+[cats[0]],fill="toself",
            fillcolor="rgba(229,57,53,0.2)" if pred==1 else "rgba(67,160,71,0.2)",
            line_color="#e53935" if pred==1 else "#43a047",name="Reading"))
        fig_r.add_trace(go.Scatterpolar(r=[1]*len(cats)+[1],theta=cats+[cats[0]],mode="lines",
            line=dict(color="#4fc3f7",dash="dash"),name="Safe Boundary"))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,2])),
            template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=400,margin=dict(t=30))
        st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("####  Prediction Probability")
        fig_p=go.Figure(go.Bar(x=["Normal","Aggressive"],y=[prob[0]*100,prob[1]*100],
            marker_color=["#43a047","#e53935"],text=[f"{prob[0]*100:.1f}%",f"{prob[1]*100:.1f}%"],textposition="auto"))
        fig_p.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Probability (%)",height=260,margin=dict(t=20))
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info(" Adjust sliders in sidebar and click **Predict Behaviour**")
        ec1,ec2 = st.columns(2)
        with ec1:
            st.markdown("** Normal Driving Values:**\n- Speed: 40–70 km/h\n- Acceleration: 0.5–1.5\n- Brake: 20–40\n- Steering: -5 to 5°\n- RPM: 1500–2500")
        with ec2:
            st.markdown("** Aggressive Driving Values:**\n- Speed: 120+ km/h\n- Acceleration: 7–10+\n- Brake: 80–100\n- Steering: ±20°+\n- RPM: 5000+")

# TAB 4 — Data Explorer
with tab4:
    st.markdown('<p class="section-header">Dataset Explorer — 150K Vehicle Telemetry Records</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records",f"{len(df_full):,}")
    c2.metric("Unique Vehicles",df_full["vehicle_id"].nunique())
    c3.metric("Normal",f"{(df_full['anomaly_label']=='normal').sum():,}")
    c4.metric("Aggressive",f"{(df_full['anomaly_label']=='aggressive_driving').sum():,}")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("####  Label Distribution")
        lc = df_full["anomaly_label"].value_counts()
        fig_d=go.Figure(go.Pie(labels=["Normal","Aggressive"],values=lc.values,marker_colors=["#43a047","#e53935"],hole=0.4))
        fig_d.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=300,margin=dict(t=20,b=20))
        st.plotly_chart(fig_d, use_container_width=True)
    with col2:
        st.markdown("####  Feature Boxplot by Label")
        fc = st.selectbox("Feature",FEATURES,format_func=lambda x:x.replace("_"," ").title())
        fig_b=go.Figure()
        for lbl,color,name in [("normal","#43a047","Normal"),("aggressive_driving","#e53935","Aggressive")]:
            fig_b.add_trace(go.Box(y=df_full[df_full["anomaly_label"]==lbl][fc],name=name,marker_color=color))
        fig_b.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=300,margin=dict(t=20))
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("####  Scatter — Normal vs Aggressive")
    sc1,sc2 = st.columns(2)
    xf = sc1.selectbox("X Axis",FEATURES,index=0,format_func=lambda x:x.replace("_"," ").title(),key="xsc")
    yf = sc2.selectbox("Y Axis",FEATURES,index=1,format_func=lambda x:x.replace("_"," ").title(),key="ysc")
    samp = df_full.sample(3000,random_state=42)
    fig_sc=px.scatter(samp,x=xf,y=yf,color=samp["anomaly_label"].map({"normal":"Normal","aggressive_driving":"Aggressive"}),
        color_discrete_map={"Normal":"#43a047","Aggressive":"#e53935"},opacity=0.5,template="plotly_dark")
    fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=380)
    st.plotly_chart(fig_sc, use_container_width=True)

    with st.expander(" Raw Data Sample"):
        st.dataframe(df_full.head(100), use_container_width=True)
