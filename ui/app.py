"""
app.py — Evolutionary Algorithm Feature Selection Dashboard
HOG+LBP Image Features | GA · PSO · SVM
"""

import time
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ga import run_ga
from pso import run_pso
from svm_eval import evaluate_svm

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="EA Feature Selection",
    page_icon="🧬",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #212529;
        line-height: 1.2;
    }
    .metric-delta-pos { color: #28a745; font-size: 0.85rem; }
    .metric-delta-neg { color: #dc3545; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    X = np.load(os.path.join(DATA_DIR, "XPCA_features_80.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))
    return X, y


# ─────────────────────────────────────────────────────────────
# Plotly helpers
# ─────────────────────────────────────────────────────────────
def plot_convergence(history, title="Convergence Curve", color="#4361ee"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(history))), y=history,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5),
        name="Best Accuracy",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Generation / Iteration",
        yaxis_title="CV Accuracy",
        yaxis=dict(range=[0, 1]),
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig


def plot_convergence_dual(ga_history, pso_history):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("GA Convergence", "PSO Convergence"))
    fig.add_trace(go.Scatter(
        x=list(range(len(ga_history))), y=ga_history,
        mode="lines+markers", line=dict(color="#4361ee", width=2),
        marker=dict(size=5), name="GA",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(len(pso_history))), y=pso_history,
        mode="lines+markers", line=dict(color="#f72585", width=2),
        marker=dict(size=5), name="PSO",
    ), row=1, col=2)
    fig.update_yaxes(range=[0, 1], title_text="CV Accuracy", row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=2)
    fig.update_xaxes(title_text="Generation / Iteration")
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=60, b=40),
                      template="plotly_white", showlegend=False)
    return fig


def plot_accuracy_bar(baseline_acc, result_acc, labels=("Baseline SVM", "EA Result")):
    fig = go.Figure(go.Bar(
        x=list(labels),
        y=[baseline_acc, result_acc],
        marker_color=["#6c757d", "#4361ee"],
        text=[f"{v:.4f}" for v in [baseline_acc, result_acc]],
        textposition="outside",
        width=0.4,
    ))
    fig.update_layout(
        title="Accuracy Comparison",
        yaxis=dict(range=[0, 1.1], title="Accuracy"),
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_white",
    )
    return fig


def plot_feature_bar(mask, title="Selected PCA Features", top_n=20):
    selected = np.where(np.asarray(mask) == 1)[0]
    show = selected[:top_n] if len(selected) > top_n else selected
    fig = go.Figure(go.Bar(
        x=[f"PCA-{i}" for i in show],
        y=[1] * len(show),
        marker_color="#4361ee",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="PCA Component Index",
        yaxis=dict(visible=False),
        height=300,
        margin=dict(l=40, r=20, t=50, b=60),
        template="plotly_white",
    )
    return fig


def plot_mask_heatmap(ga_mask, pso_mask):
    matrix = np.array([ga_mask, pso_mask], dtype=float)
    fig = px.imshow(
        matrix,
        color_continuous_scale=[[0, "#f0f0f0"], [1, "#4361ee"]],
        aspect="auto",
        labels=dict(x="Feature Index", color="Selected"),
        y=["GA", "PSO"],
        zmin=0, zmax=1,
    )
    fig.update_layout(
        title="GA vs PSO — Feature Mask Comparison",
        height=200,
        margin=dict(l=60, r=20, t=50, b=40),
        coloraxis_showscale=False,
    )
    return fig


def metric_card(label, value, delta=None):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-pos" if delta >= 0 else "metric-delta-neg"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="{cls}">{sign}{delta:.4f} vs baseline</div>'
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'{delta_html}</div>')


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
for key, default in [("ga_result", None), ("pso_result", None),
                     ("baseline_accuracy", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
try:
    X, y = load_data()
except FileNotFoundError as exc:
    st.error(f"Dataset file not found: {exc}\n\n"
             "Make sure `XPCA_features_80.npy` and `y_labels.npy` "
             "are in the same folder as `app.py`.")
    st.stop()

N_FEATURES = X.shape[1]   # 80


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("# 🧬 Evolutionary Algorithm Feature Selection")
st.markdown(
    "**HOG+LBP Image Features | GA · PSO · SVM** — "
    "Compare how genetic and swarm algorithms reduce 8,110 raw descriptors "
    f"(→ {N_FEATURES} PCA components) while preserving classification accuracy."
)
st.divider()

# ─────────────────────────────────────────────────────────────
# How to Use expander (always visible above tabs)
# ─────────────────────────────────────────────────────────────
with st.expander("📖 How to Use This Dashboard", expanded=False):
    st.markdown("""
**Welcome!** This dashboard lets you run evolutionary optimisation algorithms that automatically
find the best *subset* of image features for an SVM classifier.
Here is a quick step-by-step guide:

---

**Step 1 — Choose your algorithm** *(sidebar)*  
Select **GA** (Genetic Algorithm), **PSO** (Particle Swarm), or **Both**.

---

**Step 2 — Adjust the parameters** *(sidebar)*

| Parameter | What it controls |
|---|---|
| **Population Size / Particles** | How many candidate feature-subsets the algorithm explores at the same time. More = better search, slower run. |
| **Generations / Iterations** | How many rounds of evolution or search to perform. More rounds usually improve the result but take longer. |
| **Mutation Rate** *(GA only)* | Chance of randomly flipping a feature bit each generation. Keeps diversity so the algorithm doesn't get stuck. |
| **Mutation Type** *(GA)* | *Bit-Flip* = randomly toggle a bit; *Swap* = exchange two bits. Bit-Flip is more aggressive. |
| **Crossover Type** *(GA)* | *One-Point* = cut chromosomes at one place and swap halves; *Uniform* = mix bits randomly at every position; *Two-Point* = swap the middle segment. |
| **Selection Method** *(GA)* | *Tournament* = randomly pick a few solutions and keep the best; *Roulette Wheel* = better solutions get a proportionally higher chance to be selected. |
| **Inertia Weight w** *(PSO)* | Controls how much momentum a particle keeps from its previous direction. 0.7 is a balanced default — too high → slow convergence; too low → swarm clusters too fast. |
| **C1 (Cognitive)** *(PSO)* | How much each particle trusts its *own* best position found so far. Higher → more individual exploration. |
| **C2 (Social)** *(PSO)* | How much each particle follows the *swarm's* global best position. Higher → more collective convergence. |
| **Transfer Function** *(PSO)* | *S-shaped (Sigmoid)* = smooth probability curve; *V-shaped (Tanh)* = more aggressive binary flipping. |

---

**Step 3 — Click "▶ Run Optimization"** *(Tab 1)*  
Wait for the spinner to finish. Typical runtime: 20–60 seconds depending on population/iteration settings.

---

**Step 4 — Read the 4 metric cards**

| Card | Meaning |
|---|---|
| **Accuracy** | How well the SVM classifies images using *only* the selected features. Compare it to the Baseline row — higher is better. |
| **Selected Features** | How many of the 80 PCA components the algorithm chose to keep. |
| **Feature Reduction %** | How much smaller the selected feature set is compared to using all 80. E.g. 60% means only 32 features were kept. |
| **Runtime (s)** | Total wall-clock time the algorithm took to finish, in seconds. |

---

**Step 5 — Go to Tab 2 ⚖️** to run both GA and PSO with a single click and compare them head-to-head.

**Step 6 — Go to Tab 3 📊** to see *which specific* PCA components were selected, and compare GA vs PSO feature masks side-by-side.
    """)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Algorithm Controls")

    algorithm = st.radio("Choose algorithm", ["GA", "PSO", "Both"], horizontal=True)
    st.caption("GA explores via evolution. PSO explores via swarm behavior. Both runs them in sequence.")
    st.divider()

    # ── GA controls ──
    if algorithm in ("GA", "Both"):
        st.subheader("🔴 Genetic Algorithm")
        ga_pop_size    = st.slider("Population Size",  5, 50, 20, 5)
        ga_generations = st.slider("Generations",      5, 50, 15, 5)
        ga_mut_rate    = st.slider("Mutation Rate", 0.001, 0.1, 0.01, 0.001,
                                   format="%.3f")
        ga_mut_type = st.selectbox(
            "Mutation Type", ["bitflip", "swap"],
            format_func=lambda x: "Bit-Flip" if x == "bitflip" else "Swap")
        ga_cx_type = st.selectbox(
            "Crossover Type", ["two_point", "single_point", "uniform"],
            format_func=lambda x: {"two_point": "Two-Point",
                                   "single_point": "One-Point",
                                   "uniform": "Uniform"}[x])
        ga_selection = st.selectbox(
            "Selection Method", ["tournament", "roulette"],
            format_func=lambda x: "Tournament" if x == "tournament" else "Roulette Wheel")
        ga_survivor = st.selectbox(
            "Survivor Method", ["elitist", "generational"],
            format_func=lambda x: "Elitism" if x == "elitist" else "Generational")
        st.caption("💡 Start with Population=50, Generations=30 for a quick test")
        st.divider()

    # ── PSO controls ──
    if algorithm in ("PSO", "Both"):
        st.subheader("🔵 Particle Swarm Optimisation")
        pso_particles  = st.slider("Particles",  5, 50, 20, 5)
        pso_iterations = st.slider("Iterations", 5, 50, 15, 5)
        pso_w  = st.slider("Inertia Weight (w)",   0.1, 1.0, 0.7, 0.05)
        pso_c1 = st.slider("Cognitive Coeff (c1)", 0.5, 4.0, 2.0, 0.1)
        pso_c2 = st.slider("Social Coeff (c2)",    0.5, 4.0, 2.0, 0.1)
        pso_transfer = st.selectbox(
            "Transfer Function", ["sigmoid", "vshaped"],
            format_func=lambda x: "S-shaped (Sigmoid)" if x == "sigmoid" else "V-shaped (Tanh)")
        st.caption("💡 Start with Particles=50, Iterations=30, w=0.7, C1=C2=1.5")
        st.divider()

    with st.expander("ℹ️ About this project"):
        st.markdown("""
**Dataset**  
Images → HOG + LBP descriptors → 8,110 features → PCA → 80 components.

**Goal**  
Find a compact binary feature mask that preserves SVM accuracy.

**Fitness function**  
`fitness = 0.9 × accuracy + 0.1 × (1 − selected_ratio)`

**Algorithms**  
- **GA** — binary chromosome; tournament/roulette selection; one-point/uniform/two-point crossover; bit-flip/swap mutation; elitism survivor.  
- **PSO** — continuous velocity + sigmoid/V-shaped transfer → binary mask.  
- **SVM Baseline** — RBF kernel, C=1, gamma='scale', trained on ALL 80 features.
        """)

    if st.button("🗑️ Clear Results"):
        st.session_state.ga_result         = None
        st.session_state.pso_result        = None
        st.session_state.baseline_accuracy = None
        st.rerun()


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🚀 Run & Results",
    "⚖️ GA vs PSO Comparison",
    "📊 Feature Analysis",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — Run & Results
# ═══════════════════════════════════════════════════════════════
with tab1:
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        run_clicked = st.button("▶ Run Optimization", type="primary",
                                use_container_width=True)

    if run_clicked:
        # Baseline (computed once)
        if st.session_state.baseline_accuracy is None:
            with st.spinner("Computing SVM baseline on all features…"):
                bl = evaluate_svm(X, y)
                st.session_state.baseline_accuracy = bl["accuracy"]

        # ── Run GA ──
        if algorithm in ("GA", "Both"):
            try:
                with st.spinner("Running Genetic Algorithm… (this may take a minute)"):
                    t0 = time.time()
                    best_mask, best_score, history = run_ga(
                        X, y,
                        pop_size=ga_pop_size,
                        n_generations=ga_generations,
                        mutation_rate=ga_mut_rate,
                        mutation_type=ga_mut_type,
                        crossover_type=ga_cx_type,
                        selection_method=ga_selection,
                        survivor_method=ga_survivor,
                    )
                    runtime = time.time() - t0
                    svm_res = evaluate_svm(X, y, mask=best_mask)
                    n_sel   = int(best_mask.sum())
                    if n_sel == 0:
                        st.warning("GA selected 0 features — result may be unreliable.")
                    st.session_state.ga_result = dict(
                        mask=best_mask, cv_score=best_score,
                        accuracy=svm_res["accuracy"], n_sel=n_sel,
                        history=history, runtime=runtime,
                    )
            except Exception as exc:
                st.error(f"GA run failed: {exc}")

        # ── Run PSO ──
        if algorithm in ("PSO", "Both"):
            try:
                with st.spinner("Running Particle Swarm Optimisation… (this may take a minute)"):
                    t0 = time.time()
                    best_mask, best_score, history = run_pso(
                        X, y,
                        n_particles=pso_particles,
                        n_iterations=pso_iterations,
                        w=pso_w, c1=pso_c1, c2=pso_c2,
                        transfer_type=pso_transfer,
                    )
                    runtime = time.time() - t0
                    svm_res = evaluate_svm(X, y, mask=best_mask)
                    n_sel   = int(best_mask.sum())
                    if n_sel == 0:
                        st.warning("PSO selected 0 features — result may be unreliable.")
                    st.session_state.pso_result = dict(
                        mask=best_mask, cv_score=best_score,
                        accuracy=svm_res["accuracy"], n_sel=n_sel,
                        history=history, runtime=runtime,
                    )
            except Exception as exc:
                st.error(f"PSO run failed: {exc}")

    baseline_acc = st.session_state.baseline_accuracy

    # Which result(s) to show
    show = []
    if algorithm in ("GA", "Both") and st.session_state.ga_result:
        show.append(("GA",  st.session_state.ga_result,  "#4361ee"))
    if algorithm in ("PSO", "Both") and st.session_state.pso_result:
        show.append(("PSO", st.session_state.pso_result, "#f72585"))

    if not show:
        st.info("Configure parameters in the sidebar, then click **▶ Run Optimization**.")

    for alg_name, res, color in show:
        st.subheader(f"{alg_name} Results")

        acc       = res["accuracy"]
        n_sel     = res["n_sel"]
        runtime   = res["runtime"]
        reduction = (1 - n_sel / N_FEATURES) * 100
        delta     = acc - baseline_acc if baseline_acc is not None else None

        # 4 metric cards
        c1_, c2_, c3_, c4_ = st.columns(4)
        with c1_:
            delta_str = f"{delta:+.4f}" if delta is not None else None
            st.metric(
                "Accuracy", f"{acc:.4f}",
                delta=delta_str,
                help="SVM test accuracy using only the selected features. Higher is better. Baseline (all features) = ~0.83",
            )
        with c2_:
            st.metric(
                "Selected Features", f"{n_sel} / {N_FEATURES}",
                help="Number of PCA components kept by the algorithm out of 80 total",
            )
        with c3_:
            st.metric(
                "Feature Reduction", f"{reduction:.1f}%",
                help="How much smaller the feature set is compared to using all 80 features",
            )
        with c4_:
            st.metric(
                "Runtime (s)", f"{runtime:.1f}",
                help="Total time the algorithm took to run in seconds",
            )

        # Auto-interpretation box
        if baseline_acc is not None:
            pct_drop = (baseline_acc - acc) * 100
            if acc >= baseline_acc:
                interp = "✅ The algorithm matched or exceeded the baseline SVM using fewer features."
            elif acc >= baseline_acc - 0.02:
                interp = (
                    f"✅ Minor accuracy trade-off: {pct_drop:.1f}% drop in exchange "
                    f"for {reduction:.1f}% feature reduction."
                )
            else:
                interp = (
                    f"📉 Accuracy is {pct_drop:.1f}% below baseline, "
                    f"with {reduction:.1f}% fewer features selected."
                )
            st.info(interp)

        st.write("")

        # Convergence curve + comparison table
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                plot_convergence(res["history"], f"{alg_name} Convergence Curve", color),
                use_container_width=True,
            )
        with right:
            if baseline_acc is not None:
                import pandas as pd
                df_cmp = pd.DataFrame({
                    "Method":        ["Baseline SVM", alg_name],
                    "Accuracy":      [f"{baseline_acc:.4f}", f"{acc:.4f}"],
                    "Features Used": [N_FEATURES, n_sel],
                    "Reduction (%)": ["0.0 %", f"{reduction:.1f} %"],
                    "Δ vs Baseline": ["—",
                                      f"{'+'if delta>=0 else ''}{delta:.4f}"],
                })
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        # Accuracy bar chart
        if baseline_acc is not None:
            st.plotly_chart(
                plot_accuracy_bar(baseline_acc, acc,
                                  labels=("Baseline SVM",
                                          f"{alg_name} (selected)")),
                use_container_width=True,
            )

        st.divider()


# ═══════════════════════════════════════════════════════════════
# TAB 2 — GA vs PSO Comparison
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("⚖️ GA vs PSO Head-to-Head Comparison")

    _, col_btn2, _ = st.columns([2, 1, 2])
    with col_btn2:
        run_both = st.button("▶ Run Both Algorithms", type="primary",
                             use_container_width=True, key="run_both")

    if run_both:
        if st.session_state.baseline_accuracy is None:
            with st.spinner("Computing SVM baseline…"):
                bl = evaluate_svm(X, y)
                st.session_state.baseline_accuracy = bl["accuracy"]

        try:
            with st.spinner("Running GA…"):
                t0 = time.time()
                bm, bs, hist = run_ga(X, y, pop_size=20, n_generations=15)
                rt = time.time() - t0
                sr = evaluate_svm(X, y, mask=bm)
                st.session_state.ga_result = dict(
                    mask=bm, cv_score=bs, accuracy=sr["accuracy"],
                    n_sel=int(bm.sum()), history=hist, runtime=rt,
                )
        except Exception as exc:
            st.error(f"GA failed: {exc}")

        try:
            with st.spinner("Running PSO…"):
                t0 = time.time()
                bm, bs, hist = run_pso(X, y, n_particles=20, n_iterations=15)
                rt = time.time() - t0
                sr = evaluate_svm(X, y, mask=bm)
                st.session_state.pso_result = dict(
                    mask=bm, cv_score=bs, accuracy=sr["accuracy"],
                    n_sel=int(bm.sum()), history=hist, runtime=rt,
                )
        except Exception as exc:
            st.error(f"PSO failed: {exc}")

    ga  = st.session_state.ga_result
    pso = st.session_state.pso_result
    bl  = st.session_state.baseline_accuracy

    if ga is None and pso is None:
        st.info("Click **▶ Run Both Algorithms** to compare GA and PSO, "
                "or run them individually from the **🚀 Run & Results** tab.")
    else:
        import pandas as pd

        rows = []
        if bl is not None:
            rows.append({"Method": "Baseline SVM", "Accuracy": bl,
                         "Selected": N_FEATURES, "Reduction (%)": 0.0,
                         "vs Baseline": 0.0})
        for name, res in [("GA", ga), ("PSO", pso)]:
            if res is None:
                continue
            delta = res["accuracy"] - bl if bl is not None else 0.0
            rows.append({"Method": name,
                         "Accuracy": res["accuracy"],
                         "Selected": res["n_sel"],
                         "Reduction (%)": (1 - res["n_sel"] / N_FEATURES) * 100,
                         "vs Baseline": delta})

        df = pd.DataFrame(rows)

        def colour_delta(val):
            if isinstance(val, float) and val > 0:
                return "color: #28a745; font-weight:bold"
            if isinstance(val, float) and val < 0:
                return "color: #dc3545; font-weight:bold"
            return ""

        styled = (df.style
                  .format({"Accuracy": "{:.4f}",
                           "Reduction (%)": "{:.1f} %",
                           "vs Baseline": lambda x: f"+{x:.4f}" if x > 0
                                                     else f"{x:.4f}"})
                  .applymap(colour_delta, subset=["vs Baseline"]))
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Convergence side-by-side
        if ga and pso:
            st.plotly_chart(
                plot_convergence_dual(ga["history"], pso["history"]),
                use_container_width=True,
            )

            # Winner badge
            st.subheader("🏆 Winner")
            ga_red  = (1 - ga["n_sel"]  / N_FEATURES) * 100
            pso_red = (1 - pso["n_sel"] / N_FEATURES) * 100
            ga_comb  = 0.9 * ga["accuracy"]  + 0.1 * ga_red  / 100
            pso_comb = 0.9 * pso["accuracy"] + 0.1 * pso_red / 100

            if abs(ga_comb - pso_comb) < 0.002:
                st.info("🤝 **It's a tie!** Both algorithms performed equally well.")
            elif ga_comb > pso_comb:
                ga_n = ga["n_sel"]
                st.success(
                    f"🟢 **GA wins!**  Accuracy {ga['accuracy']:.4f} | "
                    f"{ga_n} features selected ({ga_red:.1f}% reduction)"
                )
            else:
                pso_n = pso["n_sel"]
                st.success(
                    f"🔵 **PSO wins!**  Accuracy {pso['accuracy']:.4f} | "
                    f"{pso_n} features selected ({pso_red:.1f}% reduction)"
                )

            with st.expander("🔍 How to interpret these results"):
                st.markdown("""
**vs Baseline column**  
This shows how much accuracy changed compared to the SVM trained on *all* 80 features.
- A **positive** number means the algorithm found a feature subset that actually *improves* accuracy — this happens because irrelevant features can add noise.
- A **negative** number means a small accuracy cost was paid to achieve feature reduction.

**Red numbers are not necessarily bad**  
In real-world systems, losing 1–2% accuracy while reducing features by 50%+ is often an excellent trade-off:
it means faster inference, less memory, and a simpler model. The goal is not always maximum accuracy — it's the *best balance* between accuracy and compactness.

**Why GA might trail PSO here**  
GA typically needs more generations than PSO to fully converge, because crossover and mutation are slower to explore the search space compared to PSO's velocity-guided movement. Try increasing GA generations to 30–50 for a fairer comparison.

**Reading the convergence curve**  
- A curve that *rises steeply then flattens* means the algorithm converged well — it found a good solution and stopped improving.
- A *completely flat line from generation 1* means early stopping kicked in (no improvement for many rounds). Try increasing Population/Particles or reducing Mutation Rate.
- A curve that is *still climbing at the last generation* means the algorithm had not finished searching — increase Generations/Iterations for a better result.
                """)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — Feature Analysis
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Feature Analysis")

    ga  = st.session_state.ga_result
    pso = st.session_state.pso_result

    if ga is None and pso is None:
        st.info("Run at least one algorithm first to see feature analysis.")
    else:
        for alg_name, res, color in [("GA", ga, "#4361ee"), ("PSO", pso, "#f72585")]:
            if res is None:
                continue
            n_sel = res["n_sel"]
            st.markdown(
                f"#### {alg_name} — {n_sel} / {N_FEATURES} PCA components selected "
                f"({(1 - n_sel / N_FEATURES) * 100:.1f}% reduction)"
            )
            st.markdown(
                metric_card("Selected PCA Components", f"{n_sel} / {N_FEATURES}"),
                unsafe_allow_html=True,
            )
            st.write("")
            st.plotly_chart(
                plot_feature_bar(res["mask"],
                                 f"{alg_name} — Top-20 Selected PCA Components"),
                use_container_width=True,
            )
            st.divider()

        # Heatmap when both ran
        if ga and pso:
            st.subheader("GA vs PSO — Full Mask Comparison")
            st.plotly_chart(
                plot_mask_heatmap(ga["mask"], pso["mask"]),
                use_container_width=True,
            )
            overlap = int(np.logical_and(ga["mask"], pso["mask"]).sum())
            st.markdown(
                f"**Mask overlap:** {overlap} features selected by **both** algorithms "
                f"out of {N_FEATURES} total PCA components."
            )
