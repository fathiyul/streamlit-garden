from __future__ import annotations

import math
try:
    from jax import numpy as jnp  # type: ignore
    BACKEND = "JAX"
except Exception:
    import numpy as jnp  # type: ignore
    BACKEND = "NumPy"
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Bayesian Inference on Probabilities", page_icon="🎯", layout="wide")

st.title("Inference on Probabilities 🎯")
st.caption("Posterior for a Bernoulli probability with a Beta prior.")


def beta_pdf(x: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """Beta(a,b) PDF computed in log-space for numerical stability.

    x is assumed in (0,1). a>0, b>0.
    """
    # Avoid log(0) at the boundaries
    x = jnp.clip(x, 1e-12, 1 - 1e-12)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    log_pdf = (a - 1.0) * jnp.log(x) + (b - 1.0) * jnp.log(1.0 - x) - log_beta
    return jnp.exp(log_pdf)


with st.sidebar:
    st.header("Prior (Beta)")
    prior_a = st.slider(
        "Prior successes (alpha)", min_value=0.1, max_value=2.0, value=1.0, step=0.1
    )
    prior_b = st.slider(
        "Prior failures (beta)", min_value=0.1, max_value=2.0, value=1.0, step=0.1
    )

    st.header("Observed data")
    obs_success = st.number_input(
        "Number of successes", min_value=0, step=1, value=0
    )
    obs_failure = st.number_input(
        "Number of failures", min_value=0, step=1, value=0
    )


# Posterior parameters
post_a = prior_a + obs_success
post_b = prior_b + obs_failure

# Summary stats
post_mean = post_a / (post_a + post_b)
post_var = (post_a * post_b) / (((post_a + post_b) ** 2) * (post_a + post_b + 1.0))
post_std = float(math.sqrt(post_var))

# Grid and density (JAX)
x = jnp.linspace(0.0, 1.0, 600)
y = beta_pdf(x, post_a, post_b)

# Convert to lists for plotting via pandas/vega-lite
curve_df = pd.DataFrame({
    "pi": [float(v) for v in x.tolist()],
    "pdf": [float(v) for v in y.tolist()],
})
lines_df = pd.DataFrame(
    {
        "pi": [
            post_mean,
            max(0.0, post_mean - post_std),
            min(1.0, post_mean + post_std),
        ],
        "kind": ["mean", "-1σ", "+1σ"],
    }
)

left, right = st.columns([3, 2])

with left:
    st.subheader("Posterior p(π | a, b)")

    spec = {
        "layer": [
            {
                "mark": {"type": "area", "opacity": 0.25, "color": "#4f8bf9"},
                "encoding": {
                    "x": {"field": "pi", "type": "quantitative", "title": "π"},
                    "y": {"field": "pdf", "type": "quantitative", "title": "p(π | a,b)"},
                },
            },
            {
                "mark": {"type": "line", "color": "#2b6cb0", "size": 2},
                "encoding": {
                    "x": {"field": "pi", "type": "quantitative"},
                    "y": {"field": "pdf", "type": "quantitative"},
                },
            },
            {
                "data": {"values": lines_df.to_dict(orient="records")},
                "mark": {"type": "rule"},
                "encoding": {
                    "x": {"field": "pi", "type": "quantitative"},
                    "color": {
                        "field": "kind",
                        "type": "nominal",
                        "legend": {"title": "Lines"},
                        "scale": {
                            "domain": ["mean", "-1σ", "+1σ"],
                            "range": ["#d62728", "#999999", "#999999"],
                        },
                    },
                },
            },
        ],
        "data": {"values": curve_df.to_dict(orient="records")},
        "width": "container",
        "height": 360,
    }

    st.vega_lite_chart(curve_df, spec, use_container_width=True)

with right:
    st.subheader("Parameters")
    st.markdown(
        f"""
        - Prior α (successes): **{prior_a:.2f}**  
        - Prior β (failures): **{prior_b:.2f}**  
        - Observed successes: **{obs_success}**  
        - Observed failures: **{obs_failure}**  
        - Posterior α: **{post_a:.2f}**  
        - Posterior β: **{post_b:.2f}**  
        - Mean (E[π]): **{post_mean:.4f}**  
        - Std dev (σ): **{post_std:.4f}**  
        - Backend: **{BACKEND}**
        """
    )

    st.caption(
        "Mean shown in red, ±1σ in grey. Shaded region is the area under the posterior density curve."
    )
