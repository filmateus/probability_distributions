# streamlit_app.py — Distribution Explorer with Formulas (LaTeX) and Plotly + Discovery from Data
# ---------------------------------------------------------------------------------
# - Plotly for Interactive PDF/PMF
# - Formulas rendered with LaTeX
# - Data upload and function to suggest/fit distributions (AIC/BIC, KS/Chi²)
# ------------------------------------------------------------------------------

import base64
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import (
    bernoulli, binom, geom, hypergeom, nbinom, poisson,
    beta as beta_dist, cauchy, chi2, expon, gumbel_r,
    f as fdist, gamma as gamma_dist, logistic, pareto,
    triang, weibull_max, t as t_dist, norm
)
from scipy import stats

st.set_page_config(page_title="Distribuições de Probabilidade e Pesquisa de Distrbuição", layout="wide")


def b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def b64dec(s: str) -> str:
    return base64.b64decode(s.encode("ascii")).decode("utf-8")

# -----------------------------------------------------
# distributins catalogy
# - Catalog of distributions with parameters, formulas, and usage examples
# -----------------------------------------------------

distributions = {
    "Bernoulli": {
        "func": bernoulli,
        "params": {"p": {"min": 0.0, "max": 1.0, "value": 0.5}},
        "formula_b64": b64("P(k) = p^{k} (1-p)^{1-k},\\quad k \\in \\{0,1\\}"),
        "usage": "Experimentos com dois resultados possíveis (sucesso/fracasso) - lançamento de uma moeda."
    },
    "Binomial": {
        "func": binom,
        "params": {"n": {"min": 1, "max": 100, "value": 10}, "p": {"min": 0.0, "max": 1.0, "value": 0.5}},
        "formula_b64": b64("P(k) = \\binom{n}{k} p^{k} (1-p)^{n-k}"),
        "usage": "Contagem de sucessos em n tentativas independentes com probabilidade p - Lançar uma moeda 10 vezes e contar quantas vezes sai “cara”?"
    },
    "Geométrica": {
        "func": geom,
        "params": {"p": {"min": 0.01, "max": 0.99, "value": 0.5}},
        "formula_b64": b64("P(X = k) = (1-p)^{k-1} p,\\quad k=1,2,\\dots"),
        "usage": "Número de tentativas até o primeiro sucesso - jogar um dado até sair o número 6"
    },
    "Hipergeométrica": {
        "func": hypergeom,  # scipy.hypergeom(M, K, n)
        "params": {"N": {"min": 10, "max": 2000, "value": 50}, "K": {"min": 1, "max": 1000, "value": 20}, "n": {"min": 1, "max": 1000, "value": 10}},
        "formula_b64": b64("P(X = k) = \\frac{\\binom{K}{k} \\, \\binom{N-K}{n-k}}{\\binom{N}{n}}"),
        "usage": "Amostragem sem reposição de uma população finita - Uma urna tem 20 bolas: 8 vermelhas (sucesso) e 12 azuis (fracasso). Ao retirar 5 bolas sem reposição, qual a probabilidade de tirar exatamente 3 vermelhas?"
    },
    "Binomial Negativa": {
        "func": nbinom,
        "params": {"n": {"min": 1, "max": 200, "value": 10}, "p": {"min": 0.01, "max": 0.99, "value": 0.5}},
        "formula_b64": b64("P(X = k) = \\binom{k+n-1}{n-1} p^{n} (1-p)^{k}"),
        "usage": "Número de falhas antes de obter n sucessos - Quantas tentativas fracassadas vou ter antes de alcançar r sucessos, se cada tentativa tem probabilidade p de sucesso?"
    },
    "Poisson": {
        "func": poisson,
        "params": {"mu": {"min": 0.1, "max": 50.0, "value": 10.0}},
        "formula_b64": b64("P(k) = \\frac{\\lambda^{k} e^{-\\lambda}}{k!},\\quad \\lambda=\\mu"),
        "usage": "Número de eventos em um intervalo (taxa constante). Se uma central telefônica recebe em média 5 chamadas por minuto, qual a probabilidade de receber exatamente 3 chamadas em um minuto?"
    },
    "Beta": {
        "func": beta_dist,
        "params": {"a": {"min": 0.1, "max": 10.0, "value": 2.0}, "b": {"min": 0.1, "max": 10.0, "value": 5.0}},
        "formula_b64": b64("f(x) = \\frac{x^{a-1} (1-x)^{b-1}}{B(a,b)},\\quad 0<x<1"),
        "usage": "Variáveis contínuas em [0,1] (proporções, taxas) - Uma CTR (Click-Through Rate) de 0.2 (20%) pode ser modelada como Beta(2, 8). Se você observar 10 cliques em 50 impressões, qual a probabilidade de a CTR real estar entre 0.15 e 0.25?"
    },
    "Cauchy": {
        "func": cauchy,
        "params": {"x0": {"min": -10.0, "max": 10.0, "value": 0.0}, "gamma": {"min": 0.1, "max": 10.0, "value": 1.0}},
        "formula_b64": b64("f(x) = \\frac{1}{\\pi\\, \\gamma\\, \\left(1 + \\left(\\tfrac{x - x_0}{\\gamma}\\right)^2\\right)}"),
        "usage": "Dados com caudas pesadas  - Fenômenos de física como ressonância ou erros de medição com outliers extremos"
    },
    "Qui-quadrado": {
        "func": chi2,
        "params": {"df": {"min": 1, "max": 200, "value": 5}},
        "formula_b64": b64("f(x) = \\frac{1}{2^{df/2} \\, \\Gamma(df/2)} x^{df/2 - 1} e^{-x/2},\\ x>0"),
        "usage": "Testes de variância e aderência -  graus de liberdade (df)"
    },
    "Exponencial": {
        "func": expon,
        "params": {"scale": {"min": 0.01, "max": 10.0, "value": 1.0}},
        "formula_b64": b64("f(x) = \\frac{1}{\\text{scale}} e^{-x/\\text{scale}},\\ x\\ge 0"),
        "usage": "Tempo entre eventos (processo de Poisson) - Se um ônibus chega a cada 10 minutos em média, qual a probabilidade de esperar menos de 5 minutos?"
    },
    "Gumbel - Valores Extremos": {
        "func": gumbel_r,
        "params": {"loc": {"min": -10.0, "max": 10.0, "value": 0.0}, "scale": {"min": 0.1, "max": 10.0, "value": 1.0}},
        "formula_b64": b64("f(x) = \\frac{1}{s} \\exp\\!(-\\tfrac{x-\\ell}{s}) \\exp\\!\\{ -\\exp\\!(-\\tfrac{x-\\ell}{s}) \\}"),
        "usage": "Modelagem de máximos - Temperaturas máximas diárias, picos de vendas, etc. - Se a temperatura máxima em uma cidade é modelada por Gumbel(0, 1), qual a probabilidade de um dia ter temperatura acima de 35°C?"
    },
    "F": {
        "func": fdist,
        "params": {"df1": {"min": 1, "max": 200, "value": 5}, "df2": {"min": 1, "max": 200, "value": 5}, "loc": {"min": -10.0, "max": 10.0, "value": 0.0}, "scale": {"min": 0.01, "max": 10.0, "value": 1.0}},
        "formula_b64": b64("f(x) = \\frac{(\\tfrac{df_1}{df_2})^{df_1/2} x^{df_1/2 - 1}}{B(\\tfrac{df_1}{2}, \\tfrac{df_2}{2}) (1 + \\tfrac{df_1}{df_2}x)^{(df_1+df_2)/2}}"),
        "usage": "Comparação de variâncias (ex.: ANOVA)."
    },
    "Gamma": {
        "func": gamma_dist,
        "params": {"a": {"min": 0.1, "max": 10.0, "value": 1.0}, "scale": {"min": 0.01, "max": 10.0, "value": 1.0}},
        "formula_b64": b64("f(x) = \\frac{x^{a-1} e^{-x/\\text{scale}}}{\\text{scale}^{a} \\Gamma(a)},\\ x>0"),
        "usage": "Tempos de espera e variáveis positivas."
    },
    "Logística": {
        "func": logistic,
        "params": {"loc": {"min": -10.0, "max": 10.0, "value": 0.0}, "scale": {"min": 0.1, "max": 10.0, "value": 1.0}},
        "formula_b64": b64("f(x) = \\frac{e^{-(x-\\ell)/s}}{s(1 + e^{-(x-\\ell)/s})^{2}}"),
        "usage": "Crescimento logístico e distribuições simétricas - Modelar a distribuição de renda, onde a maioria das pessoas ganha perto da média, mas há alguns com renda muito alta."
    },
    "Pareto": {
        "func": pareto,
        "params": {"b": {"min": 0.1, "max": 10.0, "value": 1.0}},
        "formula_b64": b64("f(x) = \\frac{b}{x^{b+1}},\\ x\\ge 1"),
        "usage": "Renda e fenômenos com caudas pesadas (escala padrão) - Distribuição de renda, onde poucos têm muito e muitos têm pouco. Se a renda segue Pareto(1), qual a probabilidade de uma pessoa ter renda acima de 5?"
    },
    "T de Student": {
        "func": t_dist,
        "params": {"df": {"min": 1, "max": 200, "value": 5}},
        "formula_b64": b64("f(x) = \\frac{\\Gamma((df+1)/2)}{\\sqrt{df\\pi}\\,\\Gamma(df/2)} (1 + x^{2}/df)^{-(df+1)/2}"),
        "usage": "Médias com variância desconhecida (amostras pequenas) - "
    },
    "Triangular": {
        "func": triang,
        "params": {"c": {"min": 0.0, "max": 1.0, "value": 0.5}, "loc": {"min": -10.0, "max": 10.0, "value": 0.0}, "scale": {"min": 0.1, "max": 10.0, "value": 1.0}},
        "formula_b64": b64("\\text{PDF triangular padrão com pico em } c \\in [0,1]"),
        "usage": "Distribuições com pico definido e limites - Exemplo: tempo de entrega de um produto onde a maioria chega no prazo, mas alguns atrasam."
    },
    "Weibull": {
        "func": weibull_max,
        "params": {"c": {"min": 0.1, "max": 10.0, "value": 1.0}, "loc": {"min": -10.0, "max": 10.0, "value": 0.0}, "scale": {"min": 0.1, "max": 10.0, "value": 1.0}},
        "formula_b64": b64("f(x) = \\frac{c}{s}(\\tfrac{x-\\ell}{s})^{c-1} e^{-(\\tfrac{x-\\ell}{s})^{c}},\\ x>\\ell"),
        "usage": "Tempos de vida e confiabilidade - Modelar a vida útil de um produto onde a maioria falha cedo, mas alguns duram muito tempo. Se a vida útil de um componente segue Weibull(1, 0, 1), qual a probabilidade de durar mais de 5 anos?"
    },
    "Normal": {
        "func": norm,
        "params": {"mu": {"min": -10.0, "max": 10.0, "value": 0.0}, "sigma": {"min": 0.1, "max": 10.0, "value": 1.0}},
        "formula_b64": b64("f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-(x-\\mu)^2/(2\\sigma^2)}"),
        "usage": "Dados simétricos em torno da média - Altura de pessoas, notas de provas, etc. - Se a altura média de uma população é 1.70m com desvio padrão 0.1m, qual a probabilidade de uma pessoa ter altura entre 1.65m e 1.75m?"
    },
}

DISCRETE = {"Bernoulli", "Binomial", "Geométrica", "Hipergeométrica", "Binomial Negativa", "Poisson"}

# -------------------------------------------
# RV construction and evaluation
# -------------------------------------------

def build_frozen(name: str, params: dict):
    f = distributions[name]["func"]
    if name == "Bernoulli":
        return f(params["p"])
    if name == "Binomial":
        return f(int(params["n"]), float(params["p"]))
    if name == "Geométrica":
        return f(float(params["p"]))
    if name == "Hipergeométrica":
        M = int(params["N"]); K = int(params["K"]); n = int(params["n"])
        return f(M, K, n)
    if name == "Binomial Negativa":
        return f(int(params["n"]), float(params["p"]))
    if name == "Poisson":
        return f(float(params["mu"]))
    if name == "Beta":
        return f(float(params["a"]), float(params["b"]))
    if name == "Cauchy":
        return f(loc=float(params["x0"]), scale=float(params["gamma"]))
    if name == "Qui-quadrado":
        return f(int(params["df"]))
    if name == "Exponencial":
        return f(scale=float(params["scale"]))
    if name == "Gumbel - Valores Extremos":
        return f(loc=float(params["loc"]), scale=float(params["scale"]))
    if name == "F":
        return f(int(params["df1"]), int(params["df2"]), loc=float(params["loc"]), scale=float(params["scale"]))
    if name == "Gamma":
        return f(float(params["a"]), scale=float(params["scale"]))
    if name == "Logística":
        return f(loc=float(params["loc"]), scale=float(params["scale"]))
    if name == "Pareto":
        return f(float(params["b"]))
    if name == "T de Student":
        return f(int(params["df"]))
    if name == "Triangular":
        return f(float(params["c"]), loc=float(params["loc"]), scale=float(params["scale"]))
    if name == "Weibull":
        return f(float(params["c"]), loc=float(params["loc"]), scale=float(params["scale"]))
    if name == "Normal":
        return f(loc=float(params["mu"]), scale=float(params["sigma"]))
    raise ValueError("Distribuição desconhecida")

# function to get support and values for plotting
def support_and_values(name: str, frozen):
    if name in DISCRETE:
        try:
            ql = int(np.floor(frozen.ppf(0.001)))
            qr = int(np.ceil(frozen.ppf(0.999)))
        except Exception:
            ql, qr = 0, 30
        if name == "Binomial":
            n = int(frozen.args[0]); ql, qr = 0, n
        if name == "Hipergeométrica":
            M, K, n = map(int, frozen.args[:3])
            ql = max(0, n + K - M); qr = min(K, n)
        xs = np.arange(max(ql, -1000), min(qr, 100000) + 1)
        ys = frozen.pmf(xs)
        return xs, ys
    else:
        try:
            ql = float(frozen.ppf(0.001)); qr = float(frozen.ppf(0.999))
        except Exception:
            ql, qr = -10.0, 10.0
        if not np.isfinite(ql): ql = -10.0
        if not np.isfinite(qr): qr = 10.0
        xs = np.linspace(ql, qr, 400)
        ys = frozen.pdf(xs)
        return xs, ys

# functions to plot distributions and data fits
def plot_distribution(name: str, xs, ys):
    fig = go.Figure()
    if name in DISCRETE:
        fig.add_bar(x=xs, y=ys, name="PMF")
        fig.update_xaxes(title_text="k")
        fig.update_yaxes(title_text="P(X=k)")
    else:
        fig.add_scatter(x=xs, y=ys, mode="lines", name="PDF")
        fig.update_xaxes(title_text="x")
        fig.update_yaxes(title_text="f(x)")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=420, title=f"{name}")
    st.plotly_chart(fig, use_container_width=True)

# Function to plot overlay data fit
def plot_overlay_data_fit(name: str, frozen, data: np.ndarray):
    fig = go.Figure()
    if name in DISCRETE:
        xs = np.arange(int(np.min(data)), int(np.max(data)) + 1)
        obs = np.array([(data == xi).sum() for xi in xs], dtype=float)
        obs = obs / max(obs.sum(), 1)
        fig.add_bar(x=xs, y=obs, name="Obs (freq. relativa)")
        try:
            pmf = frozen.pmf(xs)
            fig.add_scatter(x=xs, y=pmf, mode="lines+markers", name=f"PMF {name}")
        except Exception:
            pass
        fig.update_xaxes(title_text="k")
        fig.update_yaxes(title_text="Probabilidade")
    else:
        fig.add_histogram(x=data, histnorm="probability density", nbinsx=40, name="Obs (densidade)", opacity=0.5)
        try:
            ql = float(frozen.ppf(0.001)); qr = float(frozen.ppf(0.999))
        except Exception:
            ql, qr = np.min(data), np.max(data)
        xs = np.linspace(ql, qr, 400)
        try:
            pdf = frozen.pdf(xs)
            fig.add_scatter(x=xs, y=pdf, mode="lines", name=f"PDF {name}")
        except Exception:
            pass
        fig.update_xaxes(title_text="x")
        fig.update_yaxes(title_text="densidade")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=420, title=f"Ajuste aos dados • {name}")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------
# Automatic detection/adjustment from data
# -------------------------------------------

def is_integer_array(x: np.ndarray) -> bool:
    if x.size == 0:
        return False
    return bool(np.all(np.isfinite(x)) and np.allclose(x, np.round(x)))

# Function to summarize data statistics
def summarize_data(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {
        "n": int(x.size),
        "Min": float(np.min(x)),
        "Max": float(np.max(x)),
        "Média": float(np.mean(x)),
        "Variância": float(np.var(x, ddof=1)) if x.size > 1 else 0.0,
        "Assimetria": float(stats.skew(x)) if x.size > 2 else 0.0, #skew    
        "all_int": bool(is_integer_array(x)), # negatives number of integers
        "nonneg": bool(np.all(x >= 0)), # non-negative numbers
        "bounded_01": bool(np.all((x >= 0) & (x <= 1))), # bounded between 0 and 1
    }

AUTO_CONT = {
    "Normal": norm,
    "Exponencial": expon,
    "Gamma": gamma_dist,
    "Beta": beta_dist,
    "Logística": logistic,
    "Cauchy": cauchy,
    "Gumbel - Valores Extremos": gumbel_r,
    "Weibull": weibull_max,
    "T de Student": t_dist,
    "Qui-quadrado": chi2,
    "Pareto": pareto,
}

AUTO_DISC = {
    "Bernoulli": bernoulli,
    "Geométrica": geom,
    "Poisson": poisson,
    "Binomial Negativa": nbinom,
}


def fit_candidate(name: str, x: np.ndarray):
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return None

    if name == "Bernoulli":
        if not np.all(np.isin(x, [0, 1])):
            return None
        p_hat = np.clip(x.mean(), 1e-9, 1-1e-9)
        rv = bernoulli(p_hat)
        ll = float(np.sum(rv.logpmf(x)))
        return {"name": name, "rv": rv, "params": {"p": p_hat}, "loglike": ll, "k": 1}

    if name == "Geométrica":
        if np.min(x) < 1 or not is_integer_array(x):
            return None
        p_hat = np.clip(1.0 / np.mean(x), 1e-9, 1-1e-9)
        rv = geom(p_hat)
        ll = float(np.sum(rv.logpmf(x)))
        return {"name": name, "rv": rv, "params": {"p": p_hat}, "loglike": ll, "k": 1}

    if name == "Poisson":
        if not is_integer_array(x) or np.any(x < 0):
            return None
        mu = max(1e-9, float(np.mean(x)))
        rv = poisson(mu)
        ll = float(np.sum(rv.logpmf(x)))
        return {"name": name, "rv": rv, "params": {"mu": mu}, "loglike": ll, "k": 1}

    if name == "Binomial Negativa":
        if not is_integer_array(x) or np.any(x < 0):
            return None
        mean, var = x.mean(), x.var(ddof=1) if n > 1 else x.var()
        if var <= mean + 1e-12:
            return None
        r = (mean**2) / max(var - mean, 1e-9)
        p = r / (r + mean)
        p = np.clip(p, 1e-9, 1-1e-9)
        rv = nbinom(r, p)
        ll = float(np.sum(rv.logpmf(x)))
        return {"name": name, "rv": rv, "params": {"r": r, "p": p}, "loglike": ll, "k": 2}

    if name == "Binomial":
        if not is_integer_array(x) or np.any(x < 0):
            return None
        xmax = int(np.max(x))
        if xmax == 0:
            return None
        best = None
        for n_trials in range(xmax, xmax + 51):
            p_hat = np.clip(x.mean() / max(n_trials, 1), 1e-9, 1-1e-9)
            rv = binom(n_trials, p_hat)
            ll = float(np.sum(rv.logpmf(x)))
            cand = {"name": name, "rv": rv, "params": {"n": n_trials, "p": p_hat}, "loglike": ll, "k": 2}
            if (best is None) or (ll > best["loglike"]):
                best = cand
        return best

    if name in AUTO_CONT:
        dist = AUTO_CONT[name]
        try:
            if dist is expon or dist is gamma_dist or dist is beta_dist or dist is pareto:
                params = dist.fit(x, floc=0)
            else:
                params = dist.fit(x)
            rv = dist(*params)
            ll = float(np.sum(rv.logpdf(x)))
            if hasattr(dist, 'shapes') and dist.shapes:
                k = len(dist.shapes.split(',')) + 2
            else:
                k = len(params)
            return {"name": name, "rv": rv, "params_tuple": params, "loglike": ll, "k": k}
        except Exception:
            return None

    return None

# function to suggest distributions based on data
# - Fits multiple candidates and returns the best ones based on AIC/BIC
def suggest_distributions(x: np.ndarray, max_out: int = 5):
    x = np.asarray(x).astype(float)
    x = x[np.isfinite(x)]
    res = []
    stats_dict = summarize_data(x)
    if not stats_dict:
        return res

    candidates = []
    if stats_dict["all_int"] and stats_dict["nonneg"]:
        candidates += ["Bernoulli", "Geométrica", "Poisson", "Binomial Negativa", "Binomial"]
    else:
        candidates += list(AUTO_CONT.keys())
        if stats_dict["bounded_01"] and "Beta" not in candidates:
            candidates.append("Beta")

    for name in candidates:
        fit = fit_candidate(name, x)
        if not fit:
            continue
        k = fit["k"]; ll = fit["loglike"]
        aic = 2*k - 2*ll
        bic = k*np.log(len(x)) - 2*ll
        fit["AIC"] = aic
        fit["BIC"] = bic
        try:
            if name in DISCRETE or name in {"Binomial"}:
                xs = np.arange(int(np.min(x)), int(np.max(x)) + 1)
                obs = np.array([(x == xi).sum() for xi in xs])
                exp = fit["rv"].pmf(xs) * len(x)
                mask = exp > 0
                if mask.sum() >= 2:
                    chi = stats.chisquare(f_obs=obs[mask], f_exp=exp[mask])
                    fit["chi2_pval"] = float(chi.pvalue)
            else:
                ks = stats.kstest(x, fit["rv"].cdf)
                fit["ks_pval"] = float(ks.pvalue)
        except Exception:
            pass
        res.append(fit)

    res.sort(key=lambda d: d["BIC"])  # melhor BIC primeiro
    return res[:max_out]

# -------------------------------------------
# UI
# -------------------------------------------

st.title("Explorador de Distribuições e Pesquisa de Distribuição")

colL, colR = st.columns([1, 2])
with colL:
    name = st.selectbox("Distribuição", list(distributions.keys()), index=list(distributions.keys()).index("Normal"))

    # Sliders conforme o catálogo
    param_defs = distributions[name]["params"]
    values = {}
    for p_name, cfg in param_defs.items():
        min_v, max_v, val = cfg.get("min"), cfg.get("max"), cfg.get("value")
        step = cfg.get("step", None)
        if isinstance(val, int) and isinstance(min_v, int) and isinstance(max_v, int):
            values[p_name] = st.slider(p_name, min_value=min_v, max_value=max_v, value=val, step=step or 1)
        else:
            values[p_name] = st.slider(p_name, min_value=float(min_v), max_value=float(max_v), value=float(val), step=step or 0.01)

    st.markdown("---")
    st.caption(distributions[name]["usage"]) 

    st.markdown("---")
    st.subheader("Seus dados (opcional)")
    uploaded = st.file_uploader("Envie um CSV/XLSX com 1 coluna numérica", type=["csv", "xlsx", "xls"]) 
    df = None
    data_col = None
    x = np.array([])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if num_cols:
                data_col = st.selectbox("Coluna numérica", num_cols)
                x = df[data_col].to_numpy(dtype=float)
                stats_d = summarize_data(x)
                st.json(stats_d)
            else:
                st.info("Não encontrei coluna numérica.")
        except Exception as e:
            st.error(f"Falha ao ler arquivo: {e}")

with colR:
    st.subheader("Fórmula")
    st.latex(b64dec(distributions[name]["formula_b64"]))

    try:
        frozen = build_frozen(name, values)
        xs, ys = support_and_values(name, frozen)
        plot_distribution(name, xs, ys)
        mean, var = None, None
        try:
            mean, var = frozen.stats(moments='mv')
            st.write(f"**Média:** {float(mean):.4g}  •  **Variância:** {float(var):.4g}")
        except Exception:
            pass
    except Exception as e:
        st.error(f"Erro ao gerar a distribuição: {e}")

st.markdown("---")

# Discover distribution from your data
st.subheader("Descobrir distribuição a partir dos seus dados")
detect_btn = st.button("Descobrir agora", disabled=("x" not in globals() or (isinstance(x, np.ndarray) and x.size == 0)))
if detect_btn:
    if "x" not in globals() or (isinstance(x, np.ndarray) and x.size == 0):
        st.info("Envie dados na seção 'Seus dados (opcional)' para prosseguir.")
    else:
        results = suggest_distributions(x, max_out=6)
        if not results:
            st.warning("Não consegui ajustar candidatos aos seus dados. Verifique o suporte (valores negativos, inteiros, etc.).")
        else:
            rows = []
            for r in results:
                rows.append({
                    "Distribuição": r["name"],
                    "AIC": r.get("AIC", np.nan),
                    "BIC": r.get("BIC", np.nan),
                    "KS p-valor": r.get("ks_pval", np.nan),
                    "Chi² p-valor": r.get("chi2_pval", np.nan),
                })
            st.dataframe(pd.DataFrame(rows))

            best = results[0]
            st.success(f"Melhor candidata (BIC): **{best['name']}**")
            try:
                plot_overlay_data_fit(best['name'], best['rv'], x)
            except Exception as e:
                st.warning(f"Plot do ajuste falhou: {e}")

st.markdown("---")
st.caption("Dica: mova os controles para ver a PDF/PMF mudar. Fórmulas renderizadas com LaTeX. Gráficos com Plotly. Clique em 'Descobrir agora' após enviar dados.")
st.caption("Feito com ChatGPT e Streamlit. Distribuições de probabilidade e ajuste automático a partir de dados.")
st.caption("Referência: https://zehmatias.com/data-science/distribuicoes-de-dados/")
st.caption("Código fonte: https://github.com/filmateus/probability_distributions/blob/main/painel_distribuicoes2.py")