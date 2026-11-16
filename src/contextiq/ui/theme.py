"""Palette et composants visuels inspirés du dashboard BRVM."""

import streamlit as st


def apply_theme(dark: bool) -> None:
    """Styliser uniquement des composants fixes ; aucun texte documentaire n'est du HTML."""
    colors = (
        ("#0b100f", "#101714", "#edf3ed", "#94a59b", "#b8ed71", "#29372e", "#182217")
        if dark
        else ("#f5f7f2", "#ffffff", "#1d3025", "#526459", "#47751b", "#cbd8c8", "#e5efdc")
    )
    background, panel, text, muted, accent, border, glow = colors
    # Seules ces couleurs constantes sont interpolées ; les données utilisent st.text/st.code.
    st.html(f"""
    <style>
    .stApp {{background:radial-gradient(ellipse at 90% 0%,{glow}, {background} 50%);
      color:{text}; --text-color:{text}; --background-color:{background};
      --secondary-background-color:{panel}; color-scheme:{"dark" if dark else "light"};}}
    .block-container {{max-width:1480px;padding:2.8rem 2.6rem 3rem;}}
    [data-testid="stHeader"] {{background:transparent;}}
    [data-testid="stSidebar"] {{background:{panel};border-right:1px solid {border};}}
    .hero {{padding:12px 0 25px;border-bottom:1px solid {border};margin-bottom:25px;}}
    .eyebrow {{color:{accent};font-size:11px;letter-spacing:3px;font-weight:700;}}
    .hero h1 {{font-size:clamp(34px,4vw,54px);letter-spacing:-2px;color:{text};margin:12px 0;}}
    .hero p {{color:{muted};font-size:15px;max-width:800px;}}
    [data-testid="stMetric"] {{background:{panel};border:1px solid {border};
      border-radius:16px;padding:18px;}}
    [data-testid="stMetricValue"] {{color:{accent};font-size:30px;}}
    [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {{color:{muted};}}
    [data-testid="stCaptionContainer"] p {{color:{muted} !important;}}
    [data-testid="stChatMessage"], [data-testid="stExpander"] {{background:{panel};
      border:1px solid {border};border-radius:16px;}}
    [data-testid="stBaseButton-secondary"] {{background:{panel};color:{text};
      border-color:{border};border-radius:20px;}}
    [data-testid="stBaseButton-secondary"]:hover {{border-color:{accent};color:{accent};}}
    [data-baseweb="select"] > div, [data-baseweb="input"],
    [data-testid="stNumberInput"] input, [data-testid="stChatInput"] textarea {{
      background:{panel};color:{text};}}
    [data-baseweb="select"] > div, [data-testid="stNumberInput"] button,
    [data-testid="stChatInput"], [data-testid="stChatInput"] > div {{
      background-color:{panel} !important;color:{text} !important;}}
    [data-testid="stChatInput"] textarea {{color:{text} !important;}}
    [data-testid="stChatInput"] textarea::placeholder {{color:{muted} !important;opacity:1;}}
    [data-baseweb="popover"], [role="listbox"], [data-testid="stBottomBlockContainer"] {{
      background:{panel};color:{text};}}
    h1,h2,h3,[data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] {{color:{text};}}
    .footer {{margin-top:35px;padding-top:16px;border-top:1px solid {border};
      color:{muted};font-size:12px;}}
    @media(max-width:700px) {{.block-container {{padding:2rem 1rem;}}}}
    </style>
    """)


def header() -> None:
    """Présenter le produit avec un en-tête fixe, indépendant des données affichées."""
    st.html("""
    <div class="hero">
      <div class="eyebrow">CONTEXTIQ · DOCUMENT INTELLIGENCE</div>
      <h1>Une question. Des sources. Une décision.</h1>
      <p>Explorez votre assistant documentaire, comprenez ses réponses et comparez ses méthodes
      de recherche dans un même espace.</p>
    </div>
    """)
