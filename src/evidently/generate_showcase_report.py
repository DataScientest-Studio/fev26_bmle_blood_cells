"""
Génère un rapport HTML showcase mettant en avant les 4 métriques clés du monitoring drift.

Ordre d'importance :
  1. Data drift par feature (images)
  2. Distribution des classes prédites
  3. Drift de confidence
  4. Désaccord médecin

Usage:
    python -m src.evidently.generate_showcase_report
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evidently.drift_report import load_last_report  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "reports_html"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Palette ───────────────────────────────────────────────────────────────────
# Couleurs de fond + texte contrasté pour chaque niveau

_LEVELS = {
    "normal":   {"bg": "#dcfce7", "border": "#16a34a", "text": "#14532d",  "icon": "✅", "label": "Normal"},
    "warning":  {"bg": "#fef9c3", "border": "#ca8a04", "text": "#713f12",  "icon": "⚠️",  "label": "Warning"},
    "alerte":   {"bg": "#ffedd5", "border": "#ea580c", "text": "#7c2d12",  "icon": "🟠", "label": "Alerte"},
    "critique": {"bg": "#fee2e2", "border": "#dc2626", "text": "#7f1d1d",  "icon": "🔴", "label": "Critique"},
    "unknown":  {"bg": "#f1f5f9", "border": "#94a3b8", "text": "#334155",  "icon": "❓", "label": "Inconnu"},
}


def _lvl(level: str) -> dict:
    return _LEVELS.get(level, _LEVELS["unknown"])


def _score_to_level(score: float) -> str:
    if score >= 0.30:
        return "critique"
    if score >= 0.20:
        return "alerte"
    if score >= 0.10:
        return "warning"
    return "normal"


def _score_bar(score: float) -> str:
    pct = min(100, round(score * 100))
    lv = _lvl(_score_to_level(score))
    return (
        '<div style="background:#e2e8f0;border-radius:999px;height:10px;width:100%;margin-top:6px;">'
        f'<div style="background:{lv["border"]};width:{pct}%;height:10px;border-radius:999px;"></div>'
        "</div>"
    )


# ── Sections ──────────────────────────────────────────────────────────────────

def _section_num(num: int, title: str, subtitle: str) -> str:
    return (
        '<div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">'
        '<div style="background:#1e293b;color:#ffffff;border-radius:50%;width:38px;height:38px;'
        'display:flex;align-items:center;justify-content:center;font-weight:700;'
        'font-size:17px;flex-shrink:0;">'
        f"{num}</div>"
        '<div>'
        f'<h2 style="margin:0;font-size:19px;color:#0f172a;font-weight:700;">{title}</h2>'
        f'<p style="margin:2px 0 0;font-size:13px;color:#475569;">{subtitle}</p>'
        '</div></div>'
    )


def _card(content: str) -> str:
    return (
        '<div style="background:#ffffff;border-radius:14px;padding:28px;'
        'margin-bottom:24px;box-shadow:0 1px 6px rgba(0,0,0,0.08);'
        'border:1px solid #e2e8f0;">'
        f"{content}</div>"
    )


def _level_badge(level: str, score: float | None = None) -> str:
    lv = _lvl(level)
    score_str = f" — {score:.4f}" if score is not None else ""
    return (
        f'<span style="background:{lv["bg"]};color:{lv["text"]};border:1px solid {lv["border"]};'
        f'border-radius:6px;padding:3px 10px;font-size:13px;font-weight:600;">'
        f'{lv["icon"]} {lv["label"]}{score_str}</span>'
    )


def _alert_box(level: str, content: str) -> str:
    lv = _lvl(level)
    return (
        f'<div style="background:{lv["bg"]};border-left:5px solid {lv["border"]};'
        f'border-radius:8px;padding:16px 20px;margin-bottom:20px;">'
        f'<div style="color:{lv["text"]}">{content}</div></div>'
    )


# ── Légende ───────────────────────────────────────────────────────────────────

def _legend() -> str:
    rows = [
        (
            "✅", "Normal", "< 0.10", "#14532d", "#dcfce7", "#16a34a",
            "Aucun drift significatif détecté.",
            "Le modèle se comporte comme lors de son entraînement. Aucune action requise.",
        ),
        (
            "⚠️", "Warning", "0.10 – 0.20", "#713f12", "#fef9c3", "#ca8a04",
            "Drift léger, à surveiller.",
            "Changement marginal de distribution. Augmenter la fréquence de surveillance "
            "et vérifier si la tendance s'aggrave sur les prochains rapports.",
        ),
        (
            "🟠", "Alerte", "0.20 – 0.30", "#7c2d12", "#ffedd5", "#ea580c",
            "Drift modéré, investigation recommandée.",
            "Changement notable par rapport aux données d'entraînement. "
            "Analyser les features ou classes concernées. "
            "Envisager un ré-entraînement ou une revue du protocole d'acquisition "
            "(IVDR MDCG 2020-1).",
        ),
        (
            "🔴", "Critique", "≥ 0.30", "#7f1d1d", "#fee2e2", "#dc2626",
            "Drift sévère, action obligatoire.",
            "Déviation majeure de la distribution de référence. "
            "Investigation immédiate requise (ISO 14971 §9). "
            "Suspendre les prédictions si la sécurité patient est compromise. "
            "Ré-entraînement ou validation clinique nécessaire.",
        ),
    ]

    header = (
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        '<thead><tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">'
        '<th style="padding:12px 14px;text-align:left;color:#475569;font-weight:600;">Niveau</th>'
        '<th style="padding:12px 14px;text-align:left;color:#475569;font-weight:600;">Score</th>'
        '<th style="padding:12px 14px;text-align:left;color:#475569;font-weight:600;">Signification</th>'
        '<th style="padding:12px 14px;text-align:left;color:#475569;font-weight:600;">Action recommandée (IVDR)</th>'
        '</tr></thead><tbody>'
    )
    body = ""
    for icon, label, score_range, text_color, bg, border, meaning, action in rows:
        body += (
            f'<tr style="border-bottom:1px solid #f1f5f9;">'
            f'<td style="padding:12px 14px;white-space:nowrap;">'
            f'<span style="background:{bg};color:{text_color};border:1px solid {border};'
            f'border-radius:6px;padding:4px 12px;font-weight:700;font-size:13px;">'
            f'{icon} {label}</span></td>'
            f'<td style="padding:12px 14px;color:#0f172a;font-weight:700;'
            f'white-space:nowrap;font-size:14px;">{score_range}</td>'
            f'<td style="padding:12px 14px;color:#0f172a;font-weight:600;'
            f'font-size:13px;">{meaning}</td>'
            f'<td style="padding:12px 14px;color:#334155;font-size:13px;'
            f'line-height:1.6;">{action}</td>'
            f'</tr>'
        )

    return (
        '<div style="background:#ffffff;border-radius:14px;padding:28px;'
        'margin-bottom:24px;box-shadow:0 1px 6px rgba(0,0,0,0.08);border:1px solid #e2e8f0;">'
        '<h2 style="margin:0 0 6px;font-size:17px;color:#0f172a;font-weight:700;">'
        '📖 Seuils d\'alerte &amp; actions recommandées</h2>'
        '<p style="margin:0 0 18px;font-size:13px;color:#475569;">'
        'Définis conformément à IVDR 2017/746, MDCG 2020-1 et ISO 14971 '
        '— applicables au data drift, prediction drift et confidence drift.</p>'
        + header + body + '</tbody></table>'
        '</div>'
    )


# ── Section 1 : Data drift par feature ───────────────────────────────────────

def _section_data_drift(metrics: dict) -> str:
    dd = metrics.get("data_drift", {})
    level = dd.get("level", "unknown")
    score = dd.get("share", 0.0)
    n_drifted = dd.get("n_drifted_features", 0)
    per_feature = dd.get("per_feature", {})
    lv = _lvl(level)

    summary_content = (
        f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
        f'<span style="font-size:28px;">{lv["icon"]}</span>'
        f'<div>'
        f'<div style="font-size:24px;font-weight:800;color:{lv["text"]};">'
        f'{score:.1%} des features en drift ({n_drifted}/{len(per_feature)})</div>'
        f'<div style="font-size:13px;color:{lv["text"]};margin-top:2px;opacity:0.85;">'
        f'Niveau : <strong>{lv["label"].upper()}</strong></div>'
        f'</div></div>'
    )

    _th = 'style="padding:10px 14px;text-align:left;color:#475569;font-weight:600;border-bottom:2px solid #e2e8f0;"'  # noqa: E501
    rows = ""
    for feat, v in per_feature.items():
        fl = _lvl(_score_to_level(v["drift_score"]))
        detected_label = "Drift" if v["drift_detected"] else "Stable"
        status = f'<span style="color:{fl["text"]};font-weight:600;">{fl["icon"]} {detected_label}</span>'
        rows += (
            f'<tr style="border-bottom:1px solid #f1f5f9;">'
            f'<td style="padding:10px 14px;font-weight:600;color:#0f172a;">{feat}</td>'
            f'<td style="padding:10px 14px;color:#475569;font-size:13px;">{v["stattest"]}</td>'
            f'<td style="padding:10px 14px;min-width:160px;">'
            f'<span style="font-weight:700;color:{fl["text"]};">{v["drift_score"]:.4f}</span>'
            f'{_score_bar(v["drift_score"])}</td>'
            f'<td style="padding:10px 14px;">{status}</td>'
            f'</tr>'
        )

    table = (
        '<table style="width:100%;border-collapse:collapse;font-size:14px;">'
        '<thead><tr style="background:#f8fafc;">'
        f'<th {_th}>Feature</th>'
        f'<th {_th}>Test statistique</th>'
        f'<th {_th}>Score</th>'
        f'<th {_th}>Statut</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )

    definition = (
        '<div style="background:#f8fafc;border-radius:8px;padding:14px 18px;'
        'margin-bottom:20px;border:1px solid #e2e8f0;">'
        '<p style="font-size:13px;color:#334155;line-height:1.7;margin:0;">'
        '<strong style="color:#0f172a;">Qu\'est-ce que le data drift ?</strong> '
        'Le data drift mesure si les caractéristiques visuelles des images soumises au modèle '
        '(luminosité, contraste, canaux RGB, dimensions) ont changé par rapport aux images '
        'd\'entraînement. Un drift peut indiquer un changement de microscope, de colorant, '
        'de protocole d\'acquisition ou de population de patients.'
        '</p></div>'
    )

    return _card(
        _section_num(1, "Data Drift — Features Image",
                     "Référence : Source_full 2 400 img · Courant : prédictions en production")
        + definition
        + _alert_box(level, summary_content)
        + table
    )


# ── Section 2 : Distribution des classes ─────────────────────────────────────

def _section_pred_drift(metrics: dict) -> str:
    pd_ = metrics.get("prediction_drift", {})
    level = pd_.get("level", "unknown")
    score = pd_.get("score", 0.0)
    detected = pd_.get("detected", False)
    lv = _lvl(level)
    status = "détecté" if detected else "non détecté"

    content = (
        f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
        f'<span style="font-size:28px;">{lv["icon"]}</span>'
        f'<div style="flex:1;min-width:200px;">'
        f'<div style="font-size:24px;font-weight:800;color:{lv["text"]};">Score : {score:.4f}</div>'
        f'<div style="font-size:13px;color:{lv["text"]};margin-top:2px;opacity:0.85;">'
        f'Niveau : <strong>{lv["label"].upper()}</strong> — Drift {status}</div>'
        f'{_score_bar(score)}'
        f'</div></div>'
    )

    definition = (
        '<div style="background:#f8fafc;border-radius:8px;padding:14px 18px;'
        'margin-bottom:20px;border:1px solid #e2e8f0;">'
        '<p style="font-size:13px;color:#334155;line-height:1.7;margin:0;">'
        '<strong style="color:#0f172a;">Qu\'est-ce que le drift de distribution des classes ?</strong> '
        'Ce score mesure si la répartition des classes prédites (Basophil, Neutrophil, IG…) '
        'a changé par rapport à la distribution de référence. Un drift peut signaler un biais '
        'de sélection des patients, un changement épidémiologique, ou une dérive du comportement '
        'du modèle sur certaines classes.'
        '</p></div>'
    )

    return _card(
        _section_num(2, "Distribution des Classes Prédites",
                     "Drift sur predicted_class — Jensen-Shannon divergence")
        + definition
        + _alert_box(level, content)
    )


# ── Section 3 : Confidence drift ─────────────────────────────────────────────

def _section_confidence_drift(metrics: dict) -> str:
    score = metrics.get("prediction_drift", {}).get("confidence_drift", 0.0)
    level = _score_to_level(score)
    lv = _lvl(level)

    content = (
        f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
        f'<span style="font-size:28px;">{lv["icon"]}</span>'
        f'<div style="flex:1;min-width:200px;">'
        f'<div style="font-size:24px;font-weight:800;color:{lv["text"]};">Score : {score:.4f}</div>'
        f'<div style="font-size:13px;color:{lv["text"]};margin-top:2px;opacity:0.85;">'
        f'Niveau : <strong>{lv["label"].upper()}</strong></div>'
        f'{_score_bar(score)}'
        f'</div></div>'
    )

    definition = (
        '<div style="background:#f8fafc;border-radius:8px;padding:14px 18px;'
        'margin-bottom:20px;border:1px solid #e2e8f0;">'
        '<p style="font-size:13px;color:#334155;line-height:1.7;margin:0;">'
        '<strong style="color:#0f172a;">Qu\'est-ce que le drift de confidence ?</strong> '
        'La confidence est le score de certitude (0 à 1) que le modèle attribue à chaque prédiction. '
        'Un drift de confidence indique que le modèle est globalement moins (ou plus) certain '
        'sur les nouvelles images qu\'au moment de l\'entraînement — signal précoce de dégradation, '
        'même si les prédictions de classe semblent encore correctes.'
        '</p></div>'
    )

    return _card(
        _section_num(3, "Drift de Confidence",
                     "Distribution de la confidence du modèle — Wasserstein distance")
        + definition
        + _alert_box(level, content)
    )


# ── Section 4 : Désaccord médecin ────────────────────────────────────────────

def _section_model_drift(metrics: dict) -> str:
    md = metrics.get("model_drift", {})
    header = _section_num(
        4, "Revue du Médecin",
        "Feedback clinicien — seule métrique avec ground truth réel",
    )
    if not md:
        return _card(
            header
            + '<div style="background:#f8fafc;border-radius:8px;padding:20px;'
            'text-align:center;color:#475569;font-size:14px;">'
            'Aucun feedback médecin enregistré pour le moment.</div>'
        )

    n = md.get("n_feedback", 0)
    coverage = md.get("coverage_rate", 0.0)
    accuracy = md.get("accuracy", 0.0)
    disagree = md.get("disagree_rate", 0.0)
    level = _score_to_level(disagree)
    lv = _lvl(level)

    coverage_html = (
        '<div style="background:#f1f5f9;border-radius:8px;padding:10px 16px;'
        'margin-bottom:16px;font-size:13px;color:#475569;">'
        '<strong style="color:#0f172a;">Taux de couverture</strong> : proportion des prédictions '
        'du modèle ayant reçu un retour (accord ou désaccord) du médecin.'
        '</div>'
    )

    _card_style = "border-radius:12px;padding:20px;text-align:center;"
    cards_html = (
        coverage_html
        + '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px;">'
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;{_card_style}">'
        f'<div style="font-size:36px;font-weight:800;color:#0f172a;">{n}</div>'
        '<div style="font-size:13px;color:#475569;margin-top:6px;font-weight:500;">Feedbacks enregistrés</div>'
        '</div>'
        f'<div style="background:#dcfce7;border:1px solid #16a34a;{_card_style}">'
        f'<div style="font-size:36px;font-weight:800;color:#14532d;">{accuracy:.0%}</div>'
        '<div style="font-size:13px;color:#14532d;margin-top:6px;font-weight:500;">'
        'Taux d\'accord (images reviewées)</div>'
        '</div>'
        f'<div style="background:{lv["bg"]};border:1px solid {lv["border"]};{_card_style}">'
        f'<div style="font-size:36px;font-weight:800;color:{lv["text"]};">{disagree:.0%}</div>'
        f'<div style="font-size:13px;color:{lv["text"]};margin-top:6px;font-weight:500;">'
        f'Taux de désaccord (images reviewées) {lv["icon"]}</div>'
        '</div>'
        '</div>'
    )

    note = (
        '<div style="background:#fef9c3;border-left:5px solid #ca8a04;border-radius:8px;'
        'padding:14px 18px;font-size:13px;color:#713f12;">'
        f'⚠️ <strong>Note :</strong> Ces taux portent uniquement sur les {n} images ayant reçu '
        f'un feedback ({coverage:.0%} des prédictions). '
        'La significativité statistique est atteinte à partir de ~50 feedbacks (IVDR MDCG 2020-1).'
        '</div>'
    )

    return _card(header + cards_html + note)


# ── HTML complet ──────────────────────────────────────────────────────────────

def build_showcase_html(report: dict) -> str:
    """Construit le HTML showcase. Utilisable depuis Streamlit via st.components.v1.html()."""
    metrics = report.get("metrics", {})
    created_at = str(report.get("created_at", ""))[:19].replace("T", " ")
    n_ref = report.get("n_reference", 0)
    n_cur = report.get("n_current", 0)
    model_version = "DenseNet-121"

    header_html = (
        '<div style="background:#1e293b;border-radius:14px;padding:32px;'
        'margin-bottom:24px;">'
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;'
        'flex-wrap:wrap;gap:16px;">'
        '<div>'
        '<h1 style="font-size:26px;font-weight:800;color:#ffffff;margin:0 0 6px;">Rapport Monitoring Drift</h1>'
        '<p style="color:#94a3b8;font-size:14px;margin:0;">'
        'Blood Cell Classification · IVDR 2017/746 · ISO 14971</p>'
        '</div>'
        '<div style="text-align:right;font-size:13px;color:#cbd5e1;line-height:1.8;">'
        f'<div>Généré le <strong style="color:#ffffff;">{created_at}</strong></div>'
        f'<div>Modèle : <strong style="color:#ffffff;">{model_version}</strong></div>'
        f'<div><strong style="color:#ffffff;">{n_ref}</strong> imgs référence · '
        f'<strong style="color:#ffffff;">{n_cur}</strong> prédictions courantes</div>'
        '</div></div></div>'
    )

    footer_html = (
        '<div style="text-align:center;font-size:12px;color:#94a3b8;padding:12px 0 24px;">'
        'Seuils IVDR / ISO 14971 — warning ≥ 0.10 · alerte ≥ 0.20 · critique ≥ 0.30'
        '</div>'
    )

    return (
        "<!DOCTYPE html><html lang='fr'><head><meta charset='UTF-8'>"
        "<style>"
        "* {box-sizing:border-box;margin:0;padding:0;}"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "background:#f1f5f9;color:#0f172a;-webkit-font-smoothing:antialiased;}"
        ".container{max-width:920px;margin:0 auto;padding:28px 20px;}"
        "table th,table td{vertical-align:middle;}"
        "</style></head><body>"
        "<div class='container'>"
        + header_html
        + _legend()
        + _section_data_drift(metrics)
        + _section_pred_drift(metrics)
        + _section_confidence_drift(metrics)
        + _section_model_drift(metrics)
        + footer_html
        + "</div></body></html>"
    )


def generate_showcase_report() -> Path:
    report = load_last_report()
    if not report:
        print("❌ Aucun rapport en base — lancez d'abord generate_report()")
        sys.exit(1)

    html = build_showcase_html(report)
    out = OUTPUT_DIR / "showcase_drift_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ Rapport généré : {out}")
    print(f"   Ouvrir : file://{out.absolute()}")
    return out


if __name__ == "__main__":
    generate_showcase_report()
