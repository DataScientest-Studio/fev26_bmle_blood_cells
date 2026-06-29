"""
Envoi d'alertes email pour le monitoring drift (IVDR 2017/746).

Configuration via variables d'environnement :
  SMTP_HOST        : serveur SMTP (ex : smtp.gmail.com)
  SMTP_PORT        : port TLS (défaut : 587)
  SMTP_USER        : adresse d'envoi
  SMTP_PASSWORD    : mot de passe ou App Password Gmail
  ALERT_EMAILS     : destinataires séparés par virgule
"""

from __future__ import annotations

import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


LEVEL_COLORS = {
    "critique": "#c0392b",
    "alerte":   "#e67e22",
    "warning":  "#f1c40f",
    "normal":   "#27ae60",
    "unknown":  "#95a5a6",
}


def _level_badge(level: str) -> str:
    color = LEVEL_COLORS.get(level, "#95a5a6")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-weight:bold;">{level.upper()}</span>'
    )


def _build_html(drift_result: dict, perf_result: dict | None, generated_at: str) -> str:
    data_score  = drift_result.get("data_drift_score", 0)
    data_level  = drift_result.get("data_drift_level", "unknown")
    pred_score  = drift_result.get("pred_drift_score", 0)
    pred_level  = drift_result.get("pred_drift_level", "unknown")
    model_score = drift_result.get("model_drift_score")
    n_drifted   = drift_result.get("n_drifted_features", 0)
    n_ref       = drift_result.get("n_reference", 0)
    n_cur       = drift_result.get("n_current", 0)
    mv          = drift_result.get("model_version") or "toutes"

    perf_alerts: list[str] = perf_result.get("alerts", []) if perf_result else []

    rows = f"""
    <tr><td>Data drift score</td><td>{data_score:.3f}</td><td>{_level_badge(data_level)}</td></tr>
    <tr><td>Features driftées</td><td>{n_drifted}</td><td></td></tr>
    <tr><td>Prediction drift score</td><td>{pred_score:.3f}</td><td>{_level_badge(pred_level)}</td></tr>
    <tr><td>Model drift (désaccord médecin)</td>
        <td>{f"{model_score:.3f}" if model_score is not None else "N/A"}</td><td></td></tr>
    <tr><td>Images référence / courantes</td><td>{n_ref} / {n_cur}</td><td></td></tr>
    """

    perf_section = ""
    if perf_alerts:
        items = "".join(f"<li>{a}</li>" for a in perf_alerts)
        perf_section = f"""
        <h3 style="color:#c0392b;">⚠ Alertes performances IVDR</h3>
        <ul style="color:#c0392b;">{items}</ul>
        """

    return f"""<!DOCTYPE html>
<html><body style="font-family:Arial,sans-serif;max-width:640px;margin:auto;padding:20px;">
<h2 style="color:#2c3e50;">🔬 Alerte drift — Blood Cell Classifier</h2>
<p style="color:#7f8c8d;">Généré le {generated_at} · Version modèle : {mv}</p>

<table style="border-collapse:collapse;width:100%;">
  <thead>
    <tr style="background:#ecf0f1;">
      <th style="text-align:left;padding:8px;border:1px solid #bdc3c7;">Métrique</th>
      <th style="text-align:left;padding:8px;border:1px solid #bdc3c7;">Valeur</th>
      <th style="text-align:left;padding:8px;border:1px solid #bdc3c7;">Niveau</th>
    </tr>
  </thead>
  <tbody style="font-size:14px;">
    {rows}
  </tbody>
</table>

{perf_section}

<hr style="margin-top:30px;border:none;border-top:1px solid #ecf0f1;">
<p style="color:#95a5a6;font-size:12px;">
  Seuils IVDR/ISO 14971 : warning &gt; 0.10 | alerte &gt; 0.20 | critique &gt; 0.30<br>
  Ce message est envoyé automatiquement par le DAG Airflow <em>blood_cell_drift_monitoring</em>.
</p>
</body></html>"""


def _highest_level(drift_result: dict, perf_result: dict | None) -> str:
    order = ["normal", "warning", "alerte", "critique"]
    levels = [
        drift_result.get("data_drift_level", "normal"),
        drift_result.get("pred_drift_level", "normal"),
    ]
    if perf_result and perf_result.get("alerts"):
        levels.append("alerte")
    model_score = drift_result.get("model_drift_score")
    if model_score is not None:
        if model_score >= 0.15:
            levels.append("alerte")
        elif model_score >= 0.10:
            levels.append("warning")
    return max(levels, key=lambda lvl: order.index(lvl) if lvl in order else 0)


def send_drift_alert(
    drift_result: dict,
    perf_result: dict | None = None,
    min_level: str = "warning",
) -> bool:
    """Envoie un email d'alerte si le niveau de drift dépasse min_level.

    Returns True si un email a été envoyé, False sinon.
    """
    order = ["normal", "warning", "alerte", "critique"]
    level = _highest_level(drift_result, perf_result)

    if order.index(level) < order.index(min_level):
        return False

    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_pass = os.environ["SMTP_PASSWORD"]
    recipients = [r.strip() for r in os.environ["ALERT_EMAILS"].split(",") if r.strip()]

    if not recipients:
        raise ValueError("ALERT_EMAILS est vide — aucun destinataire configuré.")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subject = f"[{level.upper()}] Drift détecté — Blood Cell Classifier ({generated_at})"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(_build_html(drift_result, perf_result, generated_at), "html"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipients, msg.as_string())

    print(f"Email d'alerte [{level.upper()}] envoyé à {', '.join(recipients)}")
    return True
