"""
Brevo email integration for reminder confirmations and lead-time reminders.
Degrades gracefully (logs + returns False) when no API key is configured.
"""
import base64
import html
import logging
import os

import httpx

from backend.reminders.calendar_link import (
    appointment_end_iso,
    build_calendar_url,
    build_ics,
    to_utc_iso,
)
from backend.reminders.schemas import Reminder

logger = logging.getLogger("vitascan.reminders.emailer")

BREVO_URL = "https://api.brevo.com/v3/smtp/email"


def get_brevo_key() -> str | None:
    return os.getenv("BREVO_API_KEY", "").strip() or None


def get_brevo_from() -> str:
    return os.getenv("BREVO_FROM", "VitaScan <kamleshkmistry33@gmail.com>")


def _split_sender(value: str) -> tuple[str, str]:
    if "<" in value:
        name, email = (part.strip() for part in value.split("<", 1))
        return (name or "VitaScan"), email.rstrip(">").strip()
    return "VitaScan", value.strip()


def _format_appointment(reminder: Reminder) -> str:
    local = reminder.appointment_at.astimezone()
    return local.strftime("%A, %B %d, %Y at %I:%M %p")


def _build_email_body(reminder: Reminder) -> str:
    start_iso = to_utc_iso(reminder.appointment_at)
    end_iso = to_utc_iso(appointment_end_iso(reminder.appointment_at))
    link = build_calendar_url(reminder.title, start_iso, end_iso, reminder.notes)
    ics = build_ics(reminder.title, start_iso, end_iso, reminder.notes, uid=reminder.id)
    text = (
        f"Hi,\n\n"
        f"Your appointment '{reminder.title}' is scheduled for "
        f"{_format_appointment(reminder)}.\n\n"
        f"Add it to Google Calendar: {link}\n\n"
        f"(An .ics file is attached so you can import it into any calendar.)\n\n"
        f"Notes: {reminder.notes or 'None'}\n\n"
        f"-- VitaScan"
    )
    return text, link, ics


def _render_html(body: str, link: str, kind: str) -> str:
    """Wrap the appointment body in a clean, branded HTML email layout."""
    link_label = "Add to Google Calendar" if kind == "confirmation" else "View Calendar Link"
    escaped = html.escape(body).replace("\n", "<br/>")
    safe_link = html.escape(link, quote=True)
    return (
        '<div style="font-family:Arial,Helvetica,sans-serif;background:#f5f7fa;margin:0;padding:24px;">'
        '<div style="max-width:560px;margin:0 auto;background:#ffffff;border-radius:16px;'
        'overflow:hidden;border:1px solid #e5e7eb;">'
        '<div style="background:#1D61E7;color:#ffffff;padding:20px 28px;">'
        '<span style="font-size:20px;font-weight:700;">VitaScan</span>'
        "</div>"
        '<div style="padding:28px;color:#1f2937;font-size:15px;line-height:1.6;">'
        f"{escaped}"
        f'<div style="margin:24px 0;">'
        f'<a href="{safe_link}" style="display:inline-block;background:#1D61E7;color:#ffffff;'
        f'text-decoration:none;padding:12px 22px;border-radius:8px;font-weight:600;">{link_label}</a>'
        f"</div>"
        f'<p style="color:#6b7280;font-size:13px;">If the button does not work, '
        f'copy this link: <a href="{safe_link}" style="color:#1D61E7;">{html.escape(link)}</a></p>'
        "</div>"
        '<div style="background:#f1f5f9;padding:16px 28px;color:#9ca3af;font-size:12px;">'
        "This is an automated appointment reminder from VitaScan."
        "</div>"
        "</div>"
        "</div>"
    )


def _send(brevo_key: str | None, to_email: str, subject: str, html_content: str, ics: str | None = None) -> bool:
    if not brevo_key:
        logger.warning("BREVO_API_KEY not set; skipping email to %s (subject: %s)", to_email, subject)
        return False
    sender_name, sender_email = _split_sender(get_brevo_from())
    payload = {
        "sender": {"email": sender_email, "name": sender_name},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_content,
    }
    if ics is not None:
        payload["attachment"] = [
            {
                "content": base64.b64encode(ics.encode("utf-8")).decode("ascii"),
                "name": "appointment.ics",
            }
        ]
    try:
        response = httpx.post(url=BREVO_URL, json=payload, headers={"api-key": brevo_key}, timeout=15)
        if response.status_code >= 400:
            logger.warning("Brevo returned status %s: %s", response.status_code, response.text)
            return False
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Brevo request failed: %s", e)
        return False


def send_confirmation_email(reminder: Reminder, brevo_key: str | None = None) -> bool:
    """Send the immediate confirmation email with a calendar link and .ics."""
    brevo_key = brevo_key or get_brevo_key()
    body, link, ics = _build_email_body(reminder)
    html = _render_html(body, link, "confirmation")
    subject = f"Medical Appointment Reminder: {reminder.title}"
    return _send(brevo_key, reminder.email, subject, html, ics=ics)


def send_reminder_email(reminder: Reminder, brevo_key: str | None = None) -> bool:
    """Send the lead-time reminder email."""
    brevo_key = brevo_key or get_brevo_key()
    body, link, _ = _build_email_body(reminder)
    html = _render_html(body, link, "reminder")
    subject = f"Reminder: {reminder.title} on {_format_appointment(reminder)}"
    return _send(brevo_key, reminder.email, subject, html)