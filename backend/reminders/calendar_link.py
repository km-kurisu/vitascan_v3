"""
Pure helpers that turn a reminder into a Google Calendar "Add to calendar"
URL and an .ics payload. No Google API keys or network calls required.
"""
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

DEFAULT_DURATION_MINUTES = 60


def to_utc_iso(value: datetime) -> str:
    """Format a datetime as a UTC string Google Calendar links expect."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def appointment_end_iso(appointment_at: datetime) -> datetime:
    """Return the appointment end time (start + 60 minutes)."""
    return appointment_at + timedelta(minutes=DEFAULT_DURATION_MINUTES)


def build_calendar_url(title: str, start_iso: str, end_iso: str, notes: str = "") -> str:
    """Build a Google Calendar event-creation URL for the appointment."""
    base = "https://calendar.google.com/calendar/render"
    params = {
        "action": "TEMPLATE",
        "text": title,
        "dates": f"{start_iso}/{end_iso}",
        "details": notes,
    }
    querystring = "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())
    return f"{base}?{querystring}"


def build_ics(title: str, start_iso: str, end_iso: str, notes: str = "", uid: str = "") -> str:
    """Build an .ics calendar file payload for the appointment."""
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//VitaScan//Reminders//EN",
        "BEGIN:VEVENT",
        f"UID:{uid or now}-vitascan",
        f"DTSTAMP:{now}",
        f"DTSTART:{start_iso}",
        f"DTEND:{end_iso}",
        f"SUMMARY:{title}",
        f"DESCRIPTION:{notes}" if notes else "DESCRIPTION:",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(lines)
