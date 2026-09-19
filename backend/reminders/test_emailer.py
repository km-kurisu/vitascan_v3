import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import base64
from datetime import datetime, timezone
from unittest.mock import patch

from backend.reminders import emailer
from backend.reminders.schemas import Reminder


def make_reminder(**overrides):
    base = {
        "id": "uuid-1",
        "patient_id": "PAT-1",
        "email": "a@b.com",
        "title": "Checkup",
        "appointment_at": datetime(2026, 9, 15, 10, 30, tzinfo=timezone.utc),
        "lead_minutes": 60,
    }
    base.update(overrides)
    return Reminder(**base)


def test_send_confirmation_email_without_key_returns_false_and_logs():
    r = make_reminder()
    with patch.object(emailer, "logger") as mock_logger:
        result = emailer.send_confirmation_email(r, brevo_key=None)
    assert result is False
    mock_logger.warning.assert_called()


def test_send_confirmation_email_posts_to_brevo():
    r = make_reminder()
    with patch("backend.reminders.emailer.httpx.post") as mock_post:
        mock_post.return_value.status_code = 201
        result = emailer.send_confirmation_email(r, "xkeysib_abc")
    assert result is True
    _, kwargs = mock_post.call_args
    assert kwargs["url"] == emailer.BREVO_URL
    assert kwargs["headers"]["api-key"] == "xkeysib_abc"
    payload = kwargs["json"]
    assert "attachment" in payload
    attachment = payload["attachment"][0]
    assert attachment["name"] == "appointment.ics"
    assert b"BEGIN:VCALENDAR" in base64.b64decode(attachment["content"])
def test_send_reminder_email_posts_to_brevo():
    r = make_reminder()
    with patch("backend.reminders.emailer.httpx.post") as mock_post:
        mock_post.return_value.status_code = 201
        result = emailer.send_reminder_email(r, "xkeysib_abc")
    assert result is True
    _, kwargs = mock_post.call_args
    assert kwargs["url"] == emailer.BREVO_URL
    assert kwargs["headers"]["api-key"] == "xkeysib_abc"


def test_render_html_escapes_body_and_link_wraps_in_layout():
    rendered = emailer._render_html(
        "Hi,\n\nYour appointment", "https://calendar.google.com/event?a=1&b=2", "confirmation"
    )
    assert rendered.startswith("<div")
    assert "VitaScan" in rendered
    assert "Add to Google Calendar" in rendered
    assert "&amp;b=2" in rendered
    assert "Hi,<br/><br/>Your appointment" in rendered
    assert "This is an automated appointment reminder from VitaScan." in rendered


def test_render_html_uses_reminder_button_label():
    rendered = emailer._render_html("body", "https://example.com", "reminder")
    assert "View Calendar Link" in rendered
    assert "Add to Google Calendar" not in rendered

