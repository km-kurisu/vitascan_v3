import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from backend.reminders import reminders_service
from backend.reminders.schemas import ReminderCreate


def make_create(appointment_at):
    return ReminderCreate(
        patient_id="PAT-1",
        email="a@b.com",
        title="Checkup",
        appointment_at=appointment_at,
        lead_minutes=60,
    )


def test_process_due_reminders_marks_sent(monkeypatch):
    due = datetime.now(timezone.utc) - timedelta(hours=1)
    r = reminders_service.create_reminder(make_create(due))
    assert r.status == "confirmed"

    with patch("backend.reminders.reminders_service.send_reminder_email", return_value=True) as mock_send:
        count = reminders_service.process_due_reminders(brevo_key="re_abc")

    assert count == 1
    mock_send.assert_called_once()
    refetched = reminders_service.list_reminders(r.patient_id)
    assert len(refetched) == 1
    assert refetched[0].reminder_sent_at is not None
    assert refetched[0].status == "sent"


def test_process_due_reminders_skips_future(monkeypatch):
    future = datetime.now(timezone.utc) + timedelta(days=1)
    reminders_service.create_reminder(make_create(future))

    with patch("backend.reminders.reminders_service.send_reminder_email", return_value=True) as mock_send:
        count = reminders_service.process_due_reminders(brevo_key="re_abc")

    assert count == 0
    mock_send.assert_not_called()


def test_create_reminder_sends_confirmation_and_returns_link(monkeypatch):
    future = datetime.now(timezone.utc) + timedelta(days=1)
    with patch("backend.reminders.reminders_service.send_confirmation_email", return_value=True):
        r = reminders_service.create_reminder(make_create(future))
    assert r.status == "confirmed"
    assert r.confirmation_sent_at is not None
    assert "calendar.google.com" in r.calendar_link
    assert "BEGIN:VCALENDAR" in r.ics_content


def test_delete_reminder_returns_exists(monkeypatch):
    future = datetime.now(timezone.utc) + timedelta(days=1)
    with patch("backend.reminders.reminders_service.send_confirmation_email", return_value=True):
        r = reminders_service.create_reminder(make_create(future))
    assert reminders_service.delete_reminder(r.id) is True
    assert reminders_service.delete_reminder(r.id) is False