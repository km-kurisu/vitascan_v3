import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datetime import datetime, timezone

from backend.reminders.schemas import Reminder, ReminderCreate


def test_reminder_create_defaults():
    rc = ReminderCreate(
        patient_id="PAT-1",
        email="a@b.com",
        title="Checkup",
        appointment_at=datetime(2026, 9, 15, 10, 30, tzinfo=timezone.utc),
    )
    assert rc.notes == ""
    assert rc.lead_minutes == 1440


def test_reminder_defaults():
    r = Reminder(
        id="uuid-1",
        patient_id="PAT-1",
        email="a@b.com",
        title="Checkup",
        appointment_at=datetime(2026, 9, 15, 10, 30, tzinfo=timezone.utc),
        lead_minutes=60,
    )
    assert r.status == "confirmed"
    assert r.confirmation_sent_at is None
    assert r.reminder_sent_at is None
    assert r.calendar_link == ""