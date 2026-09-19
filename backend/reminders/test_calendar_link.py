import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from datetime import datetime, timezone

from backend.reminders.calendar_link import (
    appointment_end_iso,
    build_calendar_url,
    build_ics,
    to_utc_iso,
)


def test_to_utc_iso():
    dt = datetime(2026, 9, 15, 10, 30, tzinfo=timezone.utc)
    assert to_utc_iso(dt) == "20260915T103000Z"


def test_appointment_end_iso_is_sixty_minutes_later():
    dt = datetime(2026, 9, 15, 10, 30, tzinfo=timezone.utc)
    assert appointment_end_iso(dt) == datetime(2026, 9, 15, 11, 30, tzinfo=timezone.utc)


def test_build_calendar_url_contains_template_fields():
    url = build_calendar_url(
        title="Follow-up with Dr. Rao",
        start_iso="20260915T103000Z",
        end_iso="20260915T113000Z",
        notes="Bring blood report",
    )
    assert url.startswith("https://calendar.google.com/calendar/render?action=TEMPLATE")
    assert "text=Follow-up%20with%20Dr.%20Rao" in url
    assert "dates=20260915T103000Z%2F20260915T113000Z" in url


def test_build_ics_contains_required_fields():
    ics = build_ics(
        title="Follow-up with Dr. Rao",
        start_iso="20260915T103000Z",
        end_iso="20260915T113000Z",
        notes="Bring blood report",
        uid="reminder-abc",
    )
    assert "BEGIN:VCALENDAR" in ics
    assert "BEGIN:VEVENT" in ics
    assert "SUMMARY:Follow-up with Dr. Rao" in ics
    assert "DTSTART:20260915T103000Z" in ics
    assert "DTEND:20260915T113000Z" in ics
    assert "UID:reminder-abc" in ics
