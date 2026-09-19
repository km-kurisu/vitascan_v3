"""
CRUD + lazy due-reminder processing backed by Supabase, with an in-memory
fallback store so the feature works without Supabase credentials.
"""
import logging
import uuid
from datetime import datetime, timedelta, timezone

from backend.reminders.calendar_link import (
    appointment_end_iso,
    build_calendar_url,
    build_ics,
    to_utc_iso,
)
from backend.reminders.emailer import send_confirmation_email, send_reminder_email
from backend.reminders.schemas import Reminder, ReminderCreate
from backend.shared.supabase_client import get_supabase_client

logger = logging.getLogger("vitascan.reminders.service")

_IN_MEMORY: dict[str, Reminder] = {}


def _compute_calendar_fields(r: Reminder) -> Reminder:
    start_iso = to_utc_iso(r.appointment_at)
    end_iso = to_utc_iso(appointment_end_iso(r.appointment_at))
    r.calendar_link = build_calendar_url(r.title, start_iso, end_iso, r.notes)
    r.ics_content = build_ics(r.title, start_iso, end_iso, r.notes, uid=r.id)
    return r


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_reminder(data: ReminderCreate) -> Reminder:
    reminder = Reminder(
        id=str(uuid.uuid4()),
        patient_id=data.patient_id,
        email=data.email,
        title=data.title,
        notes=data.notes,
        appointment_at=data.appointment_at,
        lead_minutes=data.lead_minutes,
    )
    _compute_calendar_fields(reminder)

    confirmed = send_confirmation_email(reminder)
    if confirmed:
        reminder.confirmation_sent_at = _now()
        reminder.status = "confirmed"

    client = get_supabase_client()
    if client is not None:
        try:
            client.table("reminders").insert(
                {
                    "id": reminder.id,
                    "patient_id": reminder.patient_id,
                    "email": reminder.email,
                    "title": reminder.title,
                    "notes": reminder.notes,
                    "appointment_at": reminder.appointment_at.isoformat(),
                    "lead_minutes": reminder.lead_minutes,
                    "status": reminder.status,
                    "confirmation_sent_at": reminder.confirmation_sent_at.isoformat() if reminder.confirmation_sent_at else None,
                    "reminder_sent_at": None,
                }
            ).execute()
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase insert reminder failed (falling back to memory): %s", e)
            _IN_MEMORY[reminder.id] = reminder
    else:
        _IN_MEMORY[reminder.id] = reminder

    return reminder


def list_reminders(patient_id: str) -> list[Reminder]:
    client = get_supabase_client()
    if client is not None:
        try:
            rows = client.table("reminders").select("*").eq("patient_id", patient_id).execute()
            return [_row_to_reminder(r) for r in rows.data]
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase select reminders failed (falling back to memory): %s", e)
    return [r for r in _IN_MEMORY.values() if r.patient_id == patient_id]


def delete_reminder(reminder_id: str) -> bool:
    client = get_supabase_client()
    if client is not None:
        try:
            res = client.table("reminders").delete().eq("id", reminder_id).execute()
            if res.data:
                return True
            return reminder_id in _IN_MEMORY
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase delete reminder failed (falling back to memory): %s", e)
    existed = reminder_id in _IN_MEMORY
    _IN_MEMORY.pop(reminder_id, None)
    return existed


def _row_to_reminder(row: dict) -> Reminder:
    r = Reminder(
        id=row.get("id", ""),
        patient_id=row.get("patient_id", ""),
        email=row.get("email", ""),
        title=row.get("title", ""),
        notes=row.get("notes", ""),
        appointment_at=datetime.fromisoformat(row.get("appointment_at")),
        lead_minutes=int(row.get("lead_minutes", 0)),
        status=row.get("status", "confirmed"),
        confirmation_sent_at=_parse_dt(row.get("confirmation_sent_at")),
        reminder_sent_at=_parse_dt(row.get("reminder_sent_at")),
    )
    return _compute_calendar_fields(r)


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def process_due_reminders(brevo_key: str | None = None) -> int:
    """Send lead-time reminder emails for overdue reminders (run on API calls)."""
    now = _now()
    sent_count = 0
    all_reminders = _all_reminders()
    for r in all_reminders:
        if r.reminder_sent_at is not None:
            continue
        due_at = r.appointment_at - timedelta(minutes=r.lead_minutes)
        if due_at > now:
            continue
        if send_reminder_email(r, brevo_key=brevo_key):
            r.reminder_sent_at = now
            r.status = "sent"
            _update_reminder(r)
            sent_count += 1
    return sent_count


def _all_reminders() -> list[Reminder]:
    client = get_supabase_client()
    if client is not None:
        try:
            rows = client.table("reminders").select("*").execute()
            return [_row_to_reminder(r) for r in rows.data]
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase select all reminders failed (falling back to memory): %s", e)
    return list(_IN_MEMORY.values())


def _update_reminder(r: Reminder) -> None:
    client = get_supabase_client()
    if client is not None:
        try:
            client.table("reminders").update(
                {
                    "status": r.status,
                    "reminder_sent_at": r.reminder_sent_at.isoformat() if r.reminder_sent_at else None,
                }
            ).eq("id", r.id).execute()
            return
        except Exception as e:  # noqa: BLE001
            logger.warning("Supabase update reminder failed (falling back to memory): %s", e)
    _IN_MEMORY[r.id] = r