from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, AliasChoices

class ReminderCreate(BaseModel):
    patient_id: str
    email: str
    title: str = Field(validation_alias=AliasChoices('title', 'supplement_name'))
    notes: str = Field("", validation_alias=AliasChoices('notes', 'dosage'))
    appointment_at: datetime = Field(validation_alias=AliasChoices('appointment_at', 'scheduled_at'))
    lead_minutes: int = Field(1440, description="Minutes before appointment to send reminder")

class Reminder(BaseModel):
    id: str
    patient_id: str
    email: str
    title: str
    notes: str = ""
    appointment_at: datetime
    lead_minutes: int
    status: Literal["confirmed", "sent"] = "confirmed"
    confirmation_sent_at: Optional[datetime] = None
    reminder_sent_at: Optional[datetime] = None
    calendar_link: str = ""
    ics_content: str = ""
