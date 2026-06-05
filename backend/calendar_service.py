import os
import json
import logging
import threading
import datetime
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("CalendarService")

# Thread lock to prevent race-condition bookings in multi-threaded environment
calendar_lock = threading.Lock()

class CalendarService:
    def __init__(self):
        self.calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
        self.creds_json = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_JSON")
        self.google_service = None
        self.use_mock = True
        
        # Path to local stateful mock calendar
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.mock_db_path = os.path.join(os.path.dirname(current_dir), "data", "mock_calendar.json")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.mock_db_path), exist_ok=True)
        self._init_mock_db()

        # Try to initialize real Google Calendar API client
        if self.creds_json:
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
                
                logger.info("Initializing Google Calendar client with service account credentials...")
                creds_info = json.loads(self.creds_json)
                credentials = service_account.Credentials.from_service_account_info(
                    creds_info,
                    scopes=["https://www.googleapis.com/auth/calendar"]
                )
                self.google_service = build("calendar", "v3", credentials=credentials)
                self.use_mock = False
                logger.info("Google Calendar client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to load Google Calendar API: {e}. Falling back to stateful mock calendar.")
        else:
            logger.info("No GOOGLE_CALENDAR_CREDENTIALS_JSON found. Operating in stateful Mock Calendar mode.")

    def _init_mock_db(self):
        """Initializes mock database with seed data if it doesn't exist."""
        if not os.path.exists(self.mock_db_path):
            # Seed with some busy slots
            today = datetime.date.today().isoformat()
            tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
            
            initial_data = {
                # Format: "YYYY-MM-DD": {"slot": {"attendee_email": "...", "name": "...", "event_id": "..."}}
                today: {
                    "10:00-11:00": {"attendee_email": "recruiter@scaler.com", "name": "Scaler Recruiter", "event_id": "event_seed_1"},
                    "14:00-15:00": {"attendee_email": "hiring.manager@tech.com", "name": "Tech Manager", "event_id": "event_seed_2"}
                },
                tomorrow: {
                    "11:00-12:00": {"attendee_email": "hr@faang.com", "name": "FAANG HR", "event_id": "event_seed_3"}
                }
            }
            self._save_mock_db(initial_data)

    def _load_mock_db(self) -> Dict[str, Any]:
        try:
            with open(self.mock_db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read mock calendar db: {e}")
            return {}

    def _save_mock_db(self, data: Dict[str, Any]):
        try:
            with open(self.mock_db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write mock calendar db: {e}")

    def get_work_slots(self) -> List[str]:
        """Returns standard booking slots between 9 AM and 5 PM."""
        return [
            "09:00-10:00",
            "10:00-11:00",
            "11:00-12:00",
            "12:00-13:00",
            "13:00-14:00",
            "14:00-15:00",
            "15:00-16:00",
            "16:00-17:00"
        ]

    def get_available_slots(self, date_str: str) -> List[str]:
        """Returns list of open slots (e.g. ['09:00-10:00', '13:00-14:00']) for a date YYYY-MM-DD."""
        try:
            # Parse input date
            query_date = datetime.date.fromisoformat(date_str)
        except ValueError:
            logger.error(f"Invalid date format: {date_str}")
            return []

        today = datetime.date.today()
        # Don't show slots for past dates
        if query_date < today:
            return []

        work_slots = self.get_work_slots()
        booked_slots = set()

        if self.use_mock:
            db = self._load_mock_db()
            date_bookings = db.get(date_str, {})
            booked_slots = set(date_bookings.keys())
        else:
            # Fetch from real Google Calendar
            try:
                # Calculate timeMin and timeMax
                time_min = f"{date_str}T00:00:00Z"
                time_max = f"{date_str}T23:59:59Z"
                
                events_result = self.google_service.events().list(
                    calendarId=self.calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy="startTime"
                ).execute()
                
                events = events_result.get("items", [])
                for event in events:
                    start_time = event.get("start", {}).get("dateTime", "")
                    # Convert to hour string, e.g. "2026-06-05T10:00:00+05:30" -> "10:00-11:00"
                    if start_time:
                        dt = datetime.datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                        hour_str = f"{dt.strftime('%H')}:00"
                        # Match work slots
                        for ws in work_slots:
                            if ws.startswith(hour_str):
                                booked_slots.add(ws)
            except Exception as e:
                logger.error(f"Failed to query Google Calendar API: {e}. Defaulting to empty bookings.")

        # Filter out past hours if query_date is today
        available = []
        now_dt = datetime.datetime.now()
        
        for slot in work_slots:
            if slot in booked_slots:
                continue
                
            # If checking for today, skip slots that are already in the past
            if query_date == today:
                slot_start_hour = int(slot.split("-")[0].split(":")[0])
                if now_dt.hour >= slot_start_hour:
                    continue
                    
            available.append(slot)
            
        return available

    def create_event(self, date_str: str, slot: str, attendee_email: str, attendee_name: str) -> Tuple[bool, Dict[str, Any]]:
        """Creates a calendar event. Implements double-booking conflict protection.
        
        Returns:
            (success_bool, event_details_dict)
        """
        # Ensure only one thread executes booking at a time to prevent race conditions
        with calendar_lock:
            # 1. Double check availability
            available_slots = self.get_available_slots(date_str)
            if slot not in available_slots:
                logger.warning(f"Conflict detected! Slot {slot} on {date_str} is no longer available.")
                return False, {"error": "Conflict: This slot is already booked. Please choose another time."}

            event_id = f"evt_{int(datetime.datetime.utcnow().timestamp())}"
            
            # 2. Book the event
            if self.use_mock:
                # Stateful local storage booking
                db = self._load_mock_db()
                if date_str not in db:
                    db[date_str] = {}
                    
                db[date_str][slot] = {
                    "attendee_email": attendee_email,
                    "name": attendee_name,
                    "event_id": event_id
                }
                self._save_mock_db(db)
                
                logger.info(f"Booked event (Mock) {event_id} for {attendee_name} at {slot} on {date_str}")
                return True, {
                    "event_id": event_id,
                    "date": date_str,
                    "slot": slot,
                    "attendee": attendee_name,
                    "email": attendee_email,
                    "type": "Mock Calendar Booking",
                    "status": "confirmed"
                }
            else:
                # Live Google Calendar event creation
                try:
                    start_hour, end_hour = slot.split("-")
                    start_datetime = f"{date_str}T{start_hour}:00"
                    end_datetime = f"{date_str}T{end_hour}:00"
                    
                    # Assume local timezone of user
                    local_tz = "Asia/Kolkata" # standard local timezone based on locale metadata
                    
                    event_body = {
                        "summary": f"Piyush Representative Interview with {attendee_name}",
                        "description": f"Autonomous interview booked by PersonaHire AI representative. Candidate: Piyush Bhardwaj. Recruiter: {attendee_name} ({attendee_email})",
                        "start": {
                            "dateTime": start_datetime,
                            "timeZone": local_tz
                        },
                        "end": {
                            "dateTime": end_datetime,
                            "timeZone": local_tz
                        },
                        "attendees": [
                            {"email": attendee_email, "displayName": attendee_name},
                            {"email": "piyushbhardwaj634@gmail.com", "displayName": "Piyush Bhardwaj"}
                        ],
                        "reminders": {
                            "useDefault": True
                        }
                    }
                    
                    created_event = self.google_service.events().insert(
                        calendarId=self.calendar_id,
                        body=event_body
                    ).execute()
                    
                    real_event_id = created_event.get("id")
                    logger.info(f"Booked Google Event {real_event_id} for {attendee_name} at {slot} on {date_str}")
                    return True, {
                        "event_id": real_event_id,
                        "date": date_str,
                        "slot": slot,
                        "attendee": attendee_name,
                        "email": attendee_email,
                        "type": "Google Calendar Booking",
                        "status": "confirmed",
                        "htmlLink": created_event.get("htmlLink", "")
                    }
                except Exception as e:
                    logger.error(f"Google Calendar event creation failed: {e}")
                    return False, {"error": f"API Error: Could not book event in Google Calendar ({e})"}

    def cancel_event(self, date_str: str, slot: str, event_id: str) -> bool:
        """Cancels an existing calendar booking."""
        with calendar_lock:
            if self.use_mock:
                db = self._load_mock_db()
                if date_str in db and slot in db[date_str]:
                    booking = db[date_str][slot]
                    if booking.get("event_id") == event_id:
                        del db[date_str][slot]
                        # Clean up empty days
                        if not db[date_str]:
                            del db[date_str]
                        self._save_mock_db(db)
                        logger.info(f"Canceled Mock event {event_id} at {slot} on {date_str}")
                        return True
                logger.warning(f"Could not find mock booking {event_id} to cancel.")
                return False
            else:
                try:
                    self.google_service.events().delete(
                        calendarId=self.calendar_id,
                        eventId=event_id
                    ).execute()
                    logger.info(f"Canceled Google Calendar event: {event_id}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to cancel Google Calendar event {event_id}: {e}")
                    return False

if __name__ == "__main__":
    # Test service
    service = CalendarService()
    today = datetime.date.today().isoformat()
    slots = service.get_available_slots(today)
    print(f"Available slots for today ({today}):", slots)
