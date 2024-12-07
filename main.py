import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, ValidationError, field_validator, ValidationInfo, PrivateAttr
from pydantic.main import create_model
from typing import List, Optional, Dict, Union
from enum import Enum
import instructor
from openai import OpenAI
from anthropic import Anthropic
from langsmith import wrappers, traceable

# Constants
API_ENDPOINT = "https://ocr.api.mx2.law/doc/{}/text?token={}"
COMPLAINT_API_ENDPOINT = "https://sf-sync.api.mx2.dev/doc/{}/complaint-filed?salesforce_env=prod&token={}"
API_TOKEN = st.secrets["API_KEY"]

MODEL_CONFIGS = {
    "GPT-4o": {
        "provider": "openai",
        "model": "gpt-4o"
    },
    "Claude-3-Sonnet": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022"
    },
}

# Initialize OpenAI client with instructor
# client = instructor.patch(wrappers.wrap_openai(OpenAI(api_key=st.secrets["OPENAI_API_KEY"])))
# client = instructor.from_anthropic(Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"]))
def get_client(model_name: str) -> Union[OpenAI, Anthropic]:
    config = MODEL_CONFIGS[model_name]
    if config["provider"] == "anthropic":
        return instructor.from_anthropic(Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"]))
    else:
        return instructor.patch(wrappers.wrap_openai(OpenAI(api_key=st.secrets["OPENAI_API_KEY"])))

# Enums
class CircuitCourtNumber(BaseModel):
    number: int = Field(..., description="The circuit court number, should be between 1 and 20.")
    
    @field_validator('number')
    def validate_circuit_court_number(cls, value):
        if not (1 <= value <= 20):
            raise ValueError("Circuit court number should always be between 1 and 20.")
        return value

class JuryOrNonJuryEnum(str, Enum):
    JURY = "Jury"
    NON_JURY = "Non-Jury"
    UNKNOWN = "Unknown"

class CaseManagementTrackEnum(str, Enum):
    STANDARD = "Standard"
    STREAMLINED = "Streamlined"
    DIFFERENTIATED = "Differentiated"

class PartyTypeEnum(str, Enum):
    PLAINTIFF = "Plaintiff"
    DEFENDANT = "Defendant"
    BOTH = "Both"

class ConditionEnum(str, Enum):
    AFTER = "After"
    BEFORE = "Before"
    ON = "On"

class EventTypeEnum(str, Enum):
    """All possible event types based on deadline fields with specific circuit court assignments"""
    # Trial and Court Related Deadlines
    TRIAL_START = "Trial Start Date"
    PRE_TRIAL_CONFERENCE = "Pre-Trial Conference Date"
    TRIAL_DAYS = "Trial Days"
    PERFECT_SERVICE = "Perfect Service of Process Deadline"
    SERVICE_COMPLAINT = "Service of Complaint Deadline"
    SERVICE_EXTENSION = "Service under any Extension of Time Deadline"
    SERVICE_NEW_PARTIES = "Service to Add New Parties Deadline"
    ADD_NEW_PARTIES = "Add New Parties Deadline"
    
    # Discovery Related Deadlines
    DISCOVERY_NOTICE = "Discovery Notice of Compliance Deadline"
    SERVICE_DISCOVERY_REQUESTS = "Service of Written Discovery Requests Deadline"
    ALL_DISCOVERY = "All Discovery Deadline"
    PROPOUNDING_REQUESTS = "Deadline for Propounding Requests for Production, Requests to Admit and Interrogatories"
    FACT_DISCOVERY = "Completion of Fact Discovery Deadline"
    
    # Witness Related Deadlines
    FACT_WITNESSES = "Disclosure of Fact Witnesses Deadline"
    FACT_WITNESS_NOTICE = "Fact Witness Notice of Compliance Deadline"
    EXPERT_WITNESSES = "Disclosure of Expert Witnesses Deadline"
    EXPERT_WITNESSES_CME = "Expert Witnesses and CME Deadline"
    FINAL_TRIAL_WITNESSES = "Disclosure of Final Trial Witnesses Deadline"
    TRIAL_WITNESS_NOTICE = "Final Trial Witnesses Deadline- Notice of Compliance"
    
    # Expert Related Deadlines
    PLAINTIFF_REBUTTAL = "Filing of Plaintiff Rebuttal Experts Deadline"
    DEFENDANT_REBUTTAL = "Filing of Defendant Rebuttal Experts Deadline"
    EXPERT_DISCOVERY = "Expert Discovery and Response Deadline/Discovery Deadline"
    EXPERT_OPINION = "Expert Opinion to Opposing Party Deadline"
    
    # Motion Related Deadlines
    AMEND_PLEADINGS = "Motions to Amend Pleadings Deadline"
    DISPOSITIVE_MOTIONS = "Dispositive Motions Filed and Served Deadline"
    DISPOSITIVE_MOTIONS_FILED = "Dispositive Motions Filed Deadline"
    DISPOSITIVE_MOTIONS_HEARD = "Dispositive Motion Heard Deadline"
    SUMMARY_JUDGMENT = "Summary Judgment Motions Filed and Served Deadline"
    SUMMARY_JUDGMENT_HEARD = "Motion for Summary Judgement Heard Deadline"
    PLEADING_MOTIONS = "Motions directed to the Pleading Filed and Heard Deadline"
    MOTIONS_IN_LIMINE = "Motions in Limine Noticed and Heard Deadline"
    MOTIONS_IN_LIMINE_NOTICE = "Motions in Limine Noticed Date Deadline"
    ALL_MOTIONS = "All Motions Noticed and Heard Deadline"
    OPEN_MOTION_CALENDAR = "Open Motion Calendar Deadline"
    OBJECTIONS_TO_PLEADINGS = "Objections to Pleadings Deadline"
    
    # Daubert Related Deadlines
    DAUBERT_MOTIONS = "Daubert Motions Filed and Served Deadline"
    DAUBERT_NOTICE = "Daubert Notice and Hearing Deadline"
    
    # Medical Examination Deadlines
    CME_EXAM = "CME Exam Completion Deadline"
    CME_REPORT = "CME Report Provided to Plaintiff Deadline"
    
    # Evidence and Exhibit Related Deadlines
    FACT_WITNESS_LIST = "Fact Witness and Exhibit List Deadline"
    EXHIBITS_LIST = "Exhibits List Deadline"
    SURVEILLANCE = "Disclosure of Surveillance Deadline"
    SURVEILLANCE_NOTICE = "Notice of Compliance - Disclosure of Surveillance Deadline"
    NORTHRUP_MATERIALS = "Disclosure of Northrup Impeachment Materials deadline"
    
    # Deposition Related Deadlines
    DEPO_DESIGNATIONS = "Plaintiff/Defendant exchange and filing of Notice of Depo Designations Deadline"
    COUNTER_DESIGNATIONS = "Objections and Counter Designations Deadline"
    DEPO_OBJECTIONS = "Objections to Depo Designations Notice and Heard Deadline"
    
    # Trial Preparation Deadlines
    TRIAL_PREPARATION = "Exchange Between Parties for Trial Preparation Deadline"
    TRIAL_PREPARATION_NOTICE = "Exchange Between Parties for Trial Preparation - Notice of Compliance Deadline"
    ATTORNEY_MEET = "Attorney Meet/Exchange/Inspect/Pretrial Stipulation Deadline"
    JURY_INSTRUCTIONS = "Proposed Jury Instruction and Verdict Form Deadline"
    
    # Case Management Deadlines
    AGREED_CASE_MANAGEMENT = "Filing of Agreed Case Management Plan/Order"
    CERTIFICATION_REQUIREMENT = "Compliance with Certification Requirement Uniform CMO Deadline"
    STATEMENT_FACTS = "Statement of Facts and Counterclaims Deadline"
    DISPUTED_FACTS = "Identification of Facts to be Disputed Deadline"
    IDENTIFICATION_ISSUES = "Identification of Issues Deadline"
    
    # Alternative Dispute Resolution Deadlines
    MEDIATOR_AGREED = "Mediator Date Agreed To Deadline"
    MEDIATION = "Mediation Completed Deadline"
    FABRE_DEFENDANTS = "Identification of Fabre Defendants Deadline"

# Event Circuit Mappings
EVENT_CIRCUIT_MAPPINGS = {
    # Trial and Court Related
    EventTypeEnum.TRIAL_START: set(range(1, 21)),  # All circuits
    EventTypeEnum.PRE_TRIAL_CONFERENCE: {1, 9, 11, 12, 15, 16, 17, 18, 20},
    EventTypeEnum.TRIAL_DAYS: {1, 3, 17, 18, 20},
    EventTypeEnum.PERFECT_SERVICE: {3, 4, 6, 7, 8, 9, 10, 12, 15, 16, 18},
    EventTypeEnum.SERVICE_COMPLAINT: {2, 3, 4, 10},
    EventTypeEnum.SERVICE_EXTENSION: {2, 3, 4},
    EventTypeEnum.SERVICE_NEW_PARTIES: {2},
    EventTypeEnum.ADD_NEW_PARTIES: {2, 3, 7, 8, 10},
    
    # Discovery Related
    EventTypeEnum.DISCOVERY_NOTICE: {18},
    EventTypeEnum.SERVICE_DISCOVERY_REQUESTS: {7},
    EventTypeEnum.ALL_DISCOVERY: {1, 4, 14},
    EventTypeEnum.PROPOUNDING_REQUESTS: {3},
    EventTypeEnum.FACT_DISCOVERY: {6, 7, 10, 11, 15, 17, 18, 20},
    
    # Witness Related
    EventTypeEnum.FACT_WITNESSES: {1, 7, 11, 17, 18, 20},
    EventTypeEnum.FACT_WITNESS_NOTICE: {18},
    EventTypeEnum.EXPERT_WITNESSES: {1, 3, 7, 11, 12, 13, 16, 17, 18, 20},
    EventTypeEnum.EXPERT_WITNESSES_CME: {1},
    EventTypeEnum.FINAL_TRIAL_WITNESSES: {18},
    EventTypeEnum.TRIAL_WITNESS_NOTICE: {18},
    
    # Expert Related
    EventTypeEnum.PLAINTIFF_REBUTTAL: {8, 11, 18},
    EventTypeEnum.DEFENDANT_REBUTTAL: {8, 11, 12},
    EventTypeEnum.EXPERT_DISCOVERY: {2, 3, 9, 10, 12, 13, 15, 16, 17, 18, 20},
    EventTypeEnum.EXPERT_OPINION: {17, 20},
    
    # Motion Related
    EventTypeEnum.AMEND_PLEADINGS: {6, 9, 12, 15, 17, 18, 20},
    EventTypeEnum.DISPOSITIVE_MOTIONS: {1, 3, 6, 9, 12, 16, 18},
    EventTypeEnum.DISPOSITIVE_MOTIONS_FILED: {7, 8, 13, 17, 20},
    EventTypeEnum.DISPOSITIVE_MOTIONS_HEARD: {3, 12, 16},
    EventTypeEnum.SUMMARY_JUDGMENT: {3, 18},
    EventTypeEnum.SUMMARY_JUDGMENT_HEARD: {18},
    EventTypeEnum.PLEADING_MOTIONS: {15, 16},
    EventTypeEnum.MOTIONS_IN_LIMINE: {16, 18},
    EventTypeEnum.MOTIONS_IN_LIMINE_NOTICE: {16, 18},
    EventTypeEnum.ALL_MOTIONS: {9, 12, 15, 18},
    EventTypeEnum.OPEN_MOTION_CALENDAR: {16},
    EventTypeEnum.OBJECTIONS_TO_PLEADINGS: {2, 4, 7, 9, 10, 12},
    
    # Daubert Related
    EventTypeEnum.DAUBERT_MOTIONS: {9, 11, 18},
    EventTypeEnum.DAUBERT_NOTICE: {11, 18},
    
    # Medical Examination Related
    EventTypeEnum.CME_EXAM: {1, 11, 13, 18},
    EventTypeEnum.CME_REPORT: {18},
    
    # Evidence and Exhibit Related
    EventTypeEnum.FACT_WITNESS_LIST: {3, 9, 12, 15, 16, 18},
    EventTypeEnum.EXHIBITS_LIST: {11, 17, 20},
    EventTypeEnum.SURVEILLANCE: {18},
    EventTypeEnum.SURVEILLANCE_NOTICE: {18},
    EventTypeEnum.NORTHRUP_MATERIALS: {18},
    
    # Deposition Related
    EventTypeEnum.DEPO_DESIGNATIONS: {1, 3, 11, 15, 18},
    EventTypeEnum.COUNTER_DESIGNATIONS: {18},
    EventTypeEnum.DEPO_OBJECTIONS: {18},
    
    # Trial Preparation Related
    EventTypeEnum.TRIAL_PREPARATION: {18},
    EventTypeEnum.TRIAL_PREPARATION_NOTICE: {18},
    EventTypeEnum.ATTORNEY_MEET: {1, 4, 16, 18},
    EventTypeEnum.JURY_INSTRUCTIONS: {1, 3, 11, 15},
    
    # Case Management Related
    EventTypeEnum.AGREED_CASE_MANAGEMENT: {17, 18, 20},
    EventTypeEnum.CERTIFICATION_REQUIREMENT: {14},
    EventTypeEnum.STATEMENT_FACTS: {17, 20},
    EventTypeEnum.DISPUTED_FACTS: {17, 20},
    EventTypeEnum.IDENTIFICATION_ISSUES: {17, 20},
    
    # Alternative Dispute Resolution Related
    EventTypeEnum.MEDIATOR_AGREED: {6, 18},
    EventTypeEnum.MEDIATION: {1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20},
    EventTypeEnum.FABRE_DEFENDANTS: {12, 18}
}

class EventDetail(BaseModel):
    """Model for representing an event and its sub-fields."""
    event_type: EventTypeEnum = Field(..., description="The type of event from the predefined list")
    event_details: Optional[str] = Field(None, description="The full passage of text that includes the details of the event including information about deadline, deadline conditions and deadline reference/anchor date details too.")
    party_type: Optional[PartyTypeEnum] = Field(None, description="The party involved, for eg: (Plaintiff/Defendant/Both).")
    number_of_days: Optional[int] = Field(None, description="Number of days before or after or on the anchor date present in the event details. If not found or there are no days mentioned or some other time unit mentioned, would be 0")
    condition: Optional[ConditionEnum] = Field(None, description="Relation to the anchor date (After/Before/On) mentioned in the event details. If not specified, leave as null.")
    anchor_date: Optional[datetime] = Field(
        None,
        description="IF complaint filed date value is available, this will be equal to that. ONLY use this for absolute dates present directly like complaint_filed_date or e_file_date. For dates that depend on other event dates/deadlines (e.g., discovery deadline etc), use depends_on instead."
    )
    depends_on: Optional[EventTypeEnum] = Field(
        None,
        description="The event type this deadline depends on. REQUIRED for events that depend on other deadlines"
    )
    event_date: Optional[datetime] = Field(None, description="The calculated event date. Do not set this directly unless exact date is present.")

    @field_validator('anchor_date')
    def validate_anchor_date(cls, v, values):
        """Ensure anchor_date is only used appropriately."""
        if v and values.data.get('depends_on'):
            raise ValueError("Cannot set both anchor_date and depends_on. For dates depending on other events, use depends_on.")
        return v

    @field_validator('depends_on')
    def validate_depends_on(cls, v, values):
        """Validate depends_on field."""
        if v and values.data.get('anchor_date'):
            raise ValueError("Cannot set both depends_on and anchor_date.")
        return v

    @traceable(name="Calculate Date")
    def calculate_date(self, base_date: datetime) -> Optional[datetime]:
        """Calculate date based on base_date and conditions."""
        if not base_date or not self.condition or self.number_of_days is None:
            print(f"Event Name: {self.event_type}, Base Date: {base_date}, Condition: {self.condition}, No. of Days: {self.number_of_days}")
            return None

        try:
            base_date = base_date.replace(hour=0, minute=0, second=0, microsecond=0)
            if self.condition == ConditionEnum.AFTER:
                print(f"Calculating date after {self.number_of_days} days from {base_date} for {self.event_type}")
                calculate_date = base_date + timedelta(days=(self.number_of_days))
                return calculate_date
            elif self.condition == ConditionEnum.BEFORE:
                calculate_date = base_date - timedelta(days=self.number_of_days)
                return calculate_date
            elif self.condition == ConditionEnum.ON:
                return base_date
        except Exception as e:
            print(f"Error calculating date for {self.event_type}: {e}")
            return None

    @field_validator('number_of_days')
    def validate_number_of_days(cls, value):
        if value is not None and (value < 0 or value > 1000):
            raise ValueError("Number of days must be between 0 and 1000.")
        return value

class CaseManagementOrder(BaseModel):
    circuit_court: Optional[CircuitCourtNumber] = Field(None, description="The circuit court number in which the case is being litigated, should be between 1 and 100.")
    case_number: Optional[str] = Field(None, description="The court case number, a unique identifier for the court system assigned to the case.")
    e_file_date: Optional[datetime] = Field(None, description="The calendar date on which this document was filed electronically")
    _complaint_filed_date: Optional[datetime] = PrivateAttr(default=None)
    projected_trial_date: Optional[datetime] = Field(None, description="The calendar date of the trial")
    jury_or_non_jury_trial: Optional[JuryOrNonJuryEnum] = Field(None, description="If the trial identifies as 'Jury' or 'Non-Jury' or 'Unknown'. If the text does not contain any mention, identify as 'Unknown'.")
    number_of_trial_days: Optional[int] = Field(None, description="The number of days the trial will go on for.")
    case_management_order_track: Optional[CaseManagementTrackEnum] = Field(None, description="The track the trial is on (standard/general, streamlined/expedited, differentiated/complex).")
    # case_management_plan: Optional[str] = Field(None, description="Required for 9th Circuit, indicates the type of plan being used")

    @classmethod
    def create_model_for_circuit(cls, circuit_number: int):
        # Get valid events for this circuit
        valid_events = {
            event_type for event_type, circuits in EVENT_CIRCUIT_MAPPINGS.items()
            if circuit_number in circuits
        }
        # print(f"Valid events for circuit {circuit_number}: {valid_events}")
        # Create a modified EventDetail class specific to this circuit
        class CircuitEventDetail(EventDetail):
            event_type: EventTypeEnum = Field(..., description="The type of event from the predefined list")
            depends_on: Optional[EventTypeEnum] = Field(
                None,
                description="The event type this deadline depends on based on event details. REQUIRED for events that depend on other deadlines. If dependency is present, it can only be one out of the valid events: {valid_events} ",
                json_schema_extra={
                    "enum": list(valid_events)
                }
            )

            @field_validator('depends_on')
            def validate_depends_on(cls, v):
                if v is None:
                    return v
                if v not in valid_events:
                    raise ValueError(f"Event {v} is not valid for this circuit")
                return v

            @field_validator('event_type')
            def validate_event_type(cls, v):
                if v not in valid_events:
                    raise ValueError(f"Event type {v} is not valid for this circuit")
                return v
        
        # Generate fields for valid events in this circuit    
        fields = {
            event_type.name.lower(): (Optional[CircuitEventDetail], Field(None, description=f"{event_type.value} details"))
            for event_type in valid_events
        }
        return create_model('DynamicCaseManagementOrder', **fields, __base__=cls)

    @property
    def complaint_filed_date(self) -> Optional[datetime]:
        """Get the complaint filed date."""
        return self._complaint_filed_date
    def set_complaint_filed_date(self, date: Optional[datetime]):
        """Set the complaint filed date."""
        self._complaint_filed_date = date
        # Recalculate dates when complaint filed date changes
        if date:
            self.calculate_dates()
    @traceable(name="Calculate Dates")
    def calculate_dates(self):
        """Calculate all event dates."""
        if not self.e_file_date and not self._complaint_filed_date:
            print(f"No complaint filed date set. Cannot calculate dates. {self._complaint_filed_date} or {self.e_file_date}")
            return
        
        # Get all EventDetail instances from __dict__
        events = {
            name: event for name, event in self.__dict__.items() 
            if isinstance(event, EventDetail)
        }

        # Track which events have been calculated
        calculated = set()
        # Create dependency graph for debugging
        dependency_graph = {
            event_type: event.depends_on
            for event_type, event in events.items()
        }
        # print(f"Processing events with dependencies: {dependency_graph}")

        # Loop until all events are calculated or no more progress can be made
        while len(calculated) < len(events):
            progress_made = False

            for event_type, event in events.items():
                if event_type in calculated:
                    continue

                try:
                    base_date = None
                    # Handle exact dates first
                    # if event.event_date:
                    #     calculated.add(event_type)
                    #     continue

                    # Handle anchor dates next
                    if event.anchor_date:
                        base_date = event.anchor_date
                        # event.event_date = event.anchor_date
                        # calculated.add(event_type)
                        # continue

                    elif event.depends_on:
                        # Get the dependent event
                        dep_event = events.get(event.depends_on.name.lower())
                        if not dep_event or not dep_event.event_date:
                            print(f"Warning: Event {event_type} depends on missing event {event.depends_on}")
                            continue
                        if not dep_event.event_date:
                            # Skip if dependent event not calculated yet
                            continue
                        base_date = dep_event.event_date
                        event.anchor_date = base_date
                    else:
                        # Default to filing date if no dependencies
                        base_date = self._complaint_filed_date if self._complaint_filed_date is not None else self.e_file_date

                    # Calculate the date
                    if base_date and event.condition and event.number_of_days is not None:
                        event.event_date = event.calculate_date(base_date)
                        if event.event_date:
                            calculated.add(event_type)
                            progress_made = True
                            print(f"Calculated {event_type}: {event.event_date} "
                                  f"(based on: {event.depends_on or 'anchor_date' if event.anchor_date else 'filing_date'})")

                except Exception as e:
                    print(f"Error calculating {event_type}: {str(e)}")

            # If no progress was made in this iteration, we're stuck
            if not progress_made:
                remaining = set(events.keys()) - calculated
                print(f"Warning: Could not calculate dates for: {remaining}")
                print("Dependency graph for remaining events:")
                for event_type in remaining:
                    print(f"{event_type} depends on: {dependency_graph.get(event_type)}")
                break

    def __init__(self, **data):
        super().__init__(**data)

    @field_validator('number_of_trial_days')
    def validate_trial_days(cls, value):
        if value is not None and value < 0:
            raise ValueError("Number of trial days should always be positive.")
        return value
def get_document_text(document_id: str) -> str:
    """Fetch document text from API."""
    headers = {
        'accept': 'application/json',
        # 'Authorization': f'Bearer {API_TOKEN}'
    }
    response = requests.get(API_ENDPOINT.format(document_id, API_TOKEN), headers=headers)
    if response.status_code == 200:
        return response.json()['text']
    else:
        st.error(f"Error fetching document: {response.status_code}")
        return None

def get_complaint_filed_date(document_id: str) -> Optional[datetime]:
    """Fetch complaint filed date from API."""
    headers = {
        'accept': 'application/json'
    }
    response = requests.get(COMPLAINT_API_ENDPOINT.format(document_id, API_TOKEN), headers=headers)
    if response.status_code == 200:
        complaint_date = response.json()
        if complaint_date:
            # First convert YYYY-MM-DD to MM/DD/YYYY string
            date_str = datetime.strptime(complaint_date, "%Y-%m-%d").strftime("%m/%d/%Y")
            # Then convert to datetime object in MM/DD/YYYY format
            return datetime.strptime(date_str, "%m/%d/%Y")
        else:
            return None
    return None

def process_text(text: str, complaint_date: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process text using LLM and return formatted dataframe."""
    try:
        # Get client based on selected model
        client = get_client(st.session_state.model_selector)
        model_name = MODEL_CONFIGS[st.session_state.model_selector]["model"]

        # First extraction - just get circuit number
        circuit_info = client.chat.completions.create(
            model=model_name,
            response_model=CircuitCourtNumber,
            temperature=0,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "Extract only the circuit court number from the text."},
                {"role": "user", "content": text},
            ],
        )
        
        # Get applicable events for this circuit
        circuit_num = circuit_info.number
        DynamicCMO = CaseManagementOrder.create_model_for_circuit(circuit_num)

        # Get valid events for this circuit
        valid_events = {
            event_type for event_type, circuits in EVENT_CIRCUIT_MAPPINGS.items()
            if circuit_num in circuits
        }

        try:
            # Use your existing extraction logic here
            cmo_assessment = client.chat.completions.create(
                model=model_name,
                response_model=DynamicCMO,
                temperature=0,
                max_tokens=8192,
                messages=[
                    {"role": "system", "content": f"""
                    Extract the most accurate answers to the assessment questions based on the text. The complaint_filed_date is {complaint_date}.
                    IMPORTANT: For any 'depends_on' field, you must use EXACTLY one of these valid event types:
                    {[e.value for e in valid_events]}
                    
                    Do not use raw text descriptions for dependencies.
                    """},
                    {"role": "user", "content": text},
                    # {"role": "system", "content": f"Extract the most accurate answers to the assessment questions based on the text. The complaint_filed_date is {complaint_date}."},
                    # {"role": "user", "content": text},
                ],
            )
        except ValidationError as e:
            print("Validation Error Details:")
            for error in e.errors():
                print(f"Field: {error['loc']}")
                print(f"Invalid Value: {error['input']}")
                print(f"Error: {error['msg']}\n")
            raise

        # Create instance
        cmo_data = cmo_assessment.dict()
        
        # If we have complaint_date, set it as e_file_date before creating instance
        if complaint_date:
            cmo_data['e_file_date'] = complaint_date

        # Create instance and set complaint date after LLM extraction
        cmo_instance = DynamicCMO(**cmo_assessment.dict())
        if complaint_date:
            cmo_instance.set_complaint_filed_date(complaint_date)

        cmo_instance.calculate_dates()
        
        # Convert to DataFrame
        display_rows = []
        csv_rows = []
        response_data = cmo_instance.dict()

        # Add complaint_filed_date to the display and CSV data
        if cmo_instance.complaint_filed_date:
            display_rows.append({
                "Field": "complaint_filed_date",
                "Value": format_date(cmo_instance.complaint_filed_date)
            })
            csv_rows.append({
                "Field": "complaint_filed_date",
                "Value": format_date(cmo_instance.complaint_filed_date)
            })

        for key, value in response_data.items():
            if isinstance(value, dict):  # EventDetail
                sub_dict = {
                    "Field": key,
                    **{sub_key: format_date(sub_value) if isinstance(sub_value, datetime) else sub_value 
                       for sub_key, sub_value in value.items()}
                }
                display_rows.append(sub_dict)
                # For CSV - flatten and format datetime objects
                formatted_value = {
                    k: format_date(v) if isinstance(v, datetime) else v 
                    for k, v in value.items()
                }
                csv_rows.append({
                    "Field": key,
                    "Value": json.dumps(formatted_value)
                })
            else:
                formatted_value = format_date(value) if isinstance(value, datetime) else value
                row = {"Field": key, "Value": formatted_value}
                display_rows.append(row)
                csv_rows.append(row)
        
        display_df = pd.DataFrame(display_rows)
        csv_df = pd.DataFrame(csv_rows)
        display_df.fillna('', inplace=True)
        csv_df.fillna('', inplace=True)       
        return display_df, csv_df
        
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return None, None

def format_date(date):
    """Format datetime objects to MM/DD/YYYY string format."""
    if isinstance(date, datetime):
        return date.strftime("%m/%d/%Y")
    elif isinstance(date, str):
        try:
            if 'T' in date:
                dt = datetime.fromisoformat(date)
                return dt.strftime("%m/%d/%Y")
            else:
                date_str = date.split('\n')[0].strip()
                dt = datetime.strptime(date_str, "%m/%d/%Y")
                return dt.strftime("%m/%d/%Y")
        except ValueError:
            return date
    return None

# Streamlit UI
st.title("Case Management Order Extraction")

selected_model = st.selectbox(
    "Select Model",
    options=list(MODEL_CONFIGS.keys()),
    key="model_selector"
)

# Input field for document ID with persistent key
document_id = st.text_input("Enter Document ID (e.g., aDu3c00000DYjjiCAD)", key="doc_id_input")

# Initialize session state for dataframe if not exists
if 'edited_df' not in st.session_state:
    st.session_state.edited_df = pd.DataFrame()

if st.button("Process Document"):
    if document_id:
        with st.spinner("Fetching document..."):
            text = get_document_text(document_id)
            complaint_date = get_complaint_filed_date(document_id)
            print(f"Complaint Date: {complaint_date}")
            if complaint_date:
                st.info(f"Complaint Filed Date: {complaint_date}")
            
        if text:
            with st.spinner("Processing text..."):
                display_df, csv_df = process_text(text, complaint_date)
                
            if display_df is not None and csv_df is not None:
                st.success("Document processed successfully!")
                st.session_state.edited_df = display_df.copy()
                st.session_state.csv_df = csv_df.copy()

# Display the dataframe if it exists
if not st.session_state.edited_df.empty:
    st.subheader("Extracted Information")

    edited_df = st.data_editor(
        st.session_state.edited_df,
        use_container_width=True,
        height=600,
        hide_index=True,
        key="data_editor"
    )
    # Update state only if changes occurred
    # if edited_df is not None:
    #     st.session_state.edited_df = edited_df.copy()
    # st.session_state.edited_df = edited_df
    # if edited_df is not None:
    #     st.session_state.edited_df = edited_df

    # st.dataframe(display_df, use_container_width=True)
    
    # Add download button with CSV-optimized data
    if 'csv_df' in st.session_state:
        csv = st.session_state.csv_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            f"case_management_order_{document_id}.csv",
            "text/csv",
            key='download-csv'
        )
else:
    st.warning("Please enter a document ID")