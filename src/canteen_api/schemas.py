from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    students_on_campus: int = Field(gt=0)
    menu_price_rs: float = Field(ge=15)
    temp_c: float = Field(le=75)
    exam_days_left: int = Field(ge=0)
    special_dishes: int = Field(ge=0)
    queue_minutes_yesterday: float = Field(ge=0)
    day_type: str
    block: str


class PredictionResponse(BaseModel):
    prediction: float