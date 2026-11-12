from pydantic import BaseModel, ConfigDict, Field


class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "alcohol": 13.2, "malic_acid": 2.77, "ash": 2.51,
                "alcalinity_of_ash": 18.5, "magnesium": 96.0, "total_phenols": 1.9,
                "flavanoids": 0.58, "nonflavanoid_phenols": 0.5,
                "proanthocyanins": 1.25, "color_intensity": 5.4, "hue": 0.75,
                "od280_od315": 1.8, "proline": 600.0,
            }
        }
    )


class Prediction(BaseModel):
    prediction: int
    label: str
    probabilities: dict[str, float]


class BatchRequest(BaseModel):
    instances: list[WineFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
