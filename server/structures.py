from typing import List, Literal, Union, Tuple, Optional
from pydantic import BaseModel, constr, PositiveInt, validator

class FitConfig(BaseModel):
    model_name: constr(min_length=1)
    model_clf: Union[Literal['logistic-regression'], Literal['random-forest'], Literal['boosting']]
    feature_type: Union[Literal['tf-idf'], Literal['bow']]

class PredictConfig(BaseModel):
    model_name: constr(min_length=1)
    top_n: PositiveInt

class ModelConfig(BaseModel):
    model_name: constr(min_length=1)

class Texts(BaseModel):
    values: List[str]

class Labels(BaseModel):
    values: List[str]

class Scores(BaseModel):
    values: List[float]

class Prediction(BaseModel):
    labels_list: List[Labels]
    scores_list: List[Scores]

class ReturnValue(BaseModel):
    success: bool
    message: Optional[str]
    traceback: Optional[str]

class PredictReturnValue(ReturnValue):
    prediction: Optional[Prediction] = None
