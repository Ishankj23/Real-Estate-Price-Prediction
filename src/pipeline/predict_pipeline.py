import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        Transaction_Type: str,
        Registration_type: str,
        Area: str,
        Property_Type: str,
        Property_Sub_Type: str,
        Nearest_Metro: str,
        Nearest_Mall: str,
        Nearest_Landmark: str,
        parking: str,
        Property_Size: float,
        Bedrooms: int):

        self.Transaction_Type = Transaction_Type
        self.Registration_type = Registration_type
        self.Area = Area
        self.Property_Type = Property_Type
        self.Property_Sub_Type = Property_Sub_Type
        self.Nearest_Metro = Nearest_Metro
        self.Nearest_Mall = Nearest_Mall
        self.Nearest_Landmark = Nearest_Landmark
        self.parking = parking
        self.Property_Size = Property_Size
        self.Bedrooms = Bedrooms

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Transaction_Type": [self.Transaction_Type],
                "Registration_type": [self.Registration_type],
                "Area": [self.Area],
                "Property_Type": [self.Property_Type],
                "Property_Sub_Type": [self.Property_Sub_Type],
                "Nearest_Metro": [self.Nearest_Metro],
                "Nearest_Mall": [self.Nearest_Mall],
                "Nearest_Landmark": [self.Nearest_Landmark],
                "parking": [self.parking],
                "Property_Size": [self.Property_Size],
                "Bedrooms": [self.Bedrooms],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

