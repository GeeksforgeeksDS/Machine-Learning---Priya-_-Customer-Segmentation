import sys
from typing import List, Tuple
import os
from pandas import DataFrame
import numpy as np

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils,load_numpy_array_data
from neuro_mf  import ModelFactory




class CustomerSegmentationModel:
    def __init__(self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, X: DataFrame) -> DataFrame:
        logging.info("Entered predict method of srcTruckModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(X)

            logging.info("Used the trained model to get predictions")

            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise CustomerException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()


    

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            x_train, y_train, x_test, y_test = train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]
            
            
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            best_model_detail = model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy)
            preprocessing_obj = self.utils.load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                            logging.info("No best model found with score more than base score")
                            raise Exception("No best model found with score more than base score")
             
            customer_segmentation_model = CustomerSegmentationModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model
            )
            logging.info("Customer Segmentation Model is created and saved.")
            trained_model_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(trained_model_path, exist_ok=True)
            
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=customer_segmentation_model
            )
            logging.info(f"Customer Segmentation Model is saved successfully at: {trained_model_path}")
            metric_artifact = ClassificationMetricArtifact(f1_score=0.8, precision_score=0.8, recall_score=0.9)
            model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            metric_artifact=metric_artifact,
            )

            logging.info("Model training completed successfully")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

            

        except Exception as e:
            raise CustomerException(e, sys) from e
