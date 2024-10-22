import sys
from pandas import DataFrame

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import PCAConfig
from src.exception import CustomerException
from src.logger import logging


class CreateClusters:
    def __init__(self):
        self.pca_config = PCAConfig()
        
        
    def get_dataset_using_pca(self, preprocessed_data: DataFrame):
        """
            Method Name :   get_dataset_using_pca
            Description :   This method applies PCA over the preprocessed dataset.
            
            Output      :   pca object is created and preprocessed dataset is fitted and returned 
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   0.1
            
        """
        try:
            pca_object=PCA(**self.pca_config.__dict__).fit(preprocessed_data)
            reduced_dataset = pca_object.fit_transform(preprocessed_data)
        
        
            logging.info("PCA transformation is done")
        
            return reduced_dataset
        except Exception as e:
                raise CustomerException(e,sys)
    
    def initialize_clustering(self, preprocessed_data: DataFrame) -> DataFrame:
        """
        Method Name :   initialize_clustering
        Description :   This method initiates the clustering process 
        
        Output      :   Data is clustered and the cluster names are used as lables to the preprocessed data and is returned.
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   0.1
        
        """
        try:
            logging.info("Initializing clustering...")
            
            
            reduced_dataset = self.get_dataset_using_pca(preprocessed_data)
            
            model = KMeans(n_clusters=3).fit(reduced_dataset)

            preprocessed_data[TARGET_COLUMN] = model.labels_.astype(int)
            
            logging.info("Clustering is done")
            
            return preprocessed_data
        
        except Exception as e:
            raise CustomerException(e,sys)



        
