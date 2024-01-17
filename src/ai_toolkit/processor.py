import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


class DataProcessor:
    def __init__(self, binary_classification=False):
        self.binary_classification = binary_classification

    @staticmethod
    def get_highest_correlations(
        df: pd.DataFrame, threshold=0.9, drop=False
    ) -> list[tuple]:
        """
        Returns a list of tuples containing the highest correlations in the dataframe
        @param df: The dataframe to check
        @param threshold: The minimum correlation to return
        @param drop: Whether or not to drop the columns that are highly correlated
        @return: A list of tuples containing the highest correlations
        """
        correlations = df.corr().abs().unstack().sort_values(ascending=False)
        correlations = correlations[correlations >= threshold]
        correlations = correlations[correlations < 1]
        correlations = correlations.drop_duplicates()
        if drop:
            df.drop(correlations.index, axis=1, inplace=True)
        return correlations

    def format_data(
        self, df: pd.DataFrame, labels_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Formats the data for training
        @param df: The dataframe to format
        @param labels_column: The column containing the labels
        @return: The formatted dataframe and labels
        """
        labels = df[labels_column]
        df.drop(labels_column, axis=1, inplace=True)
        if self.binary_classification:
            labels = labels.apply(lambda x: 1 if x else 0)
            assert len(labels.unique()) == 2
        assert len(labels) == len(df)
        return df, labels

    def select_from_model(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        model=LogisticRegression(),
        print_features=False,
    ) -> pd.DataFrame:
        """
        Selects the best features from the dataframe using the model
        @param df: The dataframe to select from
        @param labels: The labels for the dataframe
        @param model: The model to use for feature selection
        @return: The dataframe with the best features selected
        """
        selector = SelectFromModel(model)
        selector.fit(df, labels)
        if print_features:
            print(df.columns[selector.get_support()])
        return df[df.columns[selector.get_support()]]

    @staticmethod
    def _pipeline_step(
        transformer: any,
        x_train: pd.DataFrame | pd.Series | np.ndarray,
        x_test: pd.DataFrame | pd.Series | np.ndarray,
    ) -> tuple[
        any,
        pd.DataFrame | pd.Series | np.ndarray,
        pd.DataFrame | pd.Series | np.ndarray,
    ]:
        """
        Fits and transforms the data using the transformer
        @param transformer: The transformer to use
        @param x_train: The training data
        @param x_test: The test data
        """
        transformer.fit(x_train)
        x_train = transformer.transform(x_train)
        x_test = transformer.transform(x_test)
        return transformer, x_train, x_test

    @staticmethod
    def fit_transform_data(
        pipeline: list[any],
        x_train: pd.DataFrame | pd.Series | np.ndarray,
        x_test: pd.DataFrame | pd.Series | np.ndarray,
    ) -> tuple[
        pd.DataFrame | pd.Series | np.ndarray,
        pd.DataFrame | pd.Series | np.ndarray,
    ]:
        """
        Fits and transforms the data using the pipeline
        @param pipeline: The pipeline to use. A list of sklearn transformers
        @param x_train: The training data
        @param x_test: The test data
        """
        for transformer in pipeline:
            transformer, x_train, x_test = DataProcessor.pipeline_step(
                transformer, x_train, x_test
            )
        return x_train, x_test

    @staticmethod
    def estimator_score(
        estimator: any,
        training_data: tuple[pd.DataFrame, pd.Series],
        test_data: tuple[pd.DataFrame, pd.Series],
    ) -> float:
        """
        Fits the estimator and returns the score
        @param estimator: The estimator to fit
        @param training_data: The training data
        @param test_data: The test data
        @return: The score of the estimator
        """
        estimator.fit(*training_data)
        return estimator.score(*test_data)
