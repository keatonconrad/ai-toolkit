from comet_ml import Experiment
from contextlib import contextmanager


class CometMLClient:
    """
    A wrapper around the CometML Experiment class
    """
    
    def __init__(self, api_key: str, project_name: str, workspace: str):
        self.api_key = api_key
        self.project_name = project_name
        self.workspace = workspace

    def create_experiment(self, **kwargs) -> Experiment:
        """
        Creates a CometML experiment
        @param kwargs: Any overrides for the default settings
        @return: A CometML experiment
        """
        default_settings = {
            "log_graph": True,
            "log_code": True,
            "auto_metric_logging": True,
            "auto_param_logging": True,
            "auto_histogram_weight_logging": True,
        }
        # Update default settings with any overrides provided in kwargs
        default_settings.update(kwargs)
        return Experiment(
            api_key=self.api_key,
            project_name=self.project_name,
            workspace=self.workspace,
            **default_settings
        )

    @contextmanager
    def managed_experiment(self, **kwargs):
        """
        A context manager for CometML experiments
        @param kwargs: Any overrides for the default settings
        @return: A CometML experiment
        """
        experiment = self.create_experiment(**kwargs)
        try:
            yield experiment
        finally:
            experiment.end()
