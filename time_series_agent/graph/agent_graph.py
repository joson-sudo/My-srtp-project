"""
AgentGraph for Time Series Prediction Agent
Based on LangGraph's workflow for time series prediction
"""

import os
import pandas as pd
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
import numpy as np

logger = logging.getLogger(__name__)
from agents.preprocess_agent import PreprocessAgent
from agents.analysis_agent import AnalysisAgent
from agents.validation_agent import ValidationAgent
from agents.forecast_agent import ForecastAgent
from agents.report_agent import ReportAgent
from utils.data_utils import DataLoader, DataSplitter, DataPreprocessor
from utils.file_utils import FileManager

class TimeSeriesAgentGraph:
    """
    LangGraph-based orchestrator for time series forecasting agents.
    Each agent is a node; this class only manages orchestration and state transitions.
    """
    def __init__(self, config: Dict[str, Any], model: str = "gpt-4o", debug: bool = False):
        self.config = config
        self.model = model
        self.debug = debug
        self.file_manager = FileManager(config.get('output_dir', 'results'))
        self.path_manager = self.file_manager.path_manager
        # Instantiate agents (API key is read from environment by each agent)
        self.preprocess_agent = PreprocessAgent(model, config)
        self.analysis_agent = AnalysisAgent(model, config)
        self.validation_agent = ValidationAgent(model, config)
        self.forecast_agent = ForecastAgent(model, config)
        self.report_agent = ReportAgent(model, config)
        # Build LangGraph workflow
        self.graph = self._build_graph()

    def _create_agent_nodes(self):
        return {
            "preprocess": self._preprocess_node,
            "analyze": self._analyze_node,
            "validate": self._validate_node,
            "forecast": self._forecast_node,
            "report": self._report_node,
        }

    def _preprocess_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = self.preprocess_agent.run(state["validation_data"])
        state["preprocessed_data"] = result if isinstance(result, pd.DataFrame) else result.get("cleaned_data", state["validation_data"])
        state["preprocess_result"] = result
        return state

    def _analyze_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        visualizations = state["preprocess_result"]["visualizations"]
        result = self.analysis_agent.run(state["preprocessed_data"], visualizations)
        state["analysis_result"] = result
        logger.info("Analysis node completed for slice_id=%s", state.get("slice_info", {}).get("slice_id", "unknown"))
        return state

    def _validate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        available_models = self.config.get("models", {}).get(
            "available_models",
            self.config.get("available_models", []),
        )
        logger.info("Validation node received %s available models", len(available_models))
        
        # Pass validation data to validation agent
        validation_data = state["preprocessed_data"]
        result = self.validation_agent.run(state["analysis_result"], available_models, validation_data)
        state["validation_result"] = result
        
        # Extract selected models and hyperparameters from result
        # Result is a list of model dicts with validation scores
        state["selected_models"] = [m['model'] for m in result]
        state["best_hyperparameters"] = {m['model']: m['hyperparameters'] for m in result}
        # Also store validation scores for reference
        state["model_validation_scores"] = {m['model']: m['validation_score'] for m in result}
        state["validation_metrics"] = {m['model']: m.get('validation_metrics', {}) for m in result}
        
        return state

    def _forecast_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Pass selected models, best hyperparameters, and test data to forecast agent
        logger.info(
            "Forecast node processing slice_id=%s selected_models=%s",
            state.get("slice_info", {}).get("slice_id", "unknown"),
            state.get("selected_models", []),
        )
        
        result = self.forecast_agent.run(
            state["selected_models"], 
            state["best_hyperparameters"], 
            state["preprocessed_data"],
            state["test_data"],
            output_dir=self.config.get("output_dir", "results"),
            validation_metrics=state.get("validation_metrics", {}),
        )
        
        # print(f"Forecast node: Result keys: {list(result.keys()) if result else 'None'}")
        # if result:
        #     print(f"Forecast node: Individual predictions: {len(result.get('individual_predictions', {}))} models")
        #     print(f"Forecast node: Ensemble predictions: {'Yes' if result.get('ensemble_predictions') else 'No'}")
        #     print(f"Forecast node: Test metrics: {len(result.get('test_metrics', {}))} models")
        
        state["forecast_result"] = result
        return state

    def _report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(
            "Report node processing slice_id=%s",
            state.get("slice_info", {}).get("slice_id", "unknown"),
        )
        
        # Create experiment summary from state
        experiment_summary = {
            'slice_info': state.get('slice_info', {}),
            'preprocess_result': {
                'cleaned_data_shape': state.get('preprocessed_data', pd.DataFrame()).shape if state.get('preprocessed_data') is not None else None,
                'analysis_report': state.get('preprocess_result', {}).get('analysis_report', {}),
                'visualizations': state.get('preprocess_result', {}).get('visualizations', {}),
                'outlier_info': state.get('preprocess_result', {}).get('outlier_info', {}),
                'preprocess_config': state.get('preprocess_result', {}).get('preprocess_config', {})
            },
            'analysis_result': state.get('analysis_result', {}),
            'validation_result': {
                'selected_models': state.get('selected_models', []),
                'best_hyperparameters': state.get('best_hyperparameters', {}),
                'model_validation_scores': state.get('model_validation_scores', {})
            },
            'forecast_result': {
                'individual_predictions': state.get('forecast_result', {}).get('individual_predictions', {}),
                'ensemble_predictions': state.get('forecast_result', {}).get('ensemble_predictions', {}),
                'test_metrics': state.get('forecast_result', {}).get('test_metrics', {}),
                'forecast_metrics': state.get('forecast_result', {}).get('forecast_metrics', {}),
                'confidence_intervals': state.get('forecast_result', {}).get('confidence_intervals', {}),
                'visualizations': state.get('forecast_result', {}).get('visualizations', {})
            },
            'config': state.get('config', {})
        }
        
        report = self.report_agent.run(experiment_summary)
        state["report"] = report
        
        # IMPORTANT: Keep forecast_result in state for aggregation
        # The forecast_result should already be in state from _forecast_node
        # We don't need to modify it here, just ensure it's preserved
        
        logger.info("Report node completed for slice_id=%s", state.get("slice_info", {}).get("slice_id", "unknown"))
        
        return state

    def _build_graph(self):
        nodes = self._create_agent_nodes()
        workflow = StateGraph(dict)
        workflow.add_node("preprocess", nodes["preprocess"])
        workflow.add_node("analyze", nodes["analyze"])
        workflow.add_node("validate", nodes["validate"])
        workflow.add_node("forecast", nodes["forecast"])
        workflow.add_node("report", nodes["report"])
        workflow.add_edge("preprocess", "analyze")
        workflow.add_edge("analyze", "validate")
        workflow.add_edge("validate", "forecast")
        workflow.add_edge("forecast", "report")
        workflow.add_edge("report", END)
        workflow.set_entry_point("preprocess")
        return workflow.compile()

    def run(self) -> dict:
        logger.info("Start loading data")
        run_started_at = pd.Timestamp.utcnow()
        data_path = self.config.get('data_path')
        df = DataLoader.load_data(data_path)
        date_column = self.config.get('date_column', 'date')
        value_column = self.config.get('value_column', 'OT')
        df_ts = DataPreprocessor.convert_to_time_series(df, date_column, value_column)
        num_slices = self.config.get('num_slices', 10)
        input_length = self.config.get('input_length', 512)
        horizon = self.config.get('horizon', 96)
        slices = DataSplitter.create_slices(df_ts, num_slices, input_length, horizon)
        all_results = []
        
        import time
        delay_between_slices = int(self.config.get("slice_delay_seconds", 1))

        logger.info(
            "Processing %s slices with delay=%ss between slices",
            len(slices),
            delay_between_slices,
        )
        
        for i, s in enumerate(slices):
            slice_start_time = time.time()
            logger.info("Processing slice %s/%s (slice_id=%s)", i + 1, len(slices), s['slice_id'])
            
            validation_data = s['validation']
            test_data = s['test']
            slice_info = {
                'slice_id': s['slice_id'],
                'validation_start': s['validation_start'],
                'validation_end': s['validation_end'],
                'test_start': s['test_start'],
                'test_end': s['test_end'],
            }
            # Build initial state for this slice
            state = {
                "validation_data": validation_data,
                "test_data": test_data,
                "slice_info": slice_info,
                "config": self.config
            }
            if self.debug:
                trace = []
                for chunk in self.graph.stream(state):
                    trace.append(chunk)
                final_state = trace[-1]
            else:
                final_state = self.graph.invoke(state)
            
            all_results.append(final_state)
            
            slice_end_time = time.time()
            slice_duration = slice_end_time - slice_start_time
            logger.info(
                "Slice %s/%s completed in %.2fs (selected_models=%s)",
                i + 1,
                len(slices),
                slice_duration,
                final_state.get("selected_models", []),
            )
            
            if i < len(slices) - 1 and delay_between_slices > 0:
                time.sleep(delay_between_slices)
        
        logger.info("All %s slices processed successfully", len(slices))
        
        aggregated_results = self._aggregate_slice_results(all_results)
        run_finished_at = pd.Timestamp.utcnow()
        runtime_summary = {
            "started_at_utc": run_started_at.isoformat(),
            "finished_at_utc": run_finished_at.isoformat(),
            "duration_seconds": (run_finished_at - run_started_at).total_seconds(),
            "num_slices": len(slices),
            "llm_provider": self.config.get("llm_provider"),
            "llm_model": self.config.get("llm_model"),
        }
        
        return {
            "all_results": all_results, 
            "aggregated_results": aggregated_results,
            "report": all_results[-1].get("report") if all_results else None,
            "runtime_summary": runtime_summary,
        }
    
    def _aggregate_slice_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all slices by averaging predictions and metrics."""
        if not all_results:
            logger.warning("No slice result available for aggregation")
            return {}

        logger.info("Aggregating %s slice results", len(all_results))

        all_individual_predictions = {}
        all_ensemble_predictions = []
        all_test_metrics = {}
        all_forecast_metrics = {}
        
        for result in all_results:
            forecast_result = None

            if 'forecast_result' in result:
                forecast_result = result['forecast_result']
            elif 'report' in result and isinstance(result['report'], dict):
                if 'forecast_result' in result['report']:
                    forecast_result = result['report']['forecast_result']

            if not forecast_result:
                continue

            individual_predictions = forecast_result.get('individual_predictions', {})
            for model_name, predictions in individual_predictions.items():
                if model_name not in all_individual_predictions:
                    all_individual_predictions[model_name] = []
                all_individual_predictions[model_name].append(predictions)

            ensemble_predictions = forecast_result.get('ensemble_predictions', {})
            if ensemble_predictions and 'predictions' in ensemble_predictions:
                all_ensemble_predictions.append(ensemble_predictions['predictions'])

            test_metrics = forecast_result.get('test_metrics', {})
            for model_name, metrics in test_metrics.items():
                if model_name not in all_test_metrics:
                    all_test_metrics[model_name] = {'mse': [], 'mae': [], 'mape': []}
                all_test_metrics[model_name]['mse'].append(metrics.get('mse', float('inf')))
                all_test_metrics[model_name]['mae'].append(metrics.get('mae', float('inf')))
                all_test_metrics[model_name]['mape'].append(metrics.get('mape', float('inf')))

            forecast_metrics = forecast_result.get('forecast_metrics', {})
            for model_name, metrics in forecast_metrics.items():
                if model_name not in all_forecast_metrics:
                    all_forecast_metrics[model_name] = {'mean': [], 'std': [], 'min': [], 'max': [], 'range': []}
                all_forecast_metrics[model_name]['mean'].append(metrics.get('mean', 0))
                all_forecast_metrics[model_name]['std'].append(metrics.get('std', 0))
                all_forecast_metrics[model_name]['min'].append(metrics.get('min', 0))
                all_forecast_metrics[model_name]['max'].append(metrics.get('max', 0))
                all_forecast_metrics[model_name]['range'].append(metrics.get('range', 0))

        logger.info(
            "Aggregation source summary: individual_models=%s ensemble_slices=%s metric_models=%s",
            len(all_individual_predictions),
            len(all_ensemble_predictions),
            len(all_test_metrics),
        )

        averaged_individual_predictions = {}
        for model_name, predictions_list in all_individual_predictions.items():
            if predictions_list:
                min_len = min(len(p) for p in predictions_list)
                clipped = [p[:min_len] for p in predictions_list]
                predictions_array = np.array(clipped)
                averaged_predictions = np.mean(predictions_array, axis=0)
                averaged_individual_predictions[model_name] = averaged_predictions.tolist()

        averaged_ensemble_predictions = {}
        if all_ensemble_predictions:
            min_len = min(len(p) for p in all_ensemble_predictions)
            clipped = [p[:min_len] for p in all_ensemble_predictions]
            ensemble_array = np.array(clipped)
            averaged_ensemble = np.mean(ensemble_array, axis=0)
            averaged_ensemble_predictions = {
                'predictions': averaged_ensemble.tolist(),
                'method_used': 'average_across_slices',
                'num_slices': len(clipped)
            }

        averaged_test_metrics = {}
        for model_name, metrics_list in all_test_metrics.items():
            averaged_test_metrics[model_name] = {
                'mse': np.mean(metrics_list['mse']),
                'mae': np.mean(metrics_list['mae']),
                'mape': np.mean(metrics_list['mape'])
            }

        averaged_forecast_metrics = {}
        for model_name, metrics_list in all_forecast_metrics.items():
            averaged_forecast_metrics[model_name] = {
                'mean': np.mean(metrics_list['mean']),
                'std': np.mean(metrics_list['std']),
                'min': np.mean(metrics_list['min']),
                'max': np.mean(metrics_list['max']),
                'range': np.mean(metrics_list['range'])
            }

        aggregated_results = {
            'individual_predictions': averaged_individual_predictions,
            'ensemble_predictions': averaged_ensemble_predictions,
            'test_metrics': averaged_test_metrics,
            'forecast_metrics': averaged_forecast_metrics,
            'aggregation_info': {
                'num_slices': len(all_results),
                'aggregation_method': 'average',
                'slice_ids': [result.get('slice_info', {}).get('slice_id', i) for i, result in enumerate(all_results)]
            }
        }

        logger.info(
            "Aggregation completed: individual_models=%s ensemble_available=%s test_metric_models=%s",
            len(averaged_individual_predictions),
            bool(averaged_ensemble_predictions),
            len(averaged_test_metrics),
        )
        
        return aggregated_results 