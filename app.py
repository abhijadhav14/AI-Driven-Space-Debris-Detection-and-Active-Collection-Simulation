from __future__ import annotations

import pandas as pd

from models.avoidance_ai import build_avoidance_plan
from models.collision_model import detect_collision_events, save_collision_report
from models.trajectory_model import TrajectoryPredictor, save_predicted_positions
from preprocessing.feature_engineering import save_dataset
from simulation.orbit_simulation import (
	plot_collision_map,
	plot_multi_satellite_trajectories,
	plot_satellite_trajectory,
)


def main() -> None:
	print("Space Debris AI Project")

	print("[Phase 1] Building dataset...")
	dataset_path = save_dataset(sample_satellites=20, horizon_minutes=120, step_minutes=10)
	print(f"Dataset generated at: {dataset_path}")

	print("[Phase 2] Training trajectory model...")
	predictor = TrajectoryPredictor()
	metrics = predictor.train(dataset_path)
	print(f"Trajectory model metrics: {metrics}")

	predicted_positions_path = save_predicted_positions(predictor, dataset_path)
	print(f"Predicted positions saved at: {predicted_positions_path}")

	print("[Phase 3] Detecting collision risks...")
	predicted_df = pd.read_csv(predicted_positions_path)
	collision_df = detect_collision_events(predicted_df, threshold_km=400.0, probability_alert=0.7)
	collision_report_path = save_collision_report(
		predicted_df,
		output_path="outputs/reports/collision_risks.csv",
		threshold_km=400.0,
		probability_alert=0.7,
	)
	print(f"Collision report saved at: {collision_report_path}")

	print("[Phase 4] Building avoidance plan...")
	avoidance_path = build_avoidance_plan(collision_df)
	print(f"Avoidance plan saved at: {avoidance_path}")

	print("[Phase 5] Running visualization...")
	first_satellite = pd.read_csv(dataset_path)["satellite"].iloc[0]
	single_plot = plot_satellite_trajectory(dataset_path, first_satellite)
	multi_plot = plot_multi_satellite_trajectories(dataset_path)
	collision_plot = plot_collision_map(predicted_positions_path, collision_report_path)

	print(f"Single trajectory plot: {single_plot}")
	print(f"Multi trajectory plot: {multi_plot}")
	print(f"Collision map plot: {collision_plot}")
	print("Pipeline completed successfully.")


if __name__ == "__main__":
	main()
