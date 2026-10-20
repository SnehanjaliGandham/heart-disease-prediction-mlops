# Screenshot register

Every screenshot is a real capture of a real run, mapped to the assignment task it evidences.
`python scripts/check_evidence.py` lists which files are present and which are still missing.
**PG** = captured in the BITS Prayogshala virtual lab (lab address bar / hostname visible),
**Laptop** = Windows + Docker Desktop, **GitHub** = repository / Actions pages.

| File | Task | What it shows | Where |
|---|---|---|---|
| `task1_01_prayogshala_env.png` | 1 | Prayogshala terminal: `hostname`, `python --version`, `python scripts/check_env.py` all PASS | PG |
| `task1_02_data_download.png` | 1 | `python scripts/download_data.py`: dataset verified against the SHA-256 manifest | PG |
| `task1_03_data_cleaning.png` | 1 | `python -m src.data`: missing values, class balance, clean file saved | PG |
| `task1_04_eda_notebook.png` | 1 | `01_data_acquisition_eda.ipynb` executed: feature distributions by class | PG |
| `task1_04b_eda_distributions.png` | 1 | EDA notebook in Prayogshala: numeric feature distributions by class | PG |
| `task1_05_eda_heatmap.png` | 1 | Correlation heatmap and class balance from the EDA notebook | PG |
| `task2_01_engineered_feature.png` | 2 | Notebook 02, section 2b: `hr_pct_predicted` by diagnosis + correlations | PG |
| `task2_02_training_run.png` | 2 | `python -m src.train`: model comparison, feature ablation, selection reason | PG |
| `task2_03_model_selection.png` | 2 | Notebook 02, section 5b: CV score, over-fitting gap, with / without engineered feature | PG |
| `task2_04_roc_confusion.png` | 2 | Notebook 02: ROC curves of all models and confusion matrix of the selected one | PG |
| `task3_01_parent_run.png` | 3 | MLflow: parent run with one nested run per model | Laptop |
| `task3_02_runs_table.png` | 3 | MLflow: runs table with metrics | Laptop |
| `task3_03_compare_plot.png` | 3 | MLflow compare view: chart | Laptop |
| `task3_04_compare_params.png` | 3 | MLflow compare view: parameters | Laptop |
| `task3_05a_compare_metrics.png` | 3 | MLflow compare view: metrics (1) | Laptop |
| `task3_05b_compare_metrics.png` | 3 | MLflow compare view: metrics (2) | Laptop |
| `task3_06_lr_run.png` | 3 | MLflow: Logistic Regression run - parameters and metrics | Laptop |
| `task3_07_artifacts.png` | 3 | MLflow: run artifacts (plots, CV table, model) | Laptop |
| `task3_08_registry.png` | 3 | MLflow model registry: `heart-disease-classifier` with alias `@champion` | Laptop |
| `task3_09_artifacts.png` | 3 | MLflow: comparison artifacts on the parent run | Laptop |
| `task3_10_prayogshala_mlflow.png` | 3 | MLflow UI running in Prayogshala with the new runs | PG |
| `task3_10a_prayogshala_mlflow_lineage.png` | 3 | MLflow in Prayogshala: 15 runs with source (script / notebook) and dataset digest | PG |
| `task3_11_prayogshala_registry.png` | 3 | MLflow model registry in Prayogshala: versions 1-3, @champion on the newest | PG |
| `task4_01_quality_gate.png` | 4 | `python scripts/quality_gate.py`: every threshold PASS | PG |
| `task4_02_predict_cli.png` | 4 | `python -m src.predict` + `models/` folder (joblib, MLflow format, metadata) | PG |
| `task4_03_run_pipeline.png` | 4 | `bash scripts/run_pipeline.sh` finished: reproducible end-to-end run | PG |
| `task5_01_pipeline_green.png` | 5 | GitHub Actions: all jobs green | GitHub |
| `task5_02_test_summary.png` | 5 | Test summary on the run page | GitHub |
| `task5_03_quality_gate_and_models.png` | 5 | Quality gate and model table in the job summary | GitHub |
| `task5_04_artifacts.png` | 5 | Uploaded artifacts | GitHub |
| `task5_05_test_log.png` | 5 | pytest log in CI | GitHub |
| `task5_06_train_log.png` | 5 | Training log in CI | GitHub |
| `task5_07_readme_badge.png` | 5 | README with the CI status badge | GitHub |
| `task5_08_pr_checks_failed.png` | 5 | Pull request with a deliberately failing test: checks red | GitHub |
| `task5_09_failure_log.png` | 5 | The failing test in the CI log | GitHub |
| `task5_10_pytest_prayogshala.png` | 5 | `python -m pytest --cov` all passed in Prayogshala | PG |
| `task6_01_pytest.png` | 6 | Full test suite passing after the API was added (138 passed) | Laptop |
| `task6_02_docker_build.png` | 6 | `docker build` finished + `docker images` (image size) | Laptop |
| `task6_03_docker_ps.png` | 6 | Container `Up (healthy)` | Laptop |
| `task6_04_smoke_test.png` | 6 | `scripts/smoke_test.py`: 8/8 PASS against the container | Laptop |
| `task6_05a_request.png` | 6 | Swagger UI: `/predict` request | Laptop |
| `task6_05b_response.png` | 6 | Swagger UI: response with prediction, confidence, request id | Laptop |
| `task6_06_validation_422_a.png` | 6 | Invalid input rejected with HTTP 422 (1) | Laptop |
| `task6_06_validation_422_b.png` | 6 | Invalid input rejected with HTTP 422 (2) | Laptop |
| `task6_07_json_logs.png` | 6 | `docker logs`: one JSON line per request, no patient data | Laptop |
| `task6_08a_metrics_requests.png` | 6 | `/metrics`: request counters by route and status | Laptop |
| `task6_08b_metrics_requests.png` | 6 | `/metrics`: latency histogram per route | Laptop |
| `task6_08b_metrics_predictions.png` | 6 | `/metrics`: prediction and drift metrics | Laptop |
| `task6_08c_metrics_requests.png` | 6 | `/metrics`: probability histogram, errors, input-feature summaries | Laptop |
| `task6_08b_metrics_predictions_Table.png` | 6 | Key model metrics filtered in PowerShell | Laptop |
| `task6_09a_notebook_api.png` | 6 | Notebook 03: API and in-process predictions identical | Laptop |
| `task6_10_ci_pipeline.png` | 6 | GitHub Actions with the Docker job | GitHub |
| `task6_11_ci_docker_job.png` | 6 | Docker job steps: build, smoke test, health, non-root, publish | GitHub |
| `task6_12_ghcr_package.png` | 6 | Image published to GitHub Container Registry | GitHub |
| `task7_00_docker_desktop.png` | 7 | Docker Desktop with Kubernetes running | Laptop |
| `task7_01_deployed.png` | 7 | `kubectl get all`: 2 pods, LoadBalancer on localhost:8080, HPA | Laptop |
| `task7_02_smoke_test.png` | 7 | Smoke test through the Kubernetes LoadBalancer | Laptop |
| `task7_03_load_balancing.png` | 7 | Requests answered by both pods | Laptop |
| `task7_04_self_healing.png` | 7 | Deleted pod replaced automatically | Laptop |
| `task7_05_hpa_scaling.png` | 7 | HPA scaling 2 -> 5 pods under load | Laptop |
| `task7_05b_hpa_scale_down.png` | 7 | HPA scaling back to 2 after the load | Laptop |
| `task7_06_load_test.png` | 7 | Load test: 22,880 requests, 0 connection errors | Laptop |
| `task7_06b_top.png` | 7 | `kubectl top pods`: CPU / memory of 5 pods | Laptop |
| `task7_07a_rollout.png` | 7 | Rolling update in progress -> successfully rolled out | Laptop |
| `task7_07b_load_during_rollout.png` | 7 | 0 errors while every pod was replaced | Laptop |
| `task7_08a_broken_release.png` | 7 | Broken image: new pod `ImagePullBackOff`, old pods keep serving | Laptop |
| `task7_08b_rollback.png` | 7 | Smoke test still passing, `rollout undo` | Laptop |
| `task7_09_ci_kubernetes_job.png` | 7 | GitHub Actions job 5: deploy to a kind cluster + monitoring check | GitHub |
| `task8_01_monitoring_deployed.png` | 8 | Prometheus and Grafana pods + LoadBalancer services | Laptop |
| `task8_02_check_monitoring.png` | 8 | `scripts/check_monitoring.py`: all checks PASS | Laptop |
| `task8_03_prometheus_targets.png` | 8 | Prometheus discovered both API pods (UP) | Laptop |
| `task8_04_prometheus_alerts.png` | 8 | 7 alert rules loaded, all inactive | Laptop |
| `task8_04b_drift_rule.png` | 8 | The input-drift alert rule | Laptop |
| `task8_05_grafana_health.png` | 8 | Grafana: service health, traffic, latency | Laptop |
| `task8_05b_grafana_pods.png` | 8 | Grafana: requests, CPU and memory per pod | Laptop |
| `task8_06_grafana_model.png` | 8 | Grafana: predictions and drift with normal traffic (inside the band) | Laptop |
| `task8_07_grafana_drift.png` | 8 | Grafana: drift scores outside the band with the drifted population | Laptop |
| `task8_08_drift_alert.png` | 8 | Prometheus: `HeartApiInputDrift` FIRING | Laptop |
| `task8_09_ci.png` | 8 | GitHub Actions run with the alert-rule tests and monitoring check green | GitHub |
| `task9_01_github_repo.png` | 9 | GitHub repository home page (README, file tree, commits) | GitHub |
