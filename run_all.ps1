# FactoryGuard 6G End-to-End Execution Script
# Run this script to execute all steps sequentially.

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🛡️ STARTING FACTORYGUARD 6G PRODUCTION PIPELINE RUNNER" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Ingest and Validate Data
Write-Host "`n[STEP 1/7] Ingesting and validating raw data..." -ForegroundColor Green
.venv\Scripts\python scripts/run_data_pipeline.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 1 failed."; exit 1 }

# 2. Build Features
Write-Host "`n[STEP 2/7] Generating features (leakage-safe and defragmented)..." -ForegroundColor Green
.venv\Scripts\python scripts/build_features.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 2 failed."; exit 1 }

# 3. Label Features
Write-Host "`n[STEP 3/7] Creating proxy operational deterioration labels..." -ForegroundColor Green
.venv\Scripts\python scripts/build_proxy_labels.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 3 failed."; exit 1 }

# 4. Train Anomaly Models
Write-Host "`n[STEP 4/7] Training unsupervised anomaly detection model (Isolation Forest)..." -ForegroundColor Green
.venv\Scripts\python scripts/train_anomaly_models.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 4 failed."; exit 1 }

# 5. Optimize Hyperparameters
Write-Host "`n[STEP 5/7] Running Optuna hyperparameter optimization..." -ForegroundColor Green
.venv\Scripts\python scripts/optimize_hyperparameters.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 5 failed."; exit 1 }

# 6. Train Supervised Models & Ensembles
Write-Host "`n[STEP 6/7] Training supervised classifiers and stacking ensembles..." -ForegroundColor Green
.venv\Scripts\python scripts/train_supervised_models.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 6 failed."; exit 1 }

# 7. Generate Fused Risk Predictions
Write-Host "`n[STEP 7/7] Fusing models and generating final risk ratings..." -ForegroundColor Green
.venv\Scripts\python scripts/generate_predictions.py
if ($LASTEXITCODE -ne 0) { Write-Error "Step 7 failed."; exit 1 }

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "🎉 FACTORYGUARD 6G PIPELINE RUN COMPLETED SUCCESSFULLY!" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
