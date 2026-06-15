@echo off
REM Launcher for the redeem_watch scheduled task (1-min snapshots).
REM Registered via: schtasks /Create /TN PolymarketRedeemWatch /SC MINUTE /MO 1
REM Remove via:     schtasks /Delete /TN PolymarketRedeemWatch /F
cd /d "C:\Users\joshc\OneDrive\Documents\polymarket_bot1_btc"
".venv\Scripts\python.exe" -m tools.redeem_watch snapshot >> "data_runtime\redeem_watch.task.log" 2>&1
