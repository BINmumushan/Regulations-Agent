@echo off
setlocal
".venv\Scripts\python.exe" ingest_pdfs.py %*
endlocal
