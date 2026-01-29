from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json

app = FastAPI(title="Crystal DSS - Prototype API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).parent.resolve()
STATIC_DIR = BASE / "frontend" / "static"
DATA_DIR = STATIC_DIR / "data"

if not STATIC_DIR.exists():
    raise RuntimeError(f"Static directory not found: {STATIC_DIR}")

# Mount static files (serves index.html at root)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

def _read_json(name: str):
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found: {name}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
def api_models():
    return JSONResponse(content=_read_json("models.json"))


@app.get("/api/timeseries")
def api_timeseries():
    return JSONResponse(content=_read_json("timeseries.json"))


@app.get("/api/forecast_horizon")
def api_forecast_horizon():
    return JSONResponse(content=_read_json("forecast_horizon.json"))


@app.get("/api/filtered_correlation")
def api_filtered_correlation():
    return JSONResponse(content=_read_json("filtered_correlation.json"))


@app.get("/api/granger_results")
def api_granger_results():
    return JSONResponse(content=_read_json("granger_results.json"))


@app.get("/api/regression_results")
def api_regression_results():
    return JSONResponse(content=_read_json("regression_results.json"))


@app.get("/api/seasonality_summary")
def api_seasonality_summary():
    return JSONResponse(content=_read_json("seasonality_summary.json"))


@app.get("/api/{filename}")
def api_generic(filename: str):
    # Generic passthrough to data files in frontend/static/data
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    return JSONResponse(content=_read_json(filename))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
