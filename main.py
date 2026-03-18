from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import summarize, upload

app = FastAPI(
    title="AI Legislative Analyzer",
    description="Citizen's Dashboard for Indian Legal Documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(summarize.router, prefix="/api", tags=["Summarize"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])

@app.get("/")
def root():
    return {"message": "AI Legislative Analyzer API is running."}