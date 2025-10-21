from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="Resume Matcher AI", version="1.0")

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model from Hugging Face Hub instead of local files
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight model for Vercel
model = SentenceTransformer(MODEL_NAME)
print("✅ Model loaded successfully.")


class ResumeJobInput(BaseModel):
    resume: str
    job: str


@app.get("/")
def home():
    return {"message": "Resume AI API is running!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match")
def match_resume(input_data: ResumeJobInput):
    try:
        resume_emb = model.encode(input_data.resume, convert_to_tensor=True)
        job_emb = model.encode(input_data.job, convert_to_tensor=True)
        similarity = util.cos_sim(resume_emb, job_emb).item()
        return {"similarity_score": round(similarity, 3)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
