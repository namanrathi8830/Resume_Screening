# Resume Matcher AI (FastAPI)

An AI microservice that compares resumes and job descriptions to produce a similarity score.

## Endpoints
- GET / - Health check
- POST /match - Compare resume and job description

Example:
{
  "resume": "Data analyst skilled in Python and SQL.",
  "job": "Looking for a data analyst with Python experience."
}
