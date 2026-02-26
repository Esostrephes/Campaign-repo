# ─────────────────────────────────────────────────────────────────────────────
#  QuizRush — Single-file FastAPI app (Render-ready)
#  Set environment variables in Render dashboard:
#    DATABASE_URL  — your PostgreSQL connection string
#    OPENAI_API_KEY — your OpenAI key
# ─────────────────────────────────────────────────────────────────────────────

import os, uuid, json, random, string
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import BaseModel
from openai import OpenAI

# ─── ENV ──────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Render gives postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ─── DATABASE ─────────────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── MODELS ───────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String)
    phone = Column(String)
    score = Column(Integer, default=0)
    referral_code = Column(String, unique=True)
    referred_by = Column(String)
    retries_left = Column(Integer, default=1)
    eligible_for_leaderboard = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


Base.metadata.create_all(bind=engine)


# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    phone: str
    referred_by: str | None = None


class UserOut(BaseModel):
    id: uuid.UUID
    name: str
    score: int
    referral_code: str
    retries_left: int
    eligible_for_leaderboard: bool
    created_at: datetime

    class Config:
        from_attributes = True


class LeaderboardEntry(BaseModel):
    rank: int
    name: str
    score: int


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def generate_referral_code(length=8):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def process_referral(db: Session, new_user: User):
    if not new_user.referred_by:
        return
    referrer = db.query(User).filter(
        User.referral_code == new_user.referred_by
    ).first()
    if referrer:
        referrer.retries_left += 1
        db.commit()


# ─── AI QUIZ GENERATION ───────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    {"question": "What is the powerhouse of the cell?",
     "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi Apparatus"], "answer": "B"},
    {"question": "Which planet is closest to the Sun?",
     "options": ["Venus", "Earth", "Mercury", "Mars"], "answer": "C"},
    {"question": "What does HTTP stand for?",
     "options": ["HyperText Transfer Protocol", "High Tech Transfer Process", "Hyperlink Text Tool Protocol", "HyperText Transmission Program"], "answer": "A"},
    {"question": "Who wrote 'Romeo and Juliet'?",
     "options": ["Charles Dickens", "Jane Austen", "William Shakespeare", "Mark Twain"], "answer": "C"},
    {"question": "What is 12 × 12?",
     "options": ["132", "144", "124", "148"], "answer": "B"},
]


def generate_questions(content: str) -> list:
    if not OPENAI_API_KEY:
        return DEMO_QUESTIONS

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
Generate 5 fun multiple-choice quiz questions from the content below.

{content}

Return ONLY a valid JSON array, no markdown or explanation.
Format:
[
  {{
    "question": "...",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "answer": "A"
  }}
]
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse AI quiz questions: {e}")


# ─── SCHEDULER ────────────────────────────────────────────────────────────────
def update_leaderboard_eligibility():
    db = SessionLocal()
    try:
        users = db.query(User).filter(User.eligible_for_leaderboard == False).all()  # noqa
        for user in users:
            if user.created_at:
                if datetime.now(timezone.utc) - user.created_at >= timedelta(hours=2):
                    user.eligible_for_leaderboard = True
        db.commit()
    finally:
        db.close()


scheduler = BackgroundScheduler()
scheduler.add_job(update_leaderboard_eligibility, "interval", minutes=5)
scheduler.start()


# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="QuizRush")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── FRONTEND (served from same app) ─────────────────────────────────────────
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuizRush — Compete. Win. Rise.</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f; --surface: #12121a; --card: #1a1a26; --border: #2a2a3d;
    --accent: #7c5cfc; --accent2: #fc5c7d; --gold: #ffd700; --green: #00e5a0;
    --text: #f0f0f8; --muted: #8888aa; --radius: 16px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; overflow-x: hidden; }
  body::before, body::after { content: ''; position: fixed; border-radius: 50%; filter: blur(120px); pointer-events: none; z-index: 0; }
  body::before { width: 600px; height: 600px; background: radial-gradient(circle, rgba(124,92,252,0.15) 0%, transparent 70%); top: -200px; left: -100px; animation: drift1 12s ease-in-out infinite alternate; }
  body::after { width: 500px; height: 500px; background: radial-gradient(circle, rgba(252,92,125,0.1) 0%, transparent 70%); bottom: -100px; right: -100px; animation: drift2 15s ease-in-out infinite alternate; }
  @keyframes drift1 { to { transform: translate(80px, 60px); } }
  @keyframes drift2 { to { transform: translate(-80px, -60px); } }
  #app { position: relative; z-index: 1; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: 24px 16px 80px; }
  .screen { display: none; width: 100%; max-width: 520px; animation: fadeUp 0.4s ease forwards; }
  .screen.active { display: flex; flex-direction: column; gap: 20px; }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
  .logo { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 8px; text-align: center; }
  .logo span { color: var(--accent); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px; }
  .card-glow { border-color: rgba(124,92,252,0.4); box-shadow: 0 0 40px rgba(124,92,252,0.1); }
  .hero { text-align: center; padding: 40px 28px; background: linear-gradient(135deg, rgba(124,92,252,0.12) 0%, rgba(252,92,125,0.08) 100%); border: 1px solid rgba(124,92,252,0.25); border-radius: var(--radius); position: relative; overflow: hidden; }
  .hero::before { content: '🏆'; position: absolute; font-size: 120px; opacity: 0.06; top: -20px; right: -20px; transform: rotate(15deg); }
  .hero-badge { display: inline-block; background: rgba(124,92,252,0.2); border: 1px solid rgba(124,92,252,0.4); color: var(--accent); font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; padding: 4px 12px; border-radius: 100px; margin-bottom: 16px; }
  .hero h1 { font-family: 'Syne', sans-serif; font-size: 36px; font-weight: 800; line-height: 1.1; margin-bottom: 12px; }
  .hero h1 .highlight { background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
  .hero p { color: var(--muted); font-size: 15px; line-height: 1.6; margin-bottom: 24px; }
  .social-proof { display: flex; align-items: center; justify-content: center; gap: 12px; margin-top: 16px; }
  .avatars { display: flex; }
  .avatars span { width: 28px; height: 28px; border-radius: 50%; border: 2px solid var(--bg); margin-left: -8px; display: flex; align-items: center; justify-content: center; font-size: 12px; background: var(--card); }
  .avatars span:first-child { margin-left: 0; }
  .social-proof p { font-size: 12px; color: var(--muted); }
  .social-proof strong { color: var(--text); }
  .stats-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
  .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; text-align: center; }
  .stat-num { font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 800; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
  .stat-label { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .urgency-bar { background: linear-gradient(90deg, rgba(252,92,125,0.15), rgba(252,92,125,0.05)); border: 1px solid rgba(252,92,125,0.3); border-radius: 10px; padding: 12px 16px; display: flex; align-items: center; gap: 10px; font-size: 13px; }
  .urgency-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent2); animation: pulse 1.5s ease-in-out infinite; flex-shrink: 0; }
  @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.4); } }
  .form-group { display: flex; flex-direction: column; gap: 8px; }
  label { font-size: 13px; font-weight: 500; color: var(--muted); letter-spacing: 0.3px; }
  input { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; color: var(--text); font-family: 'DM Sans', sans-serif; font-size: 15px; transition: border-color 0.2s, box-shadow 0.2s; outline: none; }
  input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(124,92,252,0.15); }
  input::placeholder { color: #555570; }
  .btn { padding: 15px 24px; border-radius: 12px; font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; cursor: pointer; border: none; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 8px; letter-spacing: 0.3px; }
  .btn-primary { background: linear-gradient(135deg, var(--accent), #9b7cff); color: white; box-shadow: 0 4px 20px rgba(124,92,252,0.35); }
  .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(124,92,252,0.5); }
  .btn-danger { background: linear-gradient(135deg, var(--accent2), #ff8c69); color: white; box-shadow: 0 4px 20px rgba(252,92,125,0.3); }
  .btn-danger:hover { transform: translateY(-2px); }
  .btn-ghost { background: var(--surface); border: 1px solid var(--border); color: var(--muted); font-family: 'DM Sans', sans-serif; }
  .btn-ghost:hover { border-color: var(--accent); color: var(--accent); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; }
  .question-meta { display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; }
  .q-counter { font-size: 12px; color: var(--muted); font-weight: 500; }
  .timer-wrap { display: flex; align-items: center; gap: 6px; font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 800; }
  .timer-wrap.danger { color: var(--accent2); } .timer-wrap.warning { color: var(--gold); } .timer-wrap.safe { color: var(--green); }
  .progress-track { width: 100%; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 24px; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 2px; transition: width 0.4s ease; }
  .question-text { font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; line-height: 1.4; margin-bottom: 20px; }
  .options { display: flex; flex-direction: column; gap: 10px; }
  .option { padding: 14px 18px; background: var(--surface); border: 1.5px solid var(--border); border-radius: 12px; cursor: pointer; font-size: 15px; transition: all 0.2s; display: flex; align-items: center; gap: 12px; text-align: left; }
  .option-letter { width: 28px; height: 28px; border-radius: 8px; background: var(--border); display: flex; align-items: center; justify-content: center; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 12px; flex-shrink: 0; transition: all 0.2s; }
  .option:hover:not(.locked) { border-color: var(--accent); background: rgba(124,92,252,0.08); }
  .option:hover:not(.locked) .option-letter { background: var(--accent); color: white; }
  .option.correct { border-color: var(--green); background: rgba(0,229,160,0.1); }
  .option.correct .option-letter { background: var(--green); color: #000; }
  .option.wrong { border-color: var(--accent2); background: rgba(252,92,125,0.1); }
  .option.wrong .option-letter { background: var(--accent2); color: white; }
  .option.locked { cursor: default; }
  .score-flash { text-align: center; font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 700; padding: 8px 0; min-height: 30px; color: var(--green); animation: scoreIn 0.3s ease; }
  @keyframes scoreIn { from { transform: scale(1.4); opacity: 0; } to { transform: scale(1); opacity: 1; } }
  .streak-banner { background: linear-gradient(135deg, rgba(255,215,0,0.15), rgba(255,165,0,0.08)); border: 1px solid rgba(255,215,0,0.3); border-radius: 10px; padding: 10px 16px; display: flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 600; color: var(--gold); display: none; }
  .result-ring { width: 140px; height: 140px; border-radius: 50%; border: 6px solid transparent; background: conic-gradient(from 0deg, var(--accent) 0%, var(--accent2) var(--pct), var(--border) var(--pct)); display: flex; align-items: center; justify-content: center; margin: 0 auto; }
  .result-ring-inner { width: 120px; height: 120px; border-radius: 50%; background: var(--card); display: flex; flex-direction: column; align-items: center; justify-content: center; }
  .result-score-num { font-family: 'Syne', sans-serif; font-size: 34px; font-weight: 800; line-height: 1; }
  .result-score-label { font-size: 11px; color: var(--muted); }
  .result-verdict { text-align: center; font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 800; }
  .result-message { text-align: center; color: var(--muted); font-size: 14px; }
  .lb-row { display: flex; align-items: center; gap: 14px; padding: 14px; border-radius: 12px; background: var(--surface); border: 1px solid var(--border); transition: transform 0.2s; }
  .lb-row:hover { transform: translateX(4px); }
  .lb-row.top1 { border-color: var(--gold); background: rgba(255,215,0,0.05); }
  .lb-row.top2 { border-color: #c0c0c0; } .lb-row.top3 { border-color: #cd7f32; }
  .lb-rank { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 18px; width: 32px; text-align: center; }
  .lb-name { flex: 1; font-weight: 500; }
  .lb-score { font-family: 'Syne', sans-serif; font-weight: 700; color: var(--accent); }
  .referral-code-box { background: var(--surface); border: 1.5px dashed var(--accent); border-radius: 12px; padding: 14px; text-align: center; font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; letter-spacing: 4px; color: var(--accent); cursor: pointer; transition: background 0.2s; }
  .referral-code-box:hover { background: rgba(124,92,252,0.1); }
  .section-title { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; margin-bottom: 4px; }
  .divider { height: 1px; background: var(--border); }
  .text-center { text-align: center; } .text-muted { color: var(--muted); font-size: 13px; }
  .flex-between { display: flex; align-items: center; justify-content: space-between; }
  .gap-8 { display: flex; flex-direction: column; gap: 8px; }
  .toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%) translateY(80px); background: var(--card); border: 1px solid var(--border); color: var(--text); padding: 12px 20px; border-radius: 100px; font-size: 14px; font-weight: 500; z-index: 999; transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1); white-space: nowrap; }
  .toast.show { transform: translateX(-50%) translateY(0); }
  #confetti-canvas { position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 999; }
</style>
</head>
<body>
<canvas id="confetti-canvas"></canvas>
<div id="app">

  <div class="screen active" id="screen-landing">
    <div class="logo">Quiz<span>Rush</span></div>
    <div class="hero">
      <div class="hero-badge">🔥 Live Competition</div>
      <h1>Prove You're <span class="highlight">Top of Class</span></h1>
      <p>Answer fast, score big, climb the leaderboard. Your peers are already competing — are you?</p>
      <div class="social-proof">
        <div class="avatars"><span>👦</span><span>👧</span><span>🧑</span><span>👩</span></div>
        <p><strong>1,247 students</strong> competing right now</p>
      </div>
    </div>
    <div class="stats-row">
      <div class="stat-card"><div class="stat-num">5x</div><div class="stat-label">Bonus Points</div></div>
      <div class="stat-card"><div class="stat-num">30s</div><div class="stat-label">Per Question</div></div>
      <div class="stat-card"><div class="stat-num">Top 10</div><div class="stat-label">Leaderboard</div></div>
    </div>
    <div class="urgency-bar">
      <div class="urgency-dot"></div>
      <span>Competition ends in <strong id="countdown">02:47:33</strong> — don't miss out!</span>
    </div>
    <div class="card">
      <div class="section-title">Join the Competition</div>
      <p class="text-muted" style="margin-bottom:20px">Create your account to save your score</p>
      <div class="gap-8">
        <div class="form-group"><label>Your Name</label><input type="text" id="reg-name" placeholder="e.g. Alex Johnson" /></div>
        <div class="form-group"><label>Phone Number</label><input type="tel" id="reg-phone" placeholder="+1 234 567 8900" /></div>
        <div class="form-group">
          <label>Referral Code <span style="color:var(--muted)">(optional — unlocks bonus retries)</span></label>
          <input type="text" id="reg-referral" placeholder="e.g. XK8A2P1Y" style="text-transform:uppercase" />
        </div>
        <button class="btn btn-primary" onclick="handleRegister()">🚀 Start Competing</button>
      </div>
    </div>
    <button class="btn btn-ghost" onclick="showScreen('screen-leaderboard'); loadLeaderboard()">🏆 View Leaderboard</button>
  </div>

  <div class="screen" id="screen-quiz">
    <div class="logo">Quiz<span>Rush</span></div>
    <div id="streak-banner" class="streak-banner">🔥 <span id="streak-text">3x Streak! +50 bonus pts</span></div>
    <div class="card card-glow">
      <div class="question-meta">
        <span class="q-counter" id="q-counter">Question 1 of 5</span>
        <div class="timer-wrap safe" id="timer-display">⏱ 30</div>
      </div>
      <div class="progress-track"><div class="progress-fill" id="progress-fill" style="width:20%"></div></div>
      <div class="question-text" id="question-text">Loading question...</div>
      <div class="options" id="options-container"></div>
      <div class="score-flash" id="score-flash"></div>
    </div>
    <div class="card">
      <div class="flex-between">
        <div><div style="font-size:12px;color:var(--muted)">Your Score</div><div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:var(--accent)" id="live-score">0</div></div>
        <div style="text-align:right"><div style="font-size:12px;color:var(--muted)">Retries Left</div><div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:var(--gold)" id="retries-display">1</div></div>
      </div>
    </div>
  </div>

  <div class="screen" id="screen-result">
    <div class="logo">Quiz<span>Rush</span></div>
    <div class="card card-glow" style="align-items:center;gap:16px;padding:36px 28px">
      <div class="result-ring" id="result-ring" style="--pct:0%">
        <div class="result-ring-inner">
          <div class="result-score-num" id="result-score-num">0</div>
          <div class="result-score-label">pts</div>
        </div>
      </div>
      <div class="result-verdict" id="result-verdict">Amazing!</div>
      <div class="result-message" id="result-message">You answered 4/5 correctly</div>
    </div>
    <div class="stats-row">
      <div class="stat-card"><div class="stat-num" id="res-correct">0</div><div class="stat-label">Correct</div></div>
      <div class="stat-card"><div class="stat-num" id="res-streak">0</div><div class="stat-label">Best Streak</div></div>
      <div class="stat-card"><div class="stat-num" id="res-rank">#—</div><div class="stat-label">Your Rank</div></div>
    </div>
    <div class="card">
      <div class="section-title" style="margin-bottom:8px">📣 Challenge Your Friends</div>
      <p class="text-muted" style="margin-bottom:14px">Share your code — when they join, you get a bonus retry!</p>
      <div class="referral-code-box" id="referral-code-display" onclick="copyReferral()">••••••••</div>
      <p class="text-muted" style="text-align:center;margin-top:8px;font-size:12px">Tap to copy</p>
    </div>
    <button class="btn btn-danger" id="retry-btn" onclick="handleRetry()">🔄 Use a Retry</button>
    <button class="btn btn-primary" onclick="showScreen('screen-leaderboard'); loadLeaderboard()">🏆 See Leaderboard</button>
    <button class="btn btn-ghost" onclick="goHome()">Back to Home</button>
  </div>

  <div class="screen" id="screen-leaderboard">
    <div class="logo">Quiz<span>Rush</span></div>
    <div class="card" style="text-align:center;background:linear-gradient(135deg,rgba(124,92,252,0.1),rgba(252,92,125,0.06))">
      <div style="font-size:40px;margin-bottom:8px">🏆</div>
      <div class="section-title">Top Students</div>
      <p class="text-muted">Updated every 5 minutes</p>
    </div>
    <div class="gap-8" id="leaderboard-list"><p class="text-muted text-center">Loading...</p></div>
    <button class="btn btn-ghost" onclick="goHome()">← Back</button>
  </div>

</div>
<div class="toast" id="toast"></div>

<script>
const API = '';  // Same-origin — FastAPI serves this HTML

let state = {
  userId: null, userName: '', referralCode: '', retriesLeft: 1,
  questions: [], currentQ: 0, score: 0, correctCount: 0,
  streak: 0, bestStreak: 0, timerInterval: null, timeLeft: 30, answered: false,
};

const DEMO_QUESTIONS = [
  { question: "What is the powerhouse of the cell?", options: ["Nucleus","Mitochondria","Ribosome","Golgi Apparatus"], answer: "B" },
  { question: "Which planet is closest to the Sun?", options: ["Venus","Earth","Mercury","Mars"], answer: "C" },
  { question: "What does HTTP stand for?", options: ["HyperText Transfer Protocol","High Tech Transfer Process","Hyperlink Text Tool Protocol","HyperText Transmission Program"], answer: "A" },
  { question: "Who wrote 'Romeo and Juliet'?", options: ["Charles Dickens","Jane Austen","William Shakespeare","Mark Twain"], answer: "C" },
  { question: "What is 12 × 12?", options: ["132","144","124","148"], answer: "B" },
];

function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}
function goHome() { stopTimer(); resetState(); showScreen('screen-landing'); }
function resetState() { state.questions=[]; state.currentQ=0; state.score=0; state.correctCount=0; state.streak=0; state.bestStreak=0; state.answered=false; }

let countdownSecs = 167253;
setInterval(() => {
  countdownSecs = Math.max(0, countdownSecs - 1);
  const h = String(Math.floor(countdownSecs/3600)).padStart(2,'0');
  const m = String(Math.floor((countdownSecs%3600)/60)).padStart(2,'0');
  const s = String(countdownSecs%60).padStart(2,'0');
  const el = document.getElementById('countdown');
  if (el) el.textContent = `${h}:${m}:${s}`;
}, 1000);

async function handleRegister() {
  const name = document.getElementById('reg-name').value.trim();
  const phone = document.getElementById('reg-phone').value.trim();
  const referral = document.getElementById('reg-referral').value.trim();
  if (!name || !phone) return showToast('Please fill in your name and phone 👋');
  try {
    const res = await fetch(`${API}/register`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name, phone, referred_by: referral || null }) });
    const data = await res.json();
    state.userId = data.id; state.userName = data.name; state.referralCode = data.referral_code; state.retriesLeft = data.retries_left;
  } catch {
    state.userId = 'demo-'+Date.now(); state.userName = name; state.referralCode = 'DEMO1234'; state.retriesLeft = 1;
  }
  startQuiz();
}

async function startQuiz() {
  resetState(); showScreen('screen-quiz');
  document.getElementById('live-score').textContent = '0';
  document.getElementById('retries-display').textContent = state.retriesLeft;
  document.getElementById('streak-banner').style.display = 'none';
  try {
    const res = await fetch(`${API}/questions`);
    state.questions = await res.json();
  } catch { state.questions = DEMO_QUESTIONS; }
  loadQuestion();
}

function loadQuestion() {
  const q = state.questions[state.currentQ];
  if (!q) return endQuiz();
  state.answered = false; state.timeLeft = 30;
  document.getElementById('score-flash').textContent = '';
  const total = state.questions.length; const idx = state.currentQ;
  document.getElementById('q-counter').textContent = `Question ${idx+1} of ${total}`;
  document.getElementById('progress-fill').style.width = `${((idx+1)/total)*100}%`;
  document.getElementById('question-text').textContent = q.question;
  const letters = ['A','B','C','D'];
  const container = document.getElementById('options-container');
  container.innerHTML = '';
  q.options.forEach((opt, i) => {
    const div = document.createElement('div');
    div.className = 'option';
    div.innerHTML = `<span class="option-letter">${letters[i]}</span>${opt}`;
    div.onclick = () => selectAnswer(i, q.answer, div, container);
    container.appendChild(div);
  });
  startTimer();
}

function startTimer() {
  stopTimer(); updateTimerDisplay();
  state.timerInterval = setInterval(() => {
    state.timeLeft--;
    updateTimerDisplay();
    if (state.timeLeft <= 0) {
      stopTimer();
      if (!state.answered) { showToast("⏰ Time's up!"); state.streak=0; lockOptions(null, state.questions[state.currentQ].answer); setTimeout(nextQuestion, 1500); }
    }
  }, 1000);
}
function stopTimer() { clearInterval(state.timerInterval); }
function updateTimerDisplay() {
  const el = document.getElementById('timer-display');
  el.textContent = `⏱ ${state.timeLeft}`;
  el.className = 'timer-wrap ' + (state.timeLeft<=5?'danger':state.timeLeft<=10?'warning':'safe');
}

function selectAnswer(idx, correctLetter, el, container) {
  if (state.answered) return;
  state.answered = true; stopTimer();
  const letters = ['A','B','C','D'];
  const chosen = letters[idx];
  const isCorrect = chosen === correctLetter;
  lockOptions(chosen, correctLetter);
  if (isCorrect) {
    state.streak++; state.bestStreak = Math.max(state.bestStreak, state.streak); state.correctCount++;
    const timeBonus = state.timeLeft * 3; const streakBonus = state.streak >= 3 ? 50 : 0;
    const points = 100 + timeBonus + streakBonus;
    state.score += points;
    document.getElementById('live-score').textContent = state.score;
    document.getElementById('score-flash').textContent = `+${points} pts ${streakBonus?'🔥 Streak Bonus!':''}`;
    if (state.streak >= 3) { const sb=document.getElementById('streak-banner'); document.getElementById('streak-text').textContent=`${state.streak}x Streak! +50 bonus pts`; sb.style.display='flex'; }
  } else {
    state.streak = 0; document.getElementById('streak-banner').style.display='none';
    document.getElementById('score-flash').textContent = `Not quite! The answer was ${correctLetter}`;
  }
  setTimeout(nextQuestion, 1800);
}

function lockOptions(chosenLetter, correctLetter) {
  const letters = ['A','B','C','D'];
  document.querySelectorAll('.option').forEach((el,i) => {
    el.classList.add('locked');
    if (letters[i]===correctLetter) el.classList.add('correct');
    else if (letters[i]===chosenLetter) el.classList.add('wrong');
  });
}

function nextQuestion() { state.currentQ++; state.currentQ >= state.questions.length ? endQuiz() : loadQuestion(); }

async function endQuiz() {
  stopTimer(); showScreen('screen-result');
  try { await fetch(`${API}/submit-score?user_id=${state.userId}&score=${state.score}`, {method:'POST'}); } catch {}
  document.getElementById('result-score-num').textContent = state.score;
  document.getElementById('res-correct').textContent = `${state.correctCount}/5`;
  document.getElementById('res-streak').textContent = state.bestStreak;
  document.getElementById('referral-code-display').textContent = state.referralCode || 'LOADING';
  document.getElementById('retry-btn').textContent = `🔄 Use a Retry (${state.retriesLeft} left)`;
  if (state.retriesLeft <= 0) document.getElementById('retry-btn').disabled = true;
  const pct = Math.round((state.correctCount/state.questions.length)*100);
  document.getElementById('result-ring').style.setProperty('--pct',`${pct}%`);
  const verdicts=[[80,'🏆 Outstanding!',"You're among the top students!"],[60,'🎉 Great Work!',"You know your stuff!"],[40,'👍 Decent Effort',"A bit more study and you'll crush it."],[0,'💪 Keep Going',"Every expert was once a beginner."]];
  const v = verdicts.find(([min])=>pct>=min);
  document.getElementById('result-verdict').textContent = v[1];
  document.getElementById('result-message').textContent = v[2];
  document.getElementById('res-rank').textContent = `#${Math.floor(Math.random()*50)+1}`;
  if (pct >= 80) launchConfetti();
}

async function handleRetry() {
  if (state.retriesLeft <= 0) return showToast('No retries! Share your code to earn one 🎁');
  try { const res=await fetch(`${API}/retry?user_id=${state.userId}`,{method:'POST'}); const data=await res.json(); state.retriesLeft=data.retries_left??state.retriesLeft-1; }
  catch { state.retriesLeft=Math.max(0,state.retriesLeft-1); }
  startQuiz();
}

async function loadLeaderboard() {
  const list = document.getElementById('leaderboard-list');
  list.innerHTML = '<p class="text-muted text-center">Loading...</p>';
  let entries = [];
  try { const res=await fetch(`${API}/leaderboard`); entries=await res.json(); }
  catch { entries=[{rank:1,name:'Jordan K.',score:2450},{rank:2,name:'Priya S.',score:2310},{rank:3,name:'Marcus T.',score:2150},{rank:4,name:'Emma L.',score:1980},{rank:5,name:'Liam O.',score:1870}]; }
  const medals = ['🥇','🥈','🥉'];
  list.innerHTML = '';
  entries.forEach((u,i) => {
    const div=document.createElement('div');
    div.className=`lb-row ${i===0?'top1':i===1?'top2':i===2?'top3':''}`;
    div.innerHTML=`<div class="lb-rank">${medals[i]??u.rank}</div><div class="lb-name">${u.name}</div><div class="lb-score">${u.score.toLocaleString()} pts</div>`;
    list.appendChild(div);
  });
  if (state.userName) {
    const self=document.createElement('div'); self.className='lb-row'; self.style.borderColor='var(--accent)';
    self.innerHTML=`<div class="lb-rank">👤</div><div class="lb-name">${state.userName} <span style="font-size:11px;color:var(--accent)">(You)</span></div><div class="lb-score">${state.score.toLocaleString()} pts</div>`;
    list.appendChild(self);
  }
}

function copyReferral() {
  if (!state.referralCode) return;
  navigator.clipboard.writeText(state.referralCode).then(()=>showToast('Referral code copied! 🎉')).catch(()=>{});
}

function showToast(msg) {
  const t=document.getElementById('toast'); t.textContent=msg; t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),3000);
}

function launchConfetti() {
  const canvas=document.getElementById('confetti-canvas'); const ctx=canvas.getContext('2d');
  canvas.width=window.innerWidth; canvas.height=window.innerHeight;
  const particles=Array.from({length:120},()=>({x:Math.random()*canvas.width,y:-20,vx:(Math.random()-0.5)*4,vy:Math.random()*4+2,color:['#7c5cfc','#fc5c7d','#ffd700','#00e5a0','#fff'][Math.floor(Math.random()*5)],size:Math.random()*8+4,rot:Math.random()*360,rotSpeed:(Math.random()-0.5)*8}));
  let frame;
  function draw() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    particles.forEach(p=>{p.x+=p.vx;p.y+=p.vy;p.rot+=p.rotSpeed;ctx.save();ctx.translate(p.x,p.y);ctx.rotate((p.rot*Math.PI)/180);ctx.fillStyle=p.color;ctx.fillRect(-p.size/2,-p.size/2,p.size,p.size*0.5);ctx.restore();});
    if(particles.some(p=>p.y<canvas.height+50)){frame=requestAnimationFrame(draw);}
    else{ctx.clearRect(0,0,canvas.width,canvas.height);}
  }
  draw();
  setTimeout(()=>{cancelAnimationFrame(frame);ctx.clearRect(0,0,canvas.width,canvas.height);},4000);
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(content=FRONTEND_HTML)


# ─── API ROUTES ───────────────────────────────────────────────────────────────
@app.post("/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    user = User(
        id=uuid.uuid4(),
        name=payload.name,
        phone=payload.phone,
        referral_code=generate_referral_code(),
        referred_by=payload.referred_by,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    process_referral(db, user)
    return user


@app.get("/questions")
def get_questions():
    """Returns quiz questions. Swap content string for your topic."""
    content = "General knowledge covering science, geography, literature, and math."
    try:
        return generate_questions(content)
    except Exception:
        return DEMO_QUESTIONS


@app.get("/leaderboard", response_model=list[LeaderboardEntry])
def leaderboard(db: Session = Depends(get_db)):
    users = (
        db.query(User)
        .filter(User.eligible_for_leaderboard == True)  # noqa
        .order_by(User.score.desc())
        .limit(10)
        .all()
    )
    return [LeaderboardEntry(rank=i+1, name=u.name, score=u.score) for i, u in enumerate(users)]


@app.post("/retry")
def retry(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.retries_left > 0:
        user.retries_left -= 1
        db.commit()
        return {"message": "Retry granted", "retries_left": user.retries_left}
    return {"message": "No retries left. Share your referral link!", "referral_code": user.referral_code}


@app.post("/submit-score")
def submit_score(user_id: str, score: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if score > user.score:
        user.score = score
        db.commit()
    return {"message": "Score updated", "score": user.score}


@app.get("/user/{user_id}", response_model=UserOut)
def get_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
