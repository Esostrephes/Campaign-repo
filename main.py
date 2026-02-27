"""
QuizRush Campaign Engine
========================
Two modes:
  1. LEADER SETUP  → /setup  (password protected, leader fills in their profile)
  2. STUDENT QUIZ  → /       (public, AI-generated questions from leader profile)

Add to Render env vars:
  SETUP_PASSWORD   → secret password only the leader/campaign team knows
  (DATABASE_URL and OPENAI_API_KEY already set)
"""

import os, uuid, json, random, string
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import BaseModel
from openai import OpenAI

# ─── ENV ──────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SETUP_PASSWORD = os.getenv("SETUP_PASSWORD", "campaign2024")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ─── DB ───────────────────────────────────────────────────────────────────────
engine = create_engine(DATABASE_URL) if DATABASE_URL else None
SessionLocal = sessionmaker(bind=engine) if engine else None
Base = declarative_base()


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


class LeaderProfile(Base):
    __tablename__ = "leader_profile"
    id = Column(Integer, primary_key=True, default=1)
    name = Column(String, default="")
    position = Column(String, default="")
    achievements = Column(Text, default="")   # Level 1 source
    manifesto = Column(Text, default="")      # Level 2 source
    personality = Column(Text, default="")    # Level 3 source (jokes, hobbies, etc.)
    campaign_color = Column(String, default="#e63946")
    slogan = Column(String, default="")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


if engine:
    Base.metadata.create_all(bind=engine)


def get_db():
    if not SessionLocal:
        raise HTTPException(500, "Database not configured")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def generate_referral_code():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def get_leader(db: Session) -> LeaderProfile:
    leader = db.query(LeaderProfile).first()
    if not leader:
        leader = LeaderProfile(id=1)
        db.add(leader)
        db.commit()
        db.refresh(leader)
    return leader


# ─── AI QUESTION GENERATION ───────────────────────────────────────────────────
LEVEL_CONFIGS = {
    1: {
        "name": "What They've Done",
        "label": "ACHIEVEMENTS",
        "description": "Tests students on real things the leader has done",
        "tone": "factual, impressive, builds credibility",
        "trigger": "social proof, authority, track record",
        "field": "achievements"
    },
    2: {
        "name": "The Vision",
        "label": "MANIFESTO",
        "description": "Tests students on the leader's plans and promises",
        "tone": "hopeful, ambitious, future-focused",
        "trigger": "hope, belonging, future identity",
        "field": "manifesto"
    },
    3: {
        "name": "Know Your Leader",
        "label": "PERSONALITY",
        "description": "Fun questions about the leader's personality, jokes, hobbies",
        "tone": "fun, relatable, human, warm",
        "trigger": "likeability, familiarity, parasocial bond",
        "field": "personality"
    }
}


def generate_campaign_questions(leader: LeaderProfile, level: int) -> list:
    config = LEVEL_CONFIGS[level]
    content = getattr(leader, config["field"], "")

    if not content:
        return get_fallback_questions(leader.name, level)

    if not OPENAI_API_KEY:
        return get_fallback_questions(leader.name, level)

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
You are a political campaign strategist and psychology expert designing a student quiz app
to get students to vote for {leader.name} who is running for {leader.position}.

LEVEL {level} — {config['name'].upper()}
Goal: {config['description']}
Psychological triggers to use: {config['trigger']}
Tone: {config['tone']}

Source material about {leader.name}:
---
{content}
---

Generate exactly 5 multiple-choice quiz questions for this level.

Rules:
- Questions should feel engaging and natural, NOT like propaganda
- Subtly make {leader.name} look competent, likeable and the obvious choice
- Use the psychological triggers listed above naturally within the questions and answer options
- Wrong answers should be clearly wrong but not obviously biased
- For level 3 (personality), include fun/inside knowledge questions that make students feel
  "in the know" — this builds a parasocial bond with the leader
- End each question with a subtle positive reinforcement in the correct answer explanation

Return ONLY a valid JSON array, no markdown, no explanation:
[
  {{
    "question": "...",
    "options": ["Option text", "Option text", "Option text", "Option text"],
    "answer": "A|B|C|D",
    "explanation": "Brief positive explanation of why this is correct (1 sentence, subtly reinforcing)"
  }}
]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return get_fallback_questions(leader.name, level)


def get_fallback_questions(name: str, level: int) -> list:
    """Shown when leader profile isn't filled yet."""
    return [
        {
            "question": f"Why should you vote for {name}?",
            "options": [
                "They have a proven track record",
                "They have a clear vision for students",
                "They genuinely care about student welfare",
                "All of the above"
            ],
            "answer": "D",
            "explanation": f"{name} represents all of these qualities and more."
        }
    ] * 5


# ─── SCHEDULER ────────────────────────────────────────────────────────────────
def update_leaderboard_eligibility():
    if not SessionLocal:
        return
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
app = FastAPI(title="QuizRush Campaign")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    phone: str
    referred_by: str | None = None


class LeaderProfileUpdate(BaseModel):
    password: str
    name: str
    position: str
    achievements: str
    manifesto: str
    personality: str
    slogan: str
    campaign_color: str = "#e63946"


# ─── LEADER SETUP PAGE ────────────────────────────────────────────────────────
SETUP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Campaign Setup — Leader Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root { --bg:#06080f; --card:#0f1117; --border:#1e2130; --accent:#e63946; --text:#f0f0f0; --muted:#6b7280; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; padding:40px 16px; }
  .wrap { max-width:680px; margin:0 auto; display:flex; flex-direction:column; gap:24px; }
  .logo { font-family:'Syne',sans-serif; font-size:20px; font-weight:800; color:var(--accent); text-align:center; }
  h1 { font-family:'Syne',sans-serif; font-size:28px; font-weight:800; text-align:center; }
  .subtitle { color:var(--muted); text-align:center; font-size:14px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:28px; display:flex; flex-direction:column; gap:16px; }
  .card h2 { font-family:'Syne',sans-serif; font-size:16px; font-weight:700; }
  .level-badge { display:inline-block; font-size:10px; font-weight:700; letter-spacing:2px; text-transform:uppercase; padding:3px 10px; border-radius:100px; margin-bottom:4px; }
  .l1 { background:rgba(0,180,216,0.15); color:#00b4d8; border:1px solid rgba(0,180,216,0.3); }
  .l2 { background:rgba(255,165,0,0.15); color:#ffa500; border:1px solid rgba(255,165,0,0.3); }
  .l3 { background:rgba(0,200,100,0.15); color:#00c864; border:1px solid rgba(0,200,100,0.3); }
  .hint { font-size:12px; color:var(--muted); line-height:1.6; }
  label { font-size:13px; color:var(--muted); font-weight:500; }
  input, textarea { background:#0a0c14; border:1px solid var(--border); border-radius:10px; padding:14px; color:var(--text); font-family:'DM Sans',sans-serif; font-size:14px; outline:none; transition:border-color 0.2s; width:100%; }
  input:focus, textarea:focus { border-color:var(--accent); }
  textarea { min-height:140px; resize:vertical; line-height:1.6; }
  .form-group { display:flex; flex-direction:column; gap:6px; }
  .btn { padding:16px; border-radius:12px; font-family:'Syne',sans-serif; font-size:15px; font-weight:700; border:none; cursor:pointer; background:var(--accent); color:white; transition:all 0.2s; letter-spacing:0.5px; }
  .btn:hover { opacity:0.9; transform:translateY(-1px); }
  .success { background:rgba(0,200,100,0.1); border:1px solid rgba(0,200,100,0.3); border-radius:12px; padding:16px; color:#00c864; font-size:14px; text-align:center; display:none; }
  .color-row { display:flex; align-items:center; gap:12px; }
  .color-preview { width:40px; height:40px; border-radius:10px; border:2px solid var(--border); flex-shrink:0; }
</style>
</head>
<body>
<div class="wrap">
  <div class="logo">QuizRush</div>
  <h1>🎯 Campaign Setup</h1>
  <p class="subtitle">Fill in your profile — AI will generate psychologically compelling quiz questions for students</p>

  <div class="card">
    <div class="form-group"><label>Setup Password</label><input type="password" id="password" placeholder="Enter campaign password" /></div>
    <div class="form-group"><label>Your Full Name</label><input type="text" id="name" placeholder="e.g. Emmanuel Osei" /></div>
    <div class="form-group"><label>Position You're Running For</label><input type="text" id="position" placeholder="e.g. Student Union President, University of Ghana" /></div>
    <div class="form-group"><label>Campaign Slogan</label><input type="text" id="slogan" placeholder="e.g. Students First, Always." /></div>
    <div class="form-group">
      <label>Campaign Color</label>
      <div class="color-row">
        <input type="color" id="campaign_color" value="#e63946" style="width:60px;height:40px;padding:4px;cursor:pointer" onchange="document.getElementById('color-preview').style.background=this.value" />
        <div class="color-preview" id="color-preview" style="background:#e63946"></div>
        <span style="font-size:13px;color:var(--muted)">Pick your campaign colour</span>
      </div>
    </div>
  </div>

  <div class="card">
    <span class="level-badge l1">Level 1 — Achievements</span>
    <h2>What Have You Done For Students?</h2>
    <p class="hint">List everything you've achieved or contributed to student life. Past roles, projects, events organized, problems solved, money raised, policies changed — be specific and detailed. The more you write, the better the questions.</p>
    <div class="form-group">
      <textarea id="achievements" placeholder="e.g.
- Organized the 2023 inter-departmental quiz that attracted 800+ students
- Fought for and secured a 30% reduction in library printing costs
- Started a mentorship program pairing 200 freshers with final year students
- Represented students at 3 university senate meetings
- Raised GH₵15,000 for the student emergency fund..."></textarea>
    </div>
  </div>

  <div class="card">
    <span class="level-badge l2">Level 2 — Manifesto</span>
    <h2>What Will You Do If Elected?</h2>
    <p class="hint">Describe your plans, promises and vision for students. What specific changes will you make? What problems will you solve? What's your timeline? Make it bold and believable.</p>
    <div class="form-group">
      <textarea id="manifesto" placeholder="e.g.
- Free WiFi extension to all hostels within 6 months
- Launch a student mental health centre with a full-time counsellor
- Create a student startup fund of GH₵50,000 for entrepreneurs
- Negotiate a 20% discount on all campus food vendors
- Monthly open town halls so every student's voice is heard
- Partner with companies for internship placements for all 3rd years..."></textarea>
    </div>
  </div>

  <div class="card">
    <span class="level-badge l3">Level 3 — Personality</span>
    <h2>Who Are You As A Person?</h2>
    <p class="hint">This is where students connect with YOU, not just your policies. Share your hobbies, favourite music, inside jokes on campus, funny stories, what you eat, your personality quirks, things only your friends know — the more human and specific, the stronger the bond students will feel.</p>
    <div class="form-group">
      <textarea id="personality" placeholder="e.g.
- Everyone calls me 'Sark' because I once freestyled at a hall party and won
- I eat jollof rice every single Friday — it's my ritual
- I'm obsessed with Marvel movies, especially Spider-Man
- My biggest fear is public speaking (ironic right?) but I face it every day
- I once got lost on campus for 2 hours during freshers week
- My friends say I'm annoyingly punctual — I've never been late to a meeting
- I play football every Saturday morning at the sports complex
- I started reading leadership books at age 15 after watching a Mandela documentary..."></textarea>
    </div>
  </div>

  <button class="btn" onclick="saveProfile()">🚀 Save Campaign Profile & Generate Questions</button>
  <div class="success" id="success-msg">✅ Profile saved! Students will now get AI-generated questions about you. Go check the quiz at <strong>/</strong></div>
</div>

<script>
async function saveProfile() {
  const payload = {
    password: document.getElementById('password').value,
    name: document.getElementById('name').value,
    position: document.getElementById('position').value,
    slogan: document.getElementById('slogan').value,
    campaign_color: document.getElementById('campaign_color').value,
    achievements: document.getElementById('achievements').value,
    manifesto: document.getElementById('manifesto').value,
    personality: document.getElementById('personality').value,
  };

  if (!payload.password || !payload.name || !payload.achievements) {
    alert('Please fill in at least: password, name, and achievements.');
    return;
  }

  try {
    const res = await fetch('/setup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (res.ok) {
      document.getElementById('success-msg').style.display = 'block';
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      alert(data.detail || 'Something went wrong');
    }
  } catch (e) {
    alert('Could not connect to server');
  }
}
</script>
</body>
</html>"""


# ─── STUDENT QUIZ FRONTEND ────────────────────────────────────────────────────
def build_student_html(leader: LeaderProfile) -> str:
    color = leader.campaign_color or "#e63946"
    name = leader.name or "Our Candidate"
    position = leader.position or "Student Leader"
    slogan = leader.slogan or "The Right Choice."

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Do You Really Know {name}?</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #050508;
    --surface: #0d0d14;
    --card: #111118;
    --border: #1f1f2e;
    --accent: {color};
    --accent-dim: {color}22;
    --accent-glow: {color}44;
    --text: #f5f5f7;
    --muted: #6b6b80;
    --gold: #ffd60a;
    --green: #06d6a0;
    --radius: 18px;
  }}

  * {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }}

  /* Dramatic background */
  body::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
      radial-gradient(ellipse 80% 50% at 50% -20%, {color}18 0%, transparent 60%),
      radial-gradient(ellipse 40% 40% at 80% 80%, {color}08 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }}

  #app {{
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 0 16px 80px;
  }}

  .screen {{ display:none; width:100%; max-width:540px; animation: fadeUp 0.5s ease forwards; }}
  .screen.active {{ display:flex; flex-direction:column; gap:18px; padding-top:32px; }}

  @keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(30px); }}
    to {{ opacity:1; transform:translateY(0); }}
  }}

  /* ── HERO LANDING ── */
  .campaign-hero {{
    text-align: center;
    padding: 48px 24px 36px;
    background: linear-gradient(180deg, {color}15 0%, transparent 100%);
    border: 1px solid {color}30;
    border-radius: var(--radius);
    position: relative;
    overflow: hidden;
  }}
  .campaign-hero::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, {color}, transparent);
  }}

  .election-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--accent-dim);
    border: 1px solid var(--accent-glow);
    color: var(--accent);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 20px;
  }}
  .pulse-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.5s infinite;
  }}
  @keyframes pulse {{ 0%,100% {{ opacity:1;transform:scale(1); }} 50% {{ opacity:0.4;transform:scale(1.5); }} }}

  .hero-name {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 52px;
    letter-spacing: 3px;
    line-height: 1;
    margin-bottom: 8px;
    background: linear-gradient(135deg, #fff 0%, {color} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}

  .hero-position {{
    font-size: 13px;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
  }}

  .hero-slogan {{
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 24px;
    font-style: italic;
  }}

  .hero-question {{
    font-size: 15px;
    color: var(--muted);
    line-height: 1.7;
  }}
  .hero-question strong {{ color: var(--text); }}

  /* ── LEVEL SELECT ── */
  .level-cards {{ display:flex; flex-direction:column; gap:12px; }}
  .level-card {{
    padding: 20px;
    border-radius: 14px;
    border: 1.5px solid var(--border);
    background: var(--card);
    cursor: pointer;
    transition: all 0.25s;
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    overflow: hidden;
  }}
  .level-card::before {{
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--lc);
    border-radius: 3px 0 0 3px;
  }}
  .level-card:hover {{ transform: translateX(6px); border-color: var(--lc); }}
  .level-icon {{ font-size: 28px; flex-shrink:0; }}
  .level-info {{ flex:1; }}
  .level-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--lc);
    margin-bottom: 3px;
  }}
  .level-title {{ font-weight: 600; font-size: 16px; margin-bottom: 2px; }}
  .level-desc {{ font-size: 12px; color: var(--muted); }}
  .level-arrow {{ color: var(--muted); font-size: 18px; }}
  .level-locked {{ opacity: 0.35; cursor: not-allowed; }}
  .level-locked:hover {{ transform: none; }}

  /* ── QUIZ SCREEN ── */
  .quiz-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
  }}
  .quiz-level-tag {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
  }}
  .timer {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    letter-spacing: 2px;
    transition: color 0.3s;
  }}
  .timer.safe {{ color: var(--green); }}
  .timer.warn {{ color: var(--gold); }}
  .timer.danger {{ color: var(--accent); animation: shake 0.3s infinite; }}
  @keyframes shake {{ 0%,100% {{ transform:translateX(0); }} 50% {{ transform:translateX(3px); }} }}

  .progress-bar {{
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }}
  .progress-fill {{
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.5s ease;
    box-shadow: 0 0 8px var(--accent-glow);
  }}

  .question-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
  }}
  .q-num {{ font-size: 11px; color: var(--muted); margin-bottom: 10px; letter-spacing: 1px; }}
  .q-text {{
    font-size: 19px;
    font-weight: 600;
    line-height: 1.5;
    margin-bottom: 22px;
  }}

  .options {{ display:flex; flex-direction:column; gap:10px; }}
  .option {{
    padding: 14px 18px;
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 12px;
    cursor: pointer;
    font-size: 14px;
    line-height: 1.5;
    transition: all 0.2s;
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }}
  .opt-letter {{
    width: 26px; height: 26px;
    border-radius: 7px;
    background: var(--border);
    display: flex; align-items:center; justify-content:center;
    font-weight: 700; font-size: 11px;
    flex-shrink: 0;
    transition: all 0.2s;
    margin-top: 1px;
  }}
  .option:hover:not(.locked) {{ border-color: var(--accent); background: var(--accent-dim); }}
  .option:hover:not(.locked) .opt-letter {{ background: var(--accent); color: #fff; }}
  .option.correct {{ border-color: var(--green); background: rgba(6,214,160,0.08); }}
  .option.correct .opt-letter {{ background: var(--green); color: #000; }}
  .option.wrong {{ border-color: var(--accent); background: rgba(230,57,70,0.08); }}
  .option.wrong .opt-letter {{ background: var(--accent); color: #fff; }}
  .option.locked {{ cursor: default; }}

  .explanation {{
    margin-top: 14px;
    padding: 12px 16px;
    background: rgba(6,214,160,0.06);
    border: 1px solid rgba(6,214,160,0.2);
    border-radius: 10px;
    font-size: 13px;
    color: var(--green);
    line-height: 1.6;
    display: none;
  }}

  .score-pop {{
    text-align: center;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 22px;
    letter-spacing: 2px;
    min-height: 32px;
    color: var(--green);
  }}

  /* ── RESULT ── */
  .result-hero {{
    text-align: center;
    padding: 40px 24px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    position: relative;
    overflow: hidden;
  }}
  .result-hero::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, {color}, transparent);
  }}
  .result-emoji {{ font-size: 56px; margin-bottom: 12px; }}
  .result-score-big {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 72px;
    letter-spacing: 4px;
    color: var(--accent);
    line-height: 1;
  }}
  .result-label {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
  .result-verdict {{ font-size: 22px; font-weight: 700; margin-top: 12px; }}
  .result-cta {{
    margin-top: 20px;
    padding: 16px 24px;
    background: var(--accent-dim);
    border: 1px solid var(--accent-glow);
    border-radius: 12px;
    font-size: 16px;
    font-weight: 600;
    color: var(--accent);
    line-height: 1.5;
  }}

  .stat-strip {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
  }}
  .stat-box {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
  }}
  .stat-val {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 26px;
    letter-spacing: 1px;
    color: var(--accent);
  }}
  .stat-lbl {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}

  .referral-section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
  }}
  .ref-title {{ font-size: 16px; font-weight: 700; margin-bottom: 6px; }}
  .ref-sub {{ font-size: 13px; color: var(--muted); margin-bottom: 14px; line-height: 1.6; }}
  .ref-code {{
    background: var(--surface);
    border: 1.5px dashed var(--accent);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 26px;
    letter-spacing: 6px;
    color: var(--accent);
    cursor: pointer;
  }}
  .ref-code:hover {{ background: var(--accent-dim); }}

  /* ── LEADERBOARD ── */
  .lb-item {{
    display: flex; align-items: center; gap: 14px;
    padding: 14px 16px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    transition: transform 0.2s;
  }}
  .lb-item:hover {{ transform: translateX(4px); }}
  .lb-item.gold {{ border-color: var(--gold); background: rgba(255,214,10,0.04); }}
  .lb-item.silver {{ border-color: #aaa; }}
  .lb-item.bronze {{ border-color: #cd7f32; }}
  .lb-rank {{ font-family:'Bebas Neue',sans-serif; font-size:22px; width:30px; text-align:center; }}
  .lb-name {{ flex:1; font-weight:500; }}
  .lb-score {{ font-family:'Bebas Neue',sans-serif; font-size:20px; color:var(--accent); letter-spacing:1px; }}

  /* ── BUTTONS ── */
  .btn {{
    padding: 16px 24px; border-radius: 13px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 17px; letter-spacing: 1.5px;
    cursor: pointer; border: none;
    transition: all 0.2s;
    display: flex; align-items: center; justify-content: center; gap: 8px;
  }}
  .btn-primary {{
    background: var(--accent);
    color: white;
    box-shadow: 0 4px 24px var(--accent-glow);
  }}
  .btn-primary:hover {{ transform:translateY(-2px); box-shadow: 0 8px 32px var(--accent-glow); }}
  .btn-outline {{
    background: transparent;
    border: 1.5px solid var(--border);
    color: var(--muted);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    letter-spacing: 0;
  }}
  .btn-outline:hover {{ border-color: var(--accent); color: var(--accent); }}
  .btn:disabled {{ opacity:0.4; cursor:not-allowed; transform:none !important; }}

  /* ── REGISTER ── */
  .reg-card {{ background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:28px; }}
  .reg-title {{ font-size:18px; font-weight:700; margin-bottom:4px; }}
  .reg-sub {{ font-size:13px; color:var(--muted); margin-bottom:20px; }}
  .form-group {{ display:flex; flex-direction:column; gap:7px; margin-bottom:14px; }}
  label {{ font-size:12px; color:var(--muted); font-weight:600; letter-spacing:0.5px; text-transform:uppercase; }}
  input {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 13px 15px; color: var(--text); font-family:'DM Sans',sans-serif;
    font-size: 14px; outline: none; transition: border-color 0.2s; width:100%;
  }}
  input:focus {{ border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-dim); }}
  input::placeholder {{ color: #3a3a50; }}

  .toast {{
    position: fixed; bottom: 24px; left:50%;
    transform: translateX(-50%) translateY(80px);
    background: var(--card); border:1px solid var(--border);
    padding: 12px 20px; border-radius: 100px;
    font-size: 14px; font-weight: 500;
    z-index: 999; transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1);
    white-space: nowrap;
  }}
  .toast.show {{ transform: translateX(-50%) translateY(0); }}
  #confetti-canvas {{ position:fixed; inset:0; pointer-events:none; z-index:999; }}
</style>
</head>
<body>
<canvas id="confetti-canvas"></canvas>
<div id="app">

  <!-- LANDING -->
  <div class="screen active" id="s-land">
    <div class="campaign-hero">
      <div class="election-badge"><span class="pulse-dot"></span> Election 2024</div>
      <div class="hero-name">{name}</div>
      <div class="hero-position">{position}</div>
      <div class="hero-slogan">"{slogan}"</div>
      <p class="hero-question">
        Think you know your candidate?<br>
        <strong>Test yourself. Learn the truth. Make the right vote.</strong>
      </p>
    </div>

    <div class="reg-card">
      <div class="reg-title">Join the Quiz</div>
      <div class="reg-sub">Register to save your score on the leaderboard</div>
      <div class="form-group"><label>Your Name</label><input id="reg-name" type="text" placeholder="e.g. Kwame Mensah" /></div>
      <div class="form-group"><label>Phone Number</label><input id="reg-phone" type="tel" placeholder="+233 XX XXX XXXX" /></div>
      <div class="form-group"><label>Referral Code (optional)</label><input id="reg-ref" type="text" placeholder="From a friend?" style="text-transform:uppercase" /></div>
      <button class="btn btn-primary" style="width:100%;margin-top:6px" onclick="handleRegister()">START THE QUIZ →</button>
    </div>

    <button class="btn btn-outline" onclick="showScreen('s-lb'); loadLeaderboard()">🏆 View Leaderboard</button>
  </div>

  <!-- LEVEL SELECT -->
  <div class="screen" id="s-levels">
    <div style="text-align:center">
      <div style="font-family:'Bebas Neue',sans-serif;font-size:32px;letter-spacing:2px">CHOOSE YOUR LEVEL</div>
      <div style="font-size:13px;color:var(--muted);margin-top:4px">Complete all 3 to unlock the full picture</div>
    </div>
    <div class="level-cards">
      <div class="level-card" style="--lc:#00b4d8" onclick="startLevel(1)">
        <div class="level-icon">🏆</div>
        <div class="level-info">
          <div class="level-label">Level 1</div>
          <div class="level-title">What They've Done</div>
          <div class="level-desc">Real achievements. Real impact. Judge the record.</div>
        </div>
        <div class="level-arrow">›</div>
      </div>
      <div class="level-card" style="--lc:#ffa500" id="lc-2" onclick="startLevel(2)">
        <div class="level-icon">📋</div>
        <div class="level-info">
          <div class="level-label">Level 2</div>
          <div class="level-title">The Vision</div>
          <div class="level-desc">Plans, promises and the future they're building for you.</div>
        </div>
        <div class="level-arrow">›</div>
      </div>
      <div class="level-card" style="--lc:#00c864" id="lc-3" onclick="startLevel(3)">
        <div class="level-icon">😄</div>
        <div class="level-info">
          <div class="level-label">Level 3</div>
          <div class="level-title">Know Your Leader</div>
          <div class="level-desc">The human behind the campaign. Hobbies, jokes & personality.</div>
        </div>
        <div class="level-arrow">›</div>
      </div>
    </div>
    <div id="live-score-bar" style="text-align:center;font-size:13px;color:var(--muted)">
      Total Score: <span style="color:var(--accent);font-family:'Bebas Neue',sans-serif;font-size:20px" id="total-score-display">0</span> pts
    </div>
  </div>

  <!-- QUIZ -->
  <div class="screen" id="s-quiz">
    <div class="quiz-header">
      <div class="quiz-level-tag" id="quiz-level-tag">LEVEL 1</div>
      <div class="timer safe" id="timer">30</div>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="prog-fill" style="width:20%"></div></div>
    <div class="question-card">
      <div class="q-num" id="q-num">Question 1 of 5</div>
      <div class="q-text" id="q-text">Loading...</div>
      <div class="options" id="options"></div>
      <div class="explanation" id="explanation"></div>
    </div>
    <div class="score-pop" id="score-pop"></div>
  </div>

  <!-- RESULT -->
  <div class="screen" id="s-result">
    <div class="result-hero">
      <div class="result-emoji" id="res-emoji">🎉</div>
      <div class="result-score-big" id="res-score">0</div>
      <div class="result-label">POINTS THIS ROUND</div>
      <div class="result-verdict" id="res-verdict">Great effort!</div>
      <div class="result-cta" id="res-cta">Loading...</div>
    </div>
    <div class="stat-strip">
      <div class="stat-box"><div class="stat-val" id="res-correct">0/5</div><div class="stat-lbl">Correct</div></div>
      <div class="stat-box"><div class="stat-val" id="res-streak">0</div><div class="stat-lbl">Best Streak</div></div>
      <div class="stat-box"><div class="stat-val" id="res-total">0</div><div class="stat-lbl">Total Pts</div></div>
    </div>
    <div class="referral-section">
      <div class="ref-title">📣 Spread the Word</div>
      <div class="ref-sub">Share your code with fellow students — when they join, you earn bonus retries AND you help {name} get more votes!</div>
      <div class="ref-code" id="ref-code" onclick="copyRef()">••••••••</div>
      <p style="text-align:center;font-size:12px;color:var(--muted);margin-top:8px">Tap to copy</p>
    </div>
    <button class="btn btn-primary" onclick="showScreen('s-levels')">NEXT LEVEL →</button>
    <button class="btn btn-outline" onclick="showScreen('s-lb'); loadLeaderboard()">🏆 Leaderboard</button>
    <button class="btn btn-outline" onclick="goHome()">← Back to Start</button>
  </div>

  <!-- LEADERBOARD -->
  <div class="screen" id="s-lb">
    <div style="text-align:center;padding:28px 0 8px">
      <div style="font-family:'Bebas Neue',sans-serif;font-size:14px;letter-spacing:3px;color:var(--muted)">TOP SUPPORTERS</div>
      <div style="font-family:'Bebas Neue',sans-serif;font-size:36px;letter-spacing:2px">{name.upper()}</div>
    </div>
    <div id="lb-list" style="display:flex;flex-direction:column;gap:10px">
      <p style="text-align:center;color:var(--muted)">Loading...</p>
    </div>
    <button class="btn btn-outline" onclick="goHome()">← Back</button>
  </div>

</div>
<div class="toast" id="toast"></div>

<script>
const API = '';
const LEVELS = {{
  1: {{ tag:'LEVEL 1 — ACHIEVEMENTS', color:'#00b4d8' }},
  2: {{ tag:'LEVEL 2 — MANIFESTO', color:'#ffa500' }},
  3: {{ tag:'LEVEL 3 — PERSONALITY', color:'#00c864' }},
}};

const VOTE_CTAS = [
  "Every question you answered correctly proves {name} has the vision students need. Make your vote count! 🗳️",
  "You just learned why {name} is the real deal. Tell 5 friends and get them to quiz too!",
  "Knowledge is power — and now you're powered up. Vote {name} on election day! 💪",
  "You know the record. You know the vision. You know the person. The choice is clear. Vote {name}!",
  "Share this quiz with every student you know. The more who know, the more who vote right! 🔥",
];

let state = {{
  userId: null, userName: '', referralCode: '',
  retriesLeft: 1, totalScore: 0,
  currentLevel: 1, questions: [],
  currentQ: 0, levelScore: 0,
  correctCount: 0, streak: 0, bestStreak: 0,
  answered: false, timerInterval: null, timeLeft: 30,
  completedLevels: [],
}};

function showScreen(id) {{
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}}

function goHome() {{ stopTimer(); showScreen('s-land'); }}

function showToast(msg) {{
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}}

let countdownSecs = 167253;
setInterval(() => {{ countdownSecs = Math.max(0, countdownSecs-1); }}, 1000);

async function handleRegister() {{
  const name = document.getElementById('reg-name').value.trim();
  const phone = document.getElementById('reg-phone').value.trim();
  const ref = document.getElementById('reg-ref').value.trim();
  if (!name || !phone) return showToast('Please enter your name and phone 👋');
  try {{
    const res = await fetch('/register', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{name, phone, referred_by: ref||null}}) }});
    const d = await res.json();
    state.userId = d.id; state.userName = d.name;
    state.referralCode = d.referral_code; state.retriesLeft = d.retries_left;
  }} catch {{
    state.userId = 'demo-'+Date.now(); state.userName = name;
    state.referralCode = 'QUIZ'+Math.random().toString(36).slice(2,6).toUpperCase();
    state.retriesLeft = 1;
  }}
  showScreen('s-levels');
}}

async function startLevel(level) {{
  state.currentLevel = level;
  state.questions = []; state.currentQ = 0;
  state.levelScore = 0; state.correctCount = 0;
  state.streak = 0; state.bestStreak = 0; state.answered = false;
  document.getElementById('score-pop').textContent = '';
  document.getElementById('explanation').style.display = 'none';

  const tag = document.getElementById('quiz-level-tag');
  tag.textContent = LEVELS[level].tag;
  tag.style.color = LEVELS[level].color;

  showScreen('s-quiz');

  try {{
    const res = await fetch('/questions?level=' + level);
    state.questions = await res.json();
  }} catch {{
    state.questions = Array(5).fill({{
      question: "Why is {name} the best candidate for students?",
      options: ["Proven track record","Clear vision","Genuine care for students","All of the above"],
      answer: "D",
      explanation: "{name} embodies all of these qualities — that's why students trust them."
    }});
  }}
  renderQuestion();
}}

function renderQuestion() {{
  const q = state.questions[state.currentQ];
  if (!q) return endLevel();
  state.answered = false; state.timeLeft = 30;
  document.getElementById('explanation').style.display = 'none';
  document.getElementById('score-pop').textContent = '';

  const total = state.questions.length;
  document.getElementById('q-num').textContent = `Question ${{state.currentQ+1}} of ${{total}}`;
  document.getElementById('prog-fill').style.width = `${{((state.currentQ+1)/total)*100}}%`;
  document.getElementById('q-text').textContent = q.question;

  const letters = ['A','B','C','D'];
  const container = document.getElementById('options');
  container.innerHTML = '';
  (q.options || []).forEach((opt, i) => {{
    const div = document.createElement('div');
    div.className = 'option';
    div.innerHTML = `<span class="opt-letter">${{letters[i]}}</span><span>${{opt}}</span>`;
    div.onclick = () => selectAnswer(i, q.answer, q.explanation);
    container.appendChild(div);
  }});
  startTimer();
}}

function startTimer() {{
  stopTimer(); updateTimer();
  state.timerInterval = setInterval(() => {{
    state.timeLeft--;
    updateTimer();
    if (state.timeLeft <= 0) {{
      stopTimer();
      if (!state.answered) {{
        state.streak = 0;
        showToast("⏰ Time's up!");
        lockOptions(null, state.questions[state.currentQ].answer);
        showExplanation(state.questions[state.currentQ].explanation);
        setTimeout(nextQ, 2000);
      }}
    }}
  }}, 1000);
}}

function stopTimer() {{ clearInterval(state.timerInterval); }}

function updateTimer() {{
  const el = document.getElementById('timer');
  el.textContent = state.timeLeft;
  el.className = 'timer ' + (state.timeLeft <= 5 ? 'danger' : state.timeLeft <= 10 ? 'warn' : 'safe');
}}

function selectAnswer(idx, correctLetter, explanation) {{
  if (state.answered) return;
  state.answered = true; stopTimer();
  const letters = ['A','B','C','D'];
  const isCorrect = letters[idx] === correctLetter;
  lockOptions(letters[idx], correctLetter);
  showExplanation(explanation);

  if (isCorrect) {{
    state.streak++; state.bestStreak = Math.max(state.bestStreak, state.streak);
    state.correctCount++;
    const pts = 100 + (state.timeLeft * 3) + (state.streak >= 3 ? 50 : 0);
    state.levelScore += pts; state.totalScore += pts;
    document.getElementById('score-pop').textContent = `+${{pts}} pts${{state.streak >= 3 ? ' 🔥' : ''}}`;
    document.getElementById('total-score-display').textContent = state.totalScore;
  }} else {{
    state.streak = 0;
    document.getElementById('score-pop').textContent = `Correct: ${{correctLetter}}`;
  }}
  setTimeout(nextQ, 2200);
}}

function lockOptions(chosenLetter, correctLetter) {{
  const letters = ['A','B','C','D'];
  document.querySelectorAll('.option').forEach((el, i) => {{
    el.classList.add('locked');
    if (letters[i] === correctLetter) el.classList.add('correct');
    else if (letters[i] === chosenLetter) el.classList.add('wrong');
  }});
}}

function showExplanation(text) {{
  if (!text) return;
  const el = document.getElementById('explanation');
  el.textContent = '💡 ' + text;
  el.style.display = 'block';
}}

function nextQ() {{
  state.currentQ++;
  state.currentQ >= state.questions.length ? endLevel() : renderQuestion();
}}

async function endLevel() {{
  stopTimer();
  state.completedLevels.push(state.currentLevel);

  try {{
    await fetch(`/submit-score?user_id=${{state.userId}}&score=${{state.totalScore}}`, {{method:'POST'}});
  }} catch {{}}

  const pct = Math.round((state.correctCount / state.questions.length) * 100);
  const emojis = pct >= 80 ? '🏆' : pct >= 60 ? '🎉' : pct >= 40 ? '👍' : '💪';
  const verdicts = [
    [80, "Outstanding! You know this candidate inside-out!"],
    [60, "Great work! You clearly pay attention."],
    [40, "Good effort! There's more to discover."],
    [0,  "Keep going — every level teaches you more!"]
  ];
  const v = verdicts.find(([min]) => pct >= min);

  document.getElementById('res-emoji').textContent = emojis;
  document.getElementById('res-score').textContent = state.levelScore.toLocaleString();
  document.getElementById('res-verdict').textContent = v[1];
  document.getElementById('res-correct').textContent = `${{state.correctCount}}/5`;
  document.getElementById('res-streak').textContent = state.bestStreak;
  document.getElementById('res-total').textContent = state.totalScore.toLocaleString();
  document.getElementById('ref-code').textContent = state.referralCode;
  document.getElementById('res-cta').textContent = VOTE_CTAS[Math.floor(Math.random() * VOTE_CTAS.length)];

  showScreen('s-result');
  if (pct >= 80) launchConfetti();
}}

async function loadLeaderboard() {{
  const list = document.getElementById('lb-list');
  list.innerHTML = '<p style="text-align:center;color:var(--muted)">Loading...</p>';
  let entries = [];
  try {{
    const res = await fetch('/leaderboard');
    entries = await res.json();
  }} catch {{
    entries = [
      {{rank:1,name:'Ama K.',score:2800}},
      {{rank:2,name:'Kweku O.',score:2450}},
      {{rank:3,name:'Priscilla T.',score:2100}},
    ];
  }}
  const medals = ['🥇','🥈','🥉'];
  const cls = ['gold','silver','bronze'];
  list.innerHTML = '';
  entries.forEach((u,i) => {{
    const div = document.createElement('div');
    div.className = `lb-item ${{cls[i]||''}}`;
    div.innerHTML = `<div class="lb-rank">${{medals[i]||u.rank}}</div><div class="lb-name">${{u.name}}</div><div class="lb-score">${{u.score.toLocaleString()}}</div>`;
    list.appendChild(div);
  }});
  if (state.userName) {{
    const self = document.createElement('div');
    self.className = 'lb-item'; self.style.borderColor = 'var(--accent)';
    self.innerHTML = `<div class="lb-rank">👤</div><div class="lb-name">${{state.userName}} <span style="font-size:11px;color:var(--accent)">(You)</span></div><div class="lb-score">${{state.totalScore.toLocaleString()}}</div>`;
    list.appendChild(self);
  }}
}}

function copyRef() {{
  if (!state.referralCode) return;
  navigator.clipboard.writeText(state.referralCode).then(() => showToast('Code copied! Share it 🎉')).catch(() => {{}});
}}

function launchConfetti() {{
  const canvas = document.getElementById('confetti-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = window.innerWidth; canvas.height = window.innerHeight;
  const color = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  const particles = Array.from({{length:150}}, () => ({{
    x: Math.random()*canvas.width, y: -20,
    vx: (Math.random()-0.5)*5, vy: Math.random()*5+2,
    color: [color,'#ffd60a','#06d6a0','#fff','#ff6b6b'][Math.floor(Math.random()*5)],
    size: Math.random()*9+4, rot: Math.random()*360, rs: (Math.random()-0.5)*8
  }}));
  let frame;
  (function draw() {{
    ctx.clearRect(0,0,canvas.width,canvas.height);
    particles.forEach(p => {{
      p.x+=p.vx; p.y+=p.vy; p.rot+=p.rs;
      ctx.save(); ctx.translate(p.x,p.y); ctx.rotate(p.rot*Math.PI/180);
      ctx.fillStyle=p.color; ctx.fillRect(-p.size/2,-p.size/2,p.size,p.size*0.5);
      ctx.restore();
    }});
    if (particles.some(p=>p.y<canvas.height+50)) frame=requestAnimationFrame(draw);
    else ctx.clearRect(0,0,canvas.width,canvas.height);
  }})();
  setTimeout(()=>{{cancelAnimationFrame(frame);ctx.clearRect(0,0,canvas.width,canvas.height);}},5000);
}}
</script>
</body>
</html>"""


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def student_quiz(db: Session = Depends(get_db)):
    leader = get_leader(db)
    return HTMLResponse(content=build_student_html(leader))


@app.get("/setup", response_class=HTMLResponse)
def setup_page():
    return HTMLResponse(content=SETUP_HTML)


@app.post("/setup")
def save_profile(payload: LeaderProfileUpdate, db: Session = Depends(get_db)):
    if payload.password != SETUP_PASSWORD:
        raise HTTPException(status_code=403, detail="Wrong password")
    leader = get_leader(db)
    leader.name = payload.name
    leader.position = payload.position
    leader.achievements = payload.achievements
    leader.manifesto = payload.manifesto
    leader.personality = payload.personality
    leader.slogan = payload.slogan
    leader.campaign_color = payload.campaign_color
    db.commit()
    return {"message": "Profile saved successfully"}


@app.get("/questions")
def get_questions(level: int = 1, db: Session = Depends(get_db)):
    leader = get_leader(db)
    return generate_campaign_questions(leader, level)


@app.post("/register")
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

    if payload.referred_by:
        ref = db.query(User).filter(User.referral_code == payload.referred_by).first()
        if ref:
            ref.retries_left += 1
            db.commit()

    return {
        "id": str(user.id), "name": user.name,
        "referral_code": user.referral_code, "retries_left": user.retries_left
    }


@app.get("/leaderboard")
def leaderboard(db: Session = Depends(get_db)):
    users = (
        db.query(User)
        .filter(User.eligible_for_leaderboard == True)  # noqa
        .order_by(User.score.desc())
        .limit(10)
        .all()
    )
    return [{"rank": i+1, "name": u.name, "score": u.score} for i, u in enumerate(users)]


@app.post("/submit-score")
def submit_score(user_id: str, score: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    if score > user.score:
        user.score = score
        db.commit()
    return {"score": user.score}
