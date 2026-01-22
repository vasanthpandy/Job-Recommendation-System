"""
Intelligent Job Recommendation System with Dashboard (single-file Flask app)
Features added:
 - Modern Bootstrap UI with Navbar and Dashboard
 - Charts (Chart.js) for job distribution (location, titles)
 - Add / delete jobs (CSV persists)
 - Upload jobs.csv or download recommended jobs as CSV
 - Keyword highlighting in job descriptions relative to resume
 - Improved resume parsing (PDF/DOCX/TXT) with stable stream handling
 - Export recommended jobs as CSV

Run:
 pip install flask pandas scikit-learn PyPDF2 python-docx nltk
 python Intelligent_Job_Recommendation_System_with_Dashboard.py
 Open http://127.0.0.1:5000
"""

import os
import re
import io
import csv
import json
from typing  import List, Tuple
from flask import (
    Flask, request, render_template_string, send_file,
    redirect, url_for, flash, jsonify, make_response
)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- NLTK resources ----------
needed = ["punkt", "stopwords", "wordnet", "omw-1.4"]
for n in needed:
    try:
        nltk.data.find(n)
    except Exception:
        nltk.download(n)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'change_this_secret')
JOBS_CSV = 'jobs.csv'

# ---------- Utility: create sample jobs.csv ----------
def create_sample_jobs_csv(path=JOBS_CSV):
    if os.path.exists(path):
        return
    sample = [
        {'job_id': 1, 'title': 'Machine Learning Engineer', 'company': 'AI Labs', 'location': 'Remote',
         'description': 'We are looking for a Machine Learning Engineer with experience in Python, scikit-learn, TensorFlow, data preprocessing, feature engineering, and model deployment.'},
        {'job_id': 2, 'title': 'Data Scientist', 'company': 'DataCorp', 'location': 'Bangalore',
         'description': 'Data Scientist role requiring expertise in statistical modeling, Python, pandas, visualization, and business problem solving.'},
        {'job_id': 3, 'title': 'NLP Engineer', 'company': 'TextAI', 'location': 'Hyderabad',
         'description': 'Build NLP pipelines using spaCy, transformers, BERT, and experience with text preprocessing and semantics.'},
        {'job_id': 4, 'title': 'Backend Developer (Python)', 'company': 'WebWorks', 'location': 'Chennai',
         'description': 'Backend developer skilled in Python, Django/Flask, REST APIs, database design, and unit testing.'},
        {'job_id': 5, 'title': 'Software Engineer - Intern', 'company': 'StartupX', 'location': 'Remote',
         'description': 'Internship for recent grads. Skills: problem solving, Python/Java, eagerness to learn, basic ML knowledge is a plus.'}
    ]
    df = pd.DataFrame(sample)
    df.to_csv(path, index=False)
    print(f"Created sample jobs CSV at {path}")

# ---------- Read & manage jobs ----------
def read_jobs(path=JOBS_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        create_sample_jobs_csv(path)
    df = pd.read_csv(path)
    # ensure required columns
    required = {'job_id', 'title', 'company', 'location', 'description'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"jobs.csv missing columns: {missing}")
    df['combined'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    return df


def save_jobs(df: pd.DataFrame, path=JOBS_CSV):
    df.to_csv(path, index=False)

# ---------- Resume extractors ----------

def extract_text_from_pdf(stream):
    reader = PyPDF2.PdfReader(stream)
    text = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ''
            text.append(t)
        except Exception:
            continue
    return '\n'.join(text)


def extract_text_from_docx(stream):
    doc = docx.Document(stream)
    return '\n'.join([p.text for p in doc.paragraphs])


def extract_text_from_txt(stream):
    return stream.read().decode(errors='ignore')


def parse_resume(file_storage) -> str:
    filename = file_storage.filename.lower()
    # read bytes into a stable buffer
    data = file_storage.read()
    stream = io.BytesIO(data)

    if filename.endswith('.pdf'):
        return extract_text_from_pdf(stream)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(stream)
    elif filename.endswith('.txt'):
        return extract_text_from_txt(stream)
    else:
        # heuristic: try pdf, then decode
        try:
            return extract_text_from_pdf(stream)
        except Exception:
            try:
                return data.decode(errors='ignore')
            except Exception:
                return ''

# ---------- Preprocessing & highlighting ----------

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def highlight_keywords(text: str, keywords: List[str]) -> str:
    # simple safe highlight: mark whole-word matches (case-insensitive)
    if not keywords:
        return text
    uniq = sorted(set([k for k in keywords if len(k) > 1]), key=lambda x: -len(x))
    def repl(match):
        return f"<mark>{match.group(0)}</mark>"
    for kw in uniq:
        try:
            pattern = re.compile(rf'\b({re.escape(kw)})\b', flags=re.IGNORECASE)
            text = pattern.sub(repl, text)
        except re.error:
            continue
    return text

# ---------- Recommender ----------
class JobRecommender:
    def __init__(self, jobs_csv=JOBS_CSV):
        self.jobs_csv = jobs_csv
        self.jobs_df = read_jobs(jobs_csv)
        self.vectorizer = TfidfVectorizer(max_features=8000)
        clean_texts = self.jobs_df['combined'].apply(preprocess_text).tolist()
        self.job_tfidf = self.vectorizer.fit_transform(clean_texts)

    def refresh(self):
        self.jobs_df = read_jobs(self.jobs_csv)
        clean_texts = self.jobs_df['combined'].apply(preprocess_text).tolist()
        self.job_tfidf = self.vectorizer.fit_transform(clean_texts)

    def recommend(self, resume_text: str, top_n:int=5) -> List[Tuple[int, float]]:
        resume_clean = preprocess_text(resume_text)
        resume_vec = self.vectorizer.transform([resume_clean])
        sims = cosine_similarity(resume_vec, self.job_tfidf).flatten()
        top_idx = sims.argsort()[::-1][:top_n]
        results = []
        for idx in top_idx:
            results.append((int(self.jobs_df.iloc[idx]['job_id']), float(sims[idx])))
        return results

    def get_job_by_id(self, job_id:int) -> dict:
        row = self.jobs_df[self.jobs_df['job_id'] == job_id].iloc[0]
        return row.to_dict()

    def add_job(self, title, company, location, description):
        df = self.jobs_df.copy()
        next_id = int(df['job_id'].max()) + 1 if not df.empty else 1
        new = {'job_id': next_id, 'title': title, 'company': company, 'location': location, 'description': description}
        df = df.append(new, ignore_index=True)
        save_jobs(df, self.jobs_csv)
        self.refresh()

    def delete_job(self, job_id:int):
        df = self.jobs_df[self.jobs_df['job_id'] != job_id]
        save_jobs(df, self.jobs_csv)
        self.refresh()

# Initialize recommender
RECOMMENDER = JobRecommender()

# ---------- Templates (index, dashboard, jobs, results) ----------
INDEX_HTML = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Intelligent Job Recommendation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { padding-top: 72px; }
      .job-card { min-height: 140px; }
      mark { background: #fffb91; }
    </style>
  </head>
  <body>

<nav class="navbar navbar-expand-lg navbar-dark bg-primary fixed-top">
  <div class="container-fluid">
    <a class="navbar-brand" href="/">JobRecommender</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#nav" aria-controls="nav" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="nav">
      <ul class="navbar-nav me-auto mb-2 mb-lg-0">
        <li class="nav-item"><a class="nav-link" href="/">Home</a></li>
        <li class="nav-item"><a class="nav-link" href="/dashboard">Dashboard</a></li>
        <li class="nav-item"><a class="nav-link" href="/jobs">Jobs</a></li>
      </ul>
      <form class="d-flex" method="get" action="/jobs">
        <input class="form-control me-2" type="search" placeholder="Search jobs..." name="q">
        <button class="btn btn-outline-light" type="submit">Search</button>
      </form>
    </div>
  </div>
</nav>

<div class="container">
  <div class="row">
    <div class="col-md-6">
      <div class="card mb-4 shadow-sm">
        <div class="card-body">
          <h5 class="card-title">Upload your resume</h5>
          {% with messages = get_flashed_messages() %}
            {% if messages %}
              <div class="alert alert-warning">{{ messages[0] }}</div>
            {% endif %}
          {% endwith %}
          <form method="post" action="/upload" enctype="multipart/form-data">
            <div class="mb-3">
              <input class="form-control" type="file" id="resume" name="resume" required>
            </div>
            <div class="mb-3">
              <label for="topn" class="form-label">How many recommendations?</label>
              <input class="form-control" type="number" id="topn" name="topn" value="5" min="1" max="20">
            </div>
            <button type="submit" class="btn btn-success">Get Recommendations</button>
            <a class="btn btn-light" href="/dashboard">View Dashboard</a>
          </form>
        </div>
      </div>

      <div class="card shadow-sm">
        <div class="card-body">
          <h5 class="card-title">Add Job (admin)</h5>
          <form method="post" action="/add_job">
            <input class="form-control mb-2" name="title" placeholder="Title" required>
            <input class="form-control mb-2" name="company" placeholder="Company" required>
            <input class="form-control mb-2" name="location" placeholder="Location" required>
            <textarea class="form-control mb-2" name="description" placeholder="Description" rows="3" required></textarea>
            <button class="btn btn-primary">Add Job</button>
          </form>
        </div>
      </div>

    </div>

    <div class="col-md-6">
      <div class="card shadow-sm mb-4">
        <div class="card-body">
          <h5 class="card-title">Quick Stats</h5>
          <p class="mb-1">Total jobs: <strong>{{ total_jobs }}</strong></p>
          <p class="mb-1">Remote jobs: <strong>{{ remote_jobs }}</strong></p>
          <p class="mb-0">Top locations: <strong>{{ top_locations_str }}</strong></p>
          <div class="mt-3">
            <a href="/download_jobs" class="btn btn-outline-secondary btn-sm">Download Jobs CSV</a>
            <form class="d-inline" method="post" action="/upload_jobs_csv" enctype="multipart/form-data">
              <input type="file" name="jobsfile" class="form-control form-control-sm d-inline-block w-auto" required>
              <button class="btn btn-sm btn-outline-primary">Upload CSV</button>
            </form>
          </div>
        </div>
      </div>

      <div class="card shadow-sm">
        <div class="card-body">
          <h5 class="card-title">Recent Jobs</h5>
          {% for _, row in recent.iterrows() %}
            <div class="job-card p-2 border rounded mb-2">
              <strong>{{ row.title }}</strong> — {{ row.company }} <small class="text-muted">({{ row.location }})</small>
              <div class="text-muted small">{{ row.description[:140] }}{% if row.description|length > 140 %}...{% endif %}</div>
            </div>
          {% endfor %}
        </div>
      </div>

    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

DASH_HTML = '''
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
      <div class="container-fluid">
        <a class="navbar-brand" href="/">JobRecommender</a>
      </div>
    </nav>
    <div class="container py-4">
      <h3>Dashboard</h3>
      <div class="row">
        <div class="col-md-6">
          <div class="card mb-3 p-3">
            <h5>Total Jobs: {{ total_jobs }}</h5>
            <p>Remote: {{ remote_jobs }}</p>
            <p>Unique locations: {{ unique_locations }}</p>
          </div>
        </div>
        <div class="col-md-6">
          <div class="card p-3">
            <h5>Top Job Titles</h5>
            <canvas id="titlesChart"></canvas>
          </div>
        </div>
      </div>

      <div class="row mt-3">
        <div class="col-md-6">
          <div class="card p-3">
            <h5>Locations</h5>
            <canvas id="locChart"></canvas>
          </div>
        </div>
        <div class="col-md-6">
          <div class="card p-3">
            <h5>Manage Jobs</h5>
            <table class="table table-sm">
              <thead><tr><th>ID</th><th>Title</th><th>Action</th></tr></thead>
              <tbody>
                {% for _, r in jobs.iterrows() %}
                  <tr>
                    <td>{{ r.job_id }}</td>
                    <td>{{ r.title }}</td>
                    <td>
                      <form method="post" action="/delete_job" onsubmit="return confirm('Delete job?');">
                        <input type="hidden" name="job_id" value="{{ r.job_id }}">
                        <button class="btn btn-sm btn-danger">Delete</button>
                      </form>
                    </td>
                  </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
        </div>
      </div>

    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
      const titlesData = {{ titles_json | safe }};
      const locData = {{ loc_json | safe }};

      const ctxT = document.getElementById('titlesChart');
      if (ctxT) {
        new Chart(ctxT, {
          type: 'bar',
          data: {
            labels: titlesData.labels,
            datasets: [{ label: 'Count', data: titlesData.data }]
          }
        });
      }

      const ctxL = document.getElementById('locChart');
      if (ctxL) {
        new Chart(ctxL, {
          type: 'pie',
          data: {
            labels: locData.labels,
            datasets: [{ data: locData.data }]
          }
        });
      }
    </script>
  </body>
</html>
'''

JOBS_HTML = '''
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Jobs</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
      <div class="container-fluid"><a class="navbar-brand" href="/">JobRecommender</a></div>
    </nav>
    <div class="container py-4">
      <h3>Jobs Database</h3>
      <form method="get" class="mb-3">
        <div class="input-group">
          <input class="form-control" name="q" placeholder="Search title or description" value="{{ q }}">
          <button class="btn btn-outline-primary">Search</button>
        </div>
      </form>
      <table class="table table-striped">
        <thead><tr><th>ID</th><th>Title</th><th>Company</th><th>Location</th><th>Description</th></tr></thead>
        <tbody>
          {% for _, row in jobs.iterrows() %}
            <tr>
              <td>{{ row.job_id }}</td>
              <td>{{ row.title }}</td>
              <td>{{ row.company }}</td>
              <td>{{ row.location }}</td>
              <td>{{ row.description }}</td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
      <a href="/dashboard" class="btn btn-secondary">Back to Dashboard</a>
    </div>
  </body>
</html>
'''

RESULTS_HTML = '''
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Recommendations</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style> mark{ background:#fffb91; } </style>
  </head>
  <body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
      <div class="container-fluid"><a class="navbar-brand" href="/">JobRecommender</a></div>
    </nav>
    <div class="container py-4">
      <a class="btn btn-link" href="/">← Back</a>
      <h4>Top Recommendations</h4>
      <p class="text-muted">Showing top {{ detailed|length }} matched jobs.</p>

      <div class="mb-3">
        <form method="post" action="/export_recs">
          <input type="hidden" name="rec_ids" value="{{ rec_ids|join(',') }}">
          <button class="btn btn-outline-secondary btn-sm">Download Recommendations CSV</button>
        </form>
      </div>

      <ul class="list-group">
        {% for job, score, highlighted in detailed %}
          <li class="list-group-item">
            <h5>{{ job.title }} <small class="text-muted">- {{ job.company }} ({{ job.location }})</small></h5>
            <p>{{ highlighted|safe }}</p>
            <div>Match Score: <strong>{{ "{:.2f}".format(score*100) }}%</strong></div>
          </li>
        {% endfor %}
      </ul>
    </div>
  </body>
</html>
'''


# ---------- Routes ----------
@app.route('/')
def index():
    jobs_df = RECOMMENDER.jobs_df
    total_jobs = len(jobs_df)
    remote_jobs = jobs_df['location'].str.contains('remote', case=False, na=False).sum()
    top_locations = jobs_df['location'].value_counts().head(3).to_dict()
    top_locations_str = ', '.join([f"{k}({v})" for k,v in top_locations.items()])
    recent = jobs_df.sort_values('job_id', ascending=False).head(5)
    return render_template_string(INDEX_HTML, total_jobs=total_jobs, remote_jobs=remote_jobs, top_locations_str=top_locations_str, recent=recent)


@app.route('/dashboard')
def dashboard():
    jobs_df = RECOMMENDER.jobs_df
    total_jobs = len(jobs_df)
    remote_jobs = jobs_df['location'].str.contains('remote', case=False, na=False).sum()
    unique_locations = jobs_df['location'].nunique()
    # titles frequency
    titles = jobs_df['title'].value_counts().head(8)
    titles_json = {'labels': titles.index.tolist(), 'data': titles.values.tolist()}
    loc = jobs_df['location'].value_counts().head(8)
    loc_json = {'labels': loc.index.tolist(), 'data': loc.values.tolist()}
    return render_template_string(DASH_HTML, total_jobs=total_jobs, remote_jobs=remote_jobs, unique_locations=unique_locations, titles_json=json.dumps(titles_json), loc_json=json.dumps(loc_json), jobs=jobs_df)


@app.route('/jobs', methods=['GET'])
def jobs():
    q = request.args.get('q','').strip()
    df = RECOMMENDER.jobs_df
    if q:
        mask = df['title'].str.contains(q, case=False, na=False) | df['description'].str.contains(q, case=False, na=False)
        df = df[mask]
    return render_template_string(JOBS_HTML, jobs=df, q=q)


@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['resume']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    try:
        topn = int(request.form.get('topn', 5))
    except Exception:
        topn = 5

    try:
        text = parse_resume(file)
        if not text.strip():
            flash('Could not extract text from the resume. Try another format (PDF/DOCX/TXT).')
            return redirect(url_for('index'))
        results = RECOMMENDER.recommend(text, top_n=topn)
        detailed = []
        # prepare keywords from resume for highlighting
        resume_tokens = preprocess_text(text).split()
        keywords = [t for t in resume_tokens if len(t)>2][:80]
        rec_ids = []
        for job_id, score in results:
            j = RECOMMENDER.get_job_by_id(job_id)
            highlighted = highlight_keywords(j['description'], keywords)
            detailed.append((j, score, highlighted))
            rec_ids.append(str(job_id))
        return render_template_string(RESULTS_HTML, detailed=detailed, rec_ids=rec_ids)
    except Exception as e:
        flash('Error processing resume: ' + str(e))
        return redirect(url_for('index'))


@app.route('/add_job', methods=['POST'])
def add_job():
    try:
        title = request.form['title']
        company = request.form['company']
        location = request.form['location']
        description = request.form['description']
        RECOMMENDER.add_job(title, company, location, description)
        flash('Job added')
    except Exception as e:
        flash('Error adding job: ' + str(e))
    return redirect(url_for('index'))


@app.route('/delete_job', methods=['POST'])
def delete_job():
    try:
        job_id = int(request.form['job_id'])
        RECOMMENDER.delete_job(job_id)
        flash('Job deleted')
    except Exception as e:
        flash('Error deleting job: ' + str(e))
    return redirect(url_for('dashboard'))


@app.route('/download_jobs')
def download_jobs():
    df = RECOMMENDER.jobs_df
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='jobs_export.csv')


@app.route('/upload_jobs_csv', methods=['POST'])
def upload_jobs_csv():
    if 'jobsfile' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('index'))
    f = request.files['jobsfile']
    try:
        data = f.read()
        s = io.StringIO(data.decode(errors='ignore'))
        df = pd.read_csv(s)
        # ensure columns
        required = {'job_id', 'title', 'company', 'location', 'description'}
        if not required.issubset(set(df.columns)):
            flash('CSV missing required columns. Need job_id,title,company,location,description')
            return redirect(url_for('index'))
        save_jobs(df)
        RECOMMENDER.refresh()
        flash('Jobs uploaded')
    except Exception as e:
        flash('Error uploading CSV: ' + str(e))
    return redirect(url_for('index'))


@app.route('/export_recs', methods=['POST'])
def export_recs():
    rec_ids = request.form.get('rec_ids','')
    ids = [int(x) for x in rec_ids.split(',') if x.strip()]
    rows = []
    for jid in ids:
        try:
            rows.append(RECOMMENDER.get_job_by_id(jid))
        except Exception:
            continue
    if not rows:
        flash('No recommendations to export')
        return redirect(url_for('index'))
    df = pd.DataFrame(rows)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='recommended_jobs.csv')


if __name__ == '__main__':
    create_sample_jobs_csv(JOBS_CSV)
    RECOMMENDER = JobRecommender(JOBS_CSV)
    print('Starting app at http://127.0.0.1:5000')
    app.run(debug=True)
