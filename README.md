# AI Resume Screener

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An AI-based tool that analyzes resumes and ranks 
candidates based on skill match using 
Machine Learning (Logistic Regression + TF-IDF).

## Features
- Reads .txt resumes from data/ folder
- Extract key skills (Python, SQL, Azure,
JavaScript, Machine Learning, etc.)
- Scores resumes and ranks them in descending order
- Pre-trained model saved in models/ folder for reuse
- Modular code: train_model.py (training) and main.py (screening)

## Tech Stack
- Python 3.10
- Pandas, NumPy, Scikit-learn, NLTK
- Joblib (model persistence)
- Git & GitHub for version control

## Project Structure
```markdown
ai-resume-screener/

├── 📂 data/              # Sample resumes in .txt format
├── 📂 models/            # Saved ML model & vectorizer
├── 📂 src/               # Core scripts
│ ├── 🧠 train_model.py   # Train and save your model
│ └── 🎯 main.py          # Predict & rank resumes
├── 📄 requirements.txt   # Project dependencies
└── 📘 README.md          # Project overview and instructions

```

## How to Run

### 1. Clone the repository & set up a virtual environment

```bash

git clone <https://github.com/anoodh-ai/ai-resume-screener.git>
cd ai-resume-screener
python -m venv .venv
.venv\Scripts\activate   # For Windows
#.venv/bin/activate      # If you're on Mac/Linux
pip install -r requirements.txt    
```
### 2. Train the model (if needed)
```bash

python src/train_model.py
```
### 3. Add resumes & run the screening
```bash

python src/main.py
```
## Example Output
```bash
Resume Rankings (match %):
resume6.txt:  77.77% match
resume5.txt:  76.70% match
resume1.txt:  76.49% match
resume16.txt: 75.36% match
resume19.txt: 74.28% match
resume7.txt:  71.38% match
resume17.txt: 68.72% match
resume20.txt: 67.92% match
resume9.txt:  65.48% match
resume8.txt:  63.42% match
```
## Accuracy
- (Current test dataset) : ~100%
- Expected real-world accuracy: ~80–85% (Will improve as more resumes are added)


## Future Improvements

- PDF/Docx parsing
- Web UI using Flask/Streamlit
- Azure cloud deployment
- Advanced skill matching using embeddings

---
 #### 📄 License
```markdown
MIT License — see [LICENSE](LICENSE) for full details.
> This project is shared for learning and inspiration. Please credit the author (Anoodh A) if reused or modified.
```
 