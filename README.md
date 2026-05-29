# World Cup Match Outcome Predictor

A machine-learning pipeline that forecasts FIFA World Cup match results using ELO ratings, recent form, and lineup strength — with an interactive Streamlit player dashboard.

## Project Structure

```
world-cup-predictor/
├── data/
│   ├── raw/             # Downloaded source files (git-ignored)
│   └── processed/       # Feature-engineered datasets (git-ignored)
├── notebooks/           # Exploratory analysis
├── src/
│   ├── data_pipeline/   # Ingestion and preprocessing
│   ├── features/        # ELO ratings, form, lineup strength
│   ├── models/          # Training, prediction, evaluation
│   └── dashboard/       # Streamlit player dashboard
└── tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your API key (football-data.org) in a `.env` file:

```
FOOTBALL_DATA_API_KEY=your_key_here
```

## Usage

### Fetch and preprocess data

```python
from src.data_pipeline import fetch_world_cup_matches, encode_outcome

df = fetch_world_cup_matches(api_key="...")
df = encode_outcome(df)
```

### Train the model

```python
from src.models import train

model = train(df, model_path="models/xgb_model.pkl")
```

### Predict a match

```python
from src.models import load_model, predict_outcome

model = load_model()
probs = predict_outcome(model, {
    "home_elo": 1820, "away_elo": 1750, "elo_diff": 70,
    "home_form": 0.80, "away_form": 0.60,
    "home_squad_rating": 85.0, "away_squad_rating": 82.0,
    "home_attack_strength": 87.0, "away_attack_strength": 79.0,
    "home_defence_strength": 83.0, "away_defence_strength": 81.0,
})
# {"home_win": 0.54, "draw": 0.27, "away_win": 0.19}
```

### Launch the player dashboard

```bash
streamlit run src/dashboard/app.py
```

## Data Sources

- Match results: [football-data.org](https://www.football-data.org/) free-tier API
- Player ratings: FIFA / Sofascore CSVs placed in `data/raw/players.csv`

## Running Tests

```bash
pytest tests/
```
