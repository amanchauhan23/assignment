# TV Series and Movie Genre Prediction

This repository contains an end-to-end machine learning pipeline to predict the primary genre of a TV series or movie based on its title and description.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
2. **train:**
   ```bash
   python src/train.py --data_path data/tv-shows.csv --model gradient_boosting --seed 42
3. **eval:**
   ```bash
   python src/evaluate.py --model_path outputs/model.pkl --data_path data/tv-shows.csv
