# Solar flare prediction (hackathon)

Theme 5: space weather. We predict strong solar flares from sun magnetic images using a small CNN.

## Run the website

```bash
cd space-weather-flare-cnn
source .venv/bin/activate
streamlit run app/app.py
```

## Train (if needed)

```bash
python src/generate_data.py
python src/train.py
```

## Files

- `src/image_processing.py` - blur the image
- `src/generate_data.py` - make training pictures
- `src/model.py` - CNN
- `src/train.py` - train
- `app/app.py` - website

We used some AI help for the pytorch part.
