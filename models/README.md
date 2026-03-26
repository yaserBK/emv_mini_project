Place your calibration .pkl file here, e.g.:
```
  docker compose run --rm build      # generates distribution.pkl from data/train/
```

Then run:
```
  docker compose up video            # live webcam scoring
  docker compose run --rm inference  # batch scoring against data/val/
```