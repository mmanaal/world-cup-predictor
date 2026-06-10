from src.models.predict import predict_match

fixtures = [
    # June 11
    ("Mexico", "South Africa"),
    ("South Korea", "Czech Republic"),
    # June 12
    ("Canada", "Bosnia And Herzegovina"),
    ("United States", "Paraguay"),
    # June 13
    ("Qatar", "Switzerland"),
    ("Brazil", "Morocco"),
    ("Haiti", "Scotland"),
    ("Australia", "Turkey"),
    # June 14
    ("Germany", "Curaçao"),
    ("Netherlands", "Japan"),
    ("Ivory Coast", "Ecuador"),
    ("Sweden", "Tunisia"),
    # June 15
    ("Belgium", "Egypt"),
    ("Spain", "Cape Verde"),
    ("Iran", "New Zealand"),
    ("Saudi Arabia", "Uruguay"),
    # June 16
    ("France", "Senegal"),
    ("Iraq", "Norway"),
    ("Argentina", "Algeria"),
    ("Austria", "Jordan"),
    # June 17
    ("Portugal", "Dr Congo"),
    ("England", "Croatia"),
    ("Ghana", "Panama"),
    ("Uzbekistan", "Colombia"),
]

print("=" * 60)
print("  2026 WORLD CUP — ROUND 1 PREDICTIONS")
print("=" * 60)

for home, away in fixtures:
    try:
        r = predict_match(
            home, away,
            neutral=True,
            tournament_weight=3,
            is_knockout=False
        )
        print(f"\n{home} vs {away}")
        print(f"  Home win : {r['home_win_prob']*100:.1f}%")
        print(f"  Draw     : {r['draw_prob']*100:.1f}%")
        print(f"  Away win : {r['away_win_prob']*100:.1f}%")
        print(f"  Predicted: {r['predicted_outcome']} ({r['confidence']})")
        print(f"  Key factors: {', '.join(r['top_factors'])}")
    except Exception as e:
        print(f"\n{home} vs {away} — could not predict: {e}")