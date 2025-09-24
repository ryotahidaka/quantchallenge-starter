import json
from startegy_rh import Strategy, Ticker, Side

# Assume Strategy and other classes are already imported
strategy = Strategy()

# Simulate orderbook updates before events (so we have market prices to compare)
# For example, best bid = 45, best ask = 55 → market prob = 0.50
strategy.on_orderbook_update(Ticker.TEAM_A, Side.BUY, 100, 45)
strategy.on_orderbook_update(Ticker.TEAM_A, Side.SELL, 100, 55)

# Load test events
with open("example-game.json", "r") as f:
    events = json.load(f)

# Replay events into the strategy
for event in events:
    strategy.on_game_event_update(
        event_type=event["event_type"],
        home_away=event["home_away"],
        home_score=event["home_score"],
        away_score=event["away_score"],
        player_name=event["player_name"],
        substituted_player_name=event["substituted_player_name"],
        shot_type=event["shot_type"],
        assist_player=event["assist_player"],
        rebound_type=event["rebound_type"],
        coordinate_x=event["coordinate_x"],
        coordinate_y=event["coordinate_y"],
        time_seconds=event["time_seconds"]
    )
