
# Trading Algorithm

- Maintains the best bid/ask, computes mid, and posts symmetric limit orders
- Track order IDs so it can update/cancel orders when the market moves
- Limit inventory and use market orders near the end of the game to flatten # May remove
- Adjust agressiveness based on scoring events
- Reset states at the end of the game (cancel outstanding orders)

# Market Overview

## The Basketball Market
You can `BUY` or `SELL` a security that tracks the probability of the home team winning a basketball game.

The `BUY` contract is structured as follows:
- You immediately pay the market price of the contract.
- Pays out $100.00 if the home team wins
- Pays out $0.00 if the home team loses
- There are no ties in basketball games

The `SELL` contract is structured as follows:
- You immediately pay the market price of the contract.
- Pays out $0.00 if the home team wins
- Pays out $100.00 if the home team loses
- There are no ties in basketball games

Note that BUY and SELL contracts automatically cancel each other, i.e. buying a SELL contract is equivalent to selling a BUY contract.

- Buy (go long) if you believe the home team is more likely to win than the current market price suggests, or sell (go short) if you believe they're less likely to win.

# Game Events & Data Stream
## Real-Time Basketball Events
Throughout the basketball game, your algorithm will receive a continuous stream of parsed basketball events that can significantly impact the probability of the home team winning.

Each event contains detailed information about what happened in the game, including scores, player actions, timing, and spatial coordinates for shots. Your algorithm can use this information to update its assessment of win probability and adjust trading positions accordingly.

The game events are structured as an array of event objects, with each event representing a specific moment in the basketball game.

## Event Object Structure

### Required Fields
- `home_away` - Team indicator ("home", "away", or "unknown" for events not associated with a team)
- `home_score` - Integer: Home team score after this event
- `away_score` - Integer: Away team score after this event
- `event_type` - String: Type of basketball event (see Event Types below)
- `time_seconds` - Float: Game time in seconds remaining (0.0 = end of game, but events may still happen with 0.0 remaining. Officially the game ends on the END_GAME event)

### Fields That May Contain Null
- `player_name` - String: Anonymized player name ("Player 1", "Player 2", etc.). There can be up to 50 player numbers, and each player number is unique to a single player. Will be null for JUMP_BALL, TIMEOUT, START_PERIOD, END_PERIOD, END_GAME, DEADBALL, NOTHING, and UNKNOWN events. Can be null in select other situations.
- `substituted_player_name` - String: Player being substituted out (SUBSTITUTION events only)
- `shot_type` - String: Parsed shot type (SCORE/MISSED events only)
- `assist_player` - String: Player who made the assist (SCORE events only)
- `rebound_type` - String: Type of rebound ("OFFENSIVE" or "DEFENSIVE". REBOUND events only)
- `coordinate_x` - Float: X coordinate of shot location in ft (SCORE/MISSED events only)
- `coordinate_y` - Float: Y coordinate of shot location in ft (SCORE/MISSED events only)

### Event Types

- `JUMP_BALL` - Game start or jump ball situations
- `SCORE` - Successful shot/basket
- `MISSED` - Missed shot attempt
- `REBOUND` - Ball recovery after missed shot
- `STEAL` - Defensive player steals ball
- `BLOCK` - Shot blocked by defender
- `TURNOVER` - Team loses possession. Can have null player_name.
- `FOUL` - Personal or technical foul
- `TIMEOUT` - Team timeout
- `SUBSTITUTION` - Player substitution (only in some formats)
- `START_PERIOD` - Period start (only in some formats)
- `END_PERIOD` - Period end (will not appear in the last period)
- `END_GAME` - Game conclusion
- `DEADBALL` - Dead ball situation (only in some formats)
- `NOTHING` - Nothing happened except passage of time
- `UNKNOWN` - Unknown event (only in some formats)

### Time Format
`time_seconds`: Total seconds of game time remaining.

Depending on the game format, a game can have either 2400 total seconds or 2880 total seconds. IMPORTANTLY: there will be no overtime in any game, and there are no ties.