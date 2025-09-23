
# Trading Algorithm

- Maintains the best bid/ask, computes mid, and posts symmetric limit orders
- Track order IDs so it can update/cancel orders when the market moves
- Limit inventory and use market orders near the end of the game to flatten # May remove
- Adjust agressiveness based on scoring events
- Reset states at the end of the game (cancel outstanding orders)