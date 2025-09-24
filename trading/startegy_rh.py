"""
Quant Challenge 2025

Algorithmic strategy template
"""
DEFAULT_MAX_INVENTORY = 50 # Inventory
DEFAULT_ORDER_SIZE = 5     # Order Size
DEFAULT_SPREAD = 2         # Spread (price units)
DEFAULT_TICK = 1           # Price tick / rounding

from enum import Enum
from typing import Optional, Dict, Tuple
import math

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    # TEAM_A (home team)
    TEAM_A = 900

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Place a market order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    """
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Place a limit order.
    
    Parameters
    ----------
    side
        Side of order to place
    ticker
        Ticker of order to place
    quantity
        Quantity of order to place
    price
        Price of order to place
    ioc
        Immediate or cancel flag (FOK)

    Returns
    -------
    order_id
        Order ID of order placed
    """
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Cancel an order.
    
    Parameters
    ----------
    ticker
        Ticker of order to cancel
    order_id
        Order ID of order to cancel

    Returns
    -------
    success
        True if order was cancelled, False otherwise
    """
    return 0

class Strategy:
    """Template for a strategy."""

    def reset_state(self) -> None:
        """Reset the state of the strategy to the start of game position.
        
        Since the sandbox execution can start mid-game, we recommend creating a
        function which can be called from __init__ and on_game_event_update (END_GAME).

        Note: In production execution, the game will start from the beginning
        and will not be replayed.
        """
        self.positions = {}
        self.pending_orders = {}
        self.last_bid = {}
        self.last_ask = {}
        self.cash = 100000
        self.tick = 1
        self._orders = {"buy": (Ticker.TEAM_A, 0, 0), "sell": (Ticker.TEAM_A, 0, 0)}
        self.inventory = 0
        self.order_size = 10
        self.max_inventory = 50
        self.spread = 2
        self.aggressive = False
        self.time_seconds = float('inf')

    def __init__(self) -> None:
        """Your initialization code goes here."""
        self.reset_state()


    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------
    def _round_price(self, price: float) -> float:
        """ Round the price to the nearest tick """
        return round(price / self.tick) * self.tick
    
    def _compute_mid(self) -> Optional[float]:
        best_bid = self.last_bid[Ticker.TEAM_A]
        best_ask = self.last_ask[Ticker.TEAM_A]

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        if best_bid is not None:
            return best_bid + 1  # fallback spread of 1
        if best_ask is not None:
            return best_ask - 1
        return None
    
    def _place_quotes(self) -> None:
        """ Place or update symmetric buy/sell limit orders around mid-price. """
        mid = self._compute_mid()
        if mid is None:
            return
        
        # If aggressive, tighten the spread
        eff_spread = self.spread * (0.6 if self.aggressive else 1)

        buy_price = self._round_price(mid - eff_spread / 2)
        sell_price = self._round_price(mid + eff_spread / 2)

        # Reduce buy size when inventory is positive, reduce sell size when inventory is negative
        buy_size = self.order_size
        sell_size = self.order_size
        if self.inventory > 0: # We are at long: Be conservative on the buys
            buy_size = max(1.0, self.order_size * max(0, 2, 1 - abs(self.inventory) / (self.max_inventory * 1.2)))
        elif self.inventory < 0:
            sell_size = max(1.0, self.order_size * max(0, 2, 1 - abs(self.inventory) / (self.max_inventory * 1.2)))

        # If we are near the end of the game, reduce the sizes and flatten
        if self.time_seconds is not None and self.time_seconds < 120: # Last 2 mins
            buy_size = min(buy_size, max(1, self.order_size * 0.5))
            sell_size = min(sell_size, max(1, self.order_size * 0.5))

        # Place/Update buy order
        _, existing_buy_id, existing_buy_price = self._orders["buy"]
        if existing_buy_id is None or existing_buy_price != buy_price:
            # Cancel the previous buy if it exists
            if existing_buy_id is not None:
                try:
                    cancel_order(Ticker.TEAM_A, existing_buy_id)
                except Exception as E:
                    print("Error", E)
            
            # Do not post buy if it would push us over max_inventory
            if self.inventory + buy_size <= self.max_inventory:
                try:
                    order_id = place_limit_order(Side.BUY, Ticker.TEAM_A, buy_size, buy_price, ioc=False)
                    self._orders["buy"] = (Ticker.TEAM_A, order_id, buy_price)
                except Exception as E:
                    self._orders["buy"] = (Ticker.TEAM_A, None, None)

        # Place/Update sell order
        _, existing_sell_id, existing_sell_price = self._orders["sell"]
        if existing_sell_id is None or existing_sell_price != sell_price:
            # Cancel the previous buy if it exists
            if existing_sell_id is not None:
                try:
                    cancel_order(Ticker.TEAM_A, existing_sell_id)
                except Exception as E:
                    print("Error", E)
            
            # Do not post sell if it would push us below -max_inventory
            if self.inventory - sell_size >= -self.max_inventory:
                try:
                    order_id = place_limit_order(Side.SELL, Ticker.TEAM_A, sell_size, sell_price, ioc=False)
                    self._orders["sell"] = (Ticker.TEAM_A, order_id, sell_price)
                except Exception as E:
                    self._orders["sell"] = (Ticker.TEAM_A, None, None)

    def _estimate_home_win_prob(self, home_score: int, away_score: int, time_seconds: float) -> float:
        """ Simple logistic model to estimate probability of home team winning. """
        lead = home_score - away_score

        # If no time info, assume full game left
        if time_seconds is None or time_seconds <= 0:
            return 1.0 if lead > 0 else 0.0

        time_factor = math.sqrt(max(time_seconds, 1))
        prob = 1 / (1 + math.exp(-(lead / 10 + lead / time_factor)))
        return max(0.0, min(1.0, prob))


    # ----------------------------------------------------
    # End of Helpers
    # ----------------------------------------------------

    def on_trade_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever two orders match. Could be one of your orders, or two other people's orders.
        Parameters
        ----------
        ticker
            Ticker of orders that were matched
        side:
            Side of orders that were matched
        quantity
            Volume traded
        price
            Price that trade was executed at
        """
        print(f"Python Trade update: {ticker} {side} {quantity} shares @ {price}")

        if price is not None:
            if side == Side.BUY:
                self.last_bid[ticker] = max(self.last_bid.get(ticker, 0), price)
            else:
                if ticker not in self.last_ask or price < self.last_ask[ticker]:
                    self.last_ask[ticker] = price

    def on_orderbook_update(
        self, ticker: Ticker, side: Side, quantity: float, price: float
    ) -> None:
        """Called whenever the orderbook changes. This could be because of a trade, or because of a new order, or both.
        Parameters
        ----------
        ticker
            Ticker that has an orderbook update
        side
            Which orderbook was updated
        price
            Price of orderbook that has an update
        quantity
            Volume placed into orderbook
        """
        # Initialize state if not present
        if ticker not in self.last_bid:
            self.last_bid[ticker] = None
        if ticker not in self.last_ask:
            self.last_ask[ticker] = None

        if side == Side.BUY:
            # Buy side update => best bid
            if quantity == 0:
                # Order removed at this price, clear if it's best
                if self.last_bid[ticker] == price:
                    self.last_bid[ticker] = None
            else:
                if self.last_bid[ticker] is None or price > self.last_bid[ticker]:
                    self.last_bid[ticker] = price
        else:
            # Sell side update => best ask
            if quantity == 0:
                if self.last_ask[ticker] == price:
                    self.last_ask[ticker] = None
            else:
                if self.last_ask[ticker] is None or price < self.last_ask[ticker]:
                    self.last_ask[ticker] = price

        # Only place quotes if both sides are known
        if self.last_bid[ticker] is not None and self.last_ask[ticker] is not None:
            self._place_quotes()

    def on_account_update(
        self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        """Called whenever one of your orders is filled.
        Parameters
        ----------
        ticker
            Ticker of order that was fulfilled
        side
            Side of order that was fulfilled
        price
            Price that order was fulfilled at
        quantity
            Volume of order that was fulfilled
        capital_remaining
            Amount of capital after fulfilling order
        """
        # Called when one of orders is filled. Update inventory and capital and react.
        if price is None or quantity is None:
            print("Skipping account update due to None values:", price, quantity)
            return

        # Update inventory: BUY increases inventory, SELL decreases
        if side == Side.BUY:
            self.inventory += quantity
            self.cash -= price * quantity
        else:
            self.inventory -= quantity
            self.cash += price * quantity

        # Platform provided captial remaining is authoritative for cash/limits
        # Keep a local copy but avoid overwriting if None
        # capital_remaining might be used to control position sizing TODO

        print(f"Filled {side.name} {quantity} @ {price}. Inventory now {self.inventory}. Cash ~ {capital_remaining}")

        # If we've exceeded risk threshold, aggressively flatten
        if abs(self.inventory) > 0.9 * self.max_inventory:
            # Flatten by sending market order opposite to our inventory sign
            if self.inventory > 0:
                # We are at long: sell to flatten
                qty_to_flatten = min(self.inventory, self.max_inventory)
                try:
                    place_market_order(Side.SELL, ticker, qty_to_flatten)
                except Exception:
                    pass
            elif self.inventory < 0:
                qty_to_flatten = min(abs(self.inventory), self.max_inventory)
                try:
                    place_market_order(Side.BUY, ticker, qty_to_flatten)
                except Exception:
                    pass

        # After fills, update quotes
        self._place_quotes()

    
    def on_game_event_update(self,
                           event_type: str,
                           home_away: str,
                           home_score: int,
                           away_score: int,
                           player_name: Optional[str],
                           substituted_player_name: Optional[str],
                           shot_type: Optional[str],
                           assist_player: Optional[str],
                           rebound_type: Optional[str],
                           coordinate_x: Optional[float],
                           coordinate_y: Optional[float],
                           time_seconds: Optional[float]
        ) -> None:
        """Called whenever a basketball game event occurs.
        Parameters
        ----------
        event_type
            Type of event that occurred
        home_score
            Home team score after event
        away_score
            Away team score after event
        player_name (Optional)
            Player involved in event
        substituted_player_name (Optional)
            Player being substituted out
        shot_type (Optional)
            Type of shot
        assist_player (Optional)
            Player who made the assist
        rebound_type (Optional)
            Type of rebound
        coordinate_x (Optional)
            X coordinate of shot location in feet
        coordinate_y (Optional)
            Y coordinate of shot location in feet
        time_seconds (Optional)
            Game time remaining in seconds
        """
        # Use game events to alter the behaviour
        print(f"{event_type} {home_score} - {away_score}")

        if event_type == "END_GAME":
            self.reset_state()
            return
        
        if time_seconds is not None:
            self.time_seconds = time_seconds

        # Only trade if we have both bid/ask info
        if Ticker.TEAM_A not in self.last_ask or Ticker.TEAM_A not in self.last_bid:
            return
        
        # Ski
        if home_score is None or away_score is None:
            return

        # Our estimated probability
        my_prob = self._estimate_home_win_prob(home_score, away_score, time_seconds)

        # Market implied probability (use mid-price)
        best_bid = self.last_bid[Ticker.TEAM_A]
        best_ask = self.last_ask[Ticker.TEAM_A]

        # Skip trading if either side is missing
        if best_bid is None or best_ask is None:
            return

        try:
            mid_price = (best_bid + best_ask) / 2
            market_prob = mid_price / 100
        except TypeError as e:
            print(f"Skipping trade calculation due to TypeError: {e}")
            return

        print(f"Est prob: {my_prob:.2f}, Market prob: {market_prob:.2f}")

        # Decision rule with threshold
        threshold = 0.05
        if abs(my_prob - getattr(self, 'last_traded_prob', 0.0)) > 0.02:
            if my_prob > market_prob + threshold:
                print("BUY signal!")
                place_market_order(Side.BUY, Ticker.TEAM_A, 10)
            elif my_prob < market_prob - threshold:
                print("SELL signal!")
                place_market_order(Side.SELL, Ticker.TEAM_A, 10)
            self.last_traded_prob = my_prob
