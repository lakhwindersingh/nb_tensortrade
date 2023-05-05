from enum import Enum
from tensortrade.core import TimedIdentifiable

class TradeType(Enum):
    LIMIT: str = "limit"
    MARKET: str = "market"
    PUT: str = "put"
    CALL: str = "call"

    def __str__(self):
        return str(self.value)

class TradeSide(Enum):
    BUY_TO_OPEN: str = "buy_to_open"
    SELL_TO_OPEN: str = "sell_to_open"
    BUY_TO_CLOSE: str = "buy_to_close"
    SELL_TO_CLOSE: str = "sell_to_close"

    def instrument(self, pair: "TradingPair") -> "Instrument":
        return pair.base if self in [TradeSide.BUY_TO_OPEN, TradeSide.BUY_TO_CLOSE] else pair.quote

    def __str__(self):
        return str(self.value)

class OptionTrade(TimedIdentifiable):
    """A trade object for use within trading environments."""

    def __init__(self,
                 order_id: str,
                 step: int,
                 exchange_pair: 'ExchangePair',
                 side: TradeSide,
                 trade_type: TradeType,
                 quantity: 'Quantity',
                 strike_price: float,
                 expiration_date: str,
                 commission: 'Quantity'):
        """
        Arguments:
            order_id: The id of the order that created the trade.
            step: The timestep the trade was made during the trading episode.
            exchange_pair: The exchange pair of instruments in the trade.
            side: Whether the quote instrument is being bought or sold.
            size: The size of the core instrument in the trade.
            strike_price: The strike price of the option contract.
            expiration_date: The expiration date of the option contract.
            commission: The commission paid for the trade in terms of the core instrument.
        """
        super().__init__()
        self.order_id = order_id
        self.step = step
        self.exchange_pair = exchange_pair
        self.side = side
        self.type = trade_type
        self.quantity = quantity
        self.strike_price = strike_price
        self.expiration_date = expiration_date
        self.commission = commission

    @property
    def base_instrument(self) -> 'Instrument':
        return self.exchange_pair.pair.base

    @property
    def quote_instrument(self) -> 'Instrument':
        return self.exchange_pair.pair.quote

    @property
    def size(self) -> float:
        return self.quantity.size

    @property
    def commission(self) -> 'Quantity':
        return self._commission

    @commission.setter
    def commission(self, commission: 'Quantity'):
        self._commission = commission

    @property
    def is_buy_to_open(self) -> bool:
        return self.side == TradeSide.BUY_TO_OPEN

    @property
    def is_sell_to_open(self) -> bool:
        return self.side == TradeSide.SELL_TO_OPEN

    @property
    def is_buy_to_close(self) -> bool:
        return self.side == TradeSide.BUY_TO_CLOSE

    @property
    def is_sell_to_close(self) -> bool:
        return self.side == TradeSide.SELL_TO_CLOSE

    @property
    def is_limit_order(self) -> bool:
        return self.type == TradeType.LIMIT

    @property
    def is_market_order(self) -> bool:
        return self.type == TradeType.MARKET

    @property
    def is_put(self) -> bool:
        return self.type == TradeType.PUT

    @property
    def is_call(self) -> bool:
        return self.type == Trade
