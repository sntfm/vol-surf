import asyncio
import time, datetime
import functools
import yfinance as yf
from typing import Dict, List, Any, Callable
import pytz

from flatbuf import serialize_option_chains, cleanup_shared_memory

def profile_time(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"{func.__name__}: {(t1-t0)*1e6:.2f}µs")
        return result
    return wrapper

def profile_time_async(func: Callable):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = await func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"{func.__name__}: {(t1-t0)*1e6:.2f}µs")
        return result
    return wrapper

def calculate_time_to_expiry(exp_date: str) -> float:
    # Parse expiration date and set to 4:00 PM Eastern (SPX options expire at 4:00 PM ET)
    exp_dt = datetime.datetime.strptime(exp_date, '%Y-%m-%d')
    eastern = pytz.timezone('US/Eastern')
    exp_dt = eastern.localize(exp_dt.replace(hour=16, minute=0))
    
    now = datetime.datetime.now(eastern)
    diff = exp_dt - now
    days = diff.total_seconds() / (24 * 3600)  

    return max(days / 365, 0.0) 

class ConnectorYF:    
    def __init__(self, ticker_str: str, max_concurrent: int = 200, request_delay: float = 5e-100):
        self.ticker = yf.Ticker(ticker_str)
        self.expirations = self.ticker.options
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_delay = request_delay
    
    @profile_time
    def get_spot_price(self) -> float:
        try: return self.ticker.fast_info['last_price']
        except:
            try: return self.ticker.info['regularMarketPrice']
            except: return None
    
    @profile_time
    def get_rfr(self) -> float:
        try:
            try: bond = yf.Ticker("^IRX")
            except: 
                try: bond = yf.Ticker("^FVX")
                except: bond = yf.Ticker("^TNX")

            return bond.history(period="1d")['Close'].iloc[-1] / 100.0
        except: return None

    @profile_time
    def get_option_chain(self, exp_date: str) -> Dict[str, Any]:
        return self.ticker.option_chain(exp_date)

    @profile_time_async
    async def get_option_chain_async(self, exp_date: str) -> Dict[str, Any]:
        async with self.semaphore:
            await asyncio.sleep(self.request_delay)  # throttle requests
            
            @profile_time
            def blocking_call():
                try:
                    # chain = self.ticker.option_chain(exp_date)
                    chain = self.get_option_chain(exp_date)
                    spot_price = self.get_spot_price()
                    rfr = self.get_rfr()
                    if spot_price is None: raise Exception("Spot price is None")
                    if rfr is None: raise Exception("Risk-free rate is None")

                    # Calculate time to expiry
                    tau = calculate_time_to_expiry(exp_date)
                    num_options = max(len(chain.calls['strike']), len(chain.puts['strike']))

                    return {
                        'expiration': exp_date,
                        'spot_price': spot_price,
                        'tau_years': tau,
                        'rfr': [rfr] * num_options,
                        'calls_strike': chain.calls['strike'].to_numpy(),
                        'calls_bid': chain.calls['bid'].to_numpy(),
                        'calls_ask': chain.calls['ask'].to_numpy(),
                        'puts_strike': chain.puts['strike'].to_numpy(),
                        'puts_bid': chain.puts['bid'].to_numpy(),
                        'puts_ask': chain.puts['ask'].to_numpy()
                    }
                except Exception as e:
                    print(f"Failed for {exp_date}: {e}")
                    return None

            result = await asyncio.to_thread(blocking_call)
            return result

    @profile_time
    async def get_all_option_chains(self) -> List[Dict[str, Any]]:
        print(f"Found {len(self.expirations)} expirations.")
        
        tasks = [self.get_option_chain_async(exp) for exp in self.expirations]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    @profile_time
    def fetch(self) -> List[Dict[str, Any]]:
        chains = asyncio.run(self.get_all_option_chains())
        return chains

    @profile_time
    def to_shared_memory(self): serialize_option_chains(self.fetch())

    @profile_time
    def purge_shared_memory(self): cleanup_shared_memory()

if __name__ == "__main__":
    def main():
        ticker = "^SPX"
        connector = ConnectorYF(ticker)
        # print(connector.fetch()[0])
        connector.to_shared_memory()

        print("\nKeeping shared memory open. Press Ctrl+C to exit...")
        try: 
            while True: time.sleep(1)
        except KeyboardInterrupt: connector.purge_shared_memory()

    main()