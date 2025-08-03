import flatbuffers
from multiprocessing import shared_memory
import OptionData.OptionChain as OptionChain
import OptionData.OptionChainList as OptionChainList
import numpy as np

def build_chain(builder, data):
    # Create expiration string first
    exp_offset = builder.CreateString(data['expiration'])
    
    # Calls vectors
    OptionChain.StartCallsStrikeVector(builder, len(data['calls_strike']))
    for x in reversed(data['calls_strike']):
        builder.PrependFloat32(float(x))
    calls_strike = builder.EndVector()

    OptionChain.StartCallsBidVector(builder, len(data['calls_bid']))
    for x in reversed(data['calls_bid']):
        builder.PrependFloat32(float(x))
    calls_bid = builder.EndVector()

    OptionChain.StartCallsAskVector(builder, len(data['calls_ask']))
    for x in reversed(data['calls_ask']):
        builder.PrependFloat32(float(x))
    calls_ask = builder.EndVector()

    # Puts vectors
    OptionChain.StartPutsStrikeVector(builder, len(data['puts_strike']))
    for x in reversed(data['puts_strike']):
        builder.PrependFloat32(float(x))
    puts_strike = builder.EndVector()

    OptionChain.StartPutsBidVector(builder, len(data['puts_bid']))
    for x in reversed(data['puts_bid']):
        builder.PrependFloat32(float(x))
    puts_bid = builder.EndVector()

    OptionChain.StartPutsAskVector(builder, len(data['puts_ask']))
    for x in reversed(data['puts_ask']):
        builder.PrependFloat32(float(x))
    puts_ask = builder.EndVector()

    # RFR vector
    OptionChain.StartRfrVector(builder, len(data['rfr']))
    for x in reversed(data['rfr']):
        builder.PrependFloat32(float(x))
    rfr = builder.EndVector()
    
    # Start building the chain
    OptionChain.OptionChainStart(builder)
    
    # Add all fields
    OptionChain.OptionChainAddExpiration(builder, exp_offset)
    OptionChain.OptionChainAddSpotPrice(builder, float(data['spot_price']))
    OptionChain.OptionChainAddTauYears(builder, float(data['tau_years']))
    OptionChain.OptionChainAddRfr(builder, rfr)
    OptionChain.OptionChainAddCallsStrike(builder, calls_strike)
    OptionChain.OptionChainAddCallsBid(builder, calls_bid)
    OptionChain.OptionChainAddCallsAsk(builder, calls_ask)
    OptionChain.OptionChainAddPutsStrike(builder, puts_strike)
    OptionChain.OptionChainAddPutsBid(builder, puts_bid)
    OptionChain.OptionChainAddPutsAsk(builder, puts_ask)
    
    # Return the offset of the chain
    return OptionChain.OptionChainEnd(builder)

def serialize_option_chains(chain_list):
    # Validate data
    for chain in chain_list:
        if not isinstance(chain, dict):
            raise ValueError("Chain must be a dictionary")
        if not all(k in chain for k in ['expiration', 'spot_price', 'tau_years', 'rfr', 
                                      'calls_strike', 'calls_bid', 'calls_ask',
                                      'puts_strike', 'puts_bid', 'puts_ask']):
            raise ValueError("Missing required fields in chain")
        
        # Convert numpy arrays to lists
        for k in ['calls_strike', 'calls_bid', 'calls_ask', 'puts_strike', 'puts_bid', 'puts_ask']:
            if isinstance(chain[k], np.ndarray):
                chain[k] = chain[k].tolist()

    builder = flatbuffers.Builder(1024 * 1024)  # 1MB initial size
    
    # Build all OptionChains and store offsets
    chain_offsets = [build_chain(builder, data) for data in reversed(chain_list)]
    
    # Create vector of chains
    OptionChainList.StartChainsVector(builder, len(chain_offsets))
    for off in chain_offsets: builder.PrependUOffsetTRelative(off)
    chains_vec = builder.EndVector()
    
    # Create root table
    OptionChainList.OptionChainListStart(builder)
    OptionChainList.OptionChainListAddChains(builder, chains_vec)
    chains = OptionChainList.OptionChainListEnd(builder)
    
    # Finish the buffer
    builder.Finish(chains)
    buf = builder.Output()

    buf_size = len(buf)
    try:
        # Try to reuse existing shared memory if size matches
        shm = shared_memory.SharedMemory(name="option_chains")
        if shm.size >= buf_size:
            shm.buf[:buf_size] = buf
            return shm.name
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass
        
    # Create new shared memory if needed
    shm = shared_memory.SharedMemory(name="option_chains", create=True, size=buf_size)
    shm.buf[:buf_size] = buf
    return shm.name

def cleanup_shared_memory(shm_name="option_chains"):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass  # Memory segment already removed
