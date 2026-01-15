import numpy as np
from scipy import stats

# Test the exact calculation that's failing
demand_std = 10.0
lead_time = 5
service_level = 0.95

print(f"Testing calculate_safety_stock with demand_std={demand_std}, lead_time={lead_time}, service_level={service_level}")

try:
    # Convert service level to z-score
    z_score = stats.norm.ppf(service_level)
    print(f"z_score = {z_score}")
    
    # Calculate lead time demand standard deviation
    lead_time_demand_std = demand_std * np.sqrt(lead_time)
    print(f"lead_time_demand_std = {lead_time_demand_std}")
    
    # Safety stock = z * sigma_LT
    safety_stock = z_score * lead_time_demand_std
    print(f"safety_stock = {safety_stock}")
    
    # Calculate reorder point
    avg_demand_per_day = demand_std  # Simplified assumption
    reorder_point = avg_demand_per_day * lead_time + safety_stock
    print(f"reorder_point = {reorder_point}")
    
    # Test the return values
    print(f"Testing return values...")
    print(f"safety_stock type: {type(safety_stock)}, value: {safety_stock}")
    print(f"reorder_point type: {type(reorder_point)}, value: {reorder_point}")
    
    safety_stock_int = int(np.ceil(safety_stock))
    reorder_point_int = int(np.ceil(reorder_point))
    
    print(f"safety_stock_int: {safety_stock_int}")
    print(f"reorder_point_int: {reorder_point_int}")
    
    result = {
        'safety_stock': safety_stock_int,
        'reorder_point': reorder_point_int,
        'z_score': z_score,
        'service_level_achieved': service_level,
        'lead_time_demand_std': lead_time_demand_std
    }
    
    print("Success! Result:", result)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()