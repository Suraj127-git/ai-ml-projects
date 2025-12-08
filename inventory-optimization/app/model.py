import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class InventoryOptimizationModel:
    """
    Inventory Optimization Model
    Supports EOQ, safety stock calculation, ABC analysis, and multi-echelon optimization
    """
    
    def __init__(self):
        self.abc_categories = {}
        self.demand_forecasters = {}
        self.lead_time_models = {}
        self.optimization_history = []
        
    def generate_synthetic_inventory_data(self, n_products: int = 20, n_days: int = 365) -> pd.DataFrame:
        """Generate synthetic inventory and demand data"""
        np.random.seed(42)
        
        products = []
        categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home']
        
        for i in range(1, n_products + 1):
            category = np.random.choice(categories)
            
            # Generate product characteristics based on category
            if category == 'Electronics':
                unit_cost = np.random.uniform(100, 1000)
                holding_cost_rate = 0.25
                demand_rate = np.random.uniform(5, 20)
                lead_time = np.random.randint(14, 30)
            elif category == 'Clothing':
                unit_cost = np.random.uniform(20, 200)
                holding_cost_rate = 0.20
                demand_rate = np.random.uniform(10, 50)
                lead_time = np.random.randint(7, 21)
            elif category == 'Food':
                unit_cost = np.random.uniform(5, 50)
                holding_cost_rate = 0.15
                demand_rate = np.random.uniform(20, 100)
                lead_time = np.random.randint(3, 10)
            elif category == 'Books':
                unit_cost = np.random.uniform(10, 80)
                holding_cost_rate = 0.18
                demand_rate = np.random.uniform(5, 25)
                lead_time = np.random.randint(10, 20)
            else:  # Home
                unit_cost = np.random.uniform(30, 300)
                holding_cost_rate = 0.22
                demand_rate = np.random.uniform(8, 35)
                lead_time = np.random.randint(10, 25)
            
            # Generate demand with seasonality and trend
            base_demand = demand_rate
            trend = np.random.uniform(-0.01, 0.02)
            seasonality_amplitude = np.random.uniform(0.1, 0.3)
            
            daily_demand = []
            for day in range(n_days):
                trend_component = base_demand + (trend * day)
                seasonal_component = seasonality_amplitude * base_demand * np.sin(2 * np.pi * day / 365)
                noise = np.random.normal(0, base_demand * 0.1)
                
                daily_demand.append(max(0, trend_component + seasonal_component + noise))
            
            # Calculate annual demand and revenue
            annual_demand = sum(daily_demand)
            annual_revenue = annual_demand * unit_cost
            
            products.append({
                'product_id': f"PRODUCT_{i:03d}",
                'product_name': f"Product {i}",
                'category': category,
                'unit_cost': unit_cost,
                'holding_cost_rate': holding_cost_rate,
                'ordering_cost': np.random.uniform(50, 200),
                'lead_time_days': lead_time,
                'current_stock': np.random.randint(50, 500),
                'demand_rate': demand_rate,
                'demand_std': np.std(daily_demand),
                'annual_demand': annual_demand,
                'annual_revenue': annual_revenue,
                'service_level': np.random.uniform(0.90, 0.99),
                'expiration_days': np.random.choice([None, 30, 90, 180, 365], p=[0.3, 0.2, 0.2, 0.2, 0.1]),
                'min_order_quantity': np.random.randint(1, 10),
                'max_stock_capacity': np.random.randint(500, 2000),
                'supplier_reliability': np.random.uniform(0.85, 0.99)
            })
        
        return pd.DataFrame(products)
    
    def generate_historical_demand(self, products_df: pd.DataFrame, n_days: int = 365) -> pd.DataFrame:
        """Generate historical demand data for products"""
        historical_data = []
        
        for _, product in products_df.iterrows():
            base_demand = product['demand_rate']
            demand_std = product['demand_std']
            
            for day in range(n_days):
                current_date = datetime.now() - timedelta(days=n_days-day)
                
                # Generate demand with some randomness
                demand = max(0, np.random.normal(base_demand, demand_std))
                
                # Occasionally create stockouts (5% probability)
                stock_out = np.random.random() < 0.05
                lost_sales = demand * 0.3 if stock_out else 0
                
                historical_data.append({
                    'product_id': product['product_id'],
                    'date': current_date,
                    'demand': demand,
                    'stock_out': stock_out,
                    'lost_sales': lost_sales
                })
        
        return pd.DataFrame(historical_data)
    
    def calculate_eoq(self, annual_demand: float, ordering_cost: float, 
                     unit_cost: float, holding_cost_rate: float) -> Dict[str, float]:
        """Calculate Economic Order Quantity (EOQ)"""
        
        # EOQ formula: sqrt(2 * D * S / H)
        # D = annual demand, S = ordering cost, H = holding cost per unit per year
        holding_cost_per_unit = unit_cost * holding_cost_rate
        
        if holding_cost_per_unit <= 0:
            return {'error': 'Holding cost must be positive'}
        
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        
        # Calculate total costs
        ordering_cost_annual = (annual_demand / eoq) * ordering_cost
        holding_cost_annual = (eoq / 2) * holding_cost_per_unit
        total_cost = ordering_cost_annual + holding_cost_annual
        
        # Calculate cycle time and orders per year
        cycle_time = eoq / (annual_demand / 365)  # days
        orders_per_year = annual_demand / eoq
        
        return {
            'economic_order_quantity': int(np.ceil(eoq)),
            'total_cost': total_cost,
            'ordering_cost': ordering_cost_annual,
            'holding_cost': holding_cost_annual,
            'cycle_time_days': cycle_time,
            'orders_per_year': orders_per_year
        }
    
    def calculate_safety_stock(self, demand_std: float, lead_time: int, 
                               service_level: float, lead_time_std: Optional[float] = None) -> Dict[str, float]:
        """Calculate safety stock using statistical methods"""
        
        print(f"DEBUG safety_stock: demand_std={demand_std}, lead_time={lead_time}, service_level={service_level}")
        
        # Convert service level to z-score
        print(f"DEBUG: About to call stats.norm.ppf with service_level={service_level}")
        z_score = stats.norm.ppf(service_level)
        print(f"DEBUG: stats.norm.ppf returned z_score={z_score}")
        
        # Calculate lead time demand standard deviation
        if lead_time_std is None:
            # Assume lead time is constant
            print(f"DEBUG: lead_time_std is None, calculating with demand_std={demand_std}, lead_time={lead_time}")
            lead_time_demand_std = demand_std * np.sqrt(lead_time)
            print(f"DEBUG: lead_time_demand_std calculated as {lead_time_demand_std}")
        else:
            # Consider lead time variability
            avg_demand = demand_std  # Assuming demand_std represents daily demand std
            print(f"DEBUG: lead_time_std provided, calculating with avg_demand={avg_demand}")
            lead_time_demand_std = np.sqrt(
                lead_time * (demand_std ** 2) + (avg_demand ** 2) * (lead_time_std ** 2)
            )
            print(f"DEBUG: lead_time_demand_std calculated as {lead_time_demand_std}")
        
        # Safety stock = z * sigma_LT
        print(f"DEBUG: calculating safety_stock with z_score={z_score}, lead_time_demand_std={lead_time_demand_std}")
        safety_stock = z_score * lead_time_demand_std
        print(f"DEBUG: safety_stock calculated as {safety_stock}")
        
        # Calculate reorder point
        avg_demand_per_day = demand_std  # Simplified assumption
        print(f"DEBUG: calculating reorder_point with avg_demand_per_day={avg_demand_per_day}, lead_time={lead_time}, safety_stock={safety_stock}")
        reorder_point = avg_demand_per_day * lead_time + safety_stock
        print(f"DEBUG: reorder_point calculated as {reorder_point}")
        
        print(f"DEBUG: preparing return values")
        print(f"DEBUG: safety_stock={safety_stock}, reorder_point={reorder_point}, z_score={z_score}, service_level={service_level}, lead_time_demand_std={lead_time_demand_std}")
        print(f"DEBUG: types - safety_stock={type(safety_stock)}, reorder_point={type(reorder_point)}")
        
        print(f"DEBUG: about to call np.ceil on safety_stock")
        safety_stock_int = int(np.ceil(safety_stock))
        print(f"DEBUG: np.ceil(safety_stock) returned {safety_stock_int}")
        
        print(f"DEBUG: about to call np.ceil on reorder_point")
        reorder_point_int = int(np.ceil(reorder_point))
        print(f"DEBUG: np.ceil(reorder_point) returned {reorder_point_int}")
        
        return {
            'safety_stock': safety_stock_int,
            'reorder_point': reorder_point_int,
            'z_score': z_score,
            'service_level_achieved': service_level,
            'lead_time_demand_std': lead_time_demand_std
        }
    
    def perform_abc_analysis(self, products_df: pd.DataFrame, 
                           revenue_percentage_a: float = 0.8, 
                           item_percentage_a: float = 0.2) -> List[Dict[str, Any]]:
        """Perform ABC analysis based on annual revenue"""
        
        # Sort products by annual revenue (descending)
        sorted_products = products_df.sort_values('annual_revenue', ascending=False)
        
        # Calculate cumulative revenue percentage
        total_revenue = sorted_products['annual_revenue'].sum()
        sorted_products['cumulative_revenue'] = sorted_products['annual_revenue'].cumsum()
        sorted_products['revenue_percentage'] = (sorted_products['cumulative_revenue'] / total_revenue) * 100
        
        # Calculate cumulative item percentage
        sorted_products['item_percentage'] = (np.arange(len(sorted_products)) + 1) / len(sorted_products) * 100
        
        # Assign ABC categories
        abc_results = []
        
        for _, product in sorted_products.iterrows():
            revenue_pct = (product['annual_revenue'] / total_revenue) * 100
            
            if product['revenue_percentage'] <= revenue_percentage_a * 100 and \
               product['item_percentage'] <= item_percentage_a * 100:
                category = 'A'
                priority = 'High'
                policy = 'Continuous review, tight control'
            elif product['revenue_percentage'] <= 0.95 * 100:  # Next 15% revenue
                category = 'B'
                priority = 'Medium'
                policy = 'Periodic review, moderate control'
            else:
                category = 'C'
                priority = 'Low'
                policy = 'Simple control, bulk ordering'
            
            abc_results.append({
                'product_id': product['product_id'],
                'category': category,
                'annual_revenue': product['annual_revenue'],
                'revenue_percentage': revenue_pct,
                'demand_volume': product['annual_demand'],
                'optimization_priority': priority,
                'recommended_policy': policy
            })
        
        # Store results for later use
        for result in abc_results:
            self.abc_categories[result['product_id']] = result['category']
        
        return abc_results
    
    def optimize_inventory_policy(self, product_data: Dict[str, Any], 
                                optimization_method: str = 'eoq',
                                service_level: Optional[float] = None,
                                demand_forecast: Optional[List[float]] = None) -> Dict[str, Any]:
        """Optimize inventory policy for a single product"""
        
        # Extract product parameters
        # Calculate annual_demand from demand_rate if not provided
        annual_demand = product_data.get('annual_demand')
        if annual_demand is None:
            demand_rate = product_data.get('demand_rate', 0)
            annual_demand = demand_rate * 365  # Convert daily demand to annual
            print(f"DEBUG: Calculated annual_demand = {annual_demand} from demand_rate = {demand_rate}")
        
        ordering_cost = product_data['ordering_cost']
        unit_cost = product_data['unit_cost']
        holding_cost_rate = product_data['holding_cost_rate']
        lead_time = product_data['lead_time_days']
        demand_std = product_data.get('demand_std', annual_demand * 0.1 / 365)
        current_service_level = service_level or product_data.get('service_level', 0.95)
        
        print(f"DEBUG: annual_demand={annual_demand}, ordering_cost={ordering_cost}, unit_cost={unit_cost}, holding_cost_rate={holding_cost_rate}, lead_time={lead_time}, demand_std={demand_std}")
        
        # Calculate EOQ
        eoq_result = self.calculate_eoq(annual_demand, ordering_cost, unit_cost, holding_cost_rate)
        
        if 'error' in eoq_result:
            return {'error': eoq_result['error']}
        
        # Calculate safety stock and reorder point
        print(f"DEBUG: about to call calculate_safety_stock with demand_std={demand_std}, lead_time={lead_time}, current_service_level={current_service_level}")
        try:
            safety_stock_result = self.calculate_safety_stock(
                demand_std, lead_time, current_service_level
            )
            print(f"DEBUG: calculate_safety_stock returned: {safety_stock_result}")
        except Exception as e:
            print(f"DEBUG: Exception in calculate_safety_stock: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Combine results
        print(f"DEBUG: about to extract results from eoq_result and safety_stock_result")
        eoq_quantity = eoq_result['economic_order_quantity']
        print(f"DEBUG: extracted eoq_quantity={eoq_quantity}")
        safety_stock = safety_stock_result['safety_stock']
        print(f"DEBUG: extracted safety_stock={safety_stock}")
        reorder_point = safety_stock_result['reorder_point']
        print(f"DEBUG: extracted reorder_point={reorder_point}")
        
        # Calculate average inventory level
        average_inventory = (eoq_quantity / 2) + safety_stock
        
        # Calculate inventory turnover
        inventory_turnover = annual_demand / average_inventory if average_inventory > 0 else 0
        
        # Calculate days of supply
        days_of_supply = int((average_inventory / (annual_demand / 365)))
        
        # Generate recommendations
        recommendations = []
        
        print(f"DEBUG: eoq_quantity={eoq_quantity}, max_stock_capacity={product_data.get('max_stock_capacity')}")
        max_capacity = product_data.get('max_stock_capacity', float('inf'))
        print(f"DEBUG: max_capacity after get={max_capacity}")
        
        if eoq_quantity > max_capacity:
            recommendations.append("Consider increasing storage capacity or negotiating smaller order quantities")
        
        print(f"DEBUG: safety_stock={safety_stock}, eoq_quantity * 0.5={eoq_quantity * 0.5}")
        if safety_stock > eoq_quantity * 0.5:
            recommendations.append("High safety stock relative to order quantity - review service level targets")
        
        if inventory_turnover < 4:
            recommendations.append("Low inventory turnover - consider reducing order quantities")
        elif inventory_turnover > 12:
            recommendations.append("High inventory turnover - monitor stockout risk")
        
        print(f"DEBUG: supplier_reliability={product_data.get('supplier_reliability')}, current_service_level={current_service_level}")
        # Calculate confidence score based on data quality and assumptions
        confidence_score = min(1.0, product_data.get('supplier_reliability', 0.95) * current_service_level)
        print(f"DEBUG: confidence_score={confidence_score}")
        
        return {
            'product_id': product_data['product_id'],
            'optimization_method': optimization_method,
            'economic_order_quantity': eoq_quantity,
            'reorder_point': reorder_point,
            'safety_stock': safety_stock,
            'average_inventory': average_inventory,
            'total_cost': eoq_result['total_cost'],
            'ordering_cost': eoq_result['ordering_cost'],
            'holding_cost': eoq_result['holding_cost'],
            'stockout_cost': 0.0,  # Simplified assumption
            'service_level_achieved': current_service_level,
            'inventory_turnover': inventory_turnover,
            'days_of_supply': days_of_supply,
            'recommendations': recommendations,
            'confidence_score': confidence_score,
            'calculation_date': datetime.now()
        }
    
    def generate_stock_alerts(self, products_df: pd.DataFrame, 
                            historical_demand: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate stock alerts based on current inventory levels"""
        
        alerts = []
        current_date = datetime.now().date()
        
        for _, product in products_df.iterrows():
            product_id = product['product_id']
            current_stock = product['current_stock']
            
            # Get product demand history
            product_demand = historical_demand[historical_demand['product_id'] == product_id]
            
            if len(product_demand) == 0:
                continue
            
            # Calculate average daily demand
            avg_daily_demand = product_demand['demand'].mean()
            
            # Calculate reorder point (simplified)
            lead_time = product['lead_time_days']
            service_level = product['service_level']
            demand_std = product.get('demand_std', avg_daily_demand * 0.2)
            
            safety_stock_result = self.calculate_safety_stock(demand_std, lead_time, service_level)
            reorder_point = safety_stock_result['reorder_point']
            
            # Check for low stock
            if current_stock <= reorder_point:
                urgency = "high" if current_stock <= safety_stock_result['safety_stock'] else "medium"
                estimated_impact = (reorder_point - current_stock) * avg_daily_demand * product['unit_cost']
                
                alerts.append({
                    'product_id': product_id,
                    'alert_type': 'low_stock',
                    'current_stock': current_stock,
                    'recommended_action': f"Reorder {int(reorder_point * 1.2)} units immediately",
                    'urgency_level': urgency,
                    'estimated_impact': estimated_impact,
                    'action_deadline': current_date + timedelta(days=lead_time)
                })
            
            # Check for overstock
            eoq_result = self.calculate_eoq(
                product['annual_demand'], 
                product['ordering_cost'], 
                product['unit_cost'], 
                product['holding_cost_rate']
            )
            
            max_reasonable_stock = eoq_result['economic_order_quantity'] * 2
            
            if current_stock >= max_reasonable_stock:
                alerts.append({
                    'product_id': product_id,
                    'alert_type': 'overstock',
                    'current_stock': current_stock,
                    'recommended_action': f"Review ordering policy - consider promotions or discounts",
                    'urgency_level': 'low',
                    'estimated_impact': (current_stock - max_reasonable_stock) * product['unit_cost'] * product['holding_cost_rate']
                })
            
            # Check for expiring stock
            if product['expiration_days'] is not None:
                # Assume current stock has been in inventory for half the expiration period
                days_in_stock = product['expiration_days'] // 2
                days_until_expiry = product['expiration_days'] - days_in_stock
                
                if days_until_expiry <= 30:  # Alert if expiring within 30 days
                    alerts.append({
                        'product_id': product_id,
                        'alert_type': 'expiring',
                        'current_stock': current_stock,
                        'recommended_action': f"Consider discounting or promotional sales",
                        'urgency_level': 'high' if days_until_expiry <= 7 else 'medium',
                        'estimated_impact': current_stock * product['unit_cost'],
                        'action_deadline': current_date + timedelta(days=days_until_expiry)
                    })
        
        return alerts
    
    def optimize_multi_echelon(self, warehouses: List[str], products_data: List[Dict[str, Any]],
                             transportation_costs: Dict[str, float], 
                             transfer_lead_times: Dict[str, int],
                             central_warehouse_capacity: int) -> List[Dict[str, Any]]:
        """Optimize multi-echelon inventory system"""
        
        results = []
        
        for warehouse in warehouses:
            for product_data in products_data:
                product_id = product_data['product_id']
                
                # Calculate optimal stock level at each warehouse
                # Simplified approach: distribute based on demand proportion
                total_demand = sum(p['annual_demand'] for p in products_data)
                demand_proportion = product_data['annual_demand'] / total_demand
                
                # Calculate optimal stock level
                base_optimization = self.optimize_inventory_policy(product_data)
                
                if 'error' in base_optimization:
                    continue
                
                # Adjust for multi-echelon
                optimal_stock = int(base_optimization['average_inventory'] * demand_proportion)
                
                # Calculate transfer quantity from central warehouse
                current_stock = product_data['current_stock']
                transfer_quantity = max(0, optimal_stock - current_stock)
                
                # Calculate costs
                transfer_cost = transfer_quantity * transportation_costs.get(warehouse, 50)
                holding_cost = optimal_stock * product_data['unit_cost'] * product_data['holding_cost_rate']
                
                total_cost = transfer_cost + holding_cost
                
                results.append({
                    'warehouse_id': warehouse,
                    'product_id': product_id,
                    'optimal_stock_level': optimal_stock,
                    'transfer_quantity': transfer_quantity,
                    'total_cost': total_cost,
                    'service_level': base_optimization['service_level_achieved'],
                    'stock_transfer_recommendations': [
                        f"Transfer {transfer_quantity} units from central warehouse",
                        f"Expected cost savings: ${transfer_cost * 0.1:.2f}"
                    ]
                })
        
        return results
    
    def forecast_demand(self, historical_demand: List[float], periods: int = 30) -> List[float]:
        """Simple demand forecasting using moving average and trend"""
        
        if len(historical_demand) < 7:
            # Not enough data, return simple average
            avg_demand = np.mean(historical_demand) if historical_demand else 0
            return [avg_demand] * periods
        
        # Calculate moving average (7-day window)
        recent_data = historical_demand[-30:] if len(historical_demand) >= 30 else historical_demand
        
        # Simple trend calculation
        if len(recent_data) >= 14:
            first_half = np.mean(recent_data[:len(recent_data)//2])
            second_half = np.mean(recent_data[len(recent_data)//2:])
            trend = (second_half - first_half) / (len(recent_data) // 2)
        else:
            trend = 0
        
        # Generate forecasts
        last_demand = recent_data[-1]
        forecasts = []
        
        for i in range(periods):
            # Moving average with trend
            forecast = last_demand + trend * (i + 1)
            forecasts.append(max(0, forecast))
        
        return forecasts
    
    def save_model(self, filepath: str):
        """Save the optimization model"""
        model_data = {
            'abc_categories': self.abc_categories,
            'demand_forecasters': self.demand_forecasters,
            'lead_time_models': self.lead_time_models,
            'optimization_history': self.optimization_history,
            'saved_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        print(f"Inventory optimization model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the optimization model"""
        model_data = joblib.load(filepath)
        
        self.abc_categories = model_data['abc_categories']
        self.demand_forecasters = model_data['demand_forecasters']
        self.lead_time_models = model_data['lead_time_models']
        self.optimization_history = model_data['optimization_history']
        
        print(f"Inventory optimization model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the optimization model"""
        return {
            'abc_categories_count': len(self.abc_categories),
            'demand_forecasters_count': len(self.demand_forecasters),
            'optimization_history_length': len(self.optimization_history),
            'available_methods': ['EOQ', 'Safety Stock', 'ABC Analysis', 'Multi-Echelon'],
            'dependencies': {
                'statsmodels': STATSMODELS_AVAILABLE,
                'xgboost': XGBOOST_AVAILABLE
            },
            'last_updated': datetime.now().isoformat()
        }