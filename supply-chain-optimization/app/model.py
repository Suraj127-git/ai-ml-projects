import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Optional imports for advanced optimization
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .schemas import (
    SupplyChainNode, SupplyChainEdge, Product, DemandForecast,
    OptimizationRequest, OptimizationResult, OptimizationObjective,
    NodeType, TransportMode
)


class SupplyChainOptimizer:
    """Supply Chain Optimization with Linear Programming and Machine Learning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.network_data = None
        self.demand_forecaster = None
        
    def solve_network_flow_optimization(self, request: OptimizationRequest) -> OptimizationResult:
        """
        Solve network flow optimization problem
        
        Args:
            request: OptimizationRequest containing nodes, edges, and constraints
            
        Returns:
            OptimizationResult with optimal flows and costs
        """
        start_time = datetime.now()
        
        try:
            if PULP_AVAILABLE:
                return self._solve_with_pulp(request, start_time)
            else:
                return self._solve_with_fallback(request, start_time)
        except Exception as e:
            raise RuntimeError(f"Network flow optimization failed: {str(e)}")
    
    def _solve_with_pulp(self, request: OptimizationRequest, start_time: datetime) -> OptimizationResult:
        """Solve using PuLP linear programming"""
        
        # Create the optimization problem
        prob = pulp.LpProblem("Supply_Chain_Optimization", pulp.LpMinimize)
        
        # Create decision variables for flows
        flow_vars = {}
        for edge in request.edges:
            var_name = f"flow_{edge.from_node}_{edge.to_node}"
            flow_vars[(edge.from_node, edge.to_node)] = pulp.LpVariable(
                var_name, lowBound=0, cat='Continuous'
            )
        
        # Objective function
        total_cost = 0
        for edge in request.edges:
            flow_var = flow_vars[(edge.from_node, edge.to_node)]
            if request.objective == OptimizationObjective.MINIMIZE_COST:
                total_cost += flow_var * edge.transportation_cost
            elif request.objective == OptimizationObjective.MINIMIZE_TIME:
                total_cost += flow_var * edge.lead_time
            elif request.objective == OptimizationObjective.MINIMIZE_RISK:
                total_cost += flow_var * (1 - edge.reliability)
        
        prob += total_cost, "Total_Objective"
        
        # Constraints
        # Flow conservation constraints
        node_flows = {}
        for node in request.nodes:
            inflow = 0
            outflow = 0
            
            # Calculate inflow
            for edge in request.edges:
                if edge.to_node == node.node_id:
                    inflow += flow_vars[(edge.from_node, edge.to_node)]
            
            # Calculate outflow
            for edge in request.edges:
                if edge.from_node == node.node_id:
                    outflow += flow_vars[(edge.from_node, edge.to_node)]
            
            # Net flow based on node type
            if node.node_type == NodeType.SUPPLIER:
                prob += outflow - inflow <= node.capacity, f"supplier_capacity_{node.node_id}"
            elif node.node_type == NodeType.DISTRIBUTION_CENTER:
                prob += inflow - outflow == 0, f"flow_conservation_{node.node_id}"
            elif node.node_type == NodeType.RETAILER:
                # Find demand for this retailer
                demand = 0
                for demand_item in request.demands:
                    if demand_item.node_id == node.node_id:
                        demand = demand_item.quantity
                        break
                prob += inflow - outflow >= demand, f"demand_satisfaction_{node.node_id}"
        
        # Edge capacity constraints
        for edge in request.edges:
            flow_var = flow_vars[(edge.from_node, edge.to_node)]
            prob += flow_var <= edge.capacity, f"edge_capacity_{edge.from_node}_{edge.to_node}"
        
        # Solve the problem
        prob.solve()
        
        # Extract results
        optimal_flows = []
        total_cost_value = pulp.value(prob.objective)
        
        for edge in request.edges:
            flow_value = pulp.value(flow_vars[(edge.from_node, edge.to_node)])
            if flow_value and flow_value > 0:
                optimal_flows.append({
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "flow": flow_value,
                    "cost": flow_value * edge.transportation_cost
                })
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            objective_value=total_cost_value or 0,
            optimal_flows=optimal_flows,
            execution_time=execution_time,
            optimization_status=str(pulp.LpStatus[prob.status]),
            node_utilization=self._calculate_node_utilization(request, optimal_flows),
            total_cost=total_cost_value or 0,
            carbon_footprint=self._calculate_carbon_footprint(request, optimal_flows)
        )
    
    def _solve_with_fallback(self, request: OptimizationRequest, start_time: datetime) -> OptimizationResult:
        """Fallback heuristic optimization when PuLP is not available"""
        
        # Simple greedy algorithm
        optimal_flows = []
        total_cost = 0
        
        # Sort edges by cost-effectiveness
        sorted_edges = sorted(request.edges, key=lambda e: e.transportation_cost)
        
        # Track remaining demands and capacities
        remaining_demands = {d.node_id: d.quantity for d in request.demands}
        node_capacities = {n.node_id: n.capacity for n in request.nodes}
        
        for edge in sorted_edges:
            # Find maximum possible flow on this edge
            max_flow = min(edge.capacity, node_capacities.get(edge.from_node, 0))
            
            # Check if this edge can help satisfy demand
            if edge.to_node in remaining_demands and remaining_demands[edge.to_node] > 0:
                flow = min(max_flow, remaining_demands[edge.to_node])
                
                if flow > 0:
                    optimal_flows.append({
                        "from_node": edge.from_node,
                        "to_node": edge.to_node,
                        "flow": flow,
                        "cost": flow * edge.transportation_cost
                    })
                    
                    total_cost += flow * edge.transportation_cost
                    remaining_demands[edge.to_node] -= flow
                    node_capacities[edge.from_node] -= flow
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            objective_value=total_cost,
            optimal_flows=optimal_flows,
            execution_time=execution_time,
            optimization_status="heuristic_solution",
            node_utilization=self._calculate_node_utilization(request, optimal_flows),
            total_cost=total_cost,
            carbon_footprint=self._calculate_carbon_footprint(request, optimal_flows)
        )
    
    def optimize_facility_location(self, nodes: List[SupplyChainNode], 
                                 demands: List[DemandForecast], 
                                 budget_constraint: float) -> Dict:
        """
        Optimize facility locations using mixed-integer programming
        
        Args:
            nodes: List of potential facility locations
            demands: List of demand points
            budget_constraint: Maximum budget for facility setup
            
        Returns:
            Dictionary with optimal facility locations and assignments
        """
        
        if not PULP_AVAILABLE:
            return self._facility_location_heuristic(nodes, demands, budget_constraint)
        
        # Create MILP problem
        prob = pulp.LpProblem("Facility_Location", pulp.LpMinimize)
        
        # Decision variables
        facility_vars = {}
        assignment_vars = {}
        
        for node in nodes:
            if node.node_type == NodeType.DISTRIBUTION_CENTER:
                facility_vars[node.node_id] = pulp.LpVariable(
                    f"facility_{node.node_id}", cat='Binary'
                )
        
        # Create assignment variables
        for demand in demands:
            for facility_id in facility_vars:
                assignment_vars[(demand.node_id, facility_id)] = pulp.LpVariable(
                    f"assign_{demand.node_id}_{facility_id}", cat='Binary'
                )
        
        # Objective: Minimize total cost (setup + transportation)
        total_cost = 0
        
        # Facility setup costs
        for node in nodes:
            if node.node_id in facility_vars:
                total_cost += facility_vars[node.node_id] * node.setup_cost
        
        # Transportation costs
        for demand in demands:
            for facility_id in facility_vars:
                assignment_var = assignment_vars[(demand.node_id, facility_id)]
                # Calculate distance-based cost (simplified)
                transport_cost = self._calculate_transport_cost(demand.node_id, facility_id)
                total_cost += assignment_var * transport_cost * demand.quantity
        
        prob += total_cost, "Total_Cost"
        
        # Constraints
        # Budget constraint
        setup_cost_total = 0
        for node in nodes:
            if node.node_id in facility_vars:
                setup_cost_total += facility_vars[node.node_id] * node.setup_cost
        prob += setup_cost_total <= budget_constraint, "Budget_Constraint"
        
        # Demand assignment constraints
        for demand in demands:
            assignment_total = 0
            for facility_id in facility_vars:
                assignment_total += assignment_vars[(demand.node_id, facility_id)]
            prob += assignment_total == 1, f"demand_assignment_{demand.node_id}"
        
        # Facility capacity constraints
        for facility_id in facility_vars:
            demand_total = 0
            for demand in demands:
                assignment_var = assignment_vars[(demand.node_id, facility_id)]
                demand_total += assignment_var * demand.quantity
            
            # Find facility capacity
            facility_capacity = 0
            for node in nodes:
                if node.node_id == facility_id:
                    facility_capacity = node.capacity
                    break
            
            prob += demand_total <= facility_capacity * facility_vars[facility_id], f"capacity_{facility_id}"
        
        # Solve
        prob.solve()
        
        # Extract results
        selected_facilities = []
        assignments = []
        
        for facility_id in facility_vars:
            if pulp.value(facility_vars[facility_id]) > 0.5:
                selected_facilities.append(facility_id)
        
        for demand in demands:
            for facility_id in facility_vars:
                assignment_var = assignment_vars[(demand.node_id, facility_id)]
                if pulp.value(assignment_var) > 0.5:
                    assignments.append({
                        "demand_node": demand.node_id,
                        "facility": facility_id,
                        "quantity": demand.quantity
                    })
        
        return {
            "selected_facilities": selected_facilities,
            "assignments": assignments,
            "total_cost": pulp.value(prob.objective),
            "status": str(pulp.LpStatus[prob.status])
        }
    
    def train_demand_forecasting_model(self, historical_data: pd.DataFrame) -> Dict:
        """
        Train ML models for demand forecasting
        
        Args:
            historical_data: DataFrame with historical demand data
            
        Returns:
            Dictionary with model performance metrics
        """
        
        # Prepare features
        features = ['price', 'seasonality', 'promotion', 'competitor_price', 'economic_index']
        target = 'demand'
        
        if not all(col in historical_data.columns for col in features + [target]):
            # Generate synthetic features if not available
            historical_data = self._generate_synthetic_features(historical_data)
        
        X = historical_data[features]
        y = historical_data[target]
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        for model_name, model in models.items():
            if model_name == 'random_forest':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'model': model
            }
            
            # Store model and scaler
            self.models[model_name] = model
            self.scalers[model_name] = scaler if model_name == 'linear_regression' else None
        
        return results
    
    def predict_demand(self, product_features: Dict, model_type: str = 'random_forest') -> float:
        """
        Predict demand for a product using trained models
        
        Args:
            product_features: Dictionary with product features
            model_type: Type of model to use for prediction
            
        Returns:
            Predicted demand value
        """
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        model = self.models[model_type]
        scaler = self.scalers.get(model_type)
        
        # Prepare features
        feature_names = ['price', 'seasonality', 'promotion', 'competitor_price', 'economic_index']
        features = [product_features.get(name, 0) for name in feature_names]
        
        # Scale if necessary
        if scaler:
            features = scaler.transform([features])
        
        # Predict
        demand = model.predict(features)[0]
        return max(0, demand)  # Ensure non-negative demand
    
    def optimize_inventory_levels(self, demand_forecasts: List[DemandForecast], 
                                holding_cost_rate: float = 0.2,
                                shortage_cost_rate: float = 0.5) -> Dict:
        """
        Optimize inventory levels using newsvendor model
        
        Args:
            demand_forecasts: List of demand forecasts
            holding_cost_rate: Cost of holding inventory (% of value)
            shortage_cost_rate: Cost of shortage (% of value)
            
        Returns:
            Dictionary with optimal inventory levels
        """
        
        results = []
        
        for forecast in demand_forecasts:
            # Calculate critical ratio
            critical_ratio = shortage_cost_rate / (shortage_cost_rate + holding_cost_rate)
            
            # Find optimal inventory level (newsvendor solution)
            # For normal distribution: Q* = μ + z*σ where z is the z-score for critical ratio
            z_score = np.norm.ppf(critical_ratio)
            optimal_inventory = forecast.mean_demand + z_score * forecast.std_demand
            
            # Calculate expected costs
            expected_holding_cost = max(0, optimal_inventory - forecast.mean_demand) * holding_cost_rate
            expected_shortage_cost = max(0, forecast.mean_demand - optimal_inventory) * shortage_cost_rate * forecast.probability_exceeding
            
            results.append({
                "node_id": forecast.node_id,
                "product_id": forecast.product_id,
                "optimal_inventory": optimal_inventory,
                "expected_holding_cost": expected_holding_cost,
                "expected_shortage_cost": expected_shortage_cost,
                "service_level": critical_ratio
            })
        
        return {
            "inventory_optimization": results,
            "total_holding_cost": sum(r["expected_holding_cost"] for r in results),
            "total_shortage_cost": sum(r["expected_shortage_cost"] for r in results)
        }
    
    def _calculate_node_utilization(self, request: OptimizationRequest, optimal_flows: List[Dict]) -> Dict:
        """Calculate node utilization based on flows"""
        
        node_utilization = {}
        
        for node in request.nodes:
            total_inflow = 0
            total_outflow = 0
            
            for flow in optimal_flows:
                if flow["to_node"] == node.node_id:
                    total_inflow += flow["flow"]
                if flow["from_node"] == node.node_id:
                    total_outflow += flow["flow"]
            
            utilization = (total_inflow + total_outflow) / (2 * node.capacity) if node.capacity > 0 else 0
            node_utilization[node.node_id] = min(1.0, utilization)
        
        return node_utilization
    
    def _calculate_carbon_footprint(self, request: OptimizationRequest, optimal_flows: List[Dict]) -> float:
        """Calculate total carbon footprint of the supply chain"""
        
        total_carbon = 0
        
        for flow in optimal_flows:
            # Find corresponding edge
            for edge in request.edges:
                if edge.from_node == flow["from_node"] and edge.to_node == flow["to_node"]:
                    # Carbon footprint = flow * distance * carbon_factor
                    carbon_footprint = flow["flow"] * edge.distance * edge.carbon_factor
                    total_carbon += carbon_footprint
                    break
        
        return total_carbon
    
    def _calculate_transport_cost(self, from_node: str, to_node: str) -> float:
        """Calculate transportation cost between two nodes (simplified)"""
        # This is a simplified calculation - in practice, you'd use actual distance data
        base_cost = 1.0
        distance_factor = 0.1  # Cost per unit distance
        return base_cost + distance_factor
    
    def _facility_location_heuristic(self, nodes: List[SupplyChainNode], 
                                   demands: List[DemandForecast], 
                                   budget_constraint: float) -> Dict:
        """Heuristic for facility location when PuLP is not available"""
        
        # Simple greedy approach: select facilities with best cost-benefit ratio
        candidate_facilities = [n for n in nodes if n.node_type == NodeType.DISTRIBUTION_CENTER]
        selected_facilities = []
        remaining_budget = budget_constraint
        
        # Sort by setup cost efficiency (capacity per cost)
        candidate_facilities.sort(key=lambda x: x.capacity / x.setup_cost if x.setup_cost > 0 else 0, reverse=True)
        
        for facility in candidate_facilities:
            if facility.setup_cost <= remaining_budget:
                selected_facilities.append(facility.node_id)
                remaining_budget -= facility.setup_cost
        
        # Assign demands to selected facilities
        assignments = []
        for demand in demands:
            # Assign to closest facility (simplified)
            if selected_facilities:
                closest_facility = selected_facilities[0]  # Simplified
                assignments.append({
                    "demand_node": demand.node_id,
                    "facility": closest_facility,
                    "quantity": demand.quantity
                })
        
        return {
            "selected_facilities": selected_facilities,
            "assignments": assignments,
            "total_cost": budget_constraint - remaining_budget,
            "status": "heuristic_solution"
        }
    
    def _generate_synthetic_features(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic features for demand forecasting"""
        
        np.random.seed(42)
        
        # Generate synthetic features
        historical_data['price'] = np.random.uniform(10, 100, len(historical_data))
        historical_data['seasonality'] = np.sin(2 * np.pi * np.arange(len(historical_data)) / 365.25)
        historical_data['promotion'] = np.random.choice([0, 1], len(historical_data), p=[0.8, 0.2])
        historical_data['competitor_price'] = historical_data['price'] * np.random.uniform(0.8, 1.2, len(historical_data))
        historical_data['economic_index'] = np.random.uniform(0.9, 1.1, len(historical_data))
        
        # Generate synthetic demand based on features
        historical_data['demand'] = (
            100 - 0.5 * historical_data['price'] +
            20 * historical_data['seasonality'] +
            30 * historical_data['promotion'] -
            0.3 * historical_data['competitor_price'] +
            10 * historical_data['economic_index'] +
            np.random.normal(0, 10, len(historical_data))
        )
        
        # Ensure non-negative demand
        historical_data['demand'] = np.maximum(historical_data['demand'], 0)
        
        return historical_data
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'network_data': self.network_data
        }
        joblib.dump(model_data, filepath)
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.network_data = model_data['network_data']