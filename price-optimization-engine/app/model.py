import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime, timedelta
import random
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

from .schemas import (
    ProductCategory, PricingStrategy, PriceRequest, PriceResponse,
    ReinforcementLearningState, ReinforcementLearningAction, ReinforcementLearningReward,
    MarketData, PriceElasticity, CompetitiveAnalysis
)

class QLearningAgent:
    """Q-Learning agent for price optimization"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
    
    def get_state_key(self, state: ReinforcementLearningState) -> str:
        """Convert state to string key for Q-table"""
        # Discretize continuous values
        price_bucket = int(state.current_price / 10) * 10
        inventory_bucket = min(state.inventory_level // 10, 10)
        demand_bucket = max(-5, min(5, int(state.demand_trend)))
        
        return f"{state.product_id}_{price_bucket}_{inventory_bucket}_{demand_bucket}"
    
    def get_action(self, state: ReinforcementLearningState) -> ReinforcementLearningAction:
        """Select action using epsilon-greedy policy"""
        state_key = self.get_state_key(state)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random action
            action_type = random.choice(["increase", "decrease", "maintain"])
            price_change = random.uniform(-0.2, 0.2)
        else:
            # Greedy action
            q_values = self.q_table[state_key]
            if not q_values:
                action_type = random.choice(["increase", "decrease", "maintain"])
                price_change = random.uniform(-0.2, 0.2)
            else:
                best_action = max(q_values.items(), key=lambda x: x[1])[0]
                action_parts = best_action.split("_")
                action_type = action_parts[0]
                price_change = float(action_parts[1]) if len(action_parts) > 1 else 0.0
        
        return ReinforcementLearningAction(
            price_change=price_change,
            price_change_percentage=price_change,
            action_type=action_type
        )
    
    def update_q_value(self, state: ReinforcementLearningState, 
                      action: ReinforcementLearningAction, 
                      reward: ReinforcementLearningReward,
                      next_state: ReinforcementLearningState):
        """Update Q-value based on experience"""
        state_key = self.get_state_key(state)
        action_key = f"{action.action_type}_{action.price_change:.2f}"
        next_state_key = self.get_state_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action_key]
        
        # Maximum Q-value for next state
        next_q_values = self.q_table[next_state_key]
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward.total_reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action_key] = new_q
        
        # Store history
        self.state_history.append(state_key)
        self.action_history.append(action_key)
        self.reward_history.append(reward.total_reward)

class PriceOptimizationModel:
    """Price Optimization Model using Reinforcement Learning and ML"""
    
    def __init__(self):
        self.rl_agent = QLearningAgent()
        self.demand_models = {}
        self.price_elasticity_models = {}
        self.competitive_models = {}
        self.scalers = {}
        self.market_data = []
        self.is_trained = False
        
        # Initialize ML models for different aspects
        self.models = {
            'demand': RandomForestRegressor(n_estimators=100, random_state=42),
            'elasticity': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'competitive': RandomForestRegressor(n_estimators=100, random_state=42)
        }
    
    def generate_synthetic_market_data(self, n_samples: int = 1000) -> List[MarketData]:
        """Generate synthetic market data for training and testing"""
        synthetic_data = []
        categories = list(ProductCategory)
        
        for i in range(n_samples):
            category = random.choice(categories)
            
            # Generate realistic product data based on category
            if category == ProductCategory.ELECTRONICS:
                base_price = random.uniform(100, 1000)
                cost_price = base_price * random.uniform(0.6, 0.8)
            elif category == ProductCategory.CLOTHING:
                base_price = random.uniform(20, 200)
                cost_price = base_price * random.uniform(0.4, 0.7)
            elif category == ProductCategory.FOOD:
                base_price = random.uniform(5, 50)
                cost_price = base_price * random.uniform(0.3, 0.6)
            else:
                base_price = random.uniform(10, 500)
                cost_price = base_price * random.uniform(0.4, 0.8)
            
            # Generate demand with price elasticity
            price_elasticity = random.uniform(-2.5, -0.5)
            competitor_price = base_price * random.uniform(0.8, 1.2)
            
            # Demand calculation with noise
            base_demand = random.randint(10, 200)
            price_effect = price_elasticity * (base_price - competitor_price) / competitor_price
            demand = max(0, int(base_demand * (1 + price_effect) + random.randint(-10, 10)))
            
            # Market conditions
            market_share = random.uniform(0.05, 0.3)
            customer_satisfaction = random.uniform(0.7, 0.95)
            
            # Inventory level
            inventory = max(0, demand + random.randint(-20, 50))
            
            market_data = MarketData(
                product_id=f"product_{i:04d}",
                date=(datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
                price=base_price,
                demand=demand,
                competitor_price=competitor_price,
                inventory=inventory,
                market_share=market_share,
                customer_satisfaction=customer_satisfaction
            )
            
            synthetic_data.append(market_data)
        
        return synthetic_data
    
    def calculate_price_elasticity(self, price_history: List[float], 
                                 demand_history: List[float]) -> PriceElasticity:
        """Calculate price elasticity coefficient"""
        if len(price_history) < 3 or len(demand_history) < 3:
            return PriceElasticity(
                product_id="unknown",
                elasticity_coefficient=-1.0,
                elasticity_category="unknown",
                confidence_interval=[-2.0, 0.0],
                statistical_significance=0.0
            )
        
        # Calculate percentage changes
        price_changes = []
        demand_changes = []
        
        for i in range(1, len(price_history)):
            if price_history[i-1] > 0:
                price_change = (price_history[i] - price_history[i-1]) / price_history[i-1]
                demand_change = (demand_history[i] - demand_history[i-1]) / max(1, demand_history[i-1])
                
                price_changes.append(price_change)
                demand_changes.append(demand_change)
        
        if len(price_changes) < 2:
            return PriceElasticity(
                product_id="unknown",
                elasticity_coefficient=-1.0,
                elasticity_category="inelastic",
                confidence_interval=[-2.0, 0.0],
                statistical_significance=0.5
            )
        
        # Calculate elasticity using linear regression
        X = np.array(price_changes).reshape(-1, 1)
        y = np.array(demand_changes)
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            elasticity = model.coef_[0]
            
            # Calculate confidence interval
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            std_error = np.sqrt(mse / len(X))
            
            # Statistical significance (simplified)
            significance = min(1.0, max(0.0, 1.0 - (std_error / abs(elasticity + 1e-6))))
            
            # Categorize elasticity
            if abs(elasticity) > 1.5:
                category = "elastic"
            elif abs(elasticity) < 0.5:
                category = "inelastic"
            else:
                category = "unitary"
            
            return PriceElasticity(
                product_id="analyzed_product",
                elasticity_coefficient=elasticity,
                elasticity_category=category,
                confidence_interval=[elasticity - 0.5, elasticity + 0.5],
                statistical_significance=significance
            )
            
        except Exception as e:
            print(f"Elasticity calculation error: {e}")
            return PriceElasticity(
                product_id="unknown",
                elasticity_coefficient=-1.0,
                elasticity_category="unknown",
                confidence_interval=[-2.0, 0.0],
                statistical_significance=0.0
            )
    
    def analyze_competitive_position(self, product_price: float, 
                                   competitor_prices: List[float]) -> CompetitiveAnalysis:
        """Analyze competitive pricing position"""
        if not competitor_prices:
            return CompetitiveAnalysis(
                product_id="unknown",
                competitor_prices=[],
                price_position="unknown",
                competitive_advantage=0.0,
                market_pricing_trend="stable"
            )
        
        avg_competitor_price = np.mean(competitor_prices)
        min_competitor_price = np.min(competitor_prices)
        max_competitor_price = np.max(competitor_prices)
        
        # Determine price position
        if product_price < min_competitor_price * 0.95:
            price_position = "discount"
        elif product_price > max_competitor_price * 1.05:
            price_position = "premium"
        else:
            price_position = "parity"
        
        # Calculate competitive advantage
        price_advantage = (avg_competitor_price - product_price) / avg_competitor_price
        competitive_advantage = max(-1.0, min(1.0, price_advantage))
        
        # Determine market pricing trend
        if len(competitor_prices) > 1:
            price_std = np.std(competitor_prices)
            if price_std < avg_competitor_price * 0.1:
                trend = "stable"
            elif product_price < avg_competitor_price:
                trend = "competitive"
            else:
                trend = "premium"
        else:
            trend = "unknown"
        
        return CompetitiveAnalysis(
            product_id="analyzed_product",
            competitor_prices=competitor_prices,
            price_position=price_position,
            competitive_advantage=competitive_advantage,
            market_pricing_trend=trend
        )
    
    def optimize_price_with_rl(self, request: PriceRequest) -> PriceResponse:
        """Optimize price using reinforcement learning"""
        
        # Create RL state
        rl_state = ReinforcementLearningState(
            product_id=request.product_id,
            current_price=request.current_price,
            cost_price=request.cost_price,
            competitor_avg_price=np.mean(request.competitor_prices) if request.competitor_prices else request.current_price,
            inventory_level=request.inventory_level,
            demand_trend=np.mean(request.demand_history[-3:]) if len(request.demand_history) >= 3 else 0,
            seasonality_index=request.seasonality_factor,
            price_elasticity=request.price_elasticity,
            days_since_price_change=random.randint(1, 30),  # Simulated
            market_volatility=random.uniform(0.1, 0.3)  # Simulated
        )
        
        # Get action from RL agent
        action = self.rl_agent.get_action(rl_state)
        
        # Calculate new price
        new_price = request.current_price * (1 + action.price_change)
        
        # Apply business constraints
        min_price = request.cost_price * (1 + request.target_margin)
        max_price_change = request.current_price * 0.2  # 20% max change
        
        new_price = max(min_price, min(new_price, request.current_price + max_price_change))
        
        # Calculate expected outcomes
        expected_demand = self._estimate_demand(request, new_price)
        expected_revenue = new_price * expected_demand
        profit_margin = (new_price - request.cost_price) / new_price
        
        # Calculate confidence based on RL agent experience
        confidence_score = min(1.0, len(self.rl_agent.state_history) / 1000)
        
        # Generate reasoning
        reasoning = f"RL agent suggested {action.action_type} of {action.price_change_percentage:.1%} " \
                   f"based on current market conditions. Expected demand: {expected_demand:.0f} units."
        
        return PriceResponse(
            product_id=request.product_id,
            optimized_price=new_price,
            current_price=request.current_price,
            expected_revenue=expected_revenue,
            expected_demand=expected_demand,
            profit_margin=profit_margin,
            confidence_score=confidence_score,
            reasoning=reasoning,
            strategy_used=PricingStrategy.REINFORCEMENT_LEARNING,
            timestamp=datetime.now().isoformat()
        )
    
    def _estimate_demand(self, request: PriceRequest, new_price: float) -> float:
        """Estimate demand at new price"""
        # Simple demand estimation using price elasticity
        price_change = (new_price - request.current_price) / request.current_price
        
        # Base demand (average of recent history or default)
        base_demand = np.mean(request.demand_history[-5:]) if request.demand_history else 50
        
        # Apply price elasticity
        demand_change = request.price_elasticity * price_change
        estimated_demand = base_demand * (1 + demand_change)
        
        # Apply inventory effect
        inventory_effect = max(0, min(0.5, request.inventory_level / 100))
        estimated_demand *= (1 + inventory_effect * 0.1)
        
        # Apply seasonality
        estimated_demand *= request.seasonality_factor
        
        return max(0, estimated_demand)
    
    def optimize_price_with_ml(self, request: PriceRequest) -> PriceResponse:
        """Optimize price using machine learning models"""
        
        if not self.is_trained:
            # Fallback to rule-based optimization
            return self._rule_based_optimization(request)
        
        # Prepare features for ML models
        features = self._prepare_features(request)
        
        # Get predictions from different models
        demand_prediction = self.models['demand'].predict([features])[0]
        elasticity_prediction = self.models['elasticity'].predict([features])[0]
        
        # Optimize price using ML predictions
        optimal_price = self._calculate_optimal_price(request, demand_prediction, elasticity_prediction)
        
        # Calculate outcomes
        expected_demand = max(0, demand_prediction)
        expected_revenue = optimal_price * expected_demand
        profit_margin = (optimal_price - request.cost_price) / optimal_price
        
        return PriceResponse(
            product_id=request.product_id,
            optimized_price=optimal_price,
            current_price=request.current_price,
            expected_revenue=expected_revenue,
            expected_demand=expected_demand,
            profit_margin=profit_margin,
            confidence_score=0.8,  # High confidence for ML model
            reasoning="Optimized using machine learning models for demand prediction and price elasticity.",
            strategy_used=PricingStrategy.DYNAMIC_PRICING,
            timestamp=datetime.now().isoformat()
        )
    
    def _prepare_features(self, request: PriceRequest) -> List[float]:
        """Prepare features for ML models"""
        features = [
            request.current_price,
            request.cost_price,
            request.inventory_level,
            request.seasonality_factor,
            request.price_elasticity,
            request.target_margin,
            np.mean(request.competitor_prices) if request.competitor_prices else request.current_price,
            len(request.competitor_prices),
            np.mean(request.demand_history) if request.demand_history else 0,
            len(request.demand_history)
        ]
        
        # Add category encoding (simple one-hot)
        for category in ProductCategory:
            features.append(1.0 if request.category == category else 0.0)
        
        return features
    
    def _calculate_optimal_price(self, request: PriceRequest, 
                               predicted_demand: float, 
                               predicted_elasticity: float) -> float:
        """Calculate optimal price using ML predictions"""
        
        # Start with current price
        current_price = request.current_price
        
        # Apply elasticity-based adjustment
        elasticity_factor = abs(predicted_elasticity) / 2.0  # Normalize
        price_adjustment = elasticity_factor * 0.1  # Max 10% adjustment
        
        # Adjust based on demand prediction
        if predicted_demand > np.mean(request.demand_history) * 1.2:
            # High demand - increase price
            optimal_price = current_price * (1 + price_adjustment)
        elif predicted_demand < np.mean(request.demand_history) * 0.8:
            # Low demand - decrease price
            optimal_price = current_price * (1 - price_adjustment)
        else:
            # Stable demand - maintain price
            optimal_price = current_price
        
        # Apply constraints
        min_price = request.cost_price * (1 + request.target_margin)
        max_price = current_price * 1.2  # Max 20% increase
        
        optimal_price = max(min_price, min(optimal_price, max_price))
        
        return optimal_price
    
    def _rule_based_optimization(self, request: PriceRequest) -> PriceResponse:
        """Fallback rule-based price optimization"""
        
        current_price = request.current_price
        cost_price = request.cost_price
        
        # Calculate competitor-based price
        if request.competitor_prices:
            avg_competitor = np.mean(request.competitor_prices)
            competitor_based_price = avg_competitor * 0.95  # 5% below competitors
        else:
            competitor_based_price = current_price
        
        # Calculate margin-based price
        margin_based_price = cost_price / (1 - request.target_margin)
        
        # Calculate inventory-based price
        if request.inventory_level > 200:
            inventory_based_price = current_price * 0.9  # Reduce price for high inventory
        elif request.inventory_level < 20:
            inventory_based_price = current_price * 1.1  # Increase price for low inventory
        else:
            inventory_based_price = current_price
        
        # Combine all factors
        weights = [0.3, 0.3, 0.2, 0.2]  # Current, competitor, margin, inventory
        prices = [current_price, competitor_based_price, margin_based_price, inventory_based_price]
        
        optimal_price = sum(w * p for w, p in zip(weights, prices))
        
        # Apply constraints
        min_price = cost_price * 1.1  # Minimum 10% margin
        max_price = current_price * 1.15  # Max 15% increase
        
        optimal_price = max(min_price, min(optimal_price, max_price))
        
        # Calculate outcomes
        expected_demand = self._estimate_demand(request, optimal_price)
        expected_revenue = optimal_price * expected_demand
        profit_margin = (optimal_price - cost_price) / optimal_price
        
        return PriceResponse(
            product_id=request.product_id,
            optimized_price=optimal_price,
            current_price=current_price,
            expected_revenue=expected_revenue,
            expected_demand=expected_demand,
            profit_margin=profit_margin,
            confidence_score=0.6,  # Medium confidence for rule-based
            reasoning="Optimized using rule-based approach combining competitor, margin, and inventory factors.",
            strategy_used=PricingStrategy.COST_PLUS,
            timestamp=datetime.now().isoformat()
        )
    
    def train_models(self, market_data: List[MarketData]):
        """Train ML models on historical market data"""
        
        if len(market_data) < 50:
            print("Insufficient data for training. Need at least 50 samples.")
            return
        
        # Prepare training data
        df = pd.DataFrame([data.dict() for data in market_data])
        
        # Feature engineering
        df['price_competitor_ratio'] = df['price'] / (df['competitor_price'] + 1)
        df['log_price'] = np.log(df['price'] + 1)
        df['log_demand'] = np.log(df['demand'] + 1)
        
        # Prepare features
        feature_cols = ['price', 'competitor_price', 'inventory', 'price_competitor_ratio', 'log_price']
        X = df[feature_cols]
        
        # Prepare targets
        y_demand = df['demand']
        y_elasticity = self._calculate_elasticity_for_training(df)
        
        # Split data
        X_train, X_test, y_demand_train, y_demand_test = train_test_split(
            X, y_demand, test_size=0.2, random_state=42
        )
        _, _, y_elasticity_train, y_elasticity_test = train_test_split(
            X, y_elasticity, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['demand'] = scaler
        self.scalers['elasticity'] = scaler
        
        # Train models
        self.models['demand'].fit(X_train_scaled, y_demand_train)
        self.models['elasticity'].fit(X_train_scaled, y_elasticity_train)
        
        # Evaluate models
        demand_pred = self.models['demand'].predict(X_test_scaled)
        elasticity_pred = self.models['elasticity'].predict(X_test_scaled)
        
        demand_mse = mean_squared_error(y_demand_test, demand_pred)
        elasticity_mse = mean_squared_error(y_elasticity_test, elasticity_pred)
        
        print(f"Demand model MSE: {demand_mse:.2f}")
        print(f"Elasticity model MSE: {elasticity_mse:.4f}")
        
        self.is_trained = True
    
    def _calculate_elasticity_for_training(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price elasticity for training data"""
        elasticity_values = []
        
        for i in range(len(df)):
            if i == 0:
                elasticity_values.append(-1.0)  # Default
            else:
                price_change = (df.iloc[i]['price'] - df.iloc[i-1]['price']) / df.iloc[i-1]['price']
                demand_change = (df.iloc[i]['demand'] - df.iloc[i-1]['demand']) / max(1, df.iloc[i-1]['demand'])
                
                if abs(price_change) > 0.01:  # Significant price change
                    elasticity = demand_change / price_change
                    elasticity = max(-5.0, min(-0.1, elasticity))  # Bound elasticity
                else:
                    elasticity = -1.0  # Default
                
                elasticity_values.append(elasticity)
        
        return pd.Series(elasticity_values)
    
    def optimize_batch_prices(self, requests: List[PriceRequest], 
                              strategy: PricingStrategy = PricingStrategy.REINFORCEMENT_LEARNING,
                              max_price_change: float = 0.2) -> List[PriceResponse]:
        """Optimize prices for multiple products"""
        
        optimized_prices = []
        
        for request in requests:
            try:
                if strategy == PricingStrategy.REINFORCEMENT_LEARNING:
                    response = self.optimize_price_with_rl(request)
                elif strategy == PricingStrategy.DYNAMIC_PRICING and self.is_trained:
                    response = self.optimize_price_with_ml(request)
                else:
                    response = self._rule_based_optimization(request)
                
                # Apply batch constraints
                price_change = abs(response.optimized_price - request.current_price) / request.current_price
                if price_change > max_price_change:
                    # Cap the price change
                    if response.optimized_price > request.current_price:
                        response.optimized_price = request.current_price * (1 + max_price_change)
                    else:
                        response.optimized_price = request.current_price * (1 - max_price_change)
                    
                    # Recalculate outcomes
                    response.expected_demand = self._estimate_demand(request, response.optimized_price)
                    response.expected_revenue = response.optimized_price * response.expected_demand
                    response.profit_margin = (response.optimized_price - request.cost_price) / response.optimized_price
                    response.reasoning += f" Price change capped at {max_price_change:.1%}."
                
                optimized_prices.append(response)
                
            except Exception as e:
                print(f"Error optimizing price for {request.product_id}: {e}")
                # Fallback to current price
                fallback_response = PriceResponse(
                    product_id=request.product_id,
                    optimized_price=request.current_price,
                    current_price=request.current_price,
                    expected_revenue=request.current_price * 50,  # Default
                    expected_demand=50,  # Default
                    profit_margin=(request.current_price - request.cost_price) / request.current_price,
                    confidence_score=0.0,
                    reasoning=f"Error in optimization, using current price: {str(e)}",
                    strategy_used=PricingStrategy.COST_PLUS,
                    timestamp=datetime.now().isoformat()
                )
                optimized_prices.append(fallback_response)
        
        return optimized_prices