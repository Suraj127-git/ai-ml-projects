"""
Interactive training and visualization script for Inventory Optimization models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import InventoryOptimizationModel
from app.schemas import ProductData, OptimizationMethod

def setup_plotting_style():
    """Set up consistent plotting style"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

def generate_synthetic_inventory_data(n_products=20, n_days=365):
    """Generate synthetic inventory and demand data"""
    print("Generating synthetic inventory data...")
    
    model = InventoryOptimizationModel()
    products_df = model.generate_synthetic_inventory_data(n_products, n_days)
    historical_df = model.generate_historical_demand(products_df, n_days)
    
    print(f"Generated {len(products_df)} products and {len(historical_df)} historical demand records")
    return products_df, historical_df, model

def demonstrate_eoq_calculation():
    """Demonstrate EOQ calculation with different scenarios"""
    print("\n=== EOQ Calculation Demonstration ===")
    
    model = InventoryOptimizationModel()
    
    # Test different scenarios
    scenarios = [
        {
            "name": "High Volume, Low Cost",
            "annual_demand": 10000,
            "ordering_cost": 50,
            "unit_cost": 10,
            "holding_cost_rate": 0.20
        },
        {
            "name": "Low Volume, High Cost", 
            "annual_demand": 1000,
            "ordering_cost": 200,
            "unit_cost": 100,
            "holding_cost_rate": 0.25
        },
        {
            "name": "Medium Volume, Medium Cost",
            "annual_demand": 5000,
            "ordering_cost": 100,
            "unit_cost": 50,
            "holding_cost_rate": 0.22
        }
    ]
    
    results = []
    for scenario in scenarios:
        eoq_result = model.calculate_eoq(
            annual_demand=scenario["annual_demand"],
            ordering_cost=scenario["ordering_cost"],
            unit_cost=scenario["unit_cost"],
            holding_cost_rate=scenario["holding_cost_rate"]
        )
        
        results.append({
            "Scenario": scenario["name"],
            "Annual Demand": scenario["annual_demand"],
            "Unit Cost": scenario["unit_cost"],
            "EOQ": eoq_result["economic_order_quantity"],
            "Total Cost": eoq_result["total_cost"],
            "Orders per Year": eoq_result["orders_per_year"],
            "Cycle Time (days)": eoq_result["cycle_time_days"]
        })
        
        print(f"\n{scenario['name']}:")
        print(f"  Annual Demand: {scenario['annual_demand']:,} units")
        print(f"  Unit Cost: ${scenario['unit_cost']:.2f}")
        print(f"  EOQ: {eoq_result['economic_order_quantity']:,} units")
        print(f"  Total Annual Cost: ${eoq_result['total_cost']:,.2f}")
        print(f"  Orders per Year: {eoq_result['orders_per_year']:.1f}")
        print(f"  Cycle Time: {eoq_result['cycle_time_days']:.1f} days")
    
    return pd.DataFrame(results)

def demonstrate_safety_stock_analysis():
    """Demonstrate safety stock calculation with different service levels"""
    print("\n=== Safety Stock Analysis ===")
    
    model = InventoryOptimizationModel()
    
    # Test different service levels
    service_levels = [0.90, 0.95, 0.99, 0.999]
    demand_std = 5.0
    lead_time = 14
    
    results = []
    for service_level in service_levels:
        safety_result = model.calculate_safety_stock(
            demand_std=demand_std,
            lead_time=lead_time,
            service_level=service_level
        )
        
        results.append({
            "Service Level": f"{service_level:.1%}",
            "Safety Stock": safety_result["safety_stock"],
            "Reorder Point": safety_result["reorder_point"],
            "Z-Score": safety_result["z_score"]
        })
        
        print(f"\nService Level {service_level:.1%}:")
        print(f"  Safety Stock: {safety_result['safety_stock']} units")
        print(f"  Reorder Point: {safety_result['reorder_point']} units")
        print(f"  Z-Score: {safety_result['z_score']:.2f}")
    
    return pd.DataFrame(results)

def visualize_eoq_analysis(eoq_results_df):
    """Visualize EOQ analysis results"""
    print("\n=== EOQ Analysis Visualization ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # EOQ comparison
    axes[0, 0].bar(eoq_results_df['Scenario'], eoq_results_df['EOQ'], color='skyblue')
    axes[0, 0].set_title('Economic Order Quantity by Scenario')
    axes[0, 0].set_ylabel('EOQ (units)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Total cost comparison
    axes[0, 1].bar(eoq_results_df['Scenario'], eoq_results_df['Total Cost'], color='lightcoral')
    axes[0, 1].set_title('Total Annual Cost by Scenario')
    axes[0, 1].set_ylabel('Total Cost ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Orders per year
    axes[1, 0].bar(eoq_results_df['Scenario'], eoq_results_df['Orders per Year'], color='lightgreen')
    axes[1, 0].set_title('Orders per Year by Scenario')
    axes[1, 0].set_ylabel('Orders per Year')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Cycle time
    axes[1, 1].bar(eoq_results_df['Scenario'], eoq_results_df['Cycle Time (days)'], color='orange')
    axes[1, 1].set_title('Cycle Time by Scenario')
    axes[1, 1].set_ylabel('Cycle Time (days)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def visualize_safety_stock_analysis(safety_results_df):
    """Visualize safety stock analysis results"""
    print("\n=== Safety Stock Analysis Visualization ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Safety stock by service level
    service_levels_numeric = [float(sl.strip('%')) / 100 for sl in safety_results_df['Service Level']]
    axes[0].plot(service_levels_numeric, safety_results_df['Safety Stock'], 'o-', color='red', linewidth=2, markersize=8)
    axes[0].set_title('Safety Stock vs Service Level')
    axes[0].set_xlabel('Service Level')
    axes[0].set_ylabel('Safety Stock (units)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0.85, 1.0)
    
    # Z-score vs service level
    axes[1].plot(service_levels_numeric, safety_results_df['Z-Score'], 'o-', color='blue', linewidth=2, markersize=8)
    axes[1].set_title('Z-Score vs Service Level')
    axes[1].set_xlabel('Service Level')
    axes[1].set_ylabel('Z-Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0.85, 1.0)
    
    plt.tight_layout()
    plt.show()

def perform_abc_analysis_demo():
    """Demonstrate ABC analysis"""
    print("\n=== ABC Analysis Demonstration ===")
    
    model = InventoryOptimizationModel()
    products_df, historical_df, _ = generate_synthetic_inventory_data(n_products=30)
    
    # Perform ABC analysis
    abc_results = model.perform_abc_analysis(products_df)
    
    abc_df = pd.DataFrame(abc_results)
    
    print(f"ABC Analysis Results:")
    print(f"Total Products: {len(abc_results)}")
    print(f"Category A: {len(abc_df[abc_df['category'] == 'A'])} products")
    print(f"Category B: {len(abc_df[abc_df['category'] == 'B'])} products")
    print(f"Category C: {len(abc_df[abc_df['category'] == 'C'])} products")
    
    # Show top products by revenue
    print(f"\nTop 5 Products by Revenue:")
    top_products = abc_df.nlargest(5, 'annual_revenue')[['product_id', 'category', 'annual_revenue', 'optimization_priority']]
    for _, product in top_products.iterrows():
        print(f"  {product['product_id']}: Category {product['category']}, "
              f"Revenue: ${product['annual_revenue']:,.2f}, Priority: {product['optimization_priority']}")
    
    return abc_df

def visualize_abc_analysis(abc_df):
    """Visualize ABC analysis results"""
    print("\n=== ABC Analysis Visualization ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Category distribution
    category_counts = abc_df['category'].value_counts()
    axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('ABC Category Distribution')
    
    # Revenue by category
    revenue_by_category = abc_df.groupby('category')['annual_revenue'].sum()
    axes[0, 1].bar(revenue_by_category.index, revenue_by_category.values, color=['red', 'orange', 'green'])
    axes[0, 1].set_title('Total Revenue by ABC Category')
    axes[0, 1].set_ylabel('Annual Revenue ($)')
    
    # Revenue percentage by category
    total_revenue = abc_df['annual_revenue'].sum()
    revenue_percentage = abc_df.groupby('category')['annual_revenue'].sum() / total_revenue * 100
    axes[1, 0].bar(revenue_percentage.index, revenue_percentage.values, color=['red', 'orange', 'green'])
    axes[1, 0].set_title('Revenue Percentage by ABC Category')
    axes[1, 0].set_ylabel('Revenue Percentage (%)')
    
    # Pareto chart (cumulative revenue)
    abc_sorted = abc_df.sort_values('annual_revenue', ascending=False)
    abc_sorted['cumulative_revenue'] = abc_sorted['annual_revenue'].cumsum()
    abc_sorted['cumulative_percentage'] = abc_sorted['cumulative_revenue'] / total_revenue * 100
    abc_sorted['item_percentage'] = (np.arange(len(abc_sorted)) + 1) / len(abc_sorted) * 100
    
    axes[1, 1].plot(abc_sorted['item_percentage'], abc_sorted['cumulative_percentage'], 'b-', linewidth=2)
    axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% Revenue')
    axes[1, 1].axvline(x=20, color='r', linestyle='--', alpha=0.7, label='20% Items')
    axes[1, 1].set_title('ABC Pareto Analysis')
    axes[1, 1].set_xlabel('Cumulative Item Percentage (%)')
    axes[1, 1].set_ylabel('Cumulative Revenue Percentage (%)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def demonstrate_inventory_optimization():
    """Demonstrate complete inventory optimization for a product"""
    print("\n=== Complete Inventory Optimization Demonstration ===")
    
    model = InventoryOptimizationModel()
    
    # Create a sample product
    product_data = {
        'product_id': 'DEMO_PRODUCT_001',
        'product_name': 'Demo Product',
        'category': 'Electronics',
        'unit_cost': 75.0,
        'holding_cost_rate': 0.22,
        'ordering_cost': 150.0,
        'lead_time_days': 21,
        'current_stock': 150,
        'annual_demand': 3650,  # 10 units per day
        'demand_std': 15.0,  # Daily demand standard deviation
        'service_level': 0.95,
        'expiration_days': 365,
        'min_order_quantity': 1,
        'max_stock_capacity': 500,
        'supplier_reliability': 0.95
    }
    
    print(f"Product: {product_data['product_name']}")
    print(f"Unit Cost: ${product_data['unit_cost']:.2f}")
    print(f"Annual Demand: {product_data['annual_demand']:,} units")
    print(f"Lead Time: {product_data['lead_time_days']} days")
    print(f"Service Level: {product_data['service_level']:.1%}")
    
    # Perform optimization
    optimization_result = model.optimize_inventory_policy(product_data)
    
    print(f"\nOptimization Results:")
    print(f"Economic Order Quantity: {optimization_result['economic_order_quantity']} units")
    print(f"Reorder Point: {optimization_result['reorder_point']} units")
    print(f"Safety Stock: {optimization_result['safety_stock']} units")
    print(f"Average Inventory: {optimization_result['average_inventory']:.1f} units")
    print(f"Total Annual Cost: ${optimization_result['total_cost']:,.2f}")
    print(f"Ordering Cost: ${optimization_result['ordering_cost']:,.2f}")
    print(f"Holding Cost: ${optimization_result['holding_cost']:,.2f}")
    print(f"Inventory Turnover: {optimization_result['inventory_turnover']:.2f}")
    print(f"Days of Supply: {optimization_result['days_of_supply']} days")
    
    recommendations = optimization_result['recommendations']
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    return optimization_result

def visualize_inventory_optimization(optimization_result):
    """Visualize inventory optimization results"""
    print("\n=== Inventory Optimization Visualization ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cost breakdown
    costs = ['Ordering Cost', 'Holding Cost']
    cost_values = [optimization_result['ordering_cost'], optimization_result['holding_cost']]
    axes[0, 0].pie(cost_values, labels=costs, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Annual Cost Breakdown')
    
    # Inventory levels
    inventory_levels = ['Safety Stock', 'Cycle Stock']
    inventory_values = [optimization_result['safety_stock'], 
                       optimization_result['economic_order_quantity'] / 2]
    axes[0, 1].bar(inventory_levels, inventory_values, color=['red', 'blue'])
    axes[0, 1].set_title('Inventory Components')
    axes[0, 1].set_ylabel('Inventory Level (units)')
    
    # Total cost vs order quantity (theoretical)
    order_quantities = np.linspace(50, 500, 100)
    annual_demand = 3650
    ordering_cost = 150.0
    unit_cost = 75.0
    holding_cost_rate = 0.22
    
    ordering_costs = (annual_demand / order_quantities) * ordering_cost
    holding_costs = (order_quantities / 2) * unit_cost * holding_cost_rate
    total_costs = ordering_costs + holding_costs
    
    axes[1, 0].plot(order_quantities, ordering_costs, 'b-', label='Ordering Cost')
    axes[1, 0].plot(order_quantities, holding_costs, 'r-', label='Holding Cost')
    axes[1, 0].plot(order_quantities, total_costs, 'g-', label='Total Cost')
    axes[1, 0].axvline(x=optimization_result['economic_order_quantity'], 
                      color='black', linestyle='--', alpha=0.7, label='EOQ')
    axes[1, 0].set_title('Cost Analysis vs Order Quantity')
    axes[1, 0].set_xlabel('Order Quantity (units)')
    axes[1, 0].set_ylabel('Annual Cost ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Service level impact
    service_levels = np.linspace(0.8, 0.99, 20)
    safety_stocks = []
    
    from scipy import stats
    for sl in service_levels:
        z_score = stats.norm.ppf(sl)
        safety_stock = z_score * 15.0 * np.sqrt(21)  # demand_std * sqrt(lead_time)
        safety_stocks.append(safety_stock)
    
    axes[1, 1].plot(service_levels, safety_stocks, 'purple', linewidth=2)
    axes[1, 1].axvline(x=optimization_result['service_level_achieved'], 
                      color='red', linestyle='--', alpha=0.7, label='Target Service Level')
    axes[1, 1].set_title('Safety Stock vs Service Level')
    axes[1, 1].set_xlabel('Service Level')
    axes[1, 1].set_ylabel('Safety Stock (units)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main training and demonstration function"""
    print("Inventory Optimization Training and Visualization")
    print("=" * 50)
    
    # Set up plotting style
    setup_plotting_style()
    
    # Generate sample data
    products_df, historical_df, model = generate_synthetic_inventory_data()
    
    # Demonstrate EOQ calculation
    eoq_results = demonstrate_eoq_calculation()
    visualize_eoq_analysis(eoq_results)
    
    # Demonstrate safety stock analysis
    safety_results = demonstrate_safety_stock_analysis()
    visualize_safety_stock_analysis(safety_results)
    
    # Perform ABC analysis
    abc_results = perform_abc_analysis_demo()
    visualize_abc_analysis(abc_results)
    
    # Demonstrate complete inventory optimization
    optimization_result = demonstrate_inventory_optimization()
    visualize_inventory_optimization(optimization_result)
    
    print("\n" + "=" * 50)
    print("Training and visualization completed!")
    print("\nKey Insights:")
    print("1. EOQ balances ordering and holding costs to minimize total cost")
    print("2. Safety stock increases exponentially with service level")
    print("3. ABC analysis helps prioritize inventory management efforts")
    print("4. Inventory optimization considers multiple factors: demand, costs, lead time, service level")
    
    # Save model if needed
    save_model = input("\nDo you want to save the trained model? (y/n): ").lower().strip()
    if save_model == 'y':
        model_path = "inventory_optimization_model.pkl"
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()