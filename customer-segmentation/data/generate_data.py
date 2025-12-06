import pandas as pd
import numpy as np

def generate_data(n_samples=200):
    np.random.seed(42)
    
    # CustomerID
    customer_ids = np.arange(1, n_samples + 1)
    
    # Gender (0: Female, 1: Male)
    genders = np.random.choice(['Female', 'Male'], n_samples)
    
    # Age (18-70)
    ages = np.random.randint(18, 70, n_samples)
    
    # Annual Income (k$) (15-137)
    incomes = np.random.randint(15, 140, n_samples)
    
    # Spending Score (1-100)
    # Generate clusters roughly
    # 1. Low Income, Low Score
    # 2. Low Income, High Score
    # 3. Mid Income, Mid Score
    # 4. High Income, Low Score
    # 5. High Income, High Score
    
    spending_scores = []
    for i in range(n_samples):
        income = incomes[i]
        if income < 40:
            # Low income
            if np.random.random() < 0.5:
                spending_scores.append(np.random.randint(5, 40)) # Low score
            else:
                spending_scores.append(np.random.randint(60, 99)) # High score
        elif income > 80:
            # High income
            if np.random.random() < 0.5:
                spending_scores.append(np.random.randint(5, 40)) # Low score
            else:
                spending_scores.append(np.random.randint(60, 99)) # High score
        else:
            # Mid income
            spending_scores.append(np.random.randint(35, 65)) # Mid score
            
    spending_scores = np.array(spending_scores)
    
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'Annual Income (k$)': incomes,
        'Spending Score (1-100)': spending_scores
    })
    
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv('customers.csv', index=False)
    print("Data generated and saved to customers.csv")
