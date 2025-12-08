from typing import List, Dict
from .schemas import Transaction, FrequentItemset


class MarketBasketAnalyzer:
    def __init__(self):
        self.transactions = []
    
    def load_transactions(self, transactions: List[Transaction]):
        self.transactions = transactions
    
    def analyze(self, min_support: float = 0.01):
        item_counts = {}
        for transaction in self.transactions:
            for item in transaction.items:
                item_counts[item.item_id] = item_counts.get(item.item_id, 0) + 1
        
        total = len(self.transactions)
        self.frequent_itemsets = []
        for item_id, count in item_counts.items():
            support = count / total
            if support >= min_support:
                self.frequent_itemsets.append(FrequentItemset(
                    items=[item_id], support=support, count=count, length=1
                ))
    
    def get_recommendations(self, items: List[str], top_k: int = 5) -> List[Dict]:
        recommendations = []
        for itemset in self.frequent_itemsets:
            item = itemset.items[0]
            if item not in items:
                recommendations.append({
                    'item_id': item, 'score': itemset.support
                })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]