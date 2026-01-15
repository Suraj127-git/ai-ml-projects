import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.model import LeadScoringModel
    from app.schemas import LeadData
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the notebooks directory")
    sys.exit(1)

def analyze_lead_characteristics(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze lead characteristics and conversion patterns"""
    
    print("📊 Analyzing lead characteristics...")
    
    analysis = {}
    
    # Overall conversion rate
    overall_conversion_rate = df['converted'].mean()
    analysis['overall_conversion_rate'] = overall_conversion_rate
    
    # Conversion by categorical features
    categorical_features = ['company_size', 'industry', 'job_title', 'lead_source', 'budget_range', 'authority_level', 'timeline']
    
    for feature in categorical_features:
        conversion_by_feature = df.groupby(feature)['converted'].agg(['count', 'sum', 'mean']).round(3)
        conversion_by_feature.columns = ['total_leads', 'conversions', 'conversion_rate']
        analysis[f'conversion_by_{feature}'] = conversion_by_feature.to_dict('index')
    
    # Numeric feature analysis
    numeric_features = ['engagement_score', 'website_visits', 'email_opens', 'form_submissions', 'demo_requests', 'content_downloads', 'social_media_engagement', 'days_since_last_activity']
    
    for feature in numeric_features:
        converted_leads = df[df['converted'] == True][feature]
        non_converted_leads = df[df['converted'] == False][feature]
        
        analysis[f'{feature}_converted'] = {
            'mean': converted_leads.mean(),
            'median': converted_leads.median(),
            'std': converted_leads.std(),
            'min': converted_leads.min(),
            'max': converted_leads.max()
        }
        
        analysis[f'{feature}_non_converted'] = {
            'mean': non_converted_leads.mean(),
            'median': non_converted_leads.median(),
            'std': non_converted_leads.std(),
            'min': non_converted_leads.min(),
            'max': non_converted_leads.max()
        }
    
    # Pain point analysis
    all_pain_points = []
    for pain_points in df['pain_points']:
        all_pain_points.extend(pain_points)
    
    pain_point_conversion = {}
    for pain_point in set(all_pain_points):
        leads_with_pain_point = df[df['pain_points'].apply(lambda x: pain_point in x)]
        if len(leads_with_pain_point) > 0:
            pain_point_conversion[pain_point] = {
                'count': len(leads_with_pain_point),
                'conversion_rate': leads_with_pain_point['converted'].mean()
            }
    
    analysis['pain_point_conversion'] = pain_point_conversion
    
    # Competitor usage impact
    competitor_analysis = df.groupby('competitor_usage')['converted'].agg(['count', 'mean']).round(3)
    competitor_analysis.columns = ['count', 'conversion_rate']
    analysis['competitor_usage_impact'] = competitor_analysis.to_dict('index')
    
    # Marketing and sales qualification impact
    qualification_analysis = df.groupby(['marketing_qualified', 'sales_qualified'])['converted'].agg(['count', 'mean']).round(3)
    qualification_analysis.columns = ['count', 'conversion_rate']
    analysis['qualification_impact'] = qualification_analysis.to_dict('index')
    
    print(f"✅ Lead characteristics analysis completed")
    print(f"   Overall conversion rate: {overall_conversion_rate:.1%}")
    
    return analysis

def feature_importance_analysis(model, feature_names: List[str]) -> Dict[str, Any]:
    """Analyze feature importance from trained models"""
    
    print("🔍 Analyzing feature importance...")
    
    importance_data = {}
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create feature importance ranking
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        importance_data['feature_importance_ranking'] = feature_importance
        importance_data['top_features'] = feature_importance[:10]  # Top 10 features
        
        # Calculate cumulative importance
        cumulative_importance = np.cumsum([imp for _, imp in feature_importance])
        importance_data['cumulative_importance'] = cumulative_importance.tolist()
        
        # Find features that contribute to 80% of importance
        threshold_80_idx = np.where(cumulative_importance >= 0.8)[0]
        if len(threshold_80_idx) > 0:
            importance_data['features_for_80_percent'] = threshold_80_idx[0] + 1
        
        print(f"✅ Feature importance analysis completed")
        print(f"   Top feature: {feature_importance[0][0]} ({feature_importance[0][1]:.3f})")
        
    return importance_data

def model_comparison_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare performance of different models"""
    
    print("📈 Comparing model performance...")
    
    comparison = {}
    
    # Extract metrics for each model
    models = []
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    for model_name, model_results in results.items():
        if model_results is not None:
            model_comparison = {
                'model': model_name,
                'accuracy': model_results['accuracy'],
                'precision': model_results['precision'],
                'recall': model_results['recall'],
                'f1_score': model_results['f1_score'],
                'auc_roc': model_results['auc_roc']
            }
            models.append(model_comparison)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(models)
    comparison_df = comparison_df.round(3)
    
    # Find best model for each metric
    best_models = {}
    for metric in metrics:
        best_model = comparison_df.loc[comparison_df[metric].idxmax()]
        best_models[metric] = {
            'model': best_model['model'],
            'value': best_model[metric]
        }
    
    # Calculate average performance
    comparison_df['average_score'] = comparison_df[metrics].mean(axis=1)
    
    comparison['model_comparison'] = comparison_df.to_dict('records')
    comparison['best_models'] = best_models
    comparison['overall_best'] = comparison_df.loc[comparison_df['average_score'].idxmax()]
    
    print(f"✅ Model comparison completed")
    print(f"   Best overall model: {comparison['overall_best']['model']} (avg score: {comparison['overall_best']['average_score']:.3f})")
    
    return comparison

def generate_lead_scoring_insights(analysis: Dict[str, Any], model_comparison: Dict[str, Any]) -> List[str]:
    """Generate actionable insights from the analysis"""
    
    print("💡 Generating insights...")
    
    insights = []
    
    # Overall conversion insights
    overall_rate = analysis['overall_conversion_rate']
    if overall_rate < 0.1:
        insights.append(f"Low overall conversion rate ({overall_rate:.1%}) suggests need for better lead qualification")
    elif overall_rate > 0.3:
        insights.append(f"High overall conversion rate ({overall_rate:.1%}) indicates effective lead generation")
    
    # Company size insights
    company_size_conversion = analysis['conversion_by_company_size']
    best_company_size = max(company_size_conversion.keys(), key=lambda x: company_size_conversion[x]['conversion_rate'])
    worst_company_size = min(company_size_conversion.keys(), key=lambda x: company_size_conversion[x]['conversion_rate'])
    
    insights.append(f"Best performing company size: {best_company_size} ({company_size_conversion[best_company_size]['conversion_rate']:.1%} conversion)")
    insights.append(f"Focus improvement on: {worst_company_size} segment ({company_size_conversion[worst_company_size]['conversion_rate']:.1%} conversion)")
    
    # Industry insights
    industry_conversion = analysis['conversion_by_industry']
    best_industry = max(industry_conversion.keys(), key=lambda x: industry_conversion[x]['conversion_rate'])
    worst_industry = min(industry_conversion.keys(), key=lambda x: industry_conversion[x]['conversion_rate'])
    
    insights.append(f"Highest converting industry: {best_industry} ({industry_conversion[best_industry]['conversion_rate']:.1%})")
    insights.append(f"Industry needing attention: {worst_industry} ({industry_conversion[worst_industry]['conversion_rate']:.1%})")
    
    # Engagement insights
    converted_engagement = analysis['engagement_score_converted']['mean']
    non_converted_engagement = analysis['engagement_score_non_converted']['mean']
    
    if converted_engagement > non_converted_engagement * 1.5:
        insights.append(f"High engagement strongly correlates with conversion ({converted_engagement:.0f} vs {non_converted_engagement:.0f})")
    
    # Authority level insights
    authority_conversion = analysis['conversion_by_authority_level']
    best_authority = max(authority_conversion.keys(), key=lambda x: authority_conversion[x]['conversion_rate'])
    
    insights.append(f"Decision makers convert best: {best_authority} ({authority_conversion[best_authority]['conversion_rate']:.1%})")
    
    # Timeline insights
    timeline_conversion = analysis['conversion_by_timeline']
    urgent_timelines = ['immediate', '1_month', '3_months']
    urgent_conversion = [timeline_conversion[t]['conversion_rate'] for t in urgent_timelines if t in timeline_conversion]
    if urgent_conversion:
        avg_urgent_conversion = sum(urgent_conversion) / len(urgent_conversion)
        insights.append(f"Urgent timelines show higher conversion: {avg_urgent_conversion:.1%} average")
    
    # Competitor insights
    competitor_impact = analysis['competitor_usage_impact']
    if True in competitor_impact and False in competitor_impact:
        competitor_penalty = competitor_impact[False]['conversion_rate'] - competitor_impact[True]['conversion_rate']
        if competitor_penalty > 0.1:
            insights.append(f"Competitor usage significantly reduces conversion by {competitor_penalty:.1%}")
    
    # Qualification insights
    qualification_impact = analysis['qualification_impact']
    mql_sql_combinations = [(True, True), (True, False), (False, True), (False, False)]
    for mql, sql in mql_sql_combinations:
        if (mql, sql) in qualification_impact:
            rate = qualification_impact[(mql, sql)]['conversion_rate']
            status = "MQL+SQL" if mql and sql else "MQL only" if mql else "SQL only" if sql else "Unqualified"
            insights.append(f"{status} leads convert at {rate:.1%}")
    
    print(f"✅ Generated {len(insights)} insights")
    
    return insights

def save_analysis_results(analysis: Dict[str, Any], model_comparison: Dict[str, Any], 
                         insights: List[str], training_data_info: Dict[str, Any]):
    """Save analysis results to files"""
    
    print("💾 Saving analysis results...")
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save comprehensive analysis
    comprehensive_results = {
        'training_data_info': training_data_info,
        'lead_analysis': analysis,
        'model_comparison': model_comparison,
        'insights': insights,
        'analysis_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(results_dir, 'lead_scoring_analysis.json'), 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Save model comparison as CSV for easy viewing
    if 'model_comparison' in model_comparison:
        import pandas as pd
        comparison_df = pd.DataFrame(model_comparison['model_comparison'])
        comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    
    print(f"✅ Analysis results saved to {results_dir}")

def main():
    """Main training and analysis function"""
    
    print("🚀 Starting Lead Scoring Model Training and Analysis")
    print("=" * 60)
    
    # Initialize model
    lead_model = LeadScoringModel()
    
    # Generate training data
    print("\n📊 Generating training data...")
    training_data = lead_model.generate_synthetic_training_data(n_samples=2000)
    print(f"✅ Generated {len(training_data)} training samples")
    
    # Analyze lead characteristics
    print("\n🔍 Analyzing lead characteristics...")
    lead_analysis = analyze_lead_characteristics(training_data)
    
    # Train models
    print("\n🤖 Training lead scoring models...")
    training_results = lead_model.train_models(training_data)
    
    # Analyze feature importance
    print("\n🔬 Analyzing feature importance...")
    feature_analysis = {}
    for model_name, model_results in training_results.items():
        if model_results is not None and hasattr(lead_model.models[model_name], 'feature_importances_'):
            feature_analysis[model_name] = feature_importance_analysis(
                lead_model.models[model_name], 
                lead_model.feature_names
            )
    
    # Compare models
    print("\n📈 Comparing model performance...")
    model_comparison = model_comparison_analysis(training_results)
    
    # Generate insights
    print("\n💡 Generating actionable insights...")
    insights = generate_lead_scoring_insights(lead_analysis, model_comparison)
    
    # Save models
    print("\n💾 Saving trained models...")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    lead_model.save_models(os.path.join(models_dir, 'lead_scoring_models.pkl'))
    
    # Save analysis results
    training_data_info = {
        'total_samples': len(training_data),
        'conversion_rate': lead_analysis['overall_conversion_rate'],
        'date_generated': datetime.now().isoformat()
    }
    
    save_analysis_results(lead_analysis, model_comparison, insights, training_data_info)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TRAINING AND ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"📊 Training Data: {len(training_data)} samples")
    print(f"🎯 Overall Conversion Rate: {lead_analysis['overall_conversion_rate']:.1%}")
    print(f"🤖 Models Trained: {len([k for k, v in training_results.items() if v is not None])}")
    print(f"🏆 Best Model: {model_comparison['overall_best']['model']} (F1: {model_comparison['overall_best']['f1_score']:.3f})")
    
    print(f"\n📈 Top Insights:")
    for i, insight in enumerate(insights[:5], 1):
        print(f"   {i}. {insight}")
    
    print(f"\n✅ Training and analysis completed successfully!")
    print(f"📁 Models saved to: {models_dir}")
    print(f"📊 Analysis saved to: {os.path.join(os.path.dirname(models_dir), 'results')}")
    
    # Example usage
    print(f"\n💡 Example Usage:")
    print(f"   # Load trained model")
    print(f"   model = LeadScoringModel()")
    print(f"   model.load_models('models/lead_scoring_models.pkl')")
    print(f"   ")
    print(f"   # Score a lead")
    print(f"   lead_data = {{'lead_id': 'LEAD001', 'company_size': 'large', ...}}")
    print(f"   score = model.predict_lead_score(lead_data)")
    print(f"   print(f'Score: {{score[\"score\"]}}/100')")

if __name__ == "__main__":
    main()