"""
Business Intelligence Module for AeroVision-GGM 2.0
Municipal decision support, ROI calculation, and economic impact analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import BUSINESS_CONFIG

class MunicipalBusinessIntelligence:
    """Professional business intelligence for municipal clients"""
    
    def __init__(self):
        self.config = BUSINESS_CONFIG
        
    def calculate_municipal_roi(self, population, current_pm25_avg, predicted_reduction=0.20):
        """Calculate ROI for municipal air quality investment"""
        
        # Current health costs
        annual_healthcare_cost = population * self.config['health_costs']['healthcare_cost_per_person_annually']
        annual_productivity_loss = population * self.config['health_costs']['productivity_loss_per_person']
        
        total_current_cost = annual_healthcare_cost + annual_productivity_loss
        
        # Projected savings with AeroVision implementation
        projected_savings = total_current_cost * predicted_reduction
        
        # Investment cost (Premium plan recommended for municipalities)
        annual_investment = self.config['municipal_pricing']['premium_plan']
        setup_cost = self.config['municipal_pricing']['setup_cost']
        
        # ROI calculation
        net_annual_benefit = projected_savings - annual_investment
        roi_percentage = (projected_savings / annual_investment) * 100
        payback_period_months = (setup_cost / max(net_annual_benefit/12, 1))
        
        return {
            'population': population,
            'current_annual_cost': total_current_cost,
            'projected_annual_savings': projected_savings,
            'annual_investment': annual_investment,
            'net_annual_benefit': net_annual_benefit,
            'roi_percentage': roi_percentage,
            'payback_period_months': payback_period_months,
            'break_even_year': max(1, setup_cost / max(net_annual_benefit, 1))
        }
    
    def calculate_health_impact(self, population, current_pm25, target_pm25):
        """Calculate health impact of PM2.5 reduction"""
        
        # WHO guidelines: PM2.5 reduction health benefits
        pm25_reduction = max(0, current_pm25 - target_pm25)
        
        # Health benefits per μg/m³ reduction (based on WHO studies)
        mortality_reduction_per_ugm3 = 0.006  # 0.6% mortality reduction per μg/m³
        hospitalization_reduction = 0.008      # 0.8% hospitalization reduction
        
        # Calculate impacts
        annual_deaths_prevented = population * mortality_reduction_per_ugm3 * pm25_reduction * 0.001
        hospitalizations_prevented = population * hospitalization_reduction * pm25_reduction * 0.01
        
        # Economic value of health benefits
        value_of_statistical_life = 15000000  # INR (conservative estimate for India)
        hospitalization_cost = 50000          # INR per hospitalization
        
        health_economic_benefit = (
            annual_deaths_prevented * value_of_statistical_life +
            hospitalizations_prevented * hospitalization_cost
        )
        
        return {
            'pm25_reduction': pm25_reduction,
            'deaths_prevented_annually': annual_deaths_prevented,
            'hospitalizations_prevented': hospitalizations_prevented,
            'health_economic_benefit': health_economic_benefit,
            'quality_adjusted_life_years': annual_deaths_prevented * 10  # QALY estimate
        }
    
    def generate_intervention_recommendations(self, pm25_forecast, aqi_threshold=100):
        """Generate AI-powered intervention recommendations"""
        
        if not pm25_forecast:
            return []
            
        recommendations = []
        max_pm25 = max([h['predicted_pm25'] for h in pm25_forecast], default=0)
        max_aqi = max([h['predicted_aqi'] for h in pm25_forecast], default=0)
        
        if max_aqi > 200:  # Hazardous
            recommendations.extend([
                {
                    'priority': 'CRITICAL',
                    'action': 'Emergency Protocol Activation',
                    'description': 'Implement emergency measures: industrial shutdown, traffic restrictions',
                    'expected_reduction': '40%',
                    'cost_estimate': 2000000,  # INR
                    'implementation_time': '2-4 hours',
                    'health_benefit': 'Prevents severe health emergencies'
                },
                {
                    'priority': 'HIGH',
                    'action': 'Public Health Advisory',
                    'description': 'Issue health warnings, recommend indoor activities',
                    'expected_reduction': '0% (mitigation)',
                    'cost_estimate': 50000,
                    'implementation_time': '30 minutes',
                    'health_benefit': 'Reduces population exposure'
                }
            ])
            
        elif max_aqi > 150:  # Unhealthy
            recommendations.extend([
                {
                    'priority': 'HIGH',
                    'action': 'Traffic Management',
                    'description': 'Implement odd-even vehicle restrictions, optimize traffic flow',
                    'expected_reduction': '15%',
                    'cost_estimate': 500000,
                    'implementation_time': '4-6 hours',
                    'health_benefit': 'Reduces respiratory symptoms'
                },
                {
                    'priority': 'MEDIUM',
                    'action': 'Construction Regulation',
                    'description': 'Halt non-essential construction activities, dust suppression',
                    'expected_reduction': '25%',
                    'cost_estimate': 1000000,
                    'implementation_time': '2-3 hours',
                    'health_benefit': 'Reduces particulate matter'
                }
            ])
            
        elif max_aqi > 100:  # Unhealthy for Sensitive
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Proactive Monitoring',
                'description': 'Increase monitoring frequency, prepare intervention measures',
                'expected_reduction': '5%',
                'cost_estimate': 100000,
                'implementation_time': '1 hour',
                'health_benefit': 'Early warning system activation'
            })
            
        return recommendations
    
    def generate_competitive_analysis(self):
        """Generate competitive analysis for municipal presentation"""
        
        existing_solutions = self.config['competitive_analysis']['existing_solutions']
        advantages = self.config['competitive_analysis']['aerovision_advantages']
        
        comparison_matrix = {
            'features': [
                'Data Coverage',
                'AI Forecasting',
                'Municipal Integration',
                'Cost Effectiveness',
                'Deployment Speed',
                'Scalability'
            ],
            'AeroVision-GGM': [
                '400K+ Google AirView+ records',
                'Advanced ensemble AI (47.06 μg/m³ MAE)',
                'Government-ready APIs',
                'High ROI (150-300%)',
                'Immediate deployment',
                'Nationwide ready'
            ],
            'Traditional_Solutions': [
                'Limited station coverage',
                'Basic statistical models',
                'Manual integration required',
                'High operational costs',
                '6-12 months setup',
                'City-by-city deployment'
            ]
        }
        
        return {
            'comparison_matrix': comparison_matrix,
            'unique_advantages': advantages,
            'market_positioning': 'Premium AI-powered municipal air quality intelligence platform'
        }
    
    def calculate_deployment_timeline(self, city_size='large'):
        """Calculate realistic deployment timeline for municipalities"""
        
        timelines = {
            'large': {  # >1M population (like Gurugram)
                'planning_phase': '2-3 weeks',
                'system_integration': '3-4 weeks', 
                'testing_validation': '2-3 weeks',
                'staff_training': '1-2 weeks',
                'full_deployment': '8-12 weeks total',
                'cost_estimate': self.config['municipal_pricing']['premium_plan']
            },
            'medium': {  # 100K-1M population
                'planning_phase': '1-2 weeks',
                'system_integration': '2-3 weeks',
                'testing_validation': '1-2 weeks', 
                'staff_training': '1 week',
                'full_deployment': '5-8 weeks total',
                'cost_estimate': self.config['municipal_pricing']['basic_plan']
            }
        }
        
        return timelines.get(city_size, timelines['large'])
