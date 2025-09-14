# -*- coding: utf-8 -*-
"""
Model Interface for Push Notification Generation
Binds Ollama LLM with the trained ML classifier for product recommendation
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_clients, load_client_tables
from scoring import compute_expected_benefits, rank_products
from prompts import build_user_prompt, SYSTEM_PROMPT, month_of_last_full_period
from ollama_client import generate_with_guardrails
from validator import validate_push, autocorrect


class ModelInterface:
    """
    Unified interface for combining ML model predictions with Ollama LLM generation
    """
    
    def __init__(self, 
                 config_path: str = "config.yaml",
                 model_path: str = "model/model.pkl",
                 model_meta_path: str = "model/model_meta.json"):
        """
        Initialize the model interface
        
        Args:
            config_path: Path to configuration YAML
            model_path: Path to trained ML model pickle file
            model_meta_path: Path to model metadata JSON
        """
        self.config_path = config_path
        self.model_path = model_path
        self.model_meta_path = model_meta_path
        
        # Load configuration
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Load model metadata
        with open(model_meta_path, 'r', encoding='utf-8') as f:
            self.model_meta = json.load(f)
        
        # Available product classes
        self.product_classes = self.model_meta.get('classes', [
            "Депозит Мультивалютный",
            "Депозит Накопительный", 
            "Депозит Сберегательный",
            "Золотые слитки",
            "Инвестиции",
            "Карта для путешествий",
            "Кредит наличными",
            "Кредитная карта",
            "Обмен валют",
            "Премиальная карта"
        ])
        
        # Try to load ML model (with fallback if version issues)
        self.ml_model = None
        self.ml_model_available = False
        self._load_ml_model()
        
        print(f"Model Interface initialized:")
        print(f"  - Config loaded: {config_path}")
        print(f"  - ML Model available: {self.ml_model_available}")
        print(f"  - Product classes: {len(self.product_classes)}")
    
    def _load_ml_model(self):
        """Load the ML model with error handling"""
        try:
            with open(self.model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            self.ml_model_available = True
            print(f"ML model loaded successfully: {type(self.ml_model)}")
        except Exception as e:
            print(f"Warning: Could not load ML model: {e}")
            print("Will fall back to rule-based scoring")
            self.ml_model_available = False
    
    def extract_features(self, client_row: pd.Series, tx: pd.DataFrame, tr: pd.DataFrame) -> np.ndarray:
        """
        Extract features for ML model prediction
        This should match the feature extraction used during training
        """
        features = []
        
        # Client features
        age = int(client_row.get('age', 0)) if pd.notna(client_row.get('age')) else 0
        avg_balance = float(client_row.get('avg_monthly_balance_KZT', 0))
        
        features.extend([age, avg_balance])
        
        # Status encoding (one-hot)
        status = client_row.get('status', 'Стандартный клиент')
        statuses = ['Студент', 'Зарплатный клиент', 'Премиальный клиент', 'Стандартный клиент']
        status_features = [1 if status == s else 0 for s in statuses]
        features.extend(status_features)
        
        # Transaction features
        if tx is not None and not tx.empty:
            total_spend = tx['amount'].sum()
            tx_count = len(tx)
            avg_tx = total_spend / tx_count if tx_count > 0 else 0
            
            # Category spending
            cat_spend = tx.groupby('category')['amount'].sum().to_dict()
            travel_spend = sum(cat_spend.get(c, 0) for c in ['Путешествия', 'Такси', 'Отели'])
            restaurant_spend = cat_spend.get('Кафе и рестораны', 0)
            online_spend = sum(cat_spend.get(c, 0) for c in ['Смотрим дома', 'Играем дома', 'Едим дома'])
            
            features.extend([total_spend, tx_count, avg_tx, travel_spend, restaurant_spend, online_spend])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Transfer features
        if tr is not None and not tr.empty:
            fx_count = tr['type'].isin(['fx_buy', 'fx_sell']).sum()
            invest_count = tr['type'].isin(['invest_in', 'invest_out']).sum()
            gold_count = tr['type'].isin(['gold_buy_out', 'gold_sell_in']).sum()
            
            features.extend([fx_count, invest_count, gold_count])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def predict_product_ml(self, client_row: pd.Series, tx: pd.DataFrame, tr: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Predict product recommendations using ML model
        
        Returns:
            List of (product_name, probability) tuples sorted by probability
        """
        if not self.ml_model_available:
            return []
        
        try:
            features = self.extract_features(client_row, tx, tr)
            
            if hasattr(self.ml_model, 'predict_proba'):
                probabilities = self.ml_model.predict_proba(features)[0]
                predictions = list(zip(self.product_classes, probabilities))
                predictions.sort(key=lambda x: x[1], reverse=True)
                return predictions
            elif hasattr(self.ml_model, 'decision_function'):
                scores = self.ml_model.decision_function(features)[0]
                predictions = list(zip(self.product_classes, scores))
                predictions.sort(key=lambda x: x[1], reverse=True)
                return predictions
            else:
                # Just predict class
                pred_class = self.ml_model.predict(features)[0]
                return [(pred_class, 1.0)]
                
        except Exception as e:
            print(f"ML prediction failed: {e}")
            return []
    
    def predict_product_rules(self, client_row: pd.Series, tx: pd.DataFrame, tr: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Predict product recommendations using rule-based scoring
        
        Returns:
            List of (product_name, benefit_score) tuples sorted by benefit
        """
        benefits, _ = compute_expected_benefits(client_row, tx, tr, self.config)
        ranked = rank_products(benefits)
        return ranked
    
    def get_best_product(self, client_row: pd.Series, tx: pd.DataFrame, tr: pd.DataFrame, 
                        method: str = "hybrid") -> Tuple[str, float]:
        """
        Get the best product recommendation
        
        Args:
            client_row: Client information
            tx: Transaction data
            tr: Transfer data  
            method: "ml", "rules", or "hybrid"
            
        Returns:
            (product_name, confidence_score)
        """
        if method == "ml" and self.ml_model_available:
            ml_predictions = self.predict_product_ml(client_row, tx, tr)
            if ml_predictions:
                return ml_predictions[0]
        
        if method == "rules" or not self.ml_model_available:
            rule_predictions = self.predict_product_rules(client_row, tx, tr)
            if rule_predictions:
                return rule_predictions[0]
        
        if method == "hybrid":
            # Combine ML and rule-based approaches
            ml_predictions = self.predict_product_ml(client_row, tx, tr) if self.ml_model_available else []
            rule_predictions = self.predict_product_rules(client_row, tx, tr)
            
            if ml_predictions and rule_predictions:
                # Weight ML predictions higher if available
                ml_top = ml_predictions[0]
                rule_top = rule_predictions[0]
                
                # If top predictions agree, return with high confidence
                if ml_top[0] == rule_top[0]:
                    return (ml_top[0], min(ml_top[1] + rule_top[1] * 0.1, 1.0))
                else:
                    # Return ML prediction but with lower confidence
                    return (ml_top[0], ml_top[1] * 0.8)
            elif rule_predictions:
                return rule_predictions[0]
        
        # Fallback
        return ("Инвестиции", 0.1)
    
    def generate_push_notification(self, 
                                 client_row: pd.Series,
                                 tx: pd.DataFrame, 
                                 tr: pd.DataFrame,
                                 ollama_model: str = None,
                                 prediction_method: str = "hybrid") -> Dict[str, Any]:
        """
        Generate personalized push notification
        
        Args:
            client_row: Client information
            tx: Transaction data
            tr: Transfer data
            ollama_model: Ollama model name (e.g., "mistral:7b")
            prediction_method: "ml", "rules", or "hybrid"
            
        Returns:
            Dictionary with recommendation results
        """
        client_code = int(client_row['client_code'])
        
        # Get product recommendation
        best_product, confidence = self.get_best_product(client_row, tx, tr, prediction_method)
        
        # Calculate expected benefit using rule-based scoring
        benefits, facts = compute_expected_benefits(client_row, tx, tr, self.config)
        expected_benefit = benefits.get(best_product, 0.0)
        
        # Get CTA for the product
        cta_map = {
            "Карта для путешествий": self.config["cta"]["travel"],
            "Премиальная карта": self.config["cta"]["premium"],
            "Кредитная карта": self.config["cta"]["credit"],
            "Обмен валют": self.config["cta"]["fx"],
            "Депозит Мультивалютный": self.config["cta"]["deposit"],
            "Депозит Сберегательный": self.config["cta"]["deposit"],
            "Депозит Накопительный": self.config["cta"]["deposit"],
            "Инвестиции": self.config["cta"]["invest"],
            "Золотые слитки": self.config["cta"]["gold"],
        }
        cta = cta_map.get(best_product, "Посмотреть")
        
        # Build behavior summary
        behavior = self._build_behavior_summary(tx)
        ref_month = month_of_last_full_period(tx['date'] if 'date' in tx.columns else pd.Series([], dtype=str))
        
        # Generate push notification
        push_text = ""
        
        if ollama_model:
            # Use Ollama LLM for generation
            user_prompt = build_user_prompt(
                name=client_row.get('name', 'Клиент'),
                status=client_row.get('status', ''),
                age=int(client_row.get('age', 0)) if pd.notna(client_row.get('age')) else 0,
                city=client_row.get('city', ''),
                avg_balance=float(client_row.get('avg_monthly_balance_KZT', 0)),
                behavior=behavior,
                product=best_product,
                expected_benefit=expected_benefit,
                cta=cta,
                ref_month=ref_month
            )
            
            push_text = generate_with_guardrails(ollama_model, SYSTEM_PROMPT, user_prompt)
        
        # Fallback to template if LLM generation fails
        if not push_text:
            push_text = self._generate_template_push(
                client_row, behavior, best_product, expected_benefit, ref_month
            )
        
        # Validate and correct
        validation = validate_push(push_text)
        if not validation["ok"]:
            push_text = autocorrect(push_text)
            validation = validate_push(push_text)
        
        return {
            'client_code': client_code,
            'product': best_product,
            'push_notification': push_text,
            'confidence': confidence,
            'expected_benefit': expected_benefit,
            'prediction_method': prediction_method,
            'validation': validation,
            'facts': facts.get(best_product, {})
        }
    
    def _build_behavior_summary(self, tx: pd.DataFrame) -> Dict[str, Any]:
        """Build behavior summary from transactions"""
        if tx is None or tx.empty:
            return {"top_categories": [], "taxi_count": 0, "travel_sum": 0.0}
        
        cat_spend = tx.groupby("category")["amount"].sum().sort_values(ascending=False).to_dict()
        top3 = list(cat_spend)[:3]
        taxi_count = int((tx["category"] == "Такси").sum()) if "category" in tx.columns else 0
        travel_sum = float(tx.loc[tx["category"].isin(["Путешествия","Такси","Отели"]), "amount"].sum())
        
        return {
            "top_categories": top3, 
            "taxi_count": taxi_count, 
            "travel_sum": travel_sum,
            "category_spending": cat_spend
        }
    
    def _generate_template_push(self, client_row: pd.Series, behavior: Dict, 
                               product: str, benefit: float, ref_month: int) -> str:
        """Generate push using template fallback"""
        from prompts import format_kzt, RU_MONTHS_GEN
        
        name = client_row.get('name', 'Клиент')
        month_name = RU_MONTHS_GEN.get(ref_month, "последнем месяце")
        benefit_txt = format_kzt(benefit) if benefit and benefit > 0 else ""
        
        templates = {
            "Карта для путешествий": f"{name}, в {month_name} вы активно ездите на такси и путешествуете. С картой для путешествий вы вернёте до {benefit_txt} кешбэком. Это выгодное решение для ваших привычек. Открыть карту в приложении.",
            "Премиальная карта": f"{name}, у вас стабильно высокий остаток на счету и регулярные траты в ресторанах. Премиальная карта даст вам повышенный кешбэк до 4% и бесплатные снятия наличных. Оформить сейчас.",
            "Кредитная карта": f"{name}, ваши топ-категории трат — {', '.join(behavior.get('top_categories', [])[:3]) or 'продукты и рестораны'}. Кредитная карта даст вам до 10% кешбэка в любимых категориях и на онлайн-сервисы. Оформить карту.",
            "Обмен валют": f"{name}, вы часто платите в иностранной валюте. В нашем приложении выгодный обмен валют без лишних комиссий и авто-покупка по целевому курсу. Настроить обмен в приложении.",
            "Инвестиции": f"{name}, у вас есть свободные средства для роста капитала. Попробуйте инвестиции с низким порогом входа и минимальными комиссиями на старте. Это поможет вам приумножить сбережения. Открыть счёт."
        }
        
        deposit_template = f"{name}, у вас остаются свободные средства на счету. Разместите их на выгодном вкладе — это удобный способ копить деньги и получать стабильное вознаграждение каждый месяц. Открыть вклад в приложении."
        
        if product in templates:
            return templates[product]
        elif "Депозит" in product:
            return deposit_template
        else:
            return f"{name}, для диверсификации портфеля вы можете рассмотреть покупку золотых слитков 999,9 пробы. Это защитный актив, который поможет сохранить ваши сбережения. Посмотреть варианты в приложении."
    
    def process_all_clients(self, 
                           clients_csv: str = "data/clients.csv",
                           ollama_model: str = None,
                           prediction_method: str = "hybrid",
                           output_path: str = "out/push_recommendations_interface.csv") -> pd.DataFrame:
        """
        Process all clients and generate recommendations
        
        Args:
            clients_csv: Path to clients CSV file
            ollama_model: Ollama model name (optional)
            prediction_method: "ml", "rules", or "hybrid"
            output_path: Output CSV path
            
        Returns:
            DataFrame with all results
        """
        from tqdm import tqdm
        
        # Load data
        clients = load_clients(clients_csv)
        tables = load_client_tables()
        
        results = []
        
        print(f"Processing {len(clients)} clients...")
        
        for _, client_row in tqdm(clients.iterrows(), total=len(clients)):
            client_code = int(client_row['client_code'])
            
            # Get client data
            tx = tables.get(client_code, {}).get("tx", pd.DataFrame())
            tr = tables.get(client_code, {}).get("tr", pd.DataFrame())
            
            # Generate recommendation
            result = self.generate_push_notification(
                client_row, tx, tr, ollama_model, prediction_method
            )
            
            results.append(result)
        
        # Create output DataFrame
        output_df = pd.DataFrame([
            {
                'client_code': r['client_code'],
                'product': r['product'],
                'push_notification': r['push_notification']
            }
            for r in results
        ])
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Save detailed results
        detailed_path = output_path.replace('.csv', '_detailed.json')
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")
        print(f"Detailed results saved to: {detailed_path}")
        
        return output_df


def main():
    """CLI interface for the model"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Push Notification Generation Interface")
    parser.add_argument("--clients-csv", default="data/clients.csv", help="Path to clients CSV")
    parser.add_argument("--ollama-model", default=None, help="Ollama model name (e.g., mistral:7b)")
    parser.add_argument("--method", choices=["ml", "rules", "hybrid"], default="hybrid", 
                       help="Prediction method")
    parser.add_argument("--output", default="out/push_recommendations_interface.csv",
                       help="Output CSV path")
    parser.add_argument("--client-id", type=int, help="Process single client by ID")
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = ModelInterface()
    
    if args.client_id:
        # Process single client
        clients = load_clients(args.clients_csv)
        client_row = clients[clients['client_code'] == args.client_id].iloc[0]
        
        tables = load_client_tables()
        tx = tables.get(args.client_id, {}).get("tx", pd.DataFrame())
        tr = tables.get(args.client_id, {}).get("tr", pd.DataFrame())
        
        result = interface.generate_push_notification(
            client_row, tx, tr, args.ollama_model, args.method
        )
        
        print(f"\nClient {args.client_id} Recommendation:")
        print(f"Product: {result['product']}")
        print(f"Push: {result['push_notification']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Expected Benefit: {result['expected_benefit']:.2f} ₸")
        
    else:
        # Process all clients
        results_df = interface.process_all_clients(
            args.clients_csv, args.ollama_model, args.method, args.output
        )
        
        print(f"\nProcessed {len(results_df)} clients")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
