import joblib
import pandas as pd
import numpy as np

def load_revenue_model(model_path='revenue_model.joblib'):
    """리뷰 성장을 예측하는 모델을 로드합니다."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: '{model_path}' 파일을 찾을 수 없습니다. 먼저 revenue_prediction_training.py를 실행하여 모델을 학습시켜주세요.")
        return None

def predict_revenue_growth(model, ai_adoption_rate, productivity_change_percent):
    """
    AI 도입률과 생산성 변화를 바탕으로 매출 성장률을 예측합니다.
    
    Args:
        model: 학습된 LightGBM 모델
        ai_adoption_rate (float): AI 도입률 (0-100)
        productivity_change_percent (float): 생산성 변화율 (%)
        
    Returns:
        float: 예측된 매출 성장률 (%)
    """
    # 모델 입력 형식에 맞춰 데이터프레임 생성
    input_df = pd.DataFrame([{
        'ai_adoption_rate': ai_adoption_rate,
        'productivity_change_percent': productivity_change_percent
    }])
    
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    # 1. 모델 로드
    model = load_revenue_model()
    
    if model:
        print("--- Revenue Growth Prediction Example ---")
        
        # 2. 예시 데이터로 예측 수행
        test_cases = [
            {'adoption': 20.0, 'productivity': 5.0},
            {'adoption': 50.0, 'productivity': 10.0},
            {'adoption': 85.0, 'productivity': 25.0}
        ]
        
        for case in test_cases:
            pred = predict_revenue_growth(model, case['adoption'], case['productivity'])
            print(f"Input: [AI 도입률 {case['adoption']}%, 생산성 변화 {case['productivity']}%] -> Predicted Revenue Growth: {pred:.2f}%")
        
        print("\n--- Interactive Prediction ---")
        try:
            user_adoption = float(input("AI 도입률을 입력하세요 (0-100): "))
            user_productivity = float(input("생산성 변화율을 입력하세요 (%): "))
            
            user_pred = predict_revenue_growth(model, user_adoption, user_productivity)
            print(f"예측된 매출 성장률: {user_pred:.2f}%")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
