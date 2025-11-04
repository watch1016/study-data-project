
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

st.set_page_config(page_title="요인 선택 → 성적 예측", page_icon="🎯", layout="wide")
st.title("🎯 요인 선택만으로 예측하는 학생 성적 (Random Forest)")

st.markdown(
    '''
    - **원하는 시스템:** 사용자가 성적에 영향을 줄 수 있는 **요인만 선택**하면 (예: 성별, 인종, 부모 교육 수준, 점심, 시험 준비)  
      → 모델이 **수학/읽기/쓰기 점수**를 한 번에 예측합니다.
    - 이 앱은 업로드한 CSV에서 **다섯 가지 요인만**을 특성으로 사용하고, 세 과목 점수를 동시에 예측합니다.
    '''
)

with st.sidebar:
    st.header("1) 데이터 업로드")
    data_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
    st.caption("필수 컬럼: gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, math_score, reading_score, writing_score")

    st.divider()
    st.header("2) 학습 설정")
    test_size = st.slider("테스트 비율", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("RandomForest 나무 개수", 100, 600, 300, 50)
    max_depth = st.slider("최대 깊이 (None=0)", 0, 30, 0, 1)
    random_state = st.number_input("random_state", min_value=0, value=42, step=1)

if data_file is None:
    st.info("왼쪽에서 CSV를 업로드해주세요.")
    st.stop()

# 데이터 로드
try:
    df = pd.read_csv(data_file)
except Exception as e:
    st.error(f"CSV 읽기 오류: {e}")
    st.stop()

required_cols = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "math_score",
    "reading_score",
    "writing_score",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"필수 컬럼 누락: {missing}")
    st.stop()

st.subheader("데이터 미리보기")
st.write("크기:", df.shape)
st.dataframe(df.head(20), use_container_width=True)

# === 특성과 타겟 ===
factor_cols = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]
target_cols = ["math_score", "reading_score", "writing_score"]

X = df[factor_cols].copy()
y = df[target_cols].copy()

# === 파이프라인 구성 ===
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), factor_cols)],
    remainder="drop",
)

base_rf = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=None if max_depth == 0 else max_depth,
    random_state=random_state,
)
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", MultiOutputRegressor(base_rf)),
    ]
)

# === 학습/평가 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=None
)

c1, c2 = st.columns([2,1], gap="large")
with c1:
    if st.button("🚀 모델 학습/평가 실행", use_container_width=True):
        model.fit(X_train, y_train)
        y_pred = pd.DataFrame(model.predict(X_test), columns=target_cols, index=y_test.index)

        # 지표 계산
        metrics = {}
        for col in target_cols:
            r2 = r2_score(y_test[col], y_pred[col])
            rmse = mean_squared_error(y_test[col], y_pred[col], squared=False)
            metrics[col] = {"R2": r2, "RMSE": rmse}

        st.success("학습 완료!")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Math R²", f"{metrics['math_score']['R2']:.3f}")
        mcol2.metric("Reading R²", f"{metrics['reading_score']['R2']:.3f}")
        mcol3.metric("Writing R²", f"{metrics['writing_score']['R2']:.3f}")

        st.caption("RMSE (↓ 낮을수록 좋음)")
        st.write({k: round(v['RMSE'], 3) for k, v in metrics.items()})

        # 모델 저장
        joblib.dump(
            {
                "model": model,
                "factor_cols": factor_cols,
                "target_cols": target_cols,
            },
            "factors_only_model.pkl",
        )
        with open("factors_only_model.pkl", "rb") as f:
            st.download_button(
                "💾 모델 다운로드 (factors_only_model.pkl)",
                data=f.read(),
                file_name="factors_only_model.pkl",
                mime="application/octet-stream",
            )

st.divider()
st.header("🧮 요인 선택 → 세 과목 점수 예측")

# 사용 가능한 카테고리 옵션 수집
options = {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in factor_cols}

with st.form("predict_form"):
    user_input = {}
    st.subheader("내 요인 선택")
    for c in factor_cols:
        user_input[c] = st.selectbox(c, options[c], key=f"sel_{c}")

    submitted = st.form_submit_button("📈 예측 실행")
    if submitted:
        # 저장된 모델이 있으면 사용하고 없으면 즉석 학습
        if os.path.exists("factors_only_model.pkl"):
            bundle = joblib.load("factors_only_model.pkl")
            model = bundle["model"]
            used_factors = bundle["factor_cols"]
            used_targets = bundle["target_cols"]
        else:
            model.fit(X, y)
            used_factors = factor_cols
            used_targets = target_cols

        input_df = pd.DataFrame([user_input], columns=used_factors)
        pred = model.predict(input_df)[0]
        pred = np.clip(pred, 0.0, 100.0)  # 점수 범위 보정

        # 간단한 예측 불확실성(트리별 분산)을 표시 (정보용)
        try:
            per_target_stds = []
            # MultiOutputRegressor 내부의 각 타겟별 RF에서 tree 예측의 분산을 측정
            transformed = model.named_steps["preprocessor"].transform(input_df)
            for est in model.named_steps["regressor"].estimators_:
                tree_preds = np.array([t.predict(transformed) for t in est.estimators_]).ravel()
                per_target_stds.append(float(np.std(tree_preds)))
        except Exception:
            per_target_stds = [np.nan, np.nan, np.nan]

        st.success("예측 결과")
        res = pd.DataFrame([pred], columns=used_targets, index=["예상 점수"]).T
        res.columns = ["예측값"]
        if not np.isnan(per_target_stds).any():
            res["(대략)표준편차"] = [round(s, 2) for s in per_target_stds]
        st.dataframe(res, use_container_width=True)

st.caption("※ 이 모델은 요인만으로 예측하므로, 실제 점수는 학습 데이터 분포와 환경에 따라 달라질 수 있습니다.")
