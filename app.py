app.py
# ------------------------------------------------------------
# 🎯 요인 선택 → 세 과목 점수 예측 (Streamlit + RandomForest)
#   - 입력: 성별/인종/부모학력/점심/시험준비(=범주형 요인만)
#   - 출력: math_score, reading_score, writing_score 동시 예측
#   - 업로드한 CSV 기반으로 학습 및 예측
# ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------- Streamlit 기본 설정 ----------
st.set_page_config(page_title="요인 선택 → 성적 예측", page_icon="🎯", layout="wide")
st.title("🎯 요인만 선택해서 예측하는 학생 성적 (Random Forest)")

st.markdown(
    """
    아래 **CSV**를 업로드한 뒤, 사이드바에서 학습 버튼을 누르고  
    본문 폼에서 **요인만 선택**하면 세 과목 점수를 한 번에 예측합니다.
    """
)

# ---------- 사이드바 ----------
with st.sidebar:
    st.header("1) 데이터 업로드")
    file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
    st.caption(
        "필수 컬럼: "
        "`gender`, `race_ethnicity`, `parental_level_of_education`, "
        "`lunch`, `test_preparation_course`, "
        "`math_score`, `reading_score`, `writing_score`"
    )

    st.divider()
    st.header("2) 학습 설정")
    test_size = st.slider("테스트 비율", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("RandomForest 나무 개수", 100, 600, 300, 50)
    max_depth_ui = st.slider("최대 깊이 (0이면 None)", 0, 30, 0, 1)
    random_state = st.number_input("random_state", min_value=0, value=42, step=1)

# ---------- 파일 체크 ----------
if file is None:
    st.info("⬅️ 사이드바에서 CSV를 업로드해주세요.")
    st.stop()

# ---------- 데이터 로드 ----------
try:
    df = pd.read_csv(file)
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

# ---------- 특성/타겟 정의 (요인만 사용) ----------
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

# ---------- 파이프라인 ----------
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), factor_cols)],
    remainder="drop",
)
base_rf = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=None if max_depth_ui == 0 else max_depth_ui,
    random_state=random_state,
)
model = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("rf_multi", MultiOutputRegressor(base_rf)),
    ]
)

# ---------- 학습/평가 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

trained = False
if st.button("🚀 모델 학습/평가 실행", type="primary"):
    model.fit(X_train, y_train)
    y_pred = pd.DataFrame(model.predict(X_test), columns=target_cols, index=y_test.index)

    # 과목별 지표
    r2s, rmses = {}, {}
    for col in target_cols:
        r2s[col] = r2_score(y_test[col], y_pred[col])
        rmses[col] = mean_squared_error(y_test[col], y_pred[col], squared=False)

    st.success("모델 학습 완료!")
    c1, c2, c3 = st.columns(3)
    c1.metric("Math R²", f"{r2s['math_score']:.3f}")
    c2.metric("Reading R²", f"{r2s['reading_score']:.3f}")
    c3.metric("Writing R²", f"{r2s['writing_score']:.3f}")

    st.caption("RMSE (↓ 낮을수록 좋음)")
    st.write({k: round(v, 3) for k, v in rmses.items()})

    # 간단한 캐시 저장 (세션 상태에 담아 예측에서 재사용)
    st.session_state["trained_model"] = model
    st.session_state["factor_cols"] = factor_cols
    st.session_state["target_cols"] = target_cols
    trained = True

# ---------- 예측 UI ----------
st.divider()
st.header("🧮 요인 선택 → 세 과목 점수 예측")

# 각 요인의 선택지(카테고리) 구성
options = {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in factor_cols}

with st.form("predict_form"):
    st.subheader("내 요인 선택")
    user_input = {}
    for c in factor_cols:
        # 값이 하나뿐이어도 selectbox는 동작하도록 기본 index=0
        user_input[c] = st.selectbox(c, options[c], key=f"sel_{c}")

    if st.form_submit_button("📈 예측 실행"):
        # 학습된 모델이 세션에 없으면 전체 데이터로 즉석 학습
        if "trained_model" in st.session_state:
            use_model = st.session_state["trained_model"]
            used_factors = st.session_state["factor_cols"]
            used_targets = st.session_state["target_cols"]
        else:
            # 즉석 학습
            model.fit(X, y)
            use_model = model
            used_factors = factor_cols
            used_targets = target_cols

        input_df = pd.DataFrame([user_input], columns=used_factors)
        pred = use_model.predict(input_df)[0]
        pred = np.clip(pred, 0.0, 100.0)  # 점수는 0~100 범위로 보정

        st.success("예측 결과")
        out = pd.DataFrame([pred], columns=used_targets, index=["예상 점수"]).T
        out.columns = ["예측값"]
        st.dataframe(out, use_container_width=True)

# ---------- 사용법 & 주의 ----------
with st.expander("ℹ️ 사용법 & 주의사항"):
    st.markdown(
        """
        - 이 앱은 **범주형 요인만**으로 예측합니다(다른 과목 점수는 사용하지 않음).  
        - 예측 품질은 **학습 데이터량/분포**에 크게 좌우됩니다.  
        - 데이터가 작거나 불균형하면 R²가 낮거나 RMSE가 커질 수 있습니다.  
        - 점수는 보기 좋게 **0~100 범위로 클리핑**하여 표시합니다.
        """
    )
