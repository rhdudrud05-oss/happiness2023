import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm

st.set_page_config(page_title="분석2: 한국행복조사 개인 수준 분석", layout="wide")

# ── 데이터 로딩 ──────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect("happiness2023.db")
    df = pd.read_sql("SELECT * FROM happiness", conn)
    conn.close()
    return df

df = load_data()
order = ['하', '중하', '중', '중상', '상']

# ── 제목 ─────────────────────────────────────────────
st.title("분석2: 개인 수준에서 소득·음주·단체참여는 연결되는가?")
st.markdown("""
**데이터:** 한국인의 행복조사 2023 (KOSSDA)  
**대상:** MZ세대 (19~39세), n=4,626명  
**목적:** 음주가 낮은 저소득층과 참여가 낮은 저소득층이 같은 사람인지 확인
""")

st.divider()

# ── 0. 분석 배경 ──────────────────────────────────────
st.header("0. 분석 배경 및 한계")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 처음 시도: 소득을 조절변수로 설정")
    st.markdown("""
    - 가설: 소득이 낮을수록 음주→단체참여 관계가 더 강할 것
    - 상호작용항(음주 × 소득) 투입
    """)
    st.error("결과: 조절효과 유의하지 않음 (β=-0.015, p=0.361)")

with col2:
    st.markdown("#### 재설정: 소득을 독립변수로")
    st.markdown("""
    - 소득과 음주를 각각 독립적 예측변수로 설정
    - 둘 다 단체참여에 유의한 영향을 미치는지 확인
    """)
    st.success("결과: 소득(p=0.0002), 음주(p<0.001) 모두 유의")

st.divider()

# ── 1. SQL JOIN 결과 ──────────────────────────────────
st.header("1. SQL 분석 결과")

sql_code = """
-- 소득분위별 음주빈도·단체참여 평균
SELECT 소득분위,
       COUNT(*) as 인원수,
       ROUND(AVG(음주빈도), 3) as 평균음주빈도,
       ROUND(AVG(단체참여), 3) as 평균단체참여
FROM happiness
GROUP BY 소득분위
ORDER BY CASE 소득분위
    WHEN '하' THEN 1 WHEN '중하' THEN 2
    WHEN '중' THEN 3 WHEN '중상' THEN 4 WHEN '상' THEN 5
END;
"""
st.code(sql_code, language='sql')

conn = sqlite3.connect("happiness2023.db")
df_sql = pd.read_sql("""
SELECT 소득분위, COUNT(*) as 인원수,
       ROUND(AVG(음주빈도),3) as 평균음주빈도,
       ROUND(AVG(단체참여),3) as 평균단체참여
FROM happiness GROUP BY 소득분위
ORDER BY CASE 소득분위
    WHEN '하' THEN 1 WHEN '중하' THEN 2
    WHEN '중' THEN 3 WHEN '중상' THEN 4 WHEN '상' THEN 5 END
""", conn)
conn.close()
st.dataframe(df_sql, use_container_width=True)

st.divider()

# ── 2. 소득분위별 음주·참여 시각화 ────────────────────
st.header("2. 소득분위별 음주빈도·단체참여 평균")

df_group = df.groupby('소득분위')[['음주빈도', '단체참여']].mean().loc[order].reset_index()

col1, col2 = st.columns(2)
with col1:
    fig1 = px.bar(df_group, x='소득분위', y='음주빈도',
                  color='소득분위',
                  color_discrete_sequence=px.colors.sequential.Blues[2:],
                  title='소득분위별 평균 음주빈도<br><sup>1=전혀 안 마심, 6=주 4회 이상</sup>',
                  category_orders={'소득분위': order})
    fig1.update_layout(showlegend=False, yaxis=dict(range=[2.5, 3.5]))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(df_group, x='소득분위', y='단체참여',
                  color='소득분위',
                  color_discrete_sequence=px.colors.sequential.Greens[2:],
                  title='소득분위별 평균 단체참여<br><sup>1=소속 없음, 5=매우 활발히 참여</sup>',
                  category_orders={'소득분위': order})
    fig2.update_layout(showlegend=False, yaxis=dict(range=[1.6, 2.2]))
    st.plotly_chart(fig2, use_container_width=True)

st.caption("""
**해석:** 단체참여는 소득 상위층(1.973)이 하위층(1.814)보다 유의하게 높음.  
음주빈도는 소득분위 간 단조적 증가 패턴이 뚜렷하지 않으나, ANOVA 결과 집단 간 차이 유의(F=4.706, p=0.001).
""")

st.divider()

# ── 3. 비음주 저소득층 단체참여 ──────────────────────
st.header("3. '음주 낮은 저소득층 = 참여 낮은 저소득층'인가?")

df_drink = df.copy()
df_drink['음주여부'] = df_drink['음주빈도'].apply(lambda x: '비음주' if x == 1 else '음주')
df_cross = df_drink.groupby(['소득분위', '음주여부'])['단체참여'].mean().reset_index()
df_cross['소득분위'] = pd.Categorical(df_cross['소득분위'], categories=order, ordered=True)
df_cross = df_cross.sort_values('소득분위')

fig3 = px.bar(df_cross, x='소득분위', y='단체참여',
              color='음주여부',
              barmode='group',
              color_discrete_map={'비음주': '#e74c3c', '음주': '#2980b9'},
              title='소득분위 × 음주여부별 평균 단체참여',
              category_orders={'소득분위': order})
st.plotly_chart(fig3, use_container_width=True)

st.caption("""
**핵심 발견:** 모든 소득분위에서 비음주자의 단체참여가 음주자보다 낮음.  
특히 소득 하위 비음주자(1.481)는 전체 집단 중 단체참여가 가장 낮음.  
→ 음주가 낮고 소득도 낮은 집단이 참여도 가장 낮다는 패턴 확인.
""")

st.divider()

# ── 4. ANOVA ──────────────────────────────────────────
st.header("4. ANOVA: 소득분위 간 차이가 통계적으로 유의한가?")

groups_g6 = [df[df['소득분위']==g]['음주빈도'].dropna() for g in order]
f_g6, p_g6 = stats.f_oneway(*groups_g6)
groups_d3 = [df[df['소득분위']==g]['단체참여'].dropna() for g in order]
f_d3, p_d3 = stats.f_oneway(*groups_d3)

col1, col2 = st.columns(2)
with col1:
    st.metric("음주빈도 F값", f"{f_g6:.3f}")
    st.metric("음주빈도 p값", f"{p_g6:.4f}")
    st.success("소득분위 간 음주빈도 차이 유의 (p<0.05)")
with col2:
    st.metric("단체참여 F값", f"{f_d3:.3f}")
    st.metric("단체참여 p값", f"{p_d3:.4f}")
    st.success("소득분위 간 단체참여 차이 유의 (p<0.05)")

st.divider()

# ── 5. 회귀분석 ──────────────────────────────────────
st.header("5. OLS 회귀분석: 소득·음주 → 단체참여")

tab1, tab2 = st.tabs(["MZ세대 전체", "20대"])

for tab, label, age_range in zip([tab1, tab2], ['MZ세대 전체', '20대'],
                                  [(19, 39), (20, 29)]):
    with tab:
        df_sub = df[df['연령'].between(*age_range)][['소득코드', '음주빈도', '단체참여']].dropna()
        X = sm.add_constant(df_sub[['소득코드', '음주빈도']])
        model = sm.OLS(df_sub['단체참여'], X).fit()

        col1, col2, col3 = st.columns(3)
        col1.metric("R²", f"{model.rsquared:.3f}")
        col2.metric("소득 β (p값)", f"{model.params['소득코드']:.4f} (p={model.pvalues['소득코드']:.4f})")
        col3.metric("음주 β (p값)", f"{model.params['음주빈도']:.4f} (p={model.pvalues['음주빈도']:.4f})")

        st.success(f"소득과 음주빈도 모두 단체참여에 유의한 정적 영향 (p<0.05)")
        st.caption(f"n={len(df_sub)}명 | R²=0.022로 낮으나 두 변수 모두 통계적으로 유의함. 단체참여에는 본 분석 외 다양한 요인이 작용함.")

st.divider()

# ── 6. 분석 요약 ─────────────────────────────────────
st.header("6. 분석2 요약")
st.info("""
**핵심 발견:**

1. **소득분위 간 차이 유의** (ANOVA): 소득분위에 따라 음주빈도(F=4.706, p=0.001)와 단체참여(F=4.022, p=0.003) 모두 유의한 차이

2. **소득·음주 → 단체참여** (회귀분석): 소득(β=0.029, p=0.0002)과 음주빈도(β=0.106, p<0.001) 모두 단체참여에 유의한 정적 영향

3. **비음주 저소득층의 이중 취약성**: 소득 하위 + 비음주 집단의 단체참여(1.481)가 전체 집단 중 최저

**결론:** 음주가 낮은 저소득층과 참여가 낮은 저소득층은 동일 집단임이 개인 수준에서 확인됨.  
저소득 20대에게 음주는 단체참여의 주요 경로이며, 금주 트렌드는 고소득층에게는 '대안으로 이동'이지만 저소득층에게는 '고립'으로 이어질 수 있음.

**한계:** R²=0.022로 설명력이 낮아 단체참여에는 본 분석 외 다양한 요인이 존재함을 인정.  
또한 소득 조절효과는 유의하지 않아(p=0.361) 소득을 독립변수로 재설정하여 분석함.
""")
