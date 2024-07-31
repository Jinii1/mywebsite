# !pip install pyreadstat
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


raw_welfare=pd.read_spss('data/Koweps_hpwc14_2019_beta2 (1).sav')
raw_welfare

welfare=raw_welfare.copy()
welfare.shape
welfare.describe()

welfare=welfare.rename(
    columns = {
        "h14_g3": "sex",
        "h14_g4": "birth",
        "h14_g10": "marriage_type",
        "h14_g11": "religion",
        "p1402_8aq1": "income",
        "h14_eco9": "code_job",
        "h14_reg7": "code_region"
    }
)
welfare.info()

welfare=welfare[["sex", "birth", "marriage_type",
                "religion", "income", "code_job", "code_region"]]
welfare.shape

# 9-2 성별에 따른 월급 차이 p.229
# 변수 검토 및 전처리(성별, 월급) -> 변수 간 관계 분석 (성별 월급 평균표, 그래프)

# 1. 성별 변수 검토 및 전처리
welfare["sex"].dtypes
6
# welfare["sex"].isna().sum()

# 결측치가 있다면
# welfare['sex']=np.where(welfare['sex'] == 9, np.nan, welfare['sex'])

welfare["sex"] = np.where(welfare["sex"] == 1,'male', 'female')
welfare["sex"].value_counts()

# 2. 월급 변수 검토 및 전처리
welfare["income"].describe()
welfare["income"].isna().sum()

# 3. 성별 월급 평균표 만들기
sex_income = welfare.dropna(subset='income') \
                    .groupby('sex', as_index=False) \
                    .agg(mean_income = ('income', 'mean'))

sex_income

# 4. 그래프 만들기
import seaborn as sns
sns.barplot(data=sex_income, x="sex", y="mean_income",
            hue="sex")
plt.show()
plt.clf()


===================================================================================

# 숙제: 위 그래프에서 각 성별 95% 신뢰구간 계산후 그리기
# 위 아래 검정색 막대기로 표시

sex_income = welfare.dropna(subset='income') \
                    .groupby('sex', as_index=False) \
                    .agg(mean_income = ('income', 'mean'),
                         std_income=('income','std'),
                         count_income=('income', 'count'))
sex_income

# 각 성별 loc 구하기
female_mean=sex_income.iloc[0,1]
male_mean=sex_income.iloc[1,1]
female_mean
male_mean

# 각 성별 scale 구하기
female_std=sex_income.iloc[0,2]
male_std=sex_income.iloc[1,2]
female_std
male_std

# 각 성별 표본갯수 구하기
female_count=sex_income.iloc[0,3]
male_count=sex_income.iloc[1,3]
female_count
male_count

# 95% 신뢰구간 계산
from scipy.stats import norm
z_005=norm.ppf(0.95, loc=0, scale=1)
ci_female_1=female_mean - z_005 * female_std / np.sqrt(female_count)
ci_female_2=female_mean + z_005 * female_std / np.sqrt(female_count)
ci_female_1
ci_female_2

ci_male_1=male_mean - z_005 * male_std / np.sqrt(male_count)
ci_male_2=male_mean + z_005 * male_std / np.sqrt(male_count)
ci_male_1
ci_male_2

# 신뢰구간을 위 그래프에 검정색 막대기로 표시
sns.barplot(data = sex_income, x = 'sex', y = 'mean_income', hue = 'sex')
plt.plot([0, 0], [ci_female_1, ci_female_2], color='black')
plt.plot([1, 1], [ci_male_1, ci_male_2], color='red')
plt.show()
plt.clf()

===================================================================================
# 9-3 나이와 월급의 관계

# 1. 나이 변수 검토 및 전처리
welfare["birth"].describe()
sns.histplot(data=welfare, x="birth")
plt.show()
plt.clf()

welfare["birth"].isna().sum()

# 나이 파생변수 만들기
welfare=welfare.assign(age = 2019 - welfare["birth"] + 1)
welfare["age"]
sns.histplot(data=welfare, x="age")
plt.show()
plt.clf()

# 나이에 따른 월급 평균표 만들기
age_income=welfare.dropna(subset="income") \
                    .groupby("age", as_index=False) \
                    .agg(mean_income = ("income", "mean"))

sns.lineplot(data=age_income, x="age", y="mean_income")
plt.show()
plt.clf()

# 나이별 income 칼럼 na 개수 세기!
# welfare['income'] 결측치 개수를 나이별로 집계하여 새로운 데이터프레임을 생성
welfare["income"].isna().sum()

my_df=welfare.assign(income_na=welfare["income"].isna()) \
                        .groupby("age", as_index=False) \
                        .agg(n = ("income_na", "sum"))


sns.barplot(data = my_df, x="age", y="n")
plt.show()
plt.clf()

# 9-4 연령대별 수입 분석
# cut
bin_cut=np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare=welfare.assign(age_group = pd.cut(welfare["age"], 
                bins=bin_cut, 
                labels=(np.arange(12) * 10).astype(str) + "대"))

# np.version.version
# (np.arange(12) * 10).astype(str) + "대"

age_income=welfare.dropna(subset="income") \
                    .groupby("age_group", as_index=False) \
                    .agg(mean_income = ("income", "mean"))

age_income
sns.barplot(data=age_income, x="age_group", y="mean_income")
plt.show()
plt.clf()

# 판다스 데이터 프레임을 다룰 때, 변수의 타입이
# 카테고리로 설정되어 있는 경우, groupby+agg 콤보
# 안먹힘. 그래서 object 타입으로 바꿔 준 후 수행
welfare["age_group"]=welfare["age_group"].astype("object")

def my_f(vec):
    return vec.sum()

sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: my_f(x)))
    
sex_age_income


sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: np.quantile(x, q=0.96))
    
    


sex_age_income

sns.barplot(data=sex_age_income,
            x="age_group", y="mean_income", 
            hue="sex")
plt.show()
plt.clf()

# 연령대별, 성별 상위 4% 수입 찾아보세요!

import pandas as pd

# 예제 데이터 프레임 생성
data = {
    'income': [50000, 60000, 70000, None, 80000, 90000, None, 100000, 110000],
    'age_group': ['20대', '30대', '40대', '20대', '30대', '40대', None, '20대', '30대'],
    'sex': ['M', 'F', 'M', 'F', None, 'M', 'F', 'M', 'F']
}
welfare = pd.DataFrame(data)
