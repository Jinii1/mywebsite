# !pip install pyreadstat
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


raw_welfare=pd.read_spss('Koweps_hpwc14_2019_beta2 (1).sav')
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

# welfare["sex"].isna().sum()

# 결측치가 있다면
# welfare['sex']=np.where(welfare['sex'] == 9, np.nan, welfare['sex'])

welfare["sex"] = np.where(welfare["sex"] == 1,'male', 'female')
welfare["sex"].value_counts() # 열에 있는 각 고유 값의 발생 빈도를 계산

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

# 9-3 나이와 월급의 관계

# 1. 나이 변수 검토 및 전처리
welfare['birth'].describe()
sns.histplot(data=welfare, x="birth")
plt.show()
plt.clf()

welfare["birth"].isna().sum()

# 나이 파생변수 만들기
welfare=welfare.assign(age=2019-welfare['birth']+1)
sns.histplot(data=welfare, x="age")
plt.show()
plt.clf()

# 나이에 따른 월급 평균표 만들기
age_income=welfare.dropna(subset='income') \
                  .groupby('age', as_index=False) \
                  .agg(mean_income=('income', 'mean'))
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
# cut
bin_cut=np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare=welfare.assign(
    age_group = pd.cut(welfare["age"], 
                       bins=bin_cut,
                       labels=(np.arange(12) * 10).astype(str) + "대")
)

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

# 연령대와 성별에 따라 그룹화하여 각 그룹의 수입 합계 계산
def my_f(vec):
    return vec.sum()

sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: my_f(x)))
    
sex_age_income

# 연령대별, 성별 상위 4% 수입 찾아보세요!
sex_age_income = \
    welfare.dropna(subset="income") \
    .groupby(["age_group", "sex"], as_index=False) \
    .agg(top4per_income=("income", lambda x: np.quantile(x, q=0.96))) # 분위수
    
sex_age_income

sns.barplot(data=sex_age_income,
            x="age_group", y="top4per_income", 
            hue="sex")
plt.show()
plt.clf()

import pandas as pd

# 예제 데이터 프레임 생성
data = {
    'income': [50000, 60000, 70000, None, 80000, 90000, None, 100000, 110000],
    'age_group': ['20대', '30대', '40대', '20대', '30대', '40대', None, '20대', '30대'],
    'sex': ['M', 'F', 'M', 'F', None, 'M', 'F', 'M', 'F']
}
welfare = pd.DataFrame(data)
====================================================================================================
# 0731
# 9-6 직업별 월급 차이
# 1. 작업 변수 검토 및 전처리
# code_job 변수의 값은 직업 코드를 의미 -> 코드가 어떤 직업을 의미하는지 몰라서
welfare['code_job']
welfare['code_job'].value_counts()

# 코드북의 직업분류코드 목록을 이용해 직업 이름을 나타낸 변수 만듦
list_job=pd.read_excel('C:/Users/USER/Documents/LS빅데이터스쿨/mywebsite/posts/hw5/Koweps_Codebook_2019.xlsx', sheet_name='직종코드')
list_job.head()

# merge()를 이용해 list_job을 welfare에 결합
welfare=welfare.merge(list_job,
                        how='left', on='code_job')
welfare.dropna(subset=['job', 'income'])[['income', 'job']]

# 2. 직업별 월급 차이 분석
# 직업별 월급 평균표 만들기
job_income=welfare.dropna(subset=['job', 'income']) \
                  .groupby('job', as_index=False) \
                  .agg(mean_income=('income', 'mean'))
job_income.head()

# 그래프 만들기
# 월급이 많은 직업 상위 10
p10=job_income.sort_values('mean_income', ascending=False).head(10)
top10

import matplotlib.pyplot as plt
# 한글 잘 보이게 맑은 고딕 폰트 설정
plt.rcParams.update({'font.family': 'Malgun Gothic'})

sns.barplot(data=top10, y='job', x='mean_income', hue='job')
plt.show()
plt.clf()

# 월급이 적은 직업 상위 10
bottom10=job_income.sort_values('mean_income', ascending = True).head(10)
bottom10

sns.barplot(data=bottom10, y='job', x='mean_income', hue='job') \
   .set(xlim=[0, 800])
plt.show()
plt.clf()

# 9-7 성별 직업 빈도 (성별에 따라 어떤 직업이 가장 많을까?)

# 남자
job_male=welfare.dropna(subset='job') \
                .query('sex == "male"') \
                .groupby('job', as_index=False) \
                .agg(n=('job', 'count')) \
                .sort_values('n', ascending=False) \
                .head(10)
job_male

sns.barplot(data=job_male, y='job', x='n', hue='job').set(xlim=[0, 500])
plt.show()
plt.clf()

# 여자
job_female=welfare.dropna(subset='job') \
                .query('sex == "female"') \
                .groupby('job', as_index=False) \
                .agg(n=('job', 'count')) \
                .sort_values('n', ascending=False) \
                .head(10)
job_female

sns.barplot(data=job_female, y='job', x='n', hue='job').set(xlim=[0, 500])
plt.show()
plt.clf()

# 9-8 종교 유무에 따른 이혼율 (종교가 있으면 이혼을 덜할까?)
welfare['religion']=np.where(welfare['religion'] == 1, 'yes', 'no')

welfare['marriage']=np.where(welfare['marriage_type']==1, 'marriage',
                    np.where(welfare['marriage_type']==3, 'divorce', 'etc'))

# 이혼 여부 별 빈도
n_divorce=welfare.groupby('marriage', as_index=False) \
                            .agg(n=('marriage', 'count'))
                            
# p. 263

df=welfare.query('marriage_type!=5') \
    .groupby('religion', as_index=False) \
    ['marriage_type'] \
    .value_counts(normalize=True) # 핵심: 비율 구하기

df['marriage_type'].value_counts
    
# proportion에 100을 곱해서 백분율로 바꾸기
df.query('marriage_type == 1') \
    .assign(proportion=df['proportion']*100).round(1)
