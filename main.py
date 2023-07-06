import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False
import streamlit as st
import plotly
import plotly.express as px
import plotly.graph_objects as go
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pymongo
import glob
from sklearn.model_selection import train_test_split
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots

st.set_page_config(page_title='Binary Classification',  layout='wide', page_icon='chart_with_upwards_trend')

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}`
.name{
    font-family:Malgun Gothic;
    font-size:18px !important;
}
.red{
    font-family:Malgun Gothic;
    color:red;
    font-size:30px;
}
.blue{
    font-family:Malgun Gothic;
    color:blue;
    font-size:30px;
}
.title{
    font-family:Malgun Gothic;
    font-size:60px;
}

</style>
""", unsafe_allow_html=True) 

# title
t1, t2 = st.columns((0.09,1)) 
t1.image('images/심볼마크.jpg', width = 145)
t2.markdown('<p class = "title"> Binary Classification </p>', unsafe_allow_html=True)
t2.markdown("website : www.idalab.ac.kr")

st.sidebar.markdown('<p class = "name"> 7조 KKT팀 </p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class = "name"> 총괄팀장 : 장성호 </p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class = "name"> 분석팀장 : 정원희 </p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class = "name"> 개발팀장 : 김수연 </p>', unsafe_allow_html=True)
st.sidebar.markdown(' ')

option = st.sidebar.radio('데이터 선택', ['내장 데이터 사용하기','DB에서 불러오기', 'Local에서 불러오기'])
# st.sidebar.markdown(":red[빨강]")
# st.sidebar.markdown(":green[초록]")

@st.cache_data(ttl=24*60*60)
def load_db(db_name, data_name):
    uri = "개인 DB 주소 연동"
    client = MongoClient(uri, server_api=ServerApi('1'))

    database = client[db_name]
    name = 'database.' + data_name
    collection = eval(name)
    df = pd.DataFrame(list(collection.find()))
    return df

@st.cache_data(ttl=24*60*60)
def load_data(data_name):
    df = pd.read_csv(data_name, encoding = 'cp949')
    return df

@st.cache_data(ttl=24*60*60)
def page1():
    return df

@st.cache_resource(ttl=24*60*60)
def corr_plot(df, size1, size2, annot_select):
    fig = plt.figure(figsize=(size1,size2))
    sns.heatmap(df.corr(), annot=annot_select, cmap='Blues', fmt='.2f', linewidths=.5)
    return st.pyplot(fig)

@st.cache_resource(ttl=24*60*60)
def box_plot(df, x, y):
    if df[x].dtype == 'object':
        return st.error('x축 변수는 숫자형이어야 합니다.')
    else:
        fig_box = px.box(df, x=y, y=x, color=y, width = 1000, height = 600)
        return fig_box

@st.cache_resource(ttl=24*60*60)
def hist_plot(df, x, y):
    fig_hist = px.histogram(df, x=x, color = y, width = 1000, height = 600)
    fig_hist.update_layout(barmode='stack')
    st.plotly_chart(fig_hist)


with st.spinner('Updating Report...'):

    df = None
    if option == 'Local에서 불러오기':
        upload = st.file_uploader('파일 업로드', type = 'csv')
        if upload is not None:
            df = pd.read_csv(upload)
            st.write(df)
        else:
            st.error('데이터를 업로드해주세요')

    elif option == '내장 데이터 사용하기':
        file_list = glob.glob('./data/*.csv')
        directory = './data'
        file_names = [os.path.basename(path) for path in file_list]
        df_name = st.selectbox('데이터 선택', file_names)
        file_path = os.path.join(directory, df_name)  

        df = load_data(file_path)
        st.write(df)

    st.session_state['df'] = df

    if df is None:
        pass
    else:
        st.success('데이터가 성공적으로 업로드 되었습니다.')
        default_index = len(df.columns)
        y = st.selectbox(':red[타겟 변수 선택]', df.columns, index = default_index-1)

        if option == '내장 데이터 사용하기':
            if df_name == 'fordA_total.csv':
                with st.expander('Data info'):
                    st.markdown('Ford A 분석에 사용된 변수명은 Sensor 1~100입니다.')
                    st.markdown('Ford A 데이터는 Ford Classification Challenge에서 제공한 오픈 데이터셋으로부터 수집했습니다.')
                    st.markdown('Ford A 데이터의 확장자는 ARFF, text입니다.')
                    st.markdown('Ford A 데이터는 4,921개의 열, 500개의 columns을 가지고 있으며, 데이터의 총량은 30MB입니다.')
                    st.markdown('Ford A 데이터는 정상 데이터는 1, 비정상 데이터는 -1로 labeling 되어 있습니다.')
                    st.markdown('Ford A 데이터는 Sensor로부터 얻은 자동자 시스템의 주요 데이터로 시간에 따른 정상/비정상 값의 특성을 분석하고, 가공된 데이터를 학습하여 AI 모델을 개발하여 불량 제품을 분류합니다.')
                    st.markdown('Ford A 데이터를 통해 개발된 AI 모델은 유사한 공정에 적용될 수 있으며, 정상/비정상 상태에 대해 정확한 Labaling 작업을 통하여 보다 정확한 분석이 가능할 것으로 예상됩니다.')
                    st.markdown('Ford A와 Ford B 데이터는 비슷하게 생겼지만, 각 데이터 문제를 잘 해결한 SOTA알고리즘은 다릅니다.')
                    st.markdown('직접 한번 해보시는 것을 추천하겠습니다.')

            

            elif df_name == 'fordB_total.csv':
                with st.expander('Data info'):
                    st.markdown('해당 데이터는 Ford B 데이터 셋입니다.')
                    st.markdown('Ford B 데이터는 아래 URL에서 다운 및 기초적인 설명을 볼 수 있습니다.')
                    st.markdown('http://www.timeseriesclassification.com/description.php?Dataset=FordB')
                    st.markdown('Ford B 분석에 사용된 변수명은 Sensor 1~100입니다.')
                    st.markdown('Ford B 데이터는 Ford Classification Challenge에서 제공한 오픈 데이터셋으로부터 수집했습니다.')
                    st.markdown('Ford B 데이터의 확장자는 ARFF, text입니다.')
                    st.markdown('Ford B 데이터는 4,445개의 열을 가지고 있으며, 500개의 columns을 가지고 있습니다.')
                    st.markdown('Ford B 데이터는 정상 데이터는 1, 비정상 데이터는 -1로 labeling 되어 있습니다.')
                    st.markdown('Ford B 데이터는 Sensor로부터 얻은 자동자 시스템의 주요 데이터로 시간에 따른 정상/비정상 값의 특성을 분석하고, 가공된 데이터를 학습하여 AI 모델을 개발하여 불량 제품을 분류합니다.')
                    st.markdown('Ford B 데이터를 통해 개발된 AI 모델은 유사한 공정에 적용될 수 있으며, 정상/비정상 상태에 대해 정확한 Labaling 작업을 통하여 보다 정확한 분석이 가능할 것으로 예상됩니다.')
                    st.markdown('Ford A와 Ford B 데이터는 비슷하게 생겼지만, 각 데이터 문제를 잘 해결한 SOTA알고리즘은 다릅니다.')
                    st.markdown('직접 한번 해보시는 것을 추천하겠습니다.')
            
            elif df_name == 'TFTLCD.csv':
                with st.expander('Data info'):
                    st.markdown('해당 데이터는 TFTLCD 데이터 셋입니다.')
                    st.markdown('TFTLCD 데이터는 반도체 공정에 관한 데이터입니다.')
                    st.markdown('TOOL은 공정 생산 라인이라고 생각하시면 됩니다.')
                    st.markdown('동일한 레시피로 반도체를 만들지만, 4개의 다른 라인에 따라 다른 분포를 띄고 있습니다.')
                    st.markdown('PCA, t-SNE에서 실제로 멀티모달을 띄고 있고, 정상과 불량이 잘 구분되고 있습니다.')
                    st.markdown('해당 데이터는 본래 classfiication 문제가 아닌 이상탐지 문제입니다.')
                    st.markdown('정상 데이터만 가지고 모델을 학습 시키고, 비정상 데이터를 탐지하는 것이 목적이지만, 기존과 다르게 랜덤적으로 데이터를 분할하여 사용할 예정입니다.')
                    st.markdown('category feature가 포함되어 있어, one-hot encoding, label encoding을 하여 진행해도 되지만, 여기서는 numeric feature만 사용할 예정입니다.')
                    st.markdown('하나의 센서에서 측정된 값을 4개의 feature로 나누어 사용하는 특성이 있습니다.')
                    st.markdown('각 센서 값의 평균, 표준편차, 최대값, 최소값으로 분할 합니다.')
                    st.markdown('공정 데이터를 분석하다 보면 이런 특성을 가진 데이터가 많이 존재합니다.')
                    st.markdown('신호처리라는 과정을 거쳐 1초단위의 데이터가 만들어지는 상황에서 이런 경우가 생긴다고 생각하시면 됩니다.')
                    st.markdown('초당 수십 ~ 수백번 측정한 뒤, 신호처리를 하는 과정에서 나오는 값들을 1초단위로 만들어지는 것입니다.')
                    st.markdown('신호처리는 기업마다 다르기 때문에, 추후 신호처리를 하지 않은 데이터를 사용하게 된다면 여러가지 방법을 적용하여 진행하는 것을 추천드립니다.')
                    st.markdown('신호처리를 하는 것은 데이터의 특성을 잘 반영하여 진행해야 하는데, NASA 베이링 데이터를 가지고 직접 해보시는 것을 추천드립니다.')
                    st.markdown('아래는 NASA 베이링 데이터 링크입니다.')
                    st.markdown('https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing')

            elif df_name == 'Pasteurizer.csv':
                with st.expander('Data info'):
                    st.markdown('데이터셋 형태 및 수집 방법')
                    st.markdown('분석에 사용된 변수 : 살균상태, 살균온도, 양품/불량여부 ')
                    st.markdown('데이터 수집 방법 : PLC, DBMS(RDB) ')
                    st.markdown('데이터셋 파일 확장자 : csv')
                    st.markdown(' ')
                    st.markdown('데이터 개수 데이터셋 총량')
                    st.markdown('데이터 개수 : 210,794건 ')
                    st.markdown('데이터셋 총량 : 6.11MB')
                    st.markdown('정상 : 0, 불량 : 1')
                    st.markdown(' ')
                    st.markdown('분석 결과 및 시사점')
                    st.markdown('공정 중 살균기 설비 운영 값을 기준으로 제품의 최종 품질을 예측할 수 있는 AI 기법을 적용합니다.')
                    st.markdown('공정 운영변수의 변화가 품질에 미칠 영향 모델링을 도출합니다.')
                    st.markdown('살균공정의 설비운영 데이터와 최종품질검사 데이터를 수집하고, 데이터 가공/전처리, AI 모델 개발과 제조공정의 적용 및 검증을 합니다.')
                    st.markdown('열악한 중소기업에 빅데이터 및 AI 기술을 적용하여 실질적인 품질향상 및 비용절감에 기여한다는 점에서 시사하는 바가 크다고 판단됩니다.')
                    st.markdown(':red[해당 데이터는결측치가 존재합니다. 결측치를 처리한 뒤 진행해주세요.]')

        st.markdown('----')
        st.markdown('# 데이터 전처리')
        st.markdown('### 1. 결측치 처리')
        st.markdown('#### 1-1. 결측치 개수')
        if st.checkbox('결측치 개수 확인'):
            st.dataframe(df.isnull().sum(), width = 500)
            if st.checkbox('결측치 개수 시각화'):
                fig = go.Figure(data=[go.Bar(x=df.isnull().sum().index, y=df.isnull().sum().values)])
                fig.update_layout(
                    title="결측치 개수 확인",
                    xaxis_tickangle=-45,
                    xaxis=dict(tickfont=dict(size=10)),
                    yaxis=dict(title="Count"),
                    width = 1200, height = 600
                )
                st.plotly_chart(fig)




        st.markdown('----')
        st.markdown('#### 1-2. 결측치 처리')
        st.markdown('##### 1-2-1. 결측치 제거')
        drop_check = st.checkbox('결측치 제거')
        if drop_check == True:
            df = df.dropna()
            st.write(df.head(10))
        else:
            pass

        st.markdown('----')
        st.markdown('##### 1-2-2. 결측치 대체')
        fill_check = st.checkbox('결측치 대체')
        if fill_check == True:
            fill_select = st.selectbox('대체 방법 선택', ['평균', '중앙값', '최빈값', '선형보간'])
            if fill_select == '평균':
                df = df.fillna(df.mean())
                st.write(df.head(10))
            elif fill_select == '중앙값':
                df = df.fillna(df.median())
                st.write(df.head(10))
            elif fill_select == '최빈값':
                df = df.fillna(df.mode())
                st.write(df.head(10))
            elif fill_select == '선형보간':
                df = df.interpolate()
                st.write(df.head(10))
        else:
            pass
        
        st.markdown('')
        st.markdown('### 2. 데이터 분할')
        train_size = st.slider('train set 비율', 0.1, 0.9, 0.7)

        train, test = train_test_split(df, train_size = train_size, random_state = 42)
        st.write('train set : ', train.shape)
        st.write('test set : ', test.shape)
        
        y_train = train[y]
        y_test = test[y]
        x_train = train.drop(axis=1, columns = [y])
        x_test = test.drop(axis=1, columns = [y])
        


        st.markdown('----')
        st.markdown('# 데이터 시각화')
        st.markdown('### 1. Train 데이터 통계')
        st.write(train.describe())
        st.markdown('#### Train 데이터 타겟 변수 개수')
        st.write(train[y].value_counts().sort_index())

        normal_value = st.number_input('정상 상태의 레이블을 입력해주세요', value = 0)
        abnormal_value = st.number_input('비정상 상태의 레이블을 입력해주세요', value = 1)

        if st.checkbox('정상 데이터만 사용하여 모델링'):
            train = train[train[y] == normal_value]
            st.write(train[y].value_counts())
            x_train = x_train.loc[train.index]
            y_train = y_train.loc[train.index]
            
        st.markdown('### 2. Train 데이터 상관관계')
        if st.checkbox('Heatmap', help = '약간의 시간이 소요될 수 있습니다.'):
            figsize = st.slider('그래프 크기 조절', 15, 25, 20)
            annot_check = st.checkbox('상관계수 표시', help = '약간의 시간이 소요될 수 있습니다.')
            if annot_check == True:
                annot_select = True
            else:
                annot_select = False
            corr_plot(train, figsize, figsize, annot_select)

        st.markdown('### 3. Train 데이터 분포')
        st.markdown('#### 3-1. Box Plot 및 이상치 제거')
        if st.checkbox('Box Plot 확인'):
            x_box = st.selectbox('x축 변수 선택', train.columns)
            fig_box = box_plot(train, x_box, y)

            # Plotly 그래프 출력
            try:
                st.plotly_chart(fig_box, use_container_width=True)

                if st.checkbox('해당 x의 y값 범위를 정하고싶다면 버튼을 눌러주세요'):
                    # y축 범위 선택
                    y_range = st.slider("Y의 범위 지정", float(train[x_box].min()), float(train[x_box].max()), (float(train[x_box].min()), float(train[x_box].max())))

                    selected_data = train[[x_box, y]]
                    # 선택된 데이터의 인덱스 추출
                    selected_indices = selected_data[(selected_data[x_box] >= y_range[0]) & (selected_data[x_box] <= y_range[1])].index

                    st.write('선택된 데이터 개수 : ', len(selected_indices))
                    st.write('총 데이터 개수 : ', len(train))

                    # 선택된 데이터의 인덱스로 df에서 추출해서 사용
                    x_train_selected = x_train.loc[selected_indices]
                    y_train_selected = y_train.loc[selected_indices]

                    if st.checkbox('선택된 데이터 확인'):
                        st.write('x train :')
                        st.write(x_train_selected)
                        st.write('y train :')
                        st.write(y_train_selected)

                        if st.checkbox('Box plot 재확인'):
                            st.plotly_chart(box_plot(x_train_selected, x_box, y_train_selected))

                    x_train = x_train_selected
                    y_train = y_train_selected

            except:
                pass

        st.markdown('#### 3-2. Hist Plot')
        if st.checkbox('Hist Plot 확인'):
            x_hist = st.selectbox('변수 선택', train.columns)
            hist_plot(train, x_hist, y)

            # # X축 범위 선택
            # x_range = st.slider("Select X range", float(df[x_box].min()), float(df[x_box].max()), (float(df[x_box].min()), float(df[x_box].max())))

            # # 선택된 데이터의 인덱스 추출
            # selected_indices = [i for i in range(len(df)) if x_range[0] <= df[x_box][i] <= x_range[1]]

            # # 선택된 데이터의 인덱스 출력
            # st.write("Selected Indices:", selected_indices)


        
        data_trans = st.button('데이터를 다른 페이지로 전송', help = '다른 페이지에서도 현제 페이지에서 설정한 데이터를 사용할 수 있습니다.')
        if data_trans == True:
            st.session_state['x_train'] = x_train.sort_index()
            st.session_state['x_test'] = x_test.sort_index()
            st.session_state['y_train'] = y_train.sort_index()
            st.session_state['y_test'] = y_test.sort_index()
            st.session_state['normal_value'] = normal_value
            st.session_state['abnormal_value'] = abnormal_value
            st.write('데이터가 전송되었습니다.')

