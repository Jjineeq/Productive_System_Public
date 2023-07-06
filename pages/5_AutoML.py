import streamlit as st
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from pycaret.anomaly import *
import base64

@st.cache_resource(ttl=24*60*60)
def anomaly_models(df):
    total_result = []
    total_col = []
    for i in all_model.index:
        try:
            result = predict_model(create_model(i), data = df)['Anomaly']
            st.write(f"모델 {i}이(가) 성공적으로 실행되었습니다.")
            total_col.append(i)
            total_result.append(result)
        except:
            try:
                result = predict_model(create_model(i), data = np.array(df))['Anomaly']
                st.write(f"모델 {i}이(가) 성공적으로 실행되었습니다.")
                total_col.append(i)
                total_result.append(result)
            except:
                st.write(f"모델 {i}은 에러로 인해 제거했습니다.")


    total_result_df = pd.DataFrame(total_result).T
    total_result_df.columns = total_col
    
    return total_result_df

def download_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">다운로드</a>'
    return href


with st.spinner('Updating Report...'):
    try:
        x_train = st.session_state['x_train']
        x_test = st.session_state['x_test']
        y_train = st.session_state['y_train'] 
        y_test = st.session_state['y_test']
        normal_value = st.session_state['normal_value']
        abnormal_value = st.session_state['abnormal_value'] 
 

    except:
        st.error('main페이지에서 데이터를 전송해주세요.')
        st.stop()


    st.title("AutoML")

    st.markdown('### with Anomaly Detection')

    st.markdown('----')
    
    numeric_columns = x_train.select_dtypes(include=['int64', 'float64']).columns
    numeric_train = x_train[numeric_columns]
    numeric_test = x_test[numeric_columns]

    st.markdown('데이터 스케일')
    if st.checkbox('StandardScaler'):
        scaler = StandardScaler()
        numeric_train = scaler.fit_transform(numeric_train)
        numeric_test = scaler.transform(numeric_test)
        st.write('스케일 완료')
        numeric_test = pd.DataFrame(numeric_test, columns=numeric_columns)
        numeric_train = pd.DataFrame(numeric_train, columns=numeric_columns)
    st.markdown('----')

    clf = setup(data = numeric_test, normalize = True, ignore_features = ['Time'])

    all_model = models()

    st.markdown('생성된 모델 확인')

    st.write(all_model)

    if st.checkbox('모델 실행'):
        total_result_df = anomaly_models(numeric_test)


        if st.checkbox('모델 결과 확인'):
            st.write(total_result_df)
    
        if st.checkbox('모델 결과 평균값 확인'):
            st.markdown(' ')
            st.markdown('평균 결과 값 :')
            st.write(pd.DataFrame(total_result_df.mean(axis=1), columns=['mean']).T, width = 100)

            st.write('0.5 이상이면 이상치로, 0.5 미만이면 정상으로 판단')
            
            mean_result = total_result_df.mean(axis=1)
            mean_result = mean_result.apply(lambda x: 1 if x >= 0.5 else 0)
            st.markdown(' ')
            st.markdown('변환 결과 :')
            st.dataframe(pd.DataFrame(mean_result, columns=['변환 값']).T)
            st.write('예측 정상 데이터 개수 : ', len(mean_result[mean_result < 0.5]))
            st.write('예측 이상 데이터 개수 : ', len(mean_result[mean_result >= 0.5]))

            st.write('실제 정상 데이터 개수 : ', len(y_test[y_test == normal_value]))
            st.write('실제 이상 데이터 개수 : ', len(y_test[y_test == abnormal_value]))



            st.markdown('----')

            if st.checkbox('모델을 직접 선정하기 위해 모델별 claasification report 확인'):
                if abnormal_value == -1:
                    y_test = y_test.apply(lambda x: 1 if x == abnormal_value else 0)
                total_col = total_result_df.columns
                for i in total_col:
                    st.write(i + '의 classification report :')
                    st.code('Model Report:\n  ' + classification_report(y_test, total_result_df[i]))

                if st.checkbox('모델 선택'):
                    select_model = st.multiselect('최종 결과에 반영할 모델 선택 :', total_col)
                    
                    mean_result = total_result_df[select_model].mean(axis=1)
                    mean_result = mean_result.apply(lambda x: 1 if x >= 0.5 else 0)



            st.markdown('----')

            if st.checkbox('결과 확인'):
                if abnormal_value == -1:
                    y_test = y_test.apply(lambda x: 1 if x == abnormal_value else 0)
                st.write('Confusion Matrix :')
                st.write(confusion_matrix(y_test, mean_result))
                st.markdown(' ')
                st.write('Classification Report : ')
                st.code('Model Report:\n  ' + classification_report(y_test, mean_result))

            if st.button('Result Download'):
                total_result_data = pd.DataFrame({'real':y_test.reset_index(drop = True), 'pred':mean_result})
                total_result_data.to_csv('total_result_data.csv', index=False)

                csv_link = download_csv(total_result_data, 'AutoML_Result.csv')
                st.markdown(csv_link, unsafe_allow_html=True)

        with st.expander('추가로 확인할 수 있는 내용'):
            st.markdown('Supervised learning 파트에서 Ensemble과 관련된 내용이 있었습니다.')
            st.markdown('또한 Unsupervised learning 파트에서는 threshold를 직접 컨트롤 하면서 이상치를 찾아내는 내용이 있었습니다.')
            st.markdown('이 두가지를 합쳐서 AutoML을 직접 구현해 보시는 것을 추천드립니다.')
            st.markdown('현재 여기에 있는 모델들은 모두 hyperparameter tuning을 하지 않은 모델입니다.')
            st.markdown('또한 hard voting을 통해 최종 결과를 도출하였습니다.')
            st.markdown('여기서 추가적으로 할 수 있는 내용으로는')
            st.markdown('1. 각 모델 결과를 확률 값으로 도출하고, soft voting을 통해 최종 결과를 도출')
            st.markdown('2. 각 모델 결과를 확률 값으로 도출하고, stacking ensemble을 통해 meta model을 만들어 최종 결과를 도출')
            st.markdown('위의 두가지 정도의 방법이 있을 것으로 생각됩니다.')
            st.markdown('또한 AutoML을 직접 구현하시면서 다양한 모델을 사용해보시고, 다양한 방법을 사용해보시면서 성능을 비교해보시는 것을 추천드립니다.')
            st.markdown('감사합니다.')

            
