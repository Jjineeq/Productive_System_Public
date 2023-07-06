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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
st.wide = True
import sys
sys.path.append('./functions/pyod-master/')
from pyod.models.loda import LODA
from pyod.models.cblof import CBLOF
import rrcf
import base64

def images(fath):
    col1, col2, col3 = st.columns([0.4,1,0.4])

    with col1:
        st.write(' ')

    with col2:
        st.image(fath, use_column_width = True)

    with col3:
        st.write(' ')

    return None

def download_pdf_file(file_path, file_name, ment):
    with open(file_path, "rb") as file:
        pdf_data = file.read()
    b64 = base64.b64encode(pdf_data).decode("UTF-8")
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">'+ ment +'</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">다운로드</a>'
    return href


@st.cache_resource(ttl=24*60*60)
def rrcf_model(numeric_test, num_trees, shingle_size, tree_size):
    n = len(numeric_test)
    #shingle_size = len(numeric_test.columns)
    sample_size_range = (n // shingle_size)
    forest = []
    while len(forest) < num_trees:
        # Select random subsets of points uniformly
        ixs = np.random.choice(n, size=sample_size_range,
                            replace=False)
        ixs = ixs.reshape(1, -1)
        # Add sampled trees to forest
        trees = [rrcf.RCTree(numeric_test_value[ix], index_labels=ix) for ix in ixs]
        forest.extend(trees)

    # Compute average CoDisp
    avg_codisp = pd.Series(0.0, index=np.arange(n))
    index = np.zeros(n)
    for tree in forest:
        codisp = pd.Series({leaf : tree.codisp(leaf)
                        for leaf in tree.leaves})
        avg_codisp[codisp.index] += codisp
        np.add.at(index, codisp.index.values, 1)
    avg_codisp /= index

    return avg_codisp

@st.cache_resource(ttl=24*60*60)
def isolation_model(data, contamination_iforest, n_estimators_iforest, max_samples_iforest, bootstrap_iforest):
    iforest = IsolationForest(contamination=contamination_iforest, n_estimators=n_estimators_iforest, max_samples=max_samples_iforest, bootstrap=bootstrap_iforest, n_jobs=-1)
    isolation_pred = iforest.fit_predict(data)
    isolation_score = iforest.decision_function(data)


    return isolation_pred, isolation_score


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

    
    numeric_columns = x_train.select_dtypes(include=['int64', 'float64']).columns
    numeric_train = x_train[numeric_columns]
    numeric_test = x_test[numeric_columns]


    st.title('Unsupervised Learning (Anomaly Detection)')
    st.markdown('----')
    st.subheader('추천 데이터 셋 : Pasteurizer')
    st.markdown('----')
    st.markdown('데이터 스케일')
    if st.checkbox('StandardScaler'):
        scaler = StandardScaler()
        numeric_train = scaler.fit_transform(numeric_train)
        numeric_test = scaler.transform(numeric_test)
        st.write('스케일 완료')
        numeric_test = pd.DataFrame(numeric_test, columns=numeric_columns)
        numeric_train = pd.DataFrame(numeric_train, columns=numeric_columns)

    
    st.markdown('----')
    st.markdown('### 1. LOF(Local Outlier Factor)')

    with st.expander('LOF 설명'):
        st.markdown('LOF은 이상탐지를 위한 비지도 학습으로 많이 사용됩니다.')
        st.markdown('밀도 기반의 이상탐지 모델이며, 데이터의 밀도가 낮은 지역에 위치한 데이터를 이상치로 판단합니다.')
        st.markdown('하지만 주변 데이터의 밀도도 같이 판단하여, 밀도가 낮은 지역에 위치한 데이터가 주변 데이터의 밀도가 낮다면 이상치로 판단하지 않습니다.')
        st.markdown('해당 개념은 다른 밀도 기반 함수들에서 많이 사용되는 개념입니다.')
        st.markdown(' ')
        st.markdown('LOF 작동 원리 :')
        st.markdown(' k-distance -> P에 근접한 객체의 수 구하기 -> Reachability distance -> Local reachability distance -> LOF 정의하기')
        st.markdown(' ')
        st.markdown('1. k-distance : P와 가장 가까운 k개의 이웃을 찾습니다.')
        images('./images/lof1.png')
        st.markdown('위의 그림에서 3-distance(p) = 1.5가 된다.')
        st.markdown("즉 d(p, o') <= d(p,o)인 객체가 k개 이상, d(p, o') > d(p,o)인 객체가 k-1개 이하")
        st.markdown("단, d(p, o')은 p와 o'의 거리, d(p,o)는 p와 o의 거리")
        st.markdown(' ')
        st.markdown('2. P에 근접한 객체의 수 구하기 :')
        st.markdown('P라는 객체의 k-distance neighorhood는 data에서 p를 제외한 q들의 집합중 d(p,q)가 k-distance(p)의 거리보다 작거나 같은 객체의 수를 나타낸다.')
        st.markdown('수식으로 나타내면 아래와 같다.')
        images('./images/lof수식_4.jpg')
        st.markdown('설명을 추가적으로 작성하면, k=3이라고 했을때, p와 가장 가까운 3개의 이웃을 찾는다.')
        st.markdown('하지만 이때 만약 이웃한 5개의 점의 길이가 모두 같다고 하면 N3(p) = 5가 된다.')
        st.markdown(' ')
        
        st.markdown('3. Reachability distance :')
        st.markdown('쉽게 표현하면 점 p을 중심으로 k번째의 거리에 해당하는 점까지의 거리를 반지름으로 잡아 원을 그린 후, 원 내부에 있는 점은 반지름으로 계산한다.')
        st.markdown('그리고 원 외부에 있는 점은 실제 거리로 계산하게 된다.')
        images('./images/lof수식_1.jpg')
        st.markdown('reahability distance를 구하는 공식은 위와 같다.')
        images('./images/lof2.png')
        st.markdown('사진으로 표현하면 위와 같다.')
        st.markdown(' ')
        
        st.markdown('4. Local reachability distance :')
        st.markdown('p객체에 이웃한 점들의 집합의 reachability distance의 합이 분모가 되며, p객체의 이웃한 점들의 수가 분자가 된다.')
        images('./images/lof수식_2.jpg')
        st.markdown('수식은 위와 같다.')

        st.markdown(' ')
        st.markdown('5. LOF 정의하기 :')
        images('./images/lof수식_3.jpg')
        st.markdown('수식은 위와 같다.')
        st.markdown('간단하게 표현하면 p의 이웃 점들의 밀도까지 고려하여 객체의 score를 계산하는 것이다.')
        st.markdown('다시 말하면 p의 이웃과 거리와 이웃들의 이웃에 대한 거리를 비교하는 것이다.')
        st.markdown(' ')
        images('./images/lof5.png')
        st.markdown('case에 대해 정리한 사진을 보면 위와 같다.')
        st.markdown('case1은 밀도가 높지만, 주변의 밀도 또한 높다.')
        st.markdown('반대로 case3은 밀도가 낮지만 주변의 밀도 또한 낮다.')
        st.markdown('이러한 경우에는 낮은값/낮은값, 높은값/높은값으로 1과 유사하게 나와 이상치로 판단하지 않는다.')
        st.markdown('반대로 case2의 경우 밀도가 낮지만 주변의 밀도는 높기때문에 LOF score의 값이 높게 나온다.')

        

        
    st.markdown('### Hyperparameter Tuning')
    neighbor = st.number_input('n_neighbors (1 ~ 100)', min_value=1, max_value=100, value=20, step=1, help = '이웃의 개수 설정, 값이 커질수록 이웃의 수 증가하며 임계값이 증가함')
    contamination = st.number_input('contamination (0 ~ 0.5)', min_value=0.0, max_value=0.5, value=0.1, step=0.1, help = '이상치 비율 설정, 값이 커질수록 이상치 비율 증가함')
    lof = LocalOutlierFactor(n_neighbors=neighbor, contamination=0.1, novelty=True, n_jobs=-1)

    if st.checkbox('LOF 실행'):
        lof.fit(numeric_test)
        y_pred = lof.predict(numeric_test)
        y_prob = lof.decision_function(numeric_test)
        y_pred_score = lof.negative_outlier_factor_
        thresholds_lof = np.sort(y_pred_score)[int((1-contamination)*len(y_pred_score))]

        if normal_value == 0:
            y_change = np.where(y_test == 0, 1, -1)
        else: 
            y_change = y_test

        if st.checkbox('LOF 결과 확인'):
            st.write('Accuracy Score : ', accuracy_score(y_change, y_pred))

            st.write('Confusion Matrix : ', pd.DataFrame(confusion_matrix(y_change, y_pred), columns=[-1,1], index=[-1,1]))
            st.write('Classification Report : ' )
            st.code('Model Report:\n  ' + classification_report(y_change, y_pred))

            fpr_lof, tpr_lof, thresholds = roc_curve(y_change, y_prob, pos_label=1)
            st.write('AUC : ', auc(fpr_lof, tpr_lof))
            fig_lof = px.area(x=fpr_lof, y=tpr_lof,  title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'))
            fig_lof.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_lof.update_xaxes(constrain='domain')
            fig_lof.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig_lof)

        if st.checkbox('LOF 결과 시각화'):
            if normal_value == 0:
                y_change = np.where(y_test == 0, 1, -1)
            else:
                y_change = y_test

            fig_lof = px.line(x=range(len(y_prob)), y=-y_prob)
            
            # 실제 이상치
            true_anomaly_index = np.where(y_test == abnormal_value)[0]
            for index in true_anomaly_index:
                fig_lof.add_trace(go.Scatter(
                    x=[index],
                    y=[-y_prob[index]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        symbol='x',
                        size=10
                    ),
                    showlegend=False
                ))
            
            # 알람 발생 이상치
            predicted_anomaly_index = np.where(y_pred == -1)[0]
            for index in predicted_anomaly_index:
                fig_lof.add_trace(go.Scatter(
                    x=[index, index],
                    y=[min(-y_prob), max(-y_prob)],
                    mode='lines',
                    line=dict(color='red', width=0.5),
                    showlegend=False
                ))

            st.write('thresholds : ', -thresholds_lof)
            fig_lof.add_trace(go.Scatter(
                x=[0, len(y_pred_score)],
                y=[-thresholds_lof, -thresholds_lof],
                mode='lines',
                line=dict(color='orange', width=1),
                showlegend=False
            ))

            st.plotly_chart(fig_lof, use_container_width=True)


            st.markdown('----')
            st.write('기존 LOF threshold : ', -thresholds_lof)
            LOF_threshold = st.text_input('LOF threshold', value=1)
            LOF_threshold = float(LOF_threshold)

            lof_pred = np.where(-y_pred_score > LOF_threshold, -1, 1)
            
            fig_lof_2 = px.line(x=range(len(y_pred_score)), y=-y_pred_score)
            for index in true_anomaly_index:
                fig_lof_2.add_trace(go.Scatter(
                    x=[index],
                    y=[-y_pred_score[index]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        symbol='x',
                        size=10
                    ),
                    showlegend=False
                ))             
            
            predicted_anomaly_index = np.where(lof_pred == -1)[0]
            for index in predicted_anomaly_index:
                fig_lof_2.add_trace(go.Scatter(
                    x=[index, index],
                    y=[min(-y_pred_score), max(-y_pred_score)],
                    mode='lines',
                    line=dict(color='red', width=0.5),
                    showlegend=False
                ))
            
            
            st.write('thresholds : ', LOF_threshold)
            
            fig_lof_2.add_trace(go.Scatter(
                x=[0, len(y_pred_score)],
                y=[LOF_threshold, LOF_threshold],
                mode='lines',
                line=dict(color='orange', width=1),
                showlegend=False
            ))


            st.plotly_chart(fig_lof_2, use_container_width=True)

            if normal_value == 0:
                y_change = np.where(y_test == 0, 1, -1)
            else:
                y_change = y_test

            st.write('Accuracy Score : ', accuracy_score(y_change, lof_pred))
            
            st.write('Confusion Matrix : ', pd.DataFrame(confusion_matrix(y_change, lof_pred), columns=[-1,1], index=[-1,1]))
            
            st.write('Classification Report : ' )
            #st.code(classification_report(y_change, lof_pred))
            st.code('Model Report:\n  ' + classification_report(y_change, lof_pred))

            if st.button('Result Download'):
                total_result_data = pd.DataFrame({'real':y_change, 'pred':lof_pred})
                total_result_data.to_csv('total_result_data.csv', index=False)

                csv_link = download_csv(total_result_data, 'LOF_Result.csv')
                st.markdown(csv_link, unsafe_allow_html=True)
            



    # st.markdown('----')
    # st.markdown('### 2. LODA(Lightweight On-line Detector of Anomalies)')
    
    # contamination_loda = st.number_input('LODA contamination', min_value=0.0, max_value=0.5, value=0.1, step=0.1)
    # n_bins = st.number_input('n_bins', min_value=1, max_value=100, value=10, step=1)
    # n_random_cuts = st.number_input('n_random_cuts', min_value=1, max_value=100, value=100, step=1)
    
    
    # if st.checkbox('LODA 실행'):
    #     loda = LODA()
    #     loda.fit(numeric_train)
    #     loda_pred = loda.fit_predict(numeric_test)
        
    #     loda_decision = loda.decision_function(numeric_test)
        
    #     if normal_value == 1:
    #         y_change = np.where(y_test == 1, 1, 0)
    #     else:
    #         y_change = y_test
        
    #     if st.checkbox('LODA 결과 확인'):
    #         st.dataframe(pd.DataFrame(loda_pred, columns=['y_pred']))
    #         st.write('Accuracy Score : ', accuracy_score(y_change, loda_pred))

    #         st.write('Confusion Matrix : ', confusion_matrix(y_change, loda_pred))

    #         st.write('Classification Report : ' )
    #         st.code(classification_report(y_change, loda_pred))

    #         fpr_loda, tpr_loda, thresholds_loda = roc_curve(y_change, loda_pred)
    #         st.write('AUC : ', auc(fpr_loda, tpr_loda))

    #         fig_loda = px.area(x=fpr_loda, y=tpr_loda,  title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'))
    #         fig_loda.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    #         fig_loda.update_xaxes(constrain='domain')
    #         fig_loda.update_yaxes(scaleanchor="x", scaleratio=1)
        
    #         st.plotly_chart(fig_loda, width = 1000, height = 1000)
            
        
    #     if st.checkbox('LODA 시각화'):
    #         fig_loda = px.line(x=range(len(loda_decision)), y=loda_decision)
    #         anomaly_index = np.where(y_test == abnormal_value)[0]
    #         for index in anomaly_index:
    #             fig_loda.add_trace(go.Scatter(
    #                 x=[index, index],
    #                 y=[min(loda_decision), max(loda_decision)],
    #                 mode='lines',
    #                 line=dict(color='red', width=0.5),
    #                 showlegend=False
    #             ))
            
    #         st.plotly_chart(fig_loda, use_container_width=True)
    
    # st.markdown('----')
    # st.markdown('### 3. LOCI(Local Correlation Integral)')
    
    # contamination_loci = st.number_input('LOCI contamination', min_value=0.0, max_value=0.5, value=0.1, step=0.1)
    # alpha_loci = st.number_input('LOCI alpha', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    # k_loci = st.number_input('LOCI k', min_value=1, max_value=100, value=20, step=1)

    # loci = LOCI(contamination=contamination_loci, alpha = alpha_loci, k=k_loci)
    # if st.checkbox('LOCI 실행'):
    #     loci_pred = loci.fit_predict(numeric_test)
    #     loci_decision = loci.decision_function(numeric_test)

    #     st.write(loci_pred)
        
    #     if normal_value == 1:
    #         y_change = np.where(y_test == 1, 1, 0)
    #     else:
    #         y_change = y_test

    #     if st.checkbox('LOCI 결과 확인'):
    #         st.write('Accuracy Score : ', accuracy_score(y_change, loci_pred))

    #         st.write('Confusion Matrix : ', confusion_matrix(y_change, loci_pred))

    #         st.write('Classification Report : ' )
    #         st.code(classification_report(y_change, loci_pred))

    #         fpr_loci, tpr_loci, thresholds_loci = roc_curve(y_change, loci_pred)
    #         st.write('AUC : ', auc(fpr_loci, tpr_loci))

    #         fig_loci = px.area(x=fpr_loci, y=tpr_loci,  title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'))
    #         fig_loci.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    #         fig_loci.update_xaxes(constrain='domain')
    #         fig_loci.update_yaxes(scaleanchor="x", scaleratio=1)

    #         st.plotly_chart(fig_loci, width = 1000, height = 1000)
    
    st.markdown('----')
    st.markdown('### 2. CBLOF(Cluster-Based Local Outlier Factor)')
    with st.expander('CBLOF 설명'):
        st.markdown('CBLOF은 군집화를 기반으로 이상치를 탐지하는 모델입니다.')
        st.markdown('가장 먼저 데이터를 여러 클러스터로 나누며, 이후 대형 클러스터와 소형 클러스터로 분류하게 됩니다.')
        st.markdown('대형 클러스터에 속한 데이터는 데이터 포인트와 클러스터 중심 사이의 거리에 따라 결정됩니다.')
        st.markdown('즉 클러스터 중심에서 멀리 떨어진 데이터일수록 더 높은 이상치 점수를 받게 됩니다.')
        st.markdown('소형 클러스터에 속한 데이터는 클러스터의 크기와 해당 클러스터의 중심과의 거리에 따라 결정됩니다.')
        st.markdown(' ')
        st.markdown('Kmeans, DBSCAN의 클러스터링 방법이 구현되어 있습니다.')
        st.markdown('K-means의 경우 초기 단계에서 K개의 클러스터 중심을 임의로 선택하여 진행합니다.')
        st.markdown('이때 각 데이터 포인트를 가장 가까운 클러스터 중심에 할당하게 됩니다.')
        st.markdown('이때 할당된 클러스터 중심을 기반으로 클러스터의 중심을 다시 계산하고, 이를 반복하여 클러스터링을 진행합니다.')
        st.markdown('중심점의 위치가 변하지 않거나, 미리 설정한 반복횟수까지 진행하는것이 일반적입니다.')
        images('./images/kmeans.png')
        st.markdown('위의 그림은 kmeans의 예시입니다.')
        st.markdown(' ')
        st.markdown(' ')

        st.markdown('DBSCAN의 경우 밀도 기반의 군집화 방법입니다.')
        st.markdown('K의 개수를 미리 지정하지 않지만 각 데이터 포인트를 중심으로 할 반경인 epsilon과 몇개 이상의 데이터가 모이면 군집으로 인정할지를 나타내는 min_samples를 설정합니다.')
        st.markdown('임의의 데이터 포인트를 설정하고, 설정한 epsilon 내에 min points 이상의 데이터가 존재하면 하나의 군집으로 인정합니다.')
        st.markdown('이후 포인트 내에 epsilon 내에 있는 모든 이웃 데이터를 찾습니다.')
        st.markdown('이웃 포인트가 아직 확인하지 않은 포인트라면 현재 군집에 추가하며, 만약 이웃 포인트가 다른 군집의 포인트라면 해당 군집과 병합합니다.')
        st.markdown('위의 순서로 모든 데이터 포인트에 대해 반복해서 진행하게 됩니다.')
        images('./images/dbscan.jpg')
        st.markdown('위의 그림은 DBSCAN의 예시입니다.')




    st.markdown('### Hyperparameter Tuning')
    contamination_cblof = st.number_input('CBLOF contamination (0 ~ 0.5)', min_value=0.0, max_value=0.5, value=0.5, step=0.1, help = '이상치 비율 설정, 값이 커질수록 이상치 비율 증가함')
    CBLOF_n_clusters = st.number_input('CBLOF n_clusters (1 ~ 100)', min_value=1, max_value=100, value=8, step=1, help = '클러스터 개수 설정')
    clustering_estimator = st.selectbox('clustering_estimator', ['kmeans', 'dbscan'], help = '클러스터링 방법 설정')

    if clustering_estimator == 'kmeans':
        cblof = CBLOF(contamination=contamination_cblof, n_clusters=CBLOF_n_clusters, 
                      clustering_estimator=KMeans(n_clusters=CBLOF_n_clusters))

    elif clustering_estimator == 'dbscan':
        eplison = st.number_input('eplison', min_value=0.0, max_value=1.0, value=0.5, step=0.1, help = '반경 설정')
        min_samples = st.number_input('min_samples', min_value=1, max_value=100, value=5, step=1, help = '최소 샘플 개수 설정')
        cblof = CBLOF(contamination=contamination_cblof, n_clusters=CBLOF_n_clusters, clustering_estimator=DBSCAN(eps=eplison, min_samples=min_samples))
    
    
    if st.checkbox('CBLOF 실행'):
        try:
            cblof_pred = cblof.fit_predict(numeric_test)
            cblof_score = cblof.decision_function(numeric_test)
            st.write(cblof_pred)

            if normal_value == 1:
                y_change = np.where(y_test == 1, 0, 1)
            else:
                y_change = y_test


            st.dataframe(pd.DataFrame(cblof_pred, columns=['y_pred']))
            st.write('Accuracy Score : ', accuracy_score(y_change, cblof_pred))
            
            st.write('Confusion Matrix : ', confusion_matrix(y_change, cblof_pred))
            
            st.write('Classification Report : ' )
            st.code('Model Report:\n  ' + classification_report(y_change, cblof_pred))

            fpr_cblof, tpr_cblof, thresholds_cblof = roc_curve(y_change, cblof_score)
            st.write('AUC : ', auc(fpr_cblof, tpr_cblof))
            fig_cblof = px.area(x=fpr_cblof, y=tpr_cblof,  title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'))
            fig_cblof.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_cblof.update_xaxes(constrain='domain')
            fig_cblof.update_yaxes(scaleanchor="x", scaleratio=1)
        
            st.plotly_chart(fig_cblof, width = 1000, height = 1000)

            st.markdown('----')

            fig_cblof = px.line(x=range(len(cblof_score)), y=cblof_score)
            
            # anomaly_index = np.where(y_test == abnormal_value)[0]
            # for index in anomaly_index:
            #     fig_cblof.add_trace(go.Scatter(
            #         x=[index, index],
            #         y=[min(cblof_score), max(cblof_score)],
            #         mode='lines',
            #         line=dict(color='red', width=0.5),
            #         showlegend=False
            #     ))
                
            true_anomaly_index = np.where(y_test == abnormal_value)[0]
            for index in true_anomaly_index:
                fig_cblof.add_trace(go.Scatter(
                    x=[index],
                    y=[cblof_score[index]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        symbol='x',
                        size=10
                    ),
                    showlegend=False
                ))    
                
            predicted_anomaly_index = np.where(cblof_pred == 1)[0]
            for index in predicted_anomaly_index:
                fig_cblof.add_trace(go.Scatter(
                    x=[index, index],
                    y=[min(cblof_score), max(cblof_score)],
                    mode='lines',
                    line=dict(color='red', width=0.5),
                    showlegend=False
                ))
            
            
            st.plotly_chart(fig_cblof, use_container_width=True)


            st.markdown('----')

            cblof_threshold = st.text_input('CBLOF threshold', value=10)
            cblof_threshold = float(cblof_threshold)
            cblof_pred = np.where(cblof_score > cblof_threshold, 1, 0)
            
            fig_cblof_2 = px.line(x=range(len(cblof_score)), y=cblof_score)
            
            anomaly_index = np.where(cblof_pred == abnormal_value)[0]
            for index in anomaly_index:
                fig_cblof_2.add_trace(go.Scatter(
                    x=[index, index],
                    y=[min(cblof_score), max(cblof_score)],
                    mode='lines',
                    line=dict(color='red', width=0.5),
                    showlegend=False
                ))
                
            for index in true_anomaly_index:
                fig_cblof_2.add_trace(go.Scatter(
                    x=[index],
                    y=[cblof_score[index]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        symbol='x',
                        size=10
                    ),
                    showlegend=False
                ))
            
            st.write('thresholds : ', cblof_threshold)
            
            fig_cblof_2.add_trace(go.Scatter(
                x=[0, len(cblof_score)],
                y=[cblof_threshold, cblof_threshold],
                mode='lines',
                line=dict(color='orange', width=1),
                showlegend=False
            ))
            
            predicted_anomaly_index = np.where(cblof_pred == 1)[0]
            for index in predicted_anomaly_index:
                fig_cblof_2.add_trace(go.Scatter(
                    x=[index, index],
                    y=[min(cblof_score), max(cblof_score)],
                    mode='lines',
                    line=dict(color='red', width=0.5),
                    showlegend=False
                ))


            st.plotly_chart(fig_cblof_2, use_container_width=True)


            st.write('Accuracy Score : ', accuracy_score(y_change, cblof_pred))           
            st.write('Confusion Matrix : ', confusion_matrix(y_change, cblof_pred))
            
            st.write('Classification Report : ' )
            st.code('Model Report:\n  ' + classification_report(y_change, cblof_pred))
            
            if st.button('Result Download'):
                total_result_data = pd.DataFrame({'real':y_change, 'pred':cblof_pred})
                total_result_data.to_csv('total_result_data.csv', index=False)

                csv_link = download_csv(total_result_data, 'CBLOF_Result.csv')
                st.markdown(csv_link, unsafe_allow_html=True)
            

        except:
            st.error('clustering_estimator를 확인해주세요.')
            st.error('clustering 방법을 변경해주세요.')



    st.markdown('----')
    
    st.markdown('### 3. IForest(Isolation Forest)')
    with st.expander('IForest 설명'):
        st.markdown('IF의 기본 가정은 이상치는 특별하고 적은 기준으로 쉽게 분리가 가능하다는 것이다.')
        st.markdown('아래 그림을 보면 xi는 정상 데이터이며, x0는 이상데이터이다.')
        images('./images/if_1.jpg')
        st.markdown('xi은 많은 기준을 사용해서 하나를 고립해야 한다.')
        st.markdown('반면 x0는 적은 기준을 사용해서 데이터를 분리할 수 있다는 특징이 있다.')
        st.markdown('실제로 각 데이터를 1000번 정도 분리를 한 결과는 아래와 같다.')
        images('./images/if_2.jpg')
        st.markdown('위의 그림에서 x축은 횟수이며, y축은 각 고립까지 걸린 depth이다.')
        st.markdown('xi의 경우에는 대략 13번 정도의 기준이 필요하고, x0의 경우에는 5번정도의 기준으로 고립이 되는 것을 볼 수 있다.')
        st.markdown('tree의 개수, 즉 반복 횟수는 대략 1000번정도 되면 average path length가 안정적으로 수렴한다고 할 수 있다.')
        st.markdown('물론 이러한 IF는 single blob은 잘 구분하지만 multiple blobs, sinusoidal의 경우에는 잘 구분하지 못한다.')
        images('./images/if_3.jpg')
        st.markdown('위의 그림을 보면 빨간색으로 표시한 부분에서는 IF가 잘 구분하지 못하는 것을 볼 수 있다.')
        st.markdown('이렇게 축에 수직으로 구분할 때의 한계점을 극복하기 위해 EIF(Extended Isolation Forest)가 제안되었다.')
        images('./images/if_4.jpg')
        st.markdown('EIF의 경우 대각선으로 데이터를 분할할 수도 있으며, 위의 한계점에 대해 잘 극복한 것을 위의 그림에서 확인할 수 있다.')            
        st.markdown('IF와 EIF의 경우 아래 논문을 넣었으니 다운받아서 확인할 수 있습니다.')
        st.markdown(' ')
        st.markdown(' ')

        file_path_if = "./paper/iforest.pdf"
        file_name_if = "iforest.pdf"
        download_pdf_file(file_path_if, file_name_if, 'Iforest 논문 다운로드')
        st.markdown(' ')
        file_path_eif = "./paper/eif.pdf"
        file_name_eif = "eif.pdf"
        download_pdf_file(file_path_eif, file_name_eif, 'EIF 논문 다운로드')

        
    st.markdown('### Hyperparameter Tuning')
    contamination_iforest = st.number_input('IForest contamination (0 ~ 0.8)', min_value=0.0, max_value=0.8, value=0.3, step=0.1, help = '이상치로 판단되는 데이터의 비율')
    n_estimators_iforest = st.number_input('IForest n_estimators (100 ~ 300)', min_value=100, max_value=300, value=100, step=100, help = 'Decision Tree의 개수, 개수가 증가할 수록 앙상블할 Tree의 개수가 증가')
    max_samples_iforest = st.number_input('IForest max_samples (1 ~ 500)', min_value=1, max_value=500, value=256, step=1, help = 'Decision Tree를 만들기 위한 샘플의 개수')
    bootstrap_iforest = st.selectbox('IForest bootstrap', [True, False], help = '샘플을 무작위로 추출하며, 독립적으로 작동 무작위성과 샘플링 개념을 같이 사용')

    iforest = IsolationForest(contamination=contamination_iforest, n_estimators=n_estimators_iforest, max_samples=max_samples_iforest, bootstrap=bootstrap_iforest, n_jobs=-1)
    
    if st.checkbox('IForest 실행'):
        isolation_pred, isolation_score = isolation_model(numeric_test, contamination_iforest, n_estimators_iforest, max_samples_iforest, bootstrap_iforest)

        if st.checkbox('IForest 결과 확인'):
            if normal_value == 0:
                y_change = np.where(y_test == 0, 1, -1)
            else:
                y_change = y_test

            st.write('Accuracy Score : ', accuracy_score(y_change, isolation_pred))
            
            st.write('Confusion Matrix : ', pd.DataFrame(confusion_matrix(y_change, isolation_pred), columns=[-1,1], index=[-1,1]))
            
            st.write('Classification Report : ' )
            st.code('Model Report:\n  ' + classification_report(y_change, isolation_pred))

            fpr_isolation, tpr_isolation, thresholds_isolation = roc_curve(y_change, isolation_score)
            st.write('AUC : ', auc(fpr_isolation, tpr_isolation))
            fig_isolation = px.area(x=fpr_isolation, y=tpr_isolation,  title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'))
            fig_isolation.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_isolation.update_xaxes(constrain='domain')
            fig_isolation.update_yaxes(scaleanchor="x", scaleratio=1)
        
            st.plotly_chart(fig_isolation, width = 1000, height = 1000)

        if st.checkbox('IForest 결과 시각화'):
            if normal_value == 0:
                y_change = np.where(y_test == 0, 1, -1)
            else:
                y_change = y_test

            fig_iso = px.line(x=range(len(isolation_score)), y=-isolation_score)
                
            # 실제 이상치
            true_anomaly_index = np.where(y_test == abnormal_value)[0]
            for index in true_anomaly_index:
                fig_iso.add_trace(go.Scatter(
                    x=[index],
                    y=[-isolation_score[index]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        symbol='x',
                        size=10
                    ),
                    showlegend=False
                ))
                
            predicted_anomaly_index = np.where(isolation_pred == -1)[0]
            for index in predicted_anomaly_index:
                fig_iso.add_trace(go.Scatter(
                    x=[index, index],
                    y=[min(-isolation_score), max(-isolation_score)],
                    mode='lines',
                    line=dict(color='red', width=1),
                    showlegend=False
                ))    
                
            fig_iso.add_trace(go.Scatter(
                x=[0, len(isolation_score)],
                y=[0, 0],
                mode='lines',
                line=dict(color='orange', width=1),
                showlegend=False
            ))
            
            st.plotly_chart(fig_iso, use_container_width=True)

            st.markdown('----')
            st.write('기존 IF threshold :', 0)
            isolation_threshold = st.text_input('IF threshold', value=1)
            isolation_threshold = float(isolation_threshold)

            isol_pred = np.where(-isolation_score > isolation_threshold, -1, 1)
            
            fig_iso_2 = px.line(x=range(len(isolation_score)), y=-isolation_score)
            
            anomaly_index = np.where(y_test == abnormal_value)[0]
            for index in true_anomaly_index:
                fig_iso_2.add_trace(go.Scatter(
                    x=[index],
                    y=[-isolation_score[index]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        symbol='x',
                        size=10
                    ),
                    showlegend=False
                ))
            
            st.write('thresholds : ', isolation_threshold)
            fig_iso_2.add_trace(go.Scatter(
                x=[0, len(isolation_score)],
                y=[isolation_threshold, isolation_threshold],
                mode='lines',
                line=dict(color='orange', width=1),
                showlegend=False
            ))
            
            predicted_anomaly_index = np.where(isol_pred == -1)[0]
            for index in predicted_anomaly_index:
                fig_iso_2.add_trace(go.Scatter(
                    x=[index, index],
                    y=[min(-isolation_score), max(-isolation_score)],
                    mode='lines',
                    line=dict(color='red', width=1),
                    showlegend=False
                ))


            st.plotly_chart(fig_iso_2, use_container_width=True)

            if normal_value == 0:
                y_change = np.where(y_test == 0, 1, -1)
            else:
                y_change = y_test

            st.write('Accuracy Score : ', accuracy_score(y_change, isol_pred))
            
            st.write('Confusion Matrix : ', pd.DataFrame(confusion_matrix(y_change, isol_pred), columns=[-1,1], index=[-1,1]))
            
            st.write('Classification Report : ' )
            st.code('Model Report:\n  ' + classification_report(y_change, isol_pred))

            if st.button('Result Download'):
                total_result_data = pd.DataFrame({'real':y_change, 'pred':isol_pred})
                total_result_data.to_csv('total_result_data.csv', index=False)

                csv_link = download_csv(total_result_data, 'IF_Result.csv')
                st.markdown(csv_link, unsafe_allow_html=True)


    st.markdown('----')
    st.markdown('### 4. RRCF(Robust Random Cut Forest)')
    with st.expander('RRCF 설명'):
        st.markdown('RRCF은 실시간으로 변화하는 데이터의 트렌드에 대한 이상치를 탐지하는 장점이 있습니다.')
        st.markdown('RRCF의 핵심 아이디어는 고차원의 데이터를 적은 차원의 하위 공간으로 나누고, 각 데이터 포인트가 차지하는 공간의 크기를 측정하여 이상치 탐지를 수행하는 것입니다.')
        st.markdown('RRCF는 각 트리가 독립적으로 생성되기 때문에 병렬화가 가능하며, 트리의 개수가 많을수록 정확도가 높아집니다.')
        st.markdown('작동 방식은 트리생성 -> 이상치 점수 계산 -> 트리 업데이트 순으로 진행됩니다.')
        st.markdown('### 1. 트리생성 :')
        st.markdown('RRCF는 랜덤 컷을 사용해 데이터 공간을 반복적으로 분할하며 트리를 생성합니다.')
        st.markdown('이 과정에서 각 노드는 하위의 공간이며, 데이터는 트리의 리프 노드에 할당됩니다.')

        st.markdown('### 2. 이상치 점수 계산 :')
        st.markdown('각 데이터 포인트에 대한 이상치 점수는 해당 포인트가 위치한 리프 노드의 깊이에 비례합니다.')
        st.markdown('이때 기존 IF와 다르게 RRCF는 깊이가 깊을수록 해당 데이터가 이상치라고 판단합니다.')
        st.markdown('전통적인 decision tree와는 반대인데, 이렇게 되는 이유는 해당 데이터가 추가되면 더 많은 분할을 거쳐야만 해당 데이터를 표현하는 노드를 만들 수 있기 때문입니다.')
        st.markdown('이것을 그림으로 나타내면 아래와 같습니다.')
        images('./images/rrcf1.png')
        st.markdown('그림을 확인해보면 데이터 x가 제거되면 트리로 표현할 때 tree의 depth가 1 감소하는 것을 볼 수 있습니다.')
        st.markdown('하지만 RRCF는 깊이를 직접적으로 score로 사용하지않고, Disp(displacement)라는 개념을 사용합니다.')
        st.markdown('Disp는 위에서 데이터 x를 제거한 것에서 생각해보면 x를 제거함으로써 depth 변화의 총합이 변화 했고, 이것을 Disp(x,p)으로 표현합니다.')
        st.markdown('하지만 이러한 disp를 바로 사용하게 되면 masking에 취약해지게 됩니다.')
        st.markdown('이러한 이유는 이상 데이터들끼리 서로 뭉치게 되면 disp가 작아지게 되며, 이러한 이유로 결과적으로 모델이 잘 분류하지 못한다는 한계점이 생겨버립니다.')
        st.markdown('이러한 한계점을 극복하기 위해 Collusive Displacement(CoDisp)라는 개념을 사용하게 됩니다.')
        st.markdown('x의 집합을 C라고 하면, 우리는 데이터 셋 S에서 x만 제거하는 것이 아니라 C를 제거했을 때 발생하는 depth 변화의 총합을 이상 스코어로 사용합니다.')
        st.markdown('여기서 집합 C의 크기가 커지면 C를 제거했을 때 트리의 변화가 크게 변화함으로, Disp(x,S)가 아닌 Disp(C,s)/|C|를 사용합니다.')
        st.markdown('이러한 방식으로 CoDisp를 사용하게 되면, 이상 데이터들끼리 뭉치게 되어도 이상 스코어가 커지게 되어 masking에 취약해지는 문제를 해결할 수 있습니다.')
        st.markdown('물론 집합 C가 속한 집합을 정확하게 규정할 수 없으므로, x를 포함하는 모든 부분집합을 고려하게 됩니다.')
        st.markdown('즉 x의 위에 속해 있는 집합의 크기를 보며, 그 중 최대값을 구하고, 해당 최대값을 사용해 계산한 뒤, 평균을 내주면 최종적인 CoDisp를 구할 수 있습니다.')
        images('./images/rrcf2.jpg')

        st.markdown(' ')

        st.markdown('### 3. 트리 업데이트 :')
        st.markdown('RRCF는 실시간으로 데이터가 추가되거나 제거될 때마다 트리를 업데이트합니다.')
        st.markdown('이때 트리를 업데이트할 떄 전체 트리를 구성하는 데이터의 개수가 256개라고 생각하고 예시를 들겠습니다.')
        st.markdown('266번째 데이터가 들어오면 가장 첫번째의 데이터가 제거되고, 267번째 데이터가 들어오면 2번째 데이터가 제거되는 식으로 진행됩니다.')
        st.markdown('이런 방식으로 트리를 업데이트 하기때문에 RRCF에서는 실시간으로 변화하는 데이터의 트랜드를 반영할 수 있는 것입니다.')


    st.markdown('### Hyperparameter Tuning')
    num_trees = st.number_input('RRCF num_trees (100 ~ 500)', min_value=100, max_value=500, value=256, step=10,
                                help = 'Decision Tree의 개수, 값이 증가하면 앙상블에 포함된 Tree의 개수가 증가함')
    shingle_size = st.number_input('RRCF shingle_size (1 ~ 100)', min_value=1, max_value=100, value=8, step=1, 
                                   help = '각 Tree에서 무작위로 선택되는 sample의 개수')
    tree_size = st.number_input('RRCF tree_size (1 ~ 15)', min_value=1, max_value=15, value=8, step=1, 
                                help = 'Tree의 분할 개수, 값이 증가하면 Tree의 깊이가 깊어짐')
    
    n = len(numeric_test)
    #shingle_size = len(numeric_test.columns)
    sample_size_range = (n // shingle_size)
    
    if st.checkbox('RRCF 실행'):
        numeric_test_value = numeric_test.values

        if normal_value == 0:
            y_change = np.where(y_test == 0, 1, -1)
        else:
            y_change = y_test

        tree = rrcf.RCTree(numeric_test_value)

        if st.checkbox('Tree 확인'):
            st.code(tree)

        if st.checkbox('RRCF 시각화 및 결과 확인'):
            rrcf_score = rrcf_model(numeric_test_value, num_trees, shingle_size, tree_size)

            fig_rrcf = px.line(x=range(len(rrcf_score)), y=rrcf_score)
                
            true_anomaly_index = np.where(y_test == abnormal_value)[0]
            for index in true_anomaly_index:
                fig_rrcf.add_trace(go.Scatter(
                    x=[index],
                    y=[rrcf_score[index]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        symbol='x',
                        size=10
                    ),
                    showlegend=False
                ))
            
            st.plotly_chart(fig_rrcf, use_container_width=True)


            st.markdown('----')

            rrcf_threshold = st.text_input('threshold', value=10)
            rrcf_threshold = float(rrcf_threshold)

            st.write(len(x_test), len(rrcf_score))

            rrcf_pred = np.where(rrcf_score > rrcf_threshold, 1, 0)
            
            fig_rrcf_2 = px.line(x=range(len(rrcf_score)), y=rrcf_score)
            for index in true_anomaly_index:
                fig_rrcf_2.add_trace(go.Scatter(
                    x=[index],
                    y=[rrcf_score[index]],
                    mode='markers',
                    marker=dict(
                        color='red',
                        symbol='x',
                        size=10
                    ),
                    showlegend=False
                ))
                
            predicted_anomaly_index = np.where(rrcf_pred == 1)[0]
            for index in predicted_anomaly_index:
                fig_rrcf_2.add_trace(go.Scatter(
                    x=[index, index],
                    y=[min(rrcf_score), max(rrcf_score)],
                    mode='lines',
                    line=dict(color='red', width=0.5),
                    showlegend=False
                ))    
            
                
            st.write('thresholds : ', rrcf_threshold)
            fig_rrcf_2.add_trace(go.Scatter(
                x=[0, len(rrcf_score)],
                y=[rrcf_threshold, rrcf_threshold],
                mode='lines',
                line=dict(color='orange', width=1),
                showlegend=False
            ))


            st.plotly_chart(fig_rrcf_2, use_container_width=True)

            if normal_value == 1:
                y_change = np.where(y_test == 1, 0, 1)
            else:
                y_change = y_test

            st.write('Accuracy Score : ', accuracy_score(y_change, rrcf_pred))           
            st.write('Confusion Matrix : ', confusion_matrix(y_change, rrcf_pred))
            
            st.write('Classification Report : ' )
            st.code('Model Report:\n  ' + classification_report(y_change, rrcf_pred))

            fpr_rrcf, tpr_rrcf, thresholds_rrcf = roc_curve(y_change, rrcf_score)
            st.write('AUC : ', auc(fpr_rrcf, tpr_rrcf))
            fig_rrcf = px.area(x=fpr_rrcf, y=tpr_rrcf,  title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'))
            fig_rrcf.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_rrcf.update_xaxes(constrain='domain')
            fig_rrcf.update_yaxes(scaleanchor="x", scaleratio=1)

            st.plotly_chart(fig_rrcf, width = 1000, height = 1000)

            if st.button('Result Download'):
                total_result_data = pd.DataFrame({'real':y_change, 'pred':rrcf_pred})
                total_result_data.to_csv('total_result_data.csv', index=False)

                csv_link = download_csv(total_result_data, 'RRCF_Result.csv')
                st.markdown(csv_link, unsafe_allow_html=True)

    st.markdown('----')
    with st.expander('추가적으로 공부할 Anomaly Detection 관련 내용'):
        st.markdown('Anomaly Detection에서 다양한 모델에 아는 것도 중요합니다.')
        st.markdown('하지만 모델을 사용하기 전에 데이터가 어떻게 생겼는지, 어떤 특징을 가지고 있는지 알아야 합니다.')
        st.markdown('또한 모델링을 한 결과를 가지고 그저 이상과 정상을 구분하는 것보다, thershold를 직접 조정하고, anomaly score가 어떻게 분포하고 있는지 확인 하는 것도 중요합니다.')
        st.markdown('위의 과정에서 전부다 threshold를 직접 조절할 수 있게 구현하였으니 꼭 직접 해당 내용을 직접 구현하며, 어떠한 결과가 나오는지 확인해보세요.')
        st.markdown('Anomaly Detection이라는 것은 정상과 이상을 구분하는 것이지만 보통의 binary classification과는 다른 점이 있습니다.')
        st.markdown('우선 비정상의 데이터가 매우 적은 경우 Anomaly Decetion의 문제로 가져갑니다.')
        st.markdown('그렇게 하는 이유는, 어느 정도의 불균형이라고 한다면 over sampling, under sampling을 통해 불균형을 해소할 수 있습니다.')
        st.markdown('하지만 99:1 정도의 불균형이라면, 불균형을 해소하는 것이 불가능합니다.')
        st.markdown('또한 그 적은 불량 데이터가 불량을 대표할 수 있는가에 대한 생각도 해봐야 합니다.')
        st.markdown('')
