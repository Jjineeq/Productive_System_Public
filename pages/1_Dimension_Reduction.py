import streamlit as st
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from umap import UMAP
import base64

st.wide = True
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def Dimension_Reduction():
    st.title("Dimension Reduction")

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

@st.cache_resource
def t_sne(data):
    tsne = TSNE(n_components = 2, n_jobs=-1)
    tsne_result = tsne.fit_transform(data)
    tsne_df = pd.DataFrame(tsne_result, columns= ['1', '2'])
    tsne_df['label'] = y.values

    return tsne_df
    
        
st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
.name{
    font-family:Malgun Gothic;
    font-size:20px !important;
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
.centered-image {
    display: flex;
    justify-content: center;
}
}

</style>
""", unsafe_allow_html=True) 

st.title('Diemension Reduction')
st.markdown('----')
st.subheader('추천 데이터 셋 : TFTLCD')

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


    st.markdown('----')
    st.markdown('#### 차원 축소 Train Data 선택')
    select_data = st.selectbox('y에 따라 데이터 선택', ['정상', '전체'])
    if select_data == '정상':
        input_normal = normal_value
        input_abnormal = abnormal_value
        
        df_normal = y_train[y_train == int(input_normal)]
        select_index = df_normal.index
        x = x_train.loc[select_index]
        y = y_train.loc[select_index]
        
    else:
        x = x_train
        y = y_train
        

    numeric_columns = x.select_dtypes(include=['int64', 'float64']).columns
    numeric_train = x[numeric_columns]
    
    st.markdown('데이터 스케일')
    if st.checkbox('StandardScaler'):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numeric_data = scaler.fit_transform(numeric_train)
    else:
        numeric_data = numeric_train

    st.markdown('----')

    st.markdown('## PCA')
    with st.expander('PCA 설명'):
        st.markdown('PCA를 그림으로 표현하면 아래와 같습니다.')
        images('./images/PCA.png')
        st.markdown('PCA의 목표는 주어진 데이터의 변수(특성)들 간의 상관관계를 이용하여 새로운 변수들을 생성하는 것입니다.')
        st.markdown('PCA는 Feature extraction의 방법이며, 선형 변환을 통해 데이터를 저차원 공간으로 투영합니다.')
        st.markdown('PCA는 데이터의 분산을 최대한 보존하는 방향으로 축을 생성합니다.')
        st.markdown('PCA를 진행하기 위해서는 데이터를 표준화 해야합니다.')
        st.markdown('이후 공분산 행렬을 계산하게 됩니다. 공분산이라는 것은 변수들 간의 상관관계를 나타내며, PCA는 이 공분산 행렬을 이용하여 고유값 분해를 진행합니다.')
        st.markdown('고유값은 주성분의 중요도를 나타내며, 고유벡터는 주성분의 방향을 나타냅니다.')
        st.markdown('고유값을 기준으로 주성분을 선택하며, 주성분을 고유값이 큰 순서대로 선택하여 차원을 축소하게 됩니다.')
        st.markdown('PCA는 차원 축소를 위해 사용되는 것이지만, 데이터의 특성을 설명하는데도 사용됩니다.')
        st.markdown('PCA에서 이후 다른 분석 기법과 연결되는 것이 많습니다.')
        st.markdown('간단한 예시로, KPCA라는 커널 기반 PCA, PCA, MDS와 같은 선형의 한계점을 극복하기 위해 ISOMAP으로 발전하기도 하였습니다.')
        st.markdown('공분산의 개념이 포함되어 있어, 향후 Hoteling T2와 같은 이상치 탐지에도 사용되니 수식을 이해하시는 것을 추천드립니다.')
        st.markdown('PCA에서는 공분산이 들어가 있어, 이상치에 민감합니다. 그러기 때문에 RPCA와 같은 방법론도 나오기도 했습니다.')
        st.markdown('PCA를 할때에는 정상 데이터만을 사용해 학습한다고 하지만, 공정 데이터는 보통 깔끔하지 못해, 이상치가 포함되어 있는 경우도 있으니, 상황에 따라 사용하시면 됩니다.')
        st.markdown(' ')
        st.markdown(' ')
        st.markdown('PCA를 설명하는 DSBA 강의 링크는 아래에 있습니다.')
        st.markdown('https://www.youtube.com/watch?v=bEX6WPMiLvo')
        st.markdown(' ')
        st.markdown('다양한 데이터에 PCA를 적용하는 사례를 포함한 블로그 글입니다.')
        st.markdown('https://datascienceschool.net/02%20mathematics/03.05%20PCA.html')

    # pca_diemnsion = st.slider('차원 축소 수', 1, 5, 2)
    # pca = PCA(n_components = pca_diemnsion)

    # if st.checkbox('PCA 실행 및 시각화', help = '기본값은 2차원으로 시각화를 합니다.'):
    #     pca.fit(numeric_data)
    #     pca_result = pca.fit_transform(numeric_data)
    #     pca_df = pd.DataFrame(pca_result, columns= range(1, pca_diemnsion+1))

    #     pca_df['label'] = y.values
    #     eigenvectors = pca.components_

        
    #     st.write(pca_df)
    #     # scatter plot

        
    #     if st.checkbox('3차원 시각화'):
    #         if pca_diemnsion < 3:
    #             st.warning('차원 축소 수를 3 이상으로 설정해주세요.')
    #             st.stop()
    #         else:
    #             fig = px.scatter_3d(pca_df, x=1, y=2, z=3, color='label', width = 1200, height = 800)
    #             scale = st.slider('scale', 0, 500, 50)
                
    #             for i in range(len(eigenvectors)):
    #                 fig.add_trace(go.Scatter3d(x=[0, eigenvectors[i, 0]* scale] , y=[0, eigenvectors[i, 1] * scale], z=[0, eigenvectors[i, 2] * scale], mode='lines', name=f'Eigenvector {i+1}'))
    #     else:
    #         fig = px.scatter(pca_df, x=1, y=2, color='label', width = 1200, height = 800)
    #         scale = st.slider('scale', 1, 300, 50)
    #         for i in range(len(eigenvectors)):
    #             fig.add_trace(go.Scatter(x=[0, eigenvectors[i, 0]* scale * 10] , y=[0, eigenvectors[i, 1] * scale * 10], mode='lines', name=f'Eigenvector {i+1}'))
    #         fig.update_layout(showlegend=True)
    #     st.plotly_chart(fig, theme = 'streamlit')
    
    pca_diemnsion = st.slider('차원 축소 수', 1, 5, 2)
    pca = PCA(n_components = pca_diemnsion)

    if st.checkbox('PCA 실행 및 시각화', help = '기본값은 2차원으로 시각화를 합니다.'):
        pca.fit(numeric_data)
        pca_result = pca.fit_transform(numeric_data)
        pca_df = pd.DataFrame(pca_result, columns= range(1, pca_diemnsion+1))
        
        pca_df['label'] = y.values
        eigenvectors = pca.components_
        original_feature_names = numeric_columns
        explained_variance_ratio = pca.explained_variance_ratio_
        st.write(pca_df)
        # scatter plot
        if st.checkbox('3차원 시각화'):
            if pca_diemnsion < 3:
                st.warning('차원 축소 수를 3 이상으로 설정해주세요.')
                st.stop()
            else:
                fig = px.scatter_3d(pca_df, x=1, y=2, z=3, color='label', width = 1200, height = 800)
                scale = st.slider('scale', 0, 500, 50)
                for i in range(len(eigenvectors)):
                    for j in range(len(eigenvectors[i])):
                        fig.add_trace(go.Scatter3d(x=[0, eigenvectors[0][j]* scale] , y=[0, eigenvectors[1][j] * scale], z=[0, eigenvectors[2][j] * scale], mode='lines', name=numeric_columns[j]))



        else:
            fig = px.scatter(pca_df, x=1, y=2, color='label', width = 1200, height = 800)
            scale = st.slider('scale', 1, 300, 1)
            for i in range(len(eigenvectors)):
                for j in range(len(eigenvectors[i])):
                    fig.add_trace(go.Scatter(x=[0, eigenvectors[0][j]* scale * 10] , y=[0, eigenvectors[1][j] * scale * 10], mode='lines', name=numeric_columns[j]))
            fig.update_layout(showlegend=True)
        st.plotly_chart(fig, theme = 'streamlit')

        if st.checkbox('Show Loadings Plot'):
            loadings = pca.components_.T

            fig_loadings = go.Figure()
            
            if st.checkbox('lodings plot 3차원 시각화'):
                if pca_diemnsion < 3:
                    st.warning('차원 축소 수를 3 이상으로 설정해주세요.')
                    st.stop()
                else:
                    for i, feature in enumerate(original_feature_names):
                        #fig_loadings.add_trace(go.Scatter3d(x=[0, loadings[i, 0]], y=[0, loadings[i, 1]], z=[0, loadings[i, 2]], mode='lines', name=feature))
                        fig_loadings.add_trace(go.Scatter3d(x=[0, loadings[i, 0]], y=[0, loadings[i, 1]], z=[0, loadings[i, 2]], mode='lines'))
                        fig_loadings.add_trace(go.Scatter3d(x=[loadings[i, 0]], y=[loadings[i, 1]], z=[loadings[i, 2]], mode='markers+text', text=[feature], textposition='middle right'))

                    fig_loadings.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'), width=800, height=800)

                    
            else:
                fig_loadings.add_trace(go.Scatter(x=loadings[:, 0], y=loadings[:, 1], mode='markers+text', text=original_feature_names, textposition='top center'))
                fig_loadings.update_xaxes(title='PC1')
                fig_loadings.update_yaxes(title='PC2')
                fig_loadings.update_layout(showlegend=False, width = 800, height = 800)
                
                for i, feature in enumerate(original_feature_names):
                    fig_loadings.add_trace(go.Scatter(x=[0, loadings[i, 0]], y=[0, loadings[i, 1]], mode='lines', name=feature))
                    fig_loadings.add_annotation(x=loadings[i, 0], y=loadings[i, 1], ax=0, ay=0, xref='x', yref='y', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#636363')


            st.plotly_chart(fig_loadings, use_container_width=True)

    

    if st.checkbox('PCA 설명력'):
        pca_result = pca.fit_transform(numeric_data)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        st.write('설명력', explained_variance)
        st.write('누적 설명력', cumulative_variance)
    
    st.markdown('----')
    st.markdown('## t-SNE')
    with st.expander('t-SNE 설명'):
        st.markdown('t-SNE는 t-distributed stochastic neighbor embedding의 약자로, 고차원의 데이터를 저차원으로 축소하는 비선형 방법입니다.')
        st.markdown('데이터 포인트 간의 유사도를 측정하여 고차원 공간에서의 거리를 저차원 공간에서의 거리로 매핑합니다.')
        st.markdown('확률적인 방식을 사용하여 데이터 포인트를 새로운 차원으로 이동시킵니다.')
        st.markdown('t-SNE를 사용할때에는 데이터의 스케일을 꼭 할 필요는 없습니다.')
        st.markdown('하지만 일부 데이터, 스케일이 너무 큰 차이가 발생하는 경우에는 스케일링을 해주는 것이 좋습니다.')
        st.markdown('차원을 축소하여 데이터의 분포를 확인하는 것도 좋지만, 우선적으로 데이터의 분포를 확인하고 차원을 축소하는 것이 좋습니다.')
        st.markdown('t-SNE는 유사도를 계산하게 됩니다. 이때 사용되는 metric은 유클리디안 또는 코사인 유사도와 같은 거리 기반의 metric을 사용합니다.')
        st.markdown('각 데이터 간의 이웃한 데이터 포인트들과 유사도를 계산하여, 이를 기반으로 차원을 축소합니다.')
        st.markdown('유사도는 정규분포에서 확률 밀도에 비례하여 이웃을 선택하면, 두 포인트가 이웃으로 선택한다는 조건부 확률로 계산이 됩니다.')
        st.markdown('조건부 확률의 차이의 합을 최소화 하기 위해 t-SNE는 gradient descent 방식을 사용해서 KL divergence를 최소화 하는 방향으로 이동합니다.')
        st.markdown('KL divergence는 두 확률 분포의 차이를 계산하는 방법입니다. KL divergence은 한 확률 분포가 두번째 예상 확률 분포와 어떻게 다른지 측정하는 척도입니다.')
        st.markdown('아래는 t-SNE의 과정을 표현한 그림입니다.')
        images('./images/t_sne_2.png')
        st.markdown('하지만 t-SNE는 input feature로 사용하는데는 제한이 있습니다.')
        st.markdown('그렇기 때문에 t-SNE는 주로 시각화를 위해 사용됩니다.')
        st.markdown('t-SNE를 sklearn을 사용해서 사용할 수 있으며, multi processing을 할 수 있는 MulticoreTSNE를 사용할 수도 있습니다.')
        st.markdown('t-SNE는 모델 작동 시간이 오래 걸리기 때문에 Muti processing을 사용하는 것이 좋습니다.')
        st.markdown(' ')
        st.markdown(' ')
        st.markdown('아래는 t-SNE의 파라미터에 대해 나와있는 URL입니다.')
        st.markdown('https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html')
        st.markdown(' ')
        st.markdown('t-SNE는 t분포를 사용하는데, CLT에 의해 정규분포와 같아지는 경우가 있습니다.')
        st.markdown('CLT는 중심 극한 정리이며, 이항분포, 포아송분포, 지수분포 등에서 표본의 개수가 많아지면 정규분포에 가까워진다는 것입니다.')
        st.markdown('각 분포가 언제, 어떻게 차이가 있는지 이해하고 있는 것이 좋습니다. 이외의 분포에 대해서 공부하시는 것을 추천드립니다.')
        st.markdown('아래는 t-분포와 정규분포의 차이를 보여주는 그림입니다.')
        st.markdown('또한 정규분포를 사용해도 되는데 두 ')
        images('./images/t_sne_1.png')
        st.markdown('해당 내용은 필수적으로 알아야 하는 내용으로, 공부를 하시는 것을 추천드립니다. 아래는 CLT에 대한 URL입니다.')
        st.markdown('https://www.youtube.com/watch?v=YAlJCEDH2uY')

    if st.checkbox('t-SNE 실행 및 시각화'):
        # tsne = TSNE(n_components = 2, n_jobs=-1)
        # tsne_result = tsne.fit_transform(numeric_data)
        # tsne_df = pd.DataFrame(tsne_result, columns= ['1', '2'])
        # tsne_df['label'] = y.values
        tsne_df = t_sne(numeric_data)
        st.write(tsne_df)

        fig = px.scatter(tsne_df, x='1', y='2', color='label', width = 1200, height = 800)
        st.plotly_chart(fig, theme = 'streamlit')
    
    st.markdown('----')
    st.markdown('## UMAP')
    with st.expander('UMAP 설명'):
        st.markdown('UMAP은 Uniform Manifold Approximation and Projection의 약자로, 고차원의 데이터를 저차원으로 축소하는 방법입니다.')
        st.markdown("UMAP은 t-SNE와 유사한 원리를 가지고 있지만, 더 빠른 계산 속도와 더 좋은 보존성능을 제공하면서도 선형 구조를 유지하는 특징을 가지고 있습니다.")
        st.markdown('UMAP은 데이터를 저차원 임베딩 공간으로 변환하는 데에 주로 사용됩니다.')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('UMAP learn 라이브러리를 사용해서 UMAP을 쉽게 사용할 수 있습니다.')
        st.markdown('UMAP learn 라이브러리의 링크는 아래와 같습니다.')
        st.markdown('https://umap-learn.readthedocs.io/en/latest/api.html#umap.umap_.UMAP')
        st.markdown(' ')
        st.markdown('UMAP은 리만 기하학을 사용해서 수학적으로 증명된 부분이 있습니다. 논문을 참고하시는 것을 추천드립니다.')
        file_path_umap = "./paper/UMAP.pdf"
        file_name_umap = "UMAP.pdf"
        download_pdf_file(file_path_umap, file_name_umap, 'UMAP 논문 다운로드')

    
    if st.checkbox('UMAP 실행 및 시각화'):
        umap = UMAP(n_components=2)
        umap_result = umap.fit_transform(numeric_train)
        umap_df = pd.DataFrame(umap_result, columns= ['1', '2'])
        umap_df['label'] = y.values
        st.write(umap_df)
        fig = px.scatter(umap_df, x='1', y='2', color='label', width = 1200, height = 800)
        st.plotly_chart(fig, theme = 'streamlit')


    st.markdown('----')
    with st.expander('추가적으로 공부할 차원 축소 방법'):
        st.write('PCA, t-SNE, UMAP 이외에도 다양한 차원 축소 방법이 있습니다.')
        st.write('AE(AutoEncoder), LDA(Linear Discriminant Analysis), MDS(Multi Dimensional Scaling), ISOMAP(Isometric Feature Mapping) 등이 있습니다.')
        st.write('사설로 고차원의 데이터는 어떻게 생겼는지 알 수 없습니다.')
        st.write('공정 데이터의 경우에는 데이터의 상관관계가 높은 편이 많아, PCA로 차원을 축소해도 데이터의 특성을 잘 나타내는 경우가 많이 있습니다.')
        st.write('하지만 이것마저 실제로 해본 결과 10건중 1~2건 정도만 가능합니다.')
        st.write('어떤 방법을 사용하는 것이 특성을 잘 보존하며 차원을 축소할 수 있는지는 모르기 때문에 다양한 방법을 사용해야 합니다.')
        st.write('매우 귀찮은 과정이고, 시간이 보다 많이 소요되지만')
        st.write('다양한 방법론을 알고, 어떤 원리로 데이터를 표현하는지 알고 있다면, EDA과정을 지나 모델링에 큰 도움이 될것으로 생각합니다.')
        st.write('모델의 input에 바로 넣는 것보다 데이터의 특성 자체를 파악하는데 중점을 두는 것을 추천드립니다.')



        st.markdown('----')
        st.markdown('## AE(AutoEncoder)')
        st.markdown('AE는 AutoEncoder의 약자로, 차원 축소 방법 중 하나입니다.')
        images('./images/AE.png')
        st.markdown('AE는 위와 같이 입력 데이터를 압축하는 인코더와 압축된 데이터를 복원하는 디코더로 구성되어 있습니다.')
        st.markdown('AE는 차원 축소로도 사용이 되지만, 이상 탐지, 노이즈 제거, 이미지 생성 등에도 사용이 됩니다.')
        st.markdown('또한 AE는 이후 LSTM AE, VAE, GAN 등으로 발전되어 사용되고 있습니다.')
        st.markdown('AE를 실행하는 간단한 예시가 있는 URL은 아래와 같습니다.')
        st.markdown('https://www.kaggle.com/code/saivarunk/dimensionality-reduction-using-keras-auto-encoder')
        file_path_ae = "./paper/NLPCAbyusingANN.pdf"
        file_name_ae = "NLPCAbyusingANN.pdf"
        st.write('아래는 AE 논문입니다. 논문을 읽는 것도 좋긴하지만 AE의 경우는 너무 많이 응용이 되어 bottle neck layer의 개념만 알고 있어도 추후 다른 논문을 읽을 때 도움이 될 것으로 생각합니다.')
        download_pdf_file(file_path_ae, file_name_ae, 'AE 논문 다운로드')


        st.markdown('----')
        st.markdown('## LDA(Linear Discriminant Analysis)')
        st.markdown('LDA는 데이터를 가장 잘 분리할 수 있는 축을 찾아 데이터를 사영하는 방법입니다.')
        st.markdown('LDA는 분류 문제에서 많이 사용이 됩니다.')
        st.markdown('기존 차원 축소와 다른점은 해당 데이터의 레이블 정보가 포함된다는 것입니다.')
        st.markdown('LDA는 PCA와 유사한 방법으로, PCA는 데이터의 분산을 최대화하는 축을 찾는 것이고, LDA는 데이터의 분산을 최대화하면서 클래스 간의 거리를 최대화하는 축을 찾습니다.')
        st.markdown('LDA를 이미지로 나타낸 것은 아래와 같습니다.')
        images('./images/LDA.png')
        st.markdown('LDA를 사용해서 iris데이터를 분류하는 예시가 인터넷에 많이 나와있습니다.')
        st.markdown('아래 URL을 참고하시면 좋을 것 같습니다.')
        st.markdown('https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-18-%EC%84%A0%ED%98%95%ED%8C%90%EB%B3%84%EB%B6%84%EC%84%9DLDA')
        st.markdown('2진 분류문제에서도 사용이 가능한데, 2개의 레이블이 있으니 1개의 축으로 분류가 가능합니다.')
        st.markdown('즉 n개의 레이블이 존재하면 n-1차원으로 축소가 가능합니다.')
        st.markdown('하나의 값으로 나오게 되면, threshold를 설정해 분류를 할 수 있습니다.')

        st.markdown('----')
        st.markdown('## MDS(Multi Dimensional Scaling)')
        st.markdown('MDS는 데이터의 거리를 보존하는 방법이며, PCA와 비슷한 방법입니다.')
        st.markdown('객체의 거리를 보존하는 축을 찾는 것이 주 목적이며 PCA와 비교하는 표는 아래와 같습니다.')
        st.markdown(' ')
        표 = """
        | |주성분분석(PCA)|다차원스케일링(MDS)| 
        |:---:|:---:|:---:|
        |데이터|d차원 공간상에 있는 n개의 인스턴스 {n x d 행렬로 부터 시작} | n개의 인스턴스 간의 근접도 행렬 {Dn x n 행렬로 부터 시작}|
        |목적| 원 데이터 분산을 보존하는 기저의 부분집합 찾기 | 인스턴스의 거리 정보를 보존하는 좌표계 찾기 |
        |출력값| 1) d개의 고유벡터(eigenvectors), 2) d개의 고윳값(eigenvalues) | d차원에 있는 각 인스턴스의 좌표값 | 
        """
        st.markdown(표) 
        st.markdown(' ')
        st.markdown(' ')
        st.markdown('PCA를 적용하는 것보다 더 넓은 범위에서 사용이 가능합니다.')
        st.markdown('유사도를 측정하는 데이터에 적용을 할 수 있습니다.')
        st.markdown('즉 X(d by n)를 D(n by n)로 바꾸는 것은 가능하지만, D는 X로 바꿀 수 없다는 것입니다. (해당내용은 예시를 보는 것을 추천드립니다.)')
        st.markdown('D(n x n) -> B(n x n) -> X(d x n)의 순서로 진행되며, 기본 가정으로 모든 변수의 평균 값은 0으로 가정하며 진행됩니다.')
        st.markdown('D = distance matrix, B = inner product matrix, X = embedding matrix')
        st.markdown('즉 D -> B는 inner product matrix의 값들을 distance matrix의 값들의 선형 결합으로 표현하겠다는 것입니다.')
        st.markdown('아래는 MDS의 과정을 그림으로 표현한 것입니다.')
        images('./images/MDS.png')
        st.markdown('증명에 관한 DSBA 강의 링크는 아래와 같습니다.')
        st.markdown('https://www.youtube.com/watch?v=Yv00AT4pLC4')
        st.markdown('해당 내용을 확인하고 추후 직접 확인하시려면 아래 링크를 추천드립니다.')
        st.markdown('https://velog.io/@euisuk-chung/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%B0%A8%EC%9B%90%EC%B6%95%EC%86%8C-%EB%B3%80%EC%88%98-%EC%B6%94%EC%B6%9C%EB%B2%95-Multi-Dimensional-Scaling-MDS')



        st.markdown('----')
        st.markdown('## ISOMAP(Isometric Feature Mapping)')
        st.markdown('ISOMAP은 LLE(Locally Linear Embedding)과 함께 소개됩니다.')
        st.markdown('LLE는 앞서 설명한 MDS와 약간 다르게, 최인접 이웃의 정보, locality에 포커싱합니다.')
        st.markdown('LLE의 과정을 그림으로 표현하면 아래와 같습니다.')
        images('./images/LLE.png')
        st.markdown('k개의 nearest neighbors를 찾고, 이웃 점들간의 구조를 weighted graph로 만듭니다.')
        st.markdown('이렇게 주변의 locality를 유지하면서 임베딩 공간을 학습하게 됩니다.')
        st.markdown('이렇게 이웃의 정보만 저장하면서 보존해도 golbal structure가 보존됩니다.')
        st.markdown('즉 이웃간의 구조를 보존하다보니, 이웃의 이웃으로 점차 이동하면서 어떠한 정보가 한 축으로 학습되게 됩니다.')
        st.markdown(' ')
        st.markdown('그렇다면 ISOMAP과 어떤 차이점이 존재하는지 확인해보겠습니다.')
        st.markdown('아래 그림을 보면 A, B, C가 있습니다.')
        images('./images/ISOMAP.png')
        st.markdown('본래 Euclidean distance로 점의 거리를 계산하면 A와 같은 거리를 구하게 됩니다.')
        st.markdown('하지만 실질적으로 우리의 거리는 A를 사용하지 않습니다.')
        st.markdown('manifold에서 점들 간의 거리를 nearest neighbor graph에서의 점들간의 최단 경로로 사용합니다.')
        st.markdown('즉 B에서와 같이 이웃들을 연결해서 거리를 계산하는 것입니다.')
        st.markdown('이렇게 휘어진 띄를 평평하게 피게 되면 우리는 C와 같게 되며, 이것이 ISOMAP의 핵심입니다.')
        st.markdown('고차원의 데이터는 우리의 눈으로 볼 수 없습니다. 그렇게 때문에 어떻게 생겼는지 모르고, 어떤 거리를 사용하는 것이 좋을지 모릅니다.')
        st.markdown('그렇기 때문에 ISOMAP과 같이 manifold에 따라 거리를 계산하는 것이 좋을 수도 있습니다.')
        st.markdown('아래는 ISOMAP 논문입니다. LLE와 함께 있으니 읽어 보시면 좋을 것 같습니다.')

        file_path_isomap = "./paper/isomap.pdf"
        file_name_isomap = "isomap.pdf"
        download_pdf_file(file_path_isomap, file_name_isomap, 'ISOMAP 논문 다운로드')
        