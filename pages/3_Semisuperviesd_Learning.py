import streamlit as st
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
st.wide = True
import sys
sys.path.append('./functions/SVDD-Python-master/')
sys.path.append('./functions/pyod-master/')
from pyod.models.deep_svdd import DeepSVDD
from sklearn.preprocessing import StandardScaler
import base64

st.markdown("")
st.title('Semi-Supervised Learning')
st.markdown('----')
st.subheader('추천 데이터 셋 : TFTLCD')
# st.markdown('----')
# st.markdown('## Self learning')

def semisupervised_learning():
    st.title("Semi-Supervised Learning")

@st.cache_resource(ttl=24*60*60)
def deepsvdd(train, test, hidden_activation, output_activation, optimizer, epochs, batch_size):
    deepsvdd = DeepSVDD(hidden_activation = hidden_activation, output_activation = output_activation, optimizer = optimizer,
                        epochs = epochs, batch_size = batch_size)
    

    deepsvdd.fit(train)
    proba = deepsvdd.predict_proba(test)
    
    predict = deepsvdd.predict(test)

    return predict, proba

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
    
    st.markdown('----')
    st.markdown('데이터 스케일')
    if st.checkbox('StandardScaler'):
        scaler = StandardScaler()
        numeric_train = scaler.fit_transform(numeric_train)
        numeric_test = scaler.transform(numeric_test)
        st.write('스케일 완료')
        numeric_test = pd.DataFrame(numeric_test, columns=numeric_columns, index = x_test.index)
        numeric_train = pd.DataFrame(numeric_train, columns=numeric_columns, index = x_train.index)

    # st.markdown('Self learning은 레이블이 없는 데이터를 사용하여 모델을 학습시킨다. 이때, 레이블이 없는 데이터를 사용하여 학습시키기 때문에, 레이블이 있는 데이터를 사용하여 학습시키는 것보다 오류가 더 많이 발생할 수 있다. \
    #             이러한 오류를 줄이기 위해, Self learning은 레이블이 없는 데이터를 사용하여 학습시킨 후, 레이블이 있는 데이터를 사용하여 학습시킨다. \
    #             이때, 레이블이 있는 데이터를 사용하여 학습시킬 때, 레이블이 있는 데이터 중에서 모델이 가장 확신하는 데이터만 사용한다. 이러한 과정을 반복하여 모델을 학습시킨다.')
    
    # # Self Training
    # st.markdown('레이블이 있는 데이터를 사용하여 모델을 학습시킨 후, 레이블이 없는 데이터를 사용하여 모델을 학습시킨다. \
    #             이때, 레이블이 없는 데이터를 사용하여 모델을 학습시킬 때, 모델이 가장 확신하는 데이터만 사용한다.')
    # st.write(' ')
    # st.write('현재 Train X 데이터의 현황 :')
    # st.dataframe(numeric_train)
    # st.write('현재 Train Y 데이터의 현황 :')
    # st.write('총 '+ str(len(y_train)) + '개의 데이터 중 레이블이 존재하는 데이터는 ' + str(y_train.notnull().sum()), '개의 이며, 레이블이 없는 데이터는' + str(y_train.isnull().sum()) +  '개이다.')

    # st.markdown('만약 모든 데이터에 레이블이 있다면 일부 레이블을 제거합니다.')
    # remove = st.checkbox('일부 데이터의 레이블을 제거합니다.')
    # if remove:
    #     unlabeled_y = y_train.copy()
    #     numberinput = st.number_input('제거할 비율 입력', value = 10)
    #     random_ind = np.random.choice(y_train.index, size=int(len(y_train) / 100 * int(numberinput)), replace=False)
    #     unlabeled_y.loc[random_ind] = np.nan
    #     st.write('현재 Train Y 데이터의 현황 :')
    #     st.write('총 ' + str(len(unlabeled_y)) + '개의 데이터 중 레이블이 존재하는 데이터는 ' + str(unlabeled_y.notnull().sum()) + '개의 이며, 레이블이 없는 데이터는 ' + str(unlabeled_y.isnull().sum()) + '개이다.')

    #     labeled_x = numeric_train[y_train.notnull()]
    #     labeled_y = y_train[y_train.notnull()]
    #     st.session_state['labeled_x_state'] = labeled_x.copy()
    #     st.session_state['labeled_y_state'] = labeled_y.copy()
    #     st.session_state['unlabeld_y_state'] = unlabeled_y.copy()
    #     st.session_state['unlabeld_x_state'] = numeric_train[unlabeled_y.isnull()]


    # st.write('')
    # st.markdown('----')

    # st.markdown('## Step 1 : 레이블이 있는 데이터를 사용하여 모델을 학습시킨다.')
    # st.write('')
    # st.markdown('레이블이 있는 데이터만 사용하여 모델을 학습시킨다.')

    # if st.checkbox('모델 학습'):


    #     classifier = LogisticRegression()
    #     classifier.fit(labeled_x, labeled_y)

    #     # Self Training
    #     st.markdown('## Step 2 : 레이블이 없는 데이터를 예측하고 레이블을 부여한다.')
    #     st.write('')

    #     st_repeat = st.number_input('반복 횟수를 입력하세요.', value=1)
    #     st.markdown('레이블이 없는 데이터를 예측하고, 예측값이 주어진 threshold를 넘으면 레이블을 부여한다.')
    #     up_threshold = st.text_input('상위 Threshold 값을 입력하세요.', value=0.95)
    #     up_threshold = float(up_threshold)

    #     down_threshold = st.text_input('하위 Threshold 값을 입력하세요.', value=0.15)
    #     down_threshold = float(down_threshold)
    #     if unlabeled_y is None:
    #         st.warning('레이블이 없는 데이터가 없습니다.')
    #         st.stop()
    #     else:
    #         unlabeled_x = st.session_state['unlabeld_x_state']
    #         unlabeled_y = st.session_state['unlabeld_y_state']


    #         if st.button('다음 반복 진행'):
    #             st.write("===== Iteration =====")
    #             st.write("현재 Train Y 데이터의 현황:")
    #             st.write("총 " + str(len(labeled_y)) + "개의 데이터 중 레이블이 존재하는 데이터는 " + str(len(labeled_y)) + "개의 이며, 레이블이 없는 데이터는 " + str(len(y_train) - len(labeled_y)) + "개이다.")

    #             if unlabeled_x.empty:
    #                 st.write('모든 데이터에 레이블이 부여되었습니다.')
    #             else:
    #                 unlabeled_x = st.session_state['unlabeld_x_state']
    #                 unlabeled_pred = classifier.predict_proba(unlabeled_x)
    #                 st.write(unlabeled_pred)

    #                 # Assign labels based on probabilities and threshold
    #                 labeled_mask = (unlabeled_pred[:, 1] >= up_threshold) & (unlabeled_pred[:, 1] <= down_threshold)
    #                 labeled_y_new = np.where(labeled_mask, unlabeled_pred[:, 1], labeled_y)

    #                 labeled_x = pd.concat([labeled_x, unlabeled_x[labeled_mask]])
    #                 labeled_y = pd.concat([labeled_y, pd.Series(labeled_y_new)])

    #                 classifier.fit(labeled_x, labeled_y)

    #                 # 마지막 반복 후, 업데이트된 상태를 session_state에 저장
    #                 st.session_state['labeled_x_state'] = labeled_x.copy()
    #                 st.session_state['labeled_y_state'] = labeled_y.copy()



    st.markdown('----')

    st.markdown('정상 상태의 데이터만 사용해서 모델을 학습합니다.')
    st.markdown('현재 Y 데이터 : ')
    st.write(y_train)
    normal = normal_value
    
    normal_train_y = y_train[y_train == normal].index
    nomal_train_x = numeric_train.loc[normal_train_y]
    st.write('현재 Train X 데이터의 현황 :')
    st.dataframe(nomal_train_x)
    
    st.write('총 '+ str(len(y_train)) +'개의 train 데이터 중 ' + str(len(normal_train_y == normal)) + '개의 정상 데이터를 사용해서 학습합니다.')
    st.write('예측할 데이터의 개수는 ' + str(len(numeric_test)) + '개 입니다.')
    st.write('만약 train data에 정상 데이터 개수가 적다면, main page에서 데이터의 비율을 늘려주세요.')

    st.markdown('----')
    st.markdown('## 1. OCSVM(One-Class SVM)')
    with st.expander('OCSVM 설명'):
        st.markdown('OCSVM은 이상치 탐지, 특이치 탐지, 새로운 패턴 발견 등 다양한 분야에서 사용할 수 있습니다.')
        st.markdown('SVM의 변형으로, 최종 목표는 정상 데이터와 이상 데이터를 분류하는 hyperplane을 찾는 것입니다.')
        st.markdown('목적함수는 아래 식과 같습니다.')
        images('./images/ocsvm2.png')
        st.markdown('첫번쨰로 w의 역할은 regularization을 수행합니다. 같은 값이면, x의 변화에 따라 변동성을 작게 하는 것 입니다.')
        st.markdown('두번쨰로 이상치 영역에 존재하는 데이터들에게 패널티를 부여하는 것입니다.')
        st.markdown('여기서 nu는 이상치의 비율을 의미합니다.')
        st.markdown('앱실론은 이상치와의 거리를 의미합니다.')
        st.markdown('초평면보다 원점에 가깝게 위치한다면 앱실론의 값은 양수가 되며, 초평면보다 우너점에서 멀리 위치한다면 0이된다.')
        st.markdown('즉 이상치에 대한 패널티를 의미하게 됩니다.')
        st.markdown('세번째로 로우는 원점과 초평면의 거리를 의미합니다.')
        st.markdown('즉 이상치와 정상치의 경계를 의미하게 됩니다.')
        images('./images/ocsvm3.png')
        st.markdown('앞서 설명한 3개의 식을 종합하면 의미는 위와 같습니다.')
        st.markdown('우리는 여기서 데이터를 통해 w와 로우를 구하게 되고, nu를 입력으로 받게 됩니다.')
        st.markdown('그림으로 표현하면 아래와 같습니다.')
        images('./images/ocsvm1.png')

        st.markdown(' ')
        st.markdown('맨 처음 보여준 목적함수에 제약조건이 존재합니다.')
        st.markdown('제약조건의 형태가 등호가 아닌, 부등호입니다.')
        st.markdown('이러한 제약조건을 목적함수에 적용하기 위해 우리는 KKT(Karush-Kuhn-Tucker) 조건을 사용합니다.')
        st.markdown('위의 내용은 아래 링크를 적어두겠습니다. 한번 참고하시는 것이 좋을것 같습니다.')
        st.markdown('https://losskatsu.github.io/machine-learning/oneclass-svm/#2-one-class-svm%EC%9D%98-%EB%AA%A9%EC%A0%81')
        st.markdown('https://limitsinx.tistory.com/147')
        st.markdown(' ')
        st.markdown('아래 링크는 ocsvm, svdd에 관한 DSBA 강의입니다.')
        st.markdown('https://www.youtube.com/watch?v=OmK_GQ40yko')
        st.markdown(' ')
        st.markdown('OCSVM 논문 다운로드 :')
        file_path_ocsvm = "./paper/SVDD_SVM.pdf"
        file_name_ocsvm = "SVDD_SVM.pdf"
        download_pdf_file(file_path_ocsvm, file_name_ocsvm, 'OCSVM 논문 다운로드')


    oneclss = OneClassSVM()
    st.markdown('### Hyperparameter Tuning')
    kernel = st.selectbox('커널을 선택하세요', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                          help = '커널의 종류를 선택')
    nu = st.number_input('nu 값을 입력하세요', value=0.5, help = '이상치의 비율을 제어하는 파라미터, nu 값이 작을수록 더 많은 데이터를 이상치로 분류')
    gamma = st.number_input('gamma 값을 입력하세요', value=0.1, help = '데이터 포인트 간의 영향력을 조절하는 파라미터, gamma 값이 클수록 복잡한 결정 경계, 작을수록 부드러운 결정 경계 생성')
    degree = st.number_input('degree 값을 입력하세요', value=3, help = '다항식 커널의 차수, degree 값이 클수록 복잡한 결정 경계, 작을수록 단순한 결정 경계 생성')
    
    if st.checkbox('One-Class SVM 모델 학습'):
        
        oneclss.fit(nomal_train_x)
        svm_predict = oneclss.predict(numeric_test)
        svm_proba = oneclss.decision_function(numeric_test)

        if normal == 0:
            y_change = np.where(y_test, 1, -1)
        else:
            y_change = y_test

        if st.checkbox('One-Class SVM 결과 확인'):
            st.write('accuracy_score : ', accuracy_score(y_change, svm_predict))
            st.write('Classification Report : ' )
            st.code('Model Report:\n  ' + classification_report(y_change, svm_predict))
            st.write('confusion matrix : ')
            st.write(confusion_matrix(y_change, svm_predict))

            fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_change, svm_proba)
            st.write('AUC : ', auc(fpr_svm, tpr_svm))
            fig_svm = px.area(x=fpr_svm, y=tpr_svm,  title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'))
            fig_svm.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_svm.update_xaxes(constrain='domain')
            fig_svm.update_yaxes(scaleanchor="x", scaleratio=1)
        
            st.plotly_chart(fig_svm, width = 1000, height = 1000)
        
        if st.button('Result Download'):
            total_result_data = pd.DataFrame({'real':y_change, 'pred':svm_predict})
            total_result_data.to_csv('total_result_data.csv', index=False)

            csv_link = download_csv(total_result_data, 'OCSVM_Result.csv')
            st.markdown(csv_link, unsafe_allow_html=True)
            
            
            
    st.markdown('----')
    st.markdown('## 2. Deep SVDD')
    with st.expander('Deep-SVDD 설명'):
        st.markdown('Deep SVDD와 SVDD는 거의 동일한 알고리즘입니다.')
        st.markdown('다만 hypersphere를 구하는 feature space로 변환시에 차이점이 존재합니다.')
        st.markdown('SVDD는 kernel trick을 사용해서 고차원의 feature space에 매핑한 후, hypersphere를 구합니다.')
        images('./images/svdd.png')
        st.markdown('사진으로 설명하면 위와 같습니다.')
        st.markdown(' ')

        st.markdown('반면 Deep-SVDD의 경우, neural network를 사용해서 낮은 차원의 latent space에 매핑한 후, hypersphere를 구합니다.')
        images('./images/deepsvdd.png')
        st.markdown('사진으로 설명하면 위와 같습니다.')
        st.markdown(' ')

        st.markdown('Deep-SVDD는 neural network를 사용하기 때문에 최적의 feature를 자동으로 학습하게 됩니다.')
        st.markdown('비선형적인 고차원데이터와 같이 더욱 더 복잡한 데이터 구조에 대해서 모델링이 가능하다는 장점이 있습니다.')
        st.markdown('목적 함수와 관련된 모델링 부분은 SVDD와 거의 동일합니다.')
        st.markdown('다만 C(원점의 중심)을 미리 정해주어야 한다는 것과, hypershphere에서 약간의 차이점이 존재합니다.')
        st.markdown('해당 부분은 아래 URL을 통해서 학습하시는 것을 추천드립니다.')
        st.markdown(' ')
        st.markdown('DSBA SVDD 강의 URL :')
        st.markdown('https://www.youtube.com/watch?v=OmK_GQ40yko')
        st.markdown(' ')
        st.markdown('Deep SVDD 관련 자료 URL :')
        
        st.markdown('https://ffighting.net/deep-learning-paper-review/anomaly-detection/deep-svdd/')
        st.markdown('http://dsba.korea.ac.kr/seminar/?mod=document&uid=1327')
        st.markdown(' ')
        st.markdown('Deep-SVDD 논문 다운로드 :')
        file_path_deepsvm = "./paper/deepsvdd.pdf"
        file_name_deepsvm = "deepsvdd.pdf"
        download_pdf_file(file_path_deepsvm, file_name_deepsvm, 'Deep-SVDD 논문 다운로드')

    st.markdown('### Hyperparameter Tuning')
    hidden_activation = st.selectbox('hidden_activation을 선택하세요', ['relu', 'sigmoid', 'tanh'], help = '은닉층에서 사용되는 활성화 함수')
    output_activation = st.selectbox('output_activation을 선택하세요', ['sigmoid', 'linear'], help = '출력층에서 사용되는 활성화 함수')
    optimizer = st.selectbox('optimizer를 선택하세요', ['adam', 'sgd', 'rmsprop'], help = '가중치를 업데이트 하는데 사용되며, 손실 함수를 최소화하는 가중치를 찾음')
    epochs = st.number_input('epochs 값을 입력하세요', value=100, step = 50, help = '학습을 위해 데이터 셋을 반복하는 횟수')
    batch_size = st.number_input('batch_size 값을 입력하세요', value=32, help = '미니배치 크기, 배치 크기에 따라 학습 속도가 달라짐')


    if st.checkbox('Deep-SVDD 예측'):
        
        deep_predict, deep_proba = deepsvdd(numeric_train,numeric_test, hidden_activation= hidden_activation, 
                           output_activation= output_activation, optimizer= optimizer,
                           epochs = epochs, batch_size= batch_size)
        
        if normal_value == 1:
            y_change = np.where(y_test == 1, 1, 0)
        else:
            y_change = y_test  
        if st.checkbox('Deep-SVDD 결과 확인'):
            st.write('accuracy_score : ', accuracy_score(y_change, deep_predict))
            st.write(' ')
            st.markdown('Classification Report : ')
            st.code('Model Report:\n  ' + classification_report(y_change, deep_predict))
            st.write(' ')
            st.write('confusion matrix : ')
            st.write(confusion_matrix(y_change, deep_predict))
        
            deep_fpr, deep_tpr, thresholds = roc_curve(y_change, deep_proba[:,1])
            st.write('AUC: ', auc(deep_fpr, deep_tpr))
            fig_SVDD_roc = px.area(x = deep_fpr, y = deep_tpr,  title='ROC Curve', labels=dict(x='False Positive Rate', y='True Positive Rate'), width=1000, height=600)
            fig_SVDD_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig_SVDD_roc.update_xaxes(constrain='domain')
            fig_SVDD_roc.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig_SVDD_roc)

        if st.button('Result Download'):
            total_result_data = pd.DataFrame({'real':y_change, 'pred':deep_predict})
            total_result_data.to_csv('total_result_data.csv', index=False)

            csv_link = download_csv(total_result_data, 'Deep_SVDD_Result.csv')
            st.markdown(csv_link, unsafe_allow_html=True)

    st.markdown('----')

    with st.expander('추가적으로 공부할 것들'):
        st.markdown('SVM(Support Vector Machine)에 관한 내용을 보면 Margin을 최대화 하는 것이 중요하다고 합니다.')
        st.markdown('위에서 SVM의 변형된 형태인 OCSVM, SVDD, Deep-SVDD를 보면 각 hyperplane, hypersphere에서 정상 데이터와 비정상 데이터의 거리를 최대화 하는 것이 중요하다고 합니다.')
        st.markdown('이러한 점에서 SVM과 유사합니다. 하지만 ANN에서 보면 동일하게 데이터를 구분하는 초평면을 찾는 것이지만 margin을 최대화 하지 않습니다.')
        st.markdown('하지만 때에 따라 ann의 성능이 더 좋게 나옵니다. 물론 데이터에 따라서 다르지만 이론상으로만 본다면 margin을 최대화 하는 SVM의 구조가 더 좋아 보입니다.')
        st.markdown('margin을 최대화 한다는 것은 일반화 성능을 향상시키는 것을 의미하기 때문입니다.')
        st.markdown('ANN에서는 loss function을 최소화 하는 방향으로 학습을 진행하기 때문에 일반화 성능을 보장하지 않고, under fitting, over fitting이 발생할 수도 있지 않을까 라는 생각을 해보셨나요?')
        st.markdown('실제로, ANN이 처음 등장했을 때, 위의 내용 때문에 비판을 받은 적이 있습니다.')
        st.markdown('사진으로 확인하면 아래와 같은 논리입니다.')
        images('./images/margin_2.jpg')
        st.markdown('SVM에서는 오른쪽 그림과 같이 가장 넓게 margin을 확보하는 초평면을 찾습니다.')
        st.markdown('ANN에서도 오른쪽 그림과 같은 평면을 찾는것이 global optimal solution이라고 생각합니다.')
        st.markdown('그리고 오른쪽의 수많은 그림도 동일하게 오차는 0이지만, 일반화 성능이 좋지 않습니다.')
        st.markdown('이러한 점을 local optimal 지점이라고 생각을 합니다.')
        st.markdown('loss function으로 표현하면 아래와 같습니다.')
        images('./images/margin_3.jpg')
        st.markdown('그래서 많은 학자들이 위의 내용을 반박하기 위한 연구를 많이 진행했습니다.')
        st.markdown('그 중 하나를 소개해드릴려고 합니다.')
        st.markdown('해당 논문에서는 안장점으로 예시를 드는데 안장점에서 한 변수에 대해서 gradient는 0이 되는 지점이 있지만, 모든 변수의 방향으로 편미분 했을때 0이 되는 지점은 거의 없다는 것입니다.')
        st.markdown('그렇기 때문에 local optimal에 빠지지 않고, global optimal에 도달할 수 있다는 것입니다.')
        st.markdown('해당 내용은 개인적인 주관과 논문을 읽고 이해한 내용이라서 틀린 부분이 있을 수 있습니다.')
        st.markdown('해당 논문을 아래 첨부해두었습니다.')
        st.markdown('한번정도 읽어 보시는 것도 좋을 것 같습니다.')
        st.markdown('논문 제목은 다음과 같습니다. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization')

        file_path_margin = './paper/Identifying and attacking the saddle point problem in high-dimensional non-convex optimization.pdf'
        file_name_margin = "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization.pdf"
        download_pdf_file(file_path_margin, file_name_margin, '논문 다운로드')

            
            